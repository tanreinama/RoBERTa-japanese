import argparse
import json
import os
import numpy as np
import tensorflow as tf
import time
import tqdm
from tensorflow.core.protobuf import rewriter_config_pb2

from model import modeling
from model.modeling import BertConfig, BertModel
from run_finetune import get_masked_lm_output,get_next_sentence_output

from encode_bpe import BPEEncoder_ja

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='RoBERTa-ja_small', help='pretrained model directory.')
parser.add_argument('--input_dir', type=str, required=True, help='input texts.')
parser.add_argument('--train_by_line', action='store_true', help='split file by lines.')
parser.add_argument('--batch_size', type=int, default=1, help='training batch size.')
parser.add_argument('--run_name', type=str, default='RoBERTa-ja_classifier', help='save model name.')
parser.add_argument('--save_every', type=int, default=2, help='save every N epoch.')
parser.add_argument('--num_epochs', type=int, default=20, help='training epochs.')
parser.add_argument('--learning_rate', type=float, default=0.00001, help='learning rate.')
parser.add_argument('--gpu', default='0', help='visible gpu number.')

CHECKPOINT_DIR = 'checkpoint'

def maketree(path):
    try:
        os.makedirs(path)
    except:
        pass

def main():
    args = parser.parse_args()

    if os.path.isfile(args.model+'/hparams.json'):
        with open(args.model+'/hparams.json') as f:
            bert_config_params = json.load(f)
    else:
        raise ValueError('invalid model name.')

    vocab_size = bert_config_params['vocab_size']
    max_seq_length = bert_config_params['max_position_embeddings']
    batch_size = args.batch_size
    save_every = args.save_every
    num_epochs = args.num_epochs
    EOT_TOKEN = vocab_size - 4
    MASK_TOKEN = vocab_size - 3
    CLS_TOKEN = vocab_size - 2
    SEP_TOKEN = vocab_size - 1

    with open('ja-bpe.txt') as f:
        bpe = f.read().split('\n')

    with open('emoji.json') as f:
        emoji = json.loads(f.read())

    enc = BPEEncoder_ja(bpe, emoji)

    keys = [f for f in os.listdir(args.input_dir) if os.path.isdir(args.input_dir+'/'+f)]
    keys = sorted(keys)
    num_labels = len(keys)
    input_contexts = []
    input_keys = []
    idmapping_dict = {}
    for i,f in enumerate(keys):
        n = 0
        for t in os.listdir(f'{args.input_dir}/{f}'):
            if os.path.isfile(f'{args.input_dir}/{f}/{t}'):
                with open(f'{args.input_dir}/{f}/{t}') as fn:
                    if args.train_by_line:
                        for p in fn.readlines():
                            tokens = enc.encode(p.strip())[:max_seq_length-2]
                            tokens = [CLS_TOKEN]+tokens+[SEP_TOKEN]
                            if len(tokens) < max_seq_length:
                                tokens.extend([0]*(max_seq_length-len(tokens)))
                            input_contexts.append(tokens)
                            input_keys.append(i)
                            n += 1
                    else:
                        p = fn.read()
                        tokens = enc.encode(p.strip())[:max_seq_length-3]
                        tokens = [CLS_TOKEN]+tokens+[EOT_TOKEN,SEP_TOKEN]
                        if len(tokens) < max_seq_length:
                            tokens.extend([0]*(max_seq_length-len(tokens)))
                        input_contexts.append(tokens)
                        input_keys.append(i)
                        n += 1
        print(f'{args.input_dir}/{f} mapped for id_{i}, read {n} contexts.')
        idmapping_dict[f] = i
    input_indexs = np.random.permutation(len(input_contexts))

    bert_config = BertConfig(**bert_config_params)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.visible_device_list = args.gpu

    with tf.Session(config=config) as sess:
        input_ids = tf.placeholder(tf.int32, [None, None])
        input_mask = tf.placeholder(tf.int32, [None, None])
        segment_ids = tf.placeholder(tf.int32, [None, None])
        masked_lm_positions = tf.placeholder(tf.int32, [None, None])
        masked_lm_ids = tf.placeholder(tf.int32, [None, None])
        masked_lm_weights = tf.placeholder(tf.float32, [None, None])
        next_sentence_labels = tf.placeholder(tf.int32, [None])

        model = BertModel(
            config=bert_config,
            is_training=True,
            input_ids=input_ids,
            input_mask=input_mask,
            token_type_ids=segment_ids,
            use_one_hot_embeddings=False)

        output = model.get_sequence_output()
        (_,_,_) = get_masked_lm_output(
             bert_config, model.get_sequence_output(), model.get_embedding_table(),
             masked_lm_positions, masked_lm_ids, masked_lm_weights)
        (_,_,_) = get_next_sentence_output(
             bert_config, model.get_pooled_output(), next_sentence_labels)

        saver = tf.train.Saver()
        ckpt = tf.train.latest_checkpoint(args.model)
        saver.restore(sess, ckpt)
        train_vars = tf.trainable_variables()
        restored_weights = {}
        for i in range(len(train_vars)):
            restored_weights[train_vars[i].name] = sess.run(train_vars[i])

        labels = tf.placeholder(tf.int32, [None, ])

        output_layer = model.get_pooled_output()

        hidden_size = output_layer.shape[-1].value

        output_weights = tf.get_variable(
          "output_weights", [num_labels, hidden_size],
          initializer=tf.truncated_normal_initializer(stddev=0.02))

        output_bias = tf.get_variable(
          "output_bias", [num_labels], initializer=tf.zeros_initializer())

        with tf.variable_scope("loss"):
            output_layer = tf.nn.dropout(output_layer, keep_prob=0.9)
            logits = tf.matmul(output_layer, output_weights, transpose_b=True)
            logits = tf.nn.bias_add(logits, output_bias)
            probabilities = tf.nn.softmax(logits, axis=-1)
            log_probs = tf.nn.log_softmax(logits, axis=-1)

            one_hot_labels = tf.one_hot(labels, depth=num_labels, dtype=tf.float32)

            per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs, axis=-1)
            loss = tf.reduce_mean(per_example_loss)

            opt = tf.train.AdamOptimizer(learning_rate=args.learning_rate)
            train_vars = tf.trainable_variables()
            opt_grads = tf.gradients(loss, train_vars)
            opt_grads = list(zip(opt_grads, train_vars))
            opt_apply = opt.apply_gradients(opt_grads)
            summaries = tf.summary.scalar('loss', loss)
            summary_log = tf.summary.FileWriter(
                os.path.join(CHECKPOINT_DIR, args.run_name))

            counter = 1
            counter_path = os.path.join(CHECKPOINT_DIR, args.run_name, 'counter')
            if os.path.exists(counter_path):
                # Load the step number if we're resuming a run
                # Add 1 so we don't immediately try to save again
                with open(counter_path, 'r') as fp:
                    counter = int(fp.read()) + 1

            hparams_path = os.path.join(CHECKPOINT_DIR, args.run_name, 'hparams.json')
            maketree(os.path.join(CHECKPOINT_DIR, args.run_name))
            with open(hparams_path, 'w') as fp:
                fp.write(json.dumps(bert_config_params))
            idmaps_path = os.path.join(CHECKPOINT_DIR, args.run_name, 'idmaps.json')
            with open(idmaps_path, 'w') as fp:
                fp.write(json.dumps(idmapping_dict))

            sess.run(tf.global_variables_initializer()) # init output_weights
            restored = 0
            for k,v in restored_weights.items():
                for i in range(len(train_vars)):
                    if train_vars[i].name == k:
                        assign_op = train_vars[i].assign(v)
                        sess.run(assign_op)
                        restored += 1
            assert restored == len(restored_weights), 'fail to restore model.'
            saver = tf.train.Saver(var_list=tf.trainable_variables())

            def save():
                maketree(os.path.join(CHECKPOINT_DIR, args.run_name))
                print(
                    'Saving',
                    os.path.join(CHECKPOINT_DIR, args.run_name,
                                 'model-{}').format(counter))
                saver.save(
                    sess,
                    os.path.join(CHECKPOINT_DIR, args.run_name, 'model'),
                    global_step=counter)
                with open(counter_path, 'w') as fp:
                    fp.write(str(counter) + '\n')
            avg_loss = (0.0, 0.0)
            start_time = time.time()

            def sample_feature(i):
                last = min((i+1)*batch_size,len(input_indexs))
                _input_ids = [input_contexts[idx] for idx in input_indexs[i*batch_size:last]]
                _input_masks = [[1]*len(input_contexts[idx])+[0]*(max_seq_length-len(input_contexts[idx])) for idx in input_indexs[i*batch_size:last]]
                _segments = [[1]*len(input_contexts[idx])+[0]*(max_seq_length-len(input_contexts[idx])) for idx in input_indexs[i*batch_size:last]]
                _labels = [input_keys[idx] for idx in input_indexs[i*batch_size:last]]
                return {
                    input_ids:_input_ids,
                    input_mask:_input_masks,
                    segment_ids:_segments,
                    masked_lm_positions:np.zeros((len(_input_ids),0), dtype=np.int32),
                    masked_lm_ids:np.zeros((len(_input_ids),0), dtype=np.int32),
                    masked_lm_weights:np.ones((len(_input_ids),0), dtype=np.float32),
                    next_sentence_labels:np.zeros((len(_input_ids),), dtype=np.int32),
                    labels:_labels
                }

            try:
                for ep in range(num_epochs):
                    if ep % args.save_every == 0:
                        save()

                    prog = tqdm.tqdm(range(0,len(input_contexts)//batch_size,1))
                    for i in prog:
                        (_, v_loss, v_summary) = sess.run(
                            (opt_apply, loss, summaries),
                            feed_dict=sample_feature(i))

                        summary_log.add_summary(v_summary, counter)

                        avg_loss = (avg_loss[0] * 0.99 + v_loss,
                                    avg_loss[1] * 0.99 + 1.0)

                        prog.set_description(
                            '[{ep} | {time:2.2f}] loss={loss:2.2f} avg={avg:2.2f}'
                            .format(
                                ep=ep,
                                time=time.time() - start_time,
                                loss=v_loss,
                                avg=avg_loss[0] / avg_loss[1]))

                        counter += 1
            except KeyboardInterrupt:
                print('interrupted')
                save()

            save()


if __name__ == '__main__':
    main()
