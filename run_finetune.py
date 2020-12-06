# coding=utf-8
# BERT-Japanease Pretrain
#
#  This program pre-trains the Japanese version of BERT.
#  As for the learning method, the following FEATUREs were
#  added from the original BERT with reference to RoBERTa.
#    - dynamic mask
#    - not use NSP
#    - FULL-SENTENCES trainig
#    - optimize batch size and learning ratio
#  Use Japanese BPE Encoder for word-separation and encoding.
#
# REFERENCEs:
#   BERT(original): https://github.com/google-research/bert
#   Japanese BPE Encoder: https://github.com/tanreinama/Japanese-BPEEncoder
#   RoBERTa: https://arxiv.org/pdf/1907.11692.pdf
#
# This software includes the work that is distributed in the Apache License 2.0

import argparse
import json
import os
import numpy as np
import tensorflow as tf
import time
import tqdm
from copy import copy
from tensorflow.core.protobuf import rewriter_config_pb2

from model import modeling
from model.modeling import BertConfig, BertModel

CHECKPOINT_DIR = 'checkpoint'
SAMPLE_DIR = 'samples'

parser = argparse.ArgumentParser(
    description='Pretraining BERT(RoBERTa) on your custom dataset.',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--dataset', metavar='PATH', type=str, required=True, help='Input npz file')
parser.add_argument('--base_model', type=str, default='RoBERTa-ja_small', help='a path to a model file')

parser.add_argument('--batch_size', metavar='SIZE', type=int, default=1, help='Batch size')
parser.add_argument('--optim', type=str, default='adam', help='"adam", "adagrad", or "sgd" to use optimizer')
parser.add_argument('--learning_rate', metavar='LR', type=float, default=5e-5, help='Learning rate for optimizer')
parser.add_argument('--warmup_steps', metavar='WR', type=int, default=0, help='Learning rate warming up steps')

parser.add_argument('--run_name', type=str, default='RoBERTa-ja_finetune', help='Run id. Name of subdirectory in checkpoint/ and samples/')
parser.add_argument('--save_every', metavar='N', type=int, default=1000, help='Write a checkpoint every N steps')
parser.add_argument('--max_predictions_per_seq', default=160, type=int, help='Maximum number of mask words')

parser.add_argument('--gpu', default='0', help='visible gpu number.')

def maketree(path):
    try:
        os.makedirs(path)
    except:
        pass

def get_masked_lm_output(bert_config, input_tensor, output_weights, positions,
                         label_ids, label_weights):
    """Get loss and log probs for the masked LM."""
    input_tensor = gather_indexes(input_tensor, positions)

    with tf.variable_scope("cls/predictions"):
        # We apply one more non-linear transformation before the output layer.
        # This matrix is not used after pre-training.
        with tf.variable_scope("transform"):
            input_tensor = tf.layers.dense(
                    input_tensor,
                    units=bert_config.hidden_size,
                    activation=modeling.get_activation(bert_config.hidden_act),
                    kernel_initializer=modeling.create_initializer(
                            bert_config.initializer_range))
            input_tensor = modeling.layer_norm(input_tensor)

        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        output_bias = tf.get_variable(
                "output_bias",
                shape=[bert_config.vocab_size],
                initializer=tf.zeros_initializer())
        logits = tf.matmul(input_tensor, output_weights, transpose_b=True)
        logits = tf.nn.bias_add(logits, output_bias)
        log_probs = tf.nn.log_softmax(logits, axis=-1)

        label_ids = tf.reshape(label_ids, [-1])
        label_weights = tf.reshape(label_weights, [-1])

        one_hot_labels = tf.one_hot(
                label_ids, depth=bert_config.vocab_size, dtype=tf.float32)

        # The `positions` tensor might be zero-padded (if the sequence is too
        # short to have the maximum number of predictions). The `label_weights`
        # tensor has a value of 1.0 for every real prediction and 0.0 for the
        # padding predictions.
        per_example_loss = -tf.reduce_sum(log_probs * one_hot_labels, axis=[-1])
        numerator = tf.reduce_sum(label_weights * per_example_loss)
        denominator = tf.reduce_sum(label_weights) + 1e-5
        loss = numerator / denominator

    return (loss, per_example_loss, log_probs)


def get_next_sentence_output(bert_config, input_tensor, labels):
    """Get loss and log probs for the next sentence prediction."""

    # Simple binary classification. Note that 0 is "next sentence" and 1 is
    # "random sentence". This weight matrix is not used after pre-training.
    with tf.variable_scope("cls/seq_relationship"):
        output_weights = tf.get_variable(
                "output_weights",
                shape=[2, bert_config.hidden_size],
                initializer=modeling.create_initializer(bert_config.initializer_range))
        output_bias = tf.get_variable(
                "output_bias", shape=[2], initializer=tf.zeros_initializer())

        logits = tf.matmul(input_tensor, output_weights, transpose_b=True)
        logits = tf.nn.bias_add(logits, output_bias)
        log_probs = tf.nn.log_softmax(logits, axis=-1)
        labels = tf.reshape(labels, [-1])
        one_hot_labels = tf.one_hot(labels, depth=2, dtype=tf.float32)
        per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs, axis=-1)
        loss = tf.reduce_mean(per_example_loss)
        return (loss, per_example_loss, log_probs)


def gather_indexes(sequence_tensor, positions):
    """Gathers the vectors at the specific positions over a minibatch."""
    sequence_shape = modeling.get_shape_list(sequence_tensor, expected_rank=3)
    batch_size = sequence_shape[0]
    seq_length = sequence_shape[1]
    width = sequence_shape[2]

    flat_offsets = tf.reshape(
            tf.range(0, batch_size, dtype=tf.int32) * seq_length, [-1, 1])
    flat_positions = tf.reshape(positions + flat_offsets, [-1])
    flat_sequence_tensor = tf.reshape(sequence_tensor,
                                                                        [batch_size * seq_length, width])
    output_tensor = tf.gather(flat_sequence_tensor, flat_positions)
    return output_tensor


def main():
    args = parser.parse_args()

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.visible_device_list = args.gpu
    config.graph_options.rewrite_options.layout_optimizer = rewriter_config_pb2.RewriterConfig.OFF

    vocab_size = 20573 + 3 # [MASK] [CLS] [SEP]
    EOT_TOKEN = vocab_size - 4
    MASK_TOKEN = vocab_size - 3
    CLS_TOKEN = vocab_size - 2
    SEP_TOKEN = vocab_size - 1
    max_predictions_per_seq = args.max_predictions_per_seq
    batch_size = args.batch_size

    with tf.Session(config=config) as sess:
        input_ids = tf.placeholder(tf.int32, [batch_size, None])
        input_mask = tf.placeholder(tf.int32, [batch_size, None])
        segment_ids = tf.placeholder(tf.int32, [batch_size, None])
        masked_lm_positions = tf.placeholder(tf.int32, [batch_size, None])
        masked_lm_ids = tf.placeholder(tf.int32, [batch_size, None])
        masked_lm_weights = tf.placeholder(tf.float32, [batch_size, None])
        next_sentence_labels = tf.placeholder(tf.int32, [None])

        if os.path.isfile(args.base_model+'/hparams.json'):
            with open(args.base_model+'/hparams.json') as f:
                bert_config_params = json.loads(f.read())
        else:
            raise ValueError('invalid model name.')

        max_seq_length = bert_config_params['max_position_embeddings']

        bert_config = BertConfig(**bert_config_params)
        model = BertModel(
            config=bert_config,
            is_training=True,
            input_ids=input_ids,
            input_mask=input_mask,
            use_one_hot_embeddings=False)

        (masked_lm_loss,_,_) = get_masked_lm_output(
             bert_config, model.get_sequence_output(), model.get_embedding_table(),
             masked_lm_positions, masked_lm_ids, masked_lm_weights)
        (next_sentence_loss,_,_) = get_next_sentence_output(
             bert_config, model.get_pooled_output(), next_sentence_labels)

        loss = masked_lm_loss + next_sentence_loss

        train_vars = tf.trainable_variables()

        global_step = tf.Variable(0, trainable=False)
        if args.warmup_steps > 0:
            learning_rate = tf.compat.v1.train.polynomial_decay(
                    learning_rate=1e-10,
                    end_learning_rate=args.learning_rate,
                    global_step=global_step,
                    decay_steps=args.warmup_steps
                )
        else:
            learning_rate = args.learning_rate

        if args.optim=='adam':
            opt = tf.train.AdamOptimizer(learning_rate=learning_rate,
                                           beta1=0.9,
                                           beta2=0.98,
                                           epsilon=1e-7)
        elif args.optim=='adagrad':
            opt = tf.train.AdagradOptimizer(learning_rate=learning_rate)
        elif args.optim=='sgd':
            opt = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
        else:
            raise ValueError('invalid optimizer name.')

        train_vars = tf.trainable_variables()
        opt_grads = tf.gradients(loss, train_vars)
        opt_grads = list(zip(opt_grads, train_vars))
        opt_apply = opt.apply_gradients(opt_grads)

        summaries = tf.summary.scalar('loss', loss)
        summary_log = tf.summary.FileWriter(
            os.path.join(CHECKPOINT_DIR, args.run_name))

        saver = tf.train.Saver(
            var_list=train_vars,
            max_to_keep=5,
            keep_checkpoint_every_n_hours=2)
        sess.run(tf.global_variables_initializer())

        ckpt = tf.train.latest_checkpoint(args.base_model)
        saver.restore(sess, ckpt)
        print('Loading checkpoint', ckpt)

        print('Loading dataset...')
        global_chunks = []
        with np.load(args.dataset) as npz:
            for inditem, item in enumerate(npz.files):
                token_chunk = npz[item]
                current_token = [np.uint16(CLS_TOKEN)]
                for ind in range(0,len(token_chunk)):
                  current_token.append(np.uint16(token_chunk[ind]))
                  if len(current_token) == max_seq_length:
                      global_chunks.append(current_token)
                      current_token = [np.uint16(CLS_TOKEN)]
                  if token_chunk[ind]==EOT_TOKEN:
                      current_token.append(np.uint16(SEP_TOKEN))
                  if len(current_token) == max_seq_length:
                      global_chunks.append(current_token)
                      current_token = [np.uint16(CLS_TOKEN)]
        global_chunk_index = np.random.permutation(len(global_chunks))
        global_chunk_step = 0
        print('Training...')

        def sample_feature():
            nonlocal global_chunks,global_chunk_index,global_chunk_step
            # Use dynamic mask
            p_input_ids = []
            p_input_mask = []
            p_segment_ids = []
            p_masked_lm_positions = []
            p_masked_lm_ids = []
            p_masked_lm_weights = []
            p_next_sentence_labels = [0] * batch_size

            for b in range(batch_size): # FULL-SENTENCES
                idx = global_chunk_index[global_chunk_step]
                global_chunk_step += 1
                if global_chunk_step >= len(global_chunk_index):
                    global_chunk_step = 0
                    global_chunk_index = np.random.permutation(len(global_chunks))
                sampled_token = global_chunks[idx]
                # Make Sequence
                ids = copy(global_chunks[idx])
                masks = [1]*len(ids)
                segments = [1]*len(ids)
                # Make Masks
                mask_indexs = []
                for i in np.random.permutation(max_seq_length):
                    if ids[i] < EOT_TOKEN:
                        mask_indexs.append(i)
                    if len(mask_indexs) >= max_predictions_per_seq:
                        break

                lm_positions = []
                lm_ids = []
                lm_weights = []
                for i in sorted(mask_indexs):
                    masked_token = None
                    # 80% of the time, replace with [MASK]
                    if np.random.random() < 0.8:
                        masked_token = MASK_TOKEN # [MASK]
                    else:
                        # 10% of the time, keep original
                        if np.random.random() < 0.5:
                            masked_token = ids[i]
                        # 10% of the time, replace with random word
                        else:
                            masked_token = np.random.randint(EOT_TOKEN-1)

                    lm_positions.append(i)
                    lm_ids.append(ids[i])
                    lm_weights.append(1.0)
                    # apply mask
                    ids[i] = masked_token
                while len(lm_positions) < max_predictions_per_seq:
                    lm_positions.append(0)
                    lm_ids.append(0)
                    lm_weights.append(0.0)

                p_input_ids.append(ids)
                p_input_mask.append(masks)
                p_segment_ids.append(segments)
                p_masked_lm_positions.append(lm_positions)
                p_masked_lm_ids.append(lm_ids)
                p_masked_lm_weights.append(lm_weights)

            return {input_ids:p_input_ids,
                    input_mask:p_input_mask,
                    segment_ids:p_segment_ids,
                    masked_lm_positions:p_masked_lm_positions,
                    masked_lm_ids:p_masked_lm_ids,
                    masked_lm_weights:p_masked_lm_weights,
                    next_sentence_labels:p_next_sentence_labels}

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

        try:
            while True:
                if counter % args.save_every == 0:
                    save()

                (_, v_loss, v_summary) = sess.run(
                    (opt_apply, loss, summaries),
                    feed_dict=sample_feature())

                summary_log.add_summary(v_summary, counter)

                avg_loss = (avg_loss[0] * 0.99 + v_loss,
                            avg_loss[1] * 0.99 + 1.0)

                print(
                    '[{counter} | {time:2.2f}] loss={loss:2.2f} avg={avg:2.2f}'
                    .format(
                        counter=counter,
                        time=time.time() - start_time,
                        loss=v_loss,
                        avg=avg_loss[0] / avg_loss[1]))

                counter = counter+1
                if args.warmup_steps > 0:
                    global_step = global_step+1
        except KeyboardInterrupt:
            print('interrupted')
            save()


if __name__ == '__main__':
    main()
