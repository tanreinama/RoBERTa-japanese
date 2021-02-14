import argparse
import json
import os
import numpy as np
import pandas as pd
import tensorflow.compat.v1 as tf
import time
import tqdm
from tensorflow.core.protobuf import rewriter_config_pb2
import matplotlib.pyplot as plt

from model import modeling
from model.modeling import BertConfig, BertModel
from run_finetune import get_masked_lm_output,get_next_sentence_output

from encode_bpe import BPEEncoder_ja

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='RoBERTa-ja_classifier', help='pretrained model directory.')
parser.add_argument('--input_dir', type=str, required=True, help='input texts.')
parser.add_argument('--train_by_line', action='store_true', help='split file by lines.')
parser.add_argument('--batch_size', type=int, default=1, help='predict batch size.')
parser.add_argument('--output_file', type=str, default='predict.csv', help='result csv file to write.')
parser.add_argument('--gpu', default='0', help='visible gpu number.')

def main():
    args = parser.parse_args()

    if os.path.isfile(args.model+'/hparams.json'):
        with open(args.model+'/hparams.json') as f:
            bert_config_params = json.load(f)
    else:
        raise ValueError('invalid model name.')
    if os.path.isfile(args.model+'/idmaps.json'):
        with open(args.model+'/idmaps.json') as f:
            idmapping_dict = json.load(f)
    else:
        raise ValueError('invalid model name.')

    vocab_size = bert_config_params['vocab_size']
    max_seq_length = bert_config_params['max_position_embeddings']
    batch_size = args.batch_size
    EOT_TOKEN = vocab_size - 4
    MASK_TOKEN = vocab_size - 3
    CLS_TOKEN = vocab_size - 2
    SEP_TOKEN = vocab_size - 1

    with open('ja-bpe.txt', encoding='utf-8') as f:
        bpe = f.read().split('\n')

    with open('emoji.json', encoding='utf-8') as f:
        emoji = json.loads(f.read())

    enc = BPEEncoder_ja(bpe, emoji)

    num_labels = len(idmapping_dict)
    input_contexts = []
    input_keys = []
    input_names = []
    for f,i in idmapping_dict.items():
        n = 0
        for t in os.listdir(f'{args.input_dir}/{f}'):
            if os.path.isfile(f'{args.input_dir}/{f}/{t}'):
                with open(f'{args.input_dir}/{f}/{t}', encoding='utf-8') as fn:
                    if args.train_by_line:
                        for ln,p in enumerate(fn.readlines()):
                            tokens = enc.encode(p.strip())[:max_seq_length-3]
                            tokens = [CLS_TOKEN]+tokens+[EOT_TOKEN,SEP_TOKEN]
                            if len(tokens) < max_seq_length:
                                tokens.extend([0]*(max_seq_length-len(tokens)))
                            input_contexts.append(tokens)
                            input_keys.append(i)
                            input_names.append(f'{f}/{t}#{ln}')
                            n += 1
                    else:
                        p = fn.read()
                        tokens = enc.encode(p.strip())[:max_seq_length-2]
                        tokens = [CLS_TOKEN]+tokens+[SEP_TOKEN]
                        if len(tokens) < max_seq_length:
                            tokens.extend([0]*(max_seq_length-len(tokens)))
                        input_contexts.append(tokens)
                        input_keys.append(i)
                        input_names.append(f'{f}/{t}')
                        n += 1
        print(f'{args.input_dir}/{f} mapped for id_{i}, read {n} contexts.')
    input_indexs = np.arange(len(input_contexts))

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
            is_training=False,
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

        labels = tf.placeholder(tf.int32, [batch_size, ])

        output_layer = model.get_pooled_output()

        if int(tf.__version__[0]) > 1:
            hidden_size = output_layer.shape[-1]
        else:
            hidden_size = output_layer.shape[-1].value

        output_weights = tf.get_variable(
          "output_weights", [num_labels, hidden_size],
          initializer=tf.truncated_normal_initializer(stddev=0.02))

        output_bias = tf.get_variable(
          "output_bias", [num_labels], initializer=tf.zeros_initializer())

        logits = tf.matmul(output_layer, output_weights, transpose_b=True)
        logits = tf.nn.bias_add(logits, output_bias)
        probabilities = tf.nn.softmax(logits, axis=-1)

        saver = tf.train.Saver(var_list=tf.trainable_variables())
        ckpt = tf.train.latest_checkpoint(args.model)
        saver.restore(sess, ckpt)

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

        preds = []
        prog = tqdm.tqdm(range(0,len(input_contexts)//batch_size,1))
        for i in prog:
            prob = sess.run(
                probabilities,
                feed_dict=sample_feature(i))
            for p in prob:
                pred = np.argmax(p)
                preds.append(pred)

        pd.DataFrame({
            'id':input_names,
            'y_true':input_keys,
            'y_pred':preds
        }).to_csv(args.output_file, index=False)

        r = np.zeros((num_labels,num_labels), dtype=int)
        for t,p in zip(input_keys, preds):
            r[t,p] += 1
        fig = plt.figure(figsize=(12,6), dpi=72)
        ax = plt.matshow(r, interpolation='nearest', aspect=.5, cmap='cool')
        for (i, j), z in np.ndenumerate(r):
            if z >= 1000:
                plt.text(j-.33, i, '{:0.1f}K'.format(z/1000), ha='left', va='center', size=9, color='black')
            else:
                plt.text(j-.33, i, f'{z}', ha='left', va='center', size=9, color='black')
        pfile = args.output_file
        if args.output_file.lower().endswith('.csv'):
            pfile = args.output_file[:-4]
        plt.savefig(pfile+'_map.png')

if __name__ == '__main__':
    main()
