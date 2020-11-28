import argparse
import json
import os
import numpy as np
import tensorflow as tf
import time
import tqdm
from tensorflow.core.protobuf import rewriter_config_pb2

import pretraining.modeling as modeling
from pretraining.modeling import BertConfig, BertModel
from pretraining.train import get_masked_lm_output

from encode_bpe import BPEEncoder_ja

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='RoBERTa-ja_base')
parser.add_argument('--context', type=str, required=True)
parser.add_argument('--output_cls', default=False, action='store_true')
parser.add_argument('--split_tag', type=str, default='')
parser.add_argument('--gpu', default='0', help='visible gpu number.')
args = parser.parse_args()

with open(args.model+'/hparams.json') as f:
    bert_config_params = json.load(f)
bert_config = BertConfig(**bert_config_params)
vocab_size = bert_config_params['vocab_size']
max_seq_length = bert_config_params['max_position_embeddings']
MASK_TOKEN = vocab_size - 3
CLS_TOKEN = vocab_size - 2
SEP_TOKEN = vocab_size - 1

config = tf.ConfigProto()
config.gpu_options.visible_device_list = args.gpu

with tf.Session(config=config,graph=tf.Graph()) as sess:
    input_ids = tf.placeholder(tf.int32, [1, None])
    input_mask = tf.placeholder(tf.int32, [1, None])
    segment_ids = tf.placeholder(tf.int32, [1, None])
    masked_lm_positions = tf.placeholder(tf.int32, [1, None])
    masked_lm_ids = tf.placeholder(tf.int32, [1, None])
    masked_lm_weights = tf.placeholder(tf.float32, [1, None])

    model = BertModel(
        config=bert_config,
        is_training=True,
        input_ids=input_ids,
        input_mask=input_mask,
        token_type_ids=segment_ids,
        use_one_hot_embeddings=False)

    output = model.get_sequence_output()
    (_,_,_) = get_masked_lm_output(
         bert_config, output, model.get_embedding_table(),
         masked_lm_positions, masked_lm_ids, masked_lm_weights)

    saver = tf.train.Saver()
    ckpt = tf.train.latest_checkpoint(args.model)
    saver.restore(sess, ckpt)

    with open('ja-bpe.txt') as f:
        bpe = f.read().split('\n')

    with open('emoji.json') as f:
        emoji = json.loads(f.read())

    enc = BPEEncoder_ja(bpe, emoji)

    if args.split_tag != '':
        contexts = args.context.split(args.split_tag)
    else:
        contexts = [args.context]
    _input_ids = []
    _input_lengths = []
    for context in contexts:
        context_tokens = enc.encode(context)[:max_seq_length-2]
        _input_lengths.append(len(context_tokens))
        inputs = []
        inputs.append(CLS_TOKEN)
        inputs.extend(context_tokens)
        inputs.append(SEP_TOKEN) # [CLS]+tokens+[SEP]
        while len(inputs) < max_seq_length:
            inputs.append(0)
        _input_ids.append(inputs)
    out = sess.run(output, feed_dict={
        input_ids:_input_ids,
        input_mask:np.zeros((len(_input_ids),max_seq_length), dtype=np.int32),
        segment_ids:np.zeros((len(_input_ids),max_seq_length), dtype=np.int32),
        masked_lm_positions:np.zeros((len(_input_ids),1), dtype=np.int32),
        masked_lm_ids:np.zeros((len(_input_ids),1), dtype=np.int32),
        masked_lm_weights:np.zeros((len(_input_ids),1), dtype=np.int32)
    })
    for i in range(len(_input_ids)):
        if args.output_cls:
            output  = out[i][0] # [CLS]のベクトル
        else:
            output = out[i][1:-1] # 文章部分のベクトル全体
        print(f'input#{i}:')
        print(output)
