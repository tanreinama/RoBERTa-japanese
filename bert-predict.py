import argparse
import json
import os
import numpy as np
import tensorflow.compat.v1 as tf
import time
import tqdm
from tensorflow.core.protobuf import rewriter_config_pb2

from model import modeling
from model.modeling import BertConfig, BertModel
from run_finetune import get_masked_lm_output,get_next_sentence_output

from encode_bpe import BPEEncoder_ja

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='RoBERTa-ja_small')
parser.add_argument('--context', type=str, required=True)
parser.add_argument('--split_tag', type=str, default='')
parser.add_argument('--gpu', default='0', help='visible gpu number.')
args = parser.parse_args()

with open(args.model+'/hparams.json') as f:
    bert_config_params = json.load(f)
bert_config = BertConfig(**bert_config_params)
vocab_size = bert_config_params['vocab_size']
max_seq_length = bert_config_params['max_position_embeddings']
EOT_TOKEN = vocab_size - 4
MASK_TOKEN = vocab_size - 3
CLS_TOKEN = vocab_size - 2
SEP_TOKEN = vocab_size - 1

config = tf.ConfigProto()
config.gpu_options.visible_device_list = args.gpu

with tf.Session(config=config,graph=tf.Graph()) as sess:
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
    (_,_,log_prob) = get_masked_lm_output(
         bert_config, model.get_sequence_output(), model.get_embedding_table(),
         masked_lm_positions, masked_lm_ids, masked_lm_weights)
    (_,_,_) = get_next_sentence_output(
         bert_config, model.get_pooled_output(), next_sentence_labels)

    saver = tf.train.Saver()
    ckpt = tf.train.latest_checkpoint(args.model)
    saver.restore(sess, ckpt)

    with open('ja-bpe.txt', encoding='utf-8') as f:
        bpe = f.read().split('\n')

    with open('emoji.json', encoding='utf-8') as f:
        emoji = json.loads(f.read())

    enc = BPEEncoder_ja(bpe, emoji)

    if args.split_tag != '':
        contexts = args.context.split(args.split_tag)
    else:
        contexts = [args.context]
    _input_ids = []
    _input_masks = []
    _segments = []
    _mask_positions = []
    for context in contexts:
        context_tokens = [enc.encode(c)+[MASK_TOKEN] for c in context.split('[MASK]')]
        context_tokens = sum(context_tokens, [])
        if len(context_tokens) > 1:
            context_tokens = context_tokens[:-1]
        context_tokens = context_tokens[:max_seq_length-3]
        inputs = []
        inputs.append(CLS_TOKEN)
        inputs.extend(context_tokens)
        inputs.append(EOT_TOKEN)
        inputs.append(SEP_TOKEN) # [CLS]+tokens+<|endoftext|>+[SEP]
        input_masks = [1] * len(inputs)
        segments = [1] * len(inputs)
        while len(inputs) < max_seq_length:
            inputs.append(0)
            input_masks.append(0)
            segments.append(1)
        _input_ids.append(inputs)
        _input_masks.append(input_masks)
        _segments.append(segments)
        mask_positions = []
        for p in range(len(inputs)):
            if inputs[p] == MASK_TOKEN:
                mask_positions.append(p)
        _mask_positions.append(mask_positions)

    max_mask_count = np.max([len(c) for c in _mask_positions])
    for p in range(len(_mask_positions)):
        q = len(_mask_positions[p])
        if q < max_mask_count:
            _mask_positions[p].extend([0]*(max_mask_count-q))

    prob = sess.run(log_prob, feed_dict={
        input_ids:_input_ids,
        input_mask:_input_masks,
        segment_ids:_segments,
        masked_lm_positions:_mask_positions,
        masked_lm_ids:np.zeros((len(_input_ids),max_mask_count), dtype=np.int32),
        masked_lm_weights:np.ones((len(_input_ids),max_mask_count), dtype=np.float32),
        next_sentence_labels:np.zeros((len(_input_ids),), dtype=np.int32),
    })
    mask_count = 0
    for i in range(len(_input_ids)):
        result_token = []
        for j in range(len(_input_ids[i])):
            if CLS_TOKEN == _input_ids[i][j]:
                pass
            elif MASK_TOKEN == _input_ids[i][j]:
                result_token.append(np.argmax(prob[mask_count]))
                mask_count += 1
            elif EOT_TOKEN <= _input_ids[i][j]:
                break
            else:
                result_token.append(_input_ids[i][j])
        print(enc.decode(result_token))
