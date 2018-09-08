import tensorflow as tf
import os
import numpy as np
from copy import deepcopy

from utils import config, unpickle
from model import model_fn
from model.pipeline import predict_in_fn
from utils.arg_parser import get_argparse
from utils.preprocessor import text_to_ids, ids_to_text

def produce_text(in_txt, seq_len):
    word_id_map = unpickle(os.path.join(config['ppdata_root'], 'word_ids.pickle'))

    in_txt = 'ssss ' + in_txt
    in_ids = text_to_ids(word_id_map, in_txt)

    lang_model = tf.estimator.Estimator(model_fn,
                                        model_dir=config['model_dir'],
                                        params={
                                            'lr': config['lr'],
                                            'vocab_size': config['vocab_size'],
                                            'embedding_size': config['embedding_size'],
                                            'hidden_units': config['hidden_units'],
                                            'keep_rate': config['keep_rate'],
                                            'num_layers': config['num_layers'],
                                            'max_gradient_norm': config['max_gradient_norm']
                                        })
    new_ids = deepcopy(in_ids)
    while len(new_ids) < seq_len:
        if len(new_ids) < 10:
            predict_in_fn_ = lambda: predict_in_fn(np.expand_dims(new_ids, 0))
        else:
            predict_in_fn_ = lambda: predict_in_fn(np.expand_dims(new_ids[-10:], 0))
        preds = lang_model.predict(predict_in_fn_)
        cur_ids = next(preds)['preds']
        new_ids = new_ids + [cur_ids[0][-1]]
    return ids_to_text(word_id_map, new_ids)

def process_text(txt):
    word_list = txt.split()
    this_is_start = False
    this_is_end = False
    processed_word_list = []
    for word in word_list:
        if word == 'ssss':
            if this_is_start:
                continue
            this_is_start = True
            continue
        if word == 'eeee':
            if this_is_end:
                continue
            this_is_end = True
        if this_is_start:
            processed_word_list.append(word.capitalize())
            this_is_start = False
            continue
        if this_is_end:
            processed_word_list.append('.')
            this_is_end = False
            continue
        processed_word_list.append(word)

    return ' '.join(processed_word_list)

if __name__ == '__main__':
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    args = get_argparse()
    print(process_text(produce_text(args.in_txt, args.seq_len)))
