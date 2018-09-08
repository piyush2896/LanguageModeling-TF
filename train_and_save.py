import tensorflow as tf
import glob
import os

from utils import config, get_tensorflow_logger, pickle_obj
from utils.preprocessor import sort_seq_len
from model.pipeline import input_fn, preprocess
from model import model_fn

def main():
    print('* Preprocessing Raw Corpus')
    preprocessor = preprocess.LangModelPreprocess(config['nltk_packages'],
                                                  config['tokenizer_path'],
                                                  config['data_root'],
                                                  config['vocab_size'],
                                                  config['ppdata_root'])
    print('* Generating Sorted File')
    sort_seq_len(os.path.join(config['ppdata_root'], 'pp.txt'),
                 os.path.join(config['ppdata_root'], 'pp_sorted.txt'))

    print('* Build Logger')
    logger = get_tensorflow_logger('tensorflow.log')

    print('* Estimator Instance Created')
    train_1_input_fn = lambda: input_fn(glob.glob(os.path.join(config['ppdata_root'], 'pp_sorted.txt')),
                                        batch_size=config['batch_size'],
                                        padding_val=config['vocab_size']-1,
                                        shuffle=False)
    train_input_fn = lambda: input_fn(glob.glob(os.path.join(config['ppdata_root'], 'pp.txt')),
                                        batch_size=config['batch_size'],
                                        padding_val=config['vocab_size']-1)
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

    print('* Start Training - Training logs to tensorflow.log')
    print('\t-Training 1 Epoch over sorted sequences')
    lang_model.train(train_1_input_fn,
                     steps=config['steps_per_epoch']*1)
    print('\t-Training {} Epoch over random sequences'.format(config['epochs']-1))
    lang_model.train(train_input_fn,
                     steps=config['steps_per_epoch']*(config['epochs'] - 1))

    print('* Saving word id map')
    if os.path.isfile(os.path.join(config['ppdata_root'], 'word_ids.pickle')):
        print('\t-File {} already present'.format(os.path.join(config['ppdata_root'], 'word_ids.pickle')))
    else:
        pickle_obj(preprocessor.word_to_ids,
                os.path.join(config['ppdata_root'], 'word_ids.pickle'))

if __name__ == '__main__':
    main()
