from collections import defaultdict
import nltk
from nltk.corpus import stopwords
import glob
import os
import codecs
import re
import random
import numpy as np

class LangModelPreprocess:
    def __init__(self, 
                 nltk_packages,
                 tokenizer_path,
                 files_root,
                 vocab_size,
                 ppdata_root,
                 start_token='ssss',
                 end_token='eeee'):
        for package in nltk_packages:
            nltk.download(package)
        self.tokenizer = nltk.data.load(tokenizer_path)
        self.ppdata_root = ppdata_root
        self.start_token = start_token
        self.end_token = end_token
        self.vocab_size = vocab_size - 1
        self.files_root = files_root
        self.files = glob.glob(os.path.join(files_root, '*'))
        self._read_given_corpus()
        self.raw_sentences = self.tokenizer.tokenize(self.raw_corpus)
        self._generate_sentences()
        self._generate_word_ids()
        print('* Generated {0:,} word ids'.format(len(self.word_to_ids)))
        self.write_data()
        print('* Saved IDs to {}'.format(os.path.join(self.files_root, 'pp.txt')))

    def _get_word_counts(self):
        word_counts = defaultdict(lambda: 0)
        for sentence in self.sentences:
            for word in sentence:
                word_counts[word.lower()] += 1
        return word_counts

    def _generate_word_ids(self):
        word_counts = self._get_word_counts()
        word_counts_tuple = list(zip(word_counts.keys(), word_counts.values()))
        top_k_word_counts = sorted(word_counts_tuple,
                                   key=lambda x: x[1],
                                   reverse=True)[:self.vocab_size]
        top_k_words = [word for word, _ in top_k_word_counts]
        self.word_to_ids = dict(zip(top_k_words,
                                list(range(self.vocab_size))))
        self.id_to_words = dict(zip(list(range(self.vocab_size)),
                                    top_k_words))

    def _read_given_corpus(self):
        self.raw_corpus = u""
        for i, filename in enumerate(self.files):
            with codecs.open(filename, "r", "utf-8") as f:
                print('* Reading book {} from path {}'.format(i, filename))
                self.raw_corpus += f.read()
        print('* Corpus length = {0:,} charachters'.format(len(self.raw_corpus)))

    def _generate_sentences(self):
        def sentence_to_wlist(raw_sentence):
            clean = re.sub("[^a-zA-Z]", " ", raw_sentence)
            return clean.split()
        self.sentences = []
        for raw_sentence in self.raw_sentences:
            self.sentences.append([self.start_token] + sentence_to_wlist(raw_sentence) + [self.end_token])
        print('* Number of Tokens found {0:,}'.format(sum([len(sentence) for sentence in self.sentences])))

    def get_padded_sentences(self, min_threshold=5, max_seq_len=100):
        sentences = []
        for sentence in self.sentences:
            if len(sentence) < min_threshold or len(sentence) > max_seq_len:
                continue
            sentence_ids = []

            for word in sentence:
                if word.lower() in self.word_to_ids:
                    sentence_ids.append(self.word_to_ids[word.lower()])
                else:
                    sentence_ids.append(self.vocab_size)

            sentences.append(sentence_ids)
        return sentences

    def write_data(self):
        if not os.path.isdir(self.ppdata_root):
            os.makedirs(self.ppdata_root)
        if not os.path.isfile(os.path.join(self.ppdata_root, 'pp.txt')):
            sentences = self.get_padded_sentences()
            with open(os.path.join(self.ppdata_root, 'pp.txt'), 'w+') as f:
                for sentence in sentences:
                    for word in sentence:
                        f.write('{} '.format(word))
                    f.write('\n')
        else:
            print('File {} already exists'.format(os.path.join(self.ppdata_root, 'pp.txt')))
