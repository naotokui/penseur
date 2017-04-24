"""
Constructing and loading dictionaries
"""
import cPickle as pkl
import numpy
from collections import OrderedDict

def build_dictionary(text, max_vocab_num = 0):
    """
    Build a dictionary
    text: list of sentences (pre-tokenized)
    """
    wordcount = OrderedDict()
    for cc in text:
        words = cc.split()
        for w in words:
            if w not in wordcount:
                wordcount[w] = 0
            wordcount[w] += 1
    words = wordcount.keys()
    freqs = wordcount.values()
    sorted_idx = numpy.argsort(freqs)[::-1]

    wordcount_ = OrderedDict()
    worddict = OrderedDict()
    for idx, sidx in enumerate(sorted_idx):
        if max_vocab_num < 0 or idx < max_vocab_num:
            worddict[words[sidx]] = idx+2 # 0: <eos>, 1: <unk>
            wordcount_[words[sidx]] = wordcount[words[sidx]]
        else:
            worddict[words[sidx]] = 1 # force to be <unk>
    return worddict, wordcount_

def load_dictionary(loc='/ais/gobi3/u/rkiros/bookgen/book_dictionary_large.pkl'):
    """
    Load a dictionary
    """
    with open(loc, 'rb') as f:
        worddict = pkl.load(f)
    return worddict

def save_dictionary(worddict, wordcount, loc):
    """
    Save a dictionary to the specified location 
    """
    with open(loc, 'wb') as f:
        pkl.dump(worddict, f)
        pkl.dump(wordcount, f)


