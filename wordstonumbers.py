from itertools import chain
from collections import Counter
import numpy as np


# 1. Get the vocabulary from the sentences
x = [' I like apples ' ' I love bananas ' ' I like bananas ']
words = 15

def get_vocab(sentences):
    all_words = list(chain(*[x.lower().split() for x in sentences]))
    words, count = np.unique(all_words, return_counts=True)
    idxs = np.argsort(count)[-words:]
    vocab = ['<UNK>'] + list(words[idxs][::-1])
    print(vocab[:5], '...', vocab[-5:])

get_vocab(x)