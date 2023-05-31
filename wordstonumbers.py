from itertools import chain
from collections import Counter
import numpy as np
from torch import nn
import torch
device = "cuda" if torch.cuda.is_available() else "cpu"


def get_vocab(X):
    senlist = X.values.tolist()[0:3000]
    all_words = list(chain(*[i.lower().split() for i in senlist]))
    n_words = 75000
    words, count = np.unique(all_words, return_counts=True)
    idxs = np.argsort(count)[-n_words:]
    vocab = ['<UNK>'] + list(words[idxs][::-1])
    vocab_d = {vocab[i]: i for i in range(len(vocab))}
    return vocab_d

def sentence_to_integer_sequence(s, vocab):
    return torch.tensor([vocab[x] if x in vocab else 0 for x in s.split()], dtype=torch.long)

def sentence_to_tensor(s, vocab):
    embedding = nn.Embedding(len(vocab), 50)
    embedding.weight.data[0, :] = 0
    return embedding(sentence_to_integer_sequence(s, vocab))
