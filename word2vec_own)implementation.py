from __future__ import print_function, division
from future.utils import iteritems
from builtins import range, input



import numpy as np
import nltk
import time
import json
import operator
import sys
import os
import string
#nltk.download('brown')


from nltk.corpus import brown
import datetime
from matplotlib.pyplot import plot,ion,show
import random
from scipy.special import expit as sigmoid


KEEP_WORDS = set([
  'king', 'man', 'queen', 'woman',
  'italy', 'rome', 'france', 'paris',
  'london', 'britain', 'england',
])

def get_sentences():
    return brown.sents()

def get_sentences_with_word2idx_limit_vocab(n_vocab=2000, keep_words=KEEP_WORDS):
  sentences = get_sentences()
  indexed_sentences = []
  i = 2
  word2idx = {'START': 0, 'END': 1}
  idx2word = ['START', 'END']
  word_idx_count = {
    0: float('inf'),
    1: float('inf'),
  }
  for sentence in sentences:
    indexed_sentence = []
    for token in sentence:
      token = token.lower()
      if token not in word2idx:
        idx2word.append(token)
        word2idx[token] = i
        i += 1

      # keep track of counts for later sorting
      idx = word2idx[token]
      word_idx_count[idx] = word_idx_count.get(idx, 0) + 1

      indexed_sentence.append(idx)
    indexed_sentences.append(indexed_sentence)



  # restrict vocab size

  # set all the words I want to keep to infinity
  # so that they are included when I pick the most
  # common words
  for word in keep_words:
    word_idx_count[word2idx[word]] = float('inf')
  sorted_word_idx_count = sorted(word_idx_count.items(), key=operator.itemgetter(1), reverse=True)
  word2idx_small = {}
  new_idx = 0
  idx_new_idx_map = {}
  for idx, count in sorted_word_idx_count[:n_vocab]:
    word = idx2word[idx]
    print(word, count)
    word2idx_small[word] = new_idx
    idx_new_idx_map[idx] = new_idx
    new_idx += 1
  # let 'unknown' be the last token
  word2idx_small['UNKNOWN'] = new_idx
  unknown = new_idx

  assert('START' in word2idx_small)
  assert('END' in word2idx_small)
  for word in keep_words:
    assert(word in word2idx_small)

  # map old idx to new idx
  sentences_small = []
  for sentence in indexed_sentences:
    if len(sentence) > 1:
      new_sentence = [idx_new_idx_map[idx] if idx in idx_new_idx_map else unknown for idx in sentence]
      sentences_small.append(new_sentence)

  return sentences_small, word2idx_small



def get_negative_sampling_distribution(sentences,vocab_size):
     word_freq=np.zeros(vocab_size)
     word_count=sum(len(sentence) for sentence in sentences)
     for sentence in sentences:
         for word in sentence:
             word_freq[word]=word_freq[word]+1
     p_neg=word_freq**0.75
     p_neg = p_neg / sum(p_neg)
     p_neg[0]=1e-12
     p_neg[1]=1e-12
     assert (np.all(p_neg > 0))
     return p_neg


def get_context(pos,sentence, window):
    start = max(pos-window,0)
    end= min(pos+window,len(sentence))
    context=[]
    for ctx_pos, ctx_word_idx in enumerate(sentence[start:end],start=start):
        if ctx_pos != pos:
            context.append(ctx_word_idx)
    return context


def sgd(input_,targets,label,learning_rate, W, V):   # binary logistic regression
     activations=W[input_].dot(V[:,targets])
     prob=sigmoid(activations)
     delta_V=np.outer(W[input_],prob-label)
     V[:,targets]=V[:,targets]-learning_rate*delta_V
     delta_W= np.sum((prob-label)*V[:,targets],axis=1)
     W[input_]=W[input_]-learning_rate*delta_W
     cost =label*np.log (prob+1e-12) + (1-label)*np.log(1-prob+1e-12)
     return cost.sum(),W,V

def train_skipgram(savedir):

    vocab_size=500
    sentences,word2idx=get_sentences_with_word2idx_limit_vocab(vocab_size,KEEP_WORDS)
    vocab_size=len(word2idx)
    learning_rate =0.01
    D=50
    W=np.random.randn(vocab_size,D)
    V=np.random.randn(D,vocab_size)
    epochs=30
    #  negative distribution
    costs=[]
    p_neg=get_negative_sampling_distribution(sentences,vocab_size)
    threshold = 1e-5
    p_drop=1-np.sqrt(threshold/p_neg)
    window_size=2
    for epoch in range(epochs):
        np.random.shuffle(sentences)
        cost = 0
        counter = 0
        t0 = datetime.datetime.now()
        for sentence in sentences:
            sentence = [w for w in sentence \
                        if np.random.random() < (1 - p_drop[w])
                        ]
            if len(sentence) < 2:
                continue
            random_positions=np.random.choice(len(sentence),size=len(sentence),replace=False)
            for pos in random_positions:
                word=sentence[pos]
                context_words=get_context(pos,sentence,window_size)
                neg_word=np.random.choice(vocab_size,p=p_neg)
                targets=np.array(context_words)
                c,W,V=sgd(word,targets,1,learning_rate,W,V)
                cost=cost+c
                c,W,V=sgd(neg_word,targets,0,learning_rate,W,V)
                cost=cost+c
            counter=counter+1

        costs.append(cost)
        print('sentence complete :',counter,'Training Epoch :', epoch, 'Cost: ', cost)
        time.sleep(0.1)



    if not os.path.exists(savedir):
        os.mkdir(savedir)

    with open('%s/word2idx.json'%savedir,'w') as f:
        json.dump(word2idx,f)

    np.savez('%s/weights.npz'%savedir,'W','V')
    return costs

def load_model(savedir):
    with open('%s/word2idx.json'%savedir,'r') as f:
        word2idx=json.load(f)

    npz=np.load('%s/weights.npz'%savedir)
    W=npz['arr_0']
    V=npz['arr_1']
    return word2idx,W,V

def get_analogy_word(p1,n1,p2):






 if __name__ == '__main__':
    savedir='C:\\NLP'
    costs = train_skipgram(savedir)
    plot(costs)
    show()
    word2idx,W,V=load_model(savedir)
    print('Program terminated')

    # testing the model




















































































































































