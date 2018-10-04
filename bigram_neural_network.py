from __future__ import print_function, division
from future.utils import iteritems
from builtins import range, input



import numpy as np
import nltk

import operator
#nltk.download('brown')


from nltk.corpus import brown

import datetime
import matplotlib.pyplot as plt
import random




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



def get_sentences_with_word2idx():
    sentences=get_sentences()
    indexed_sentences = []
    i = 2
    word2idx={'START':0,'END':1}

    for sentence in sentences:
        indexed_sentence=[]
        for token in sentence:
            token=token.lower()
            if token not in word2idx:
                word2idx[token]=i
                i=i+1

            indexed_sentence.append(word2idx[token])
        indexed_sentences.append(indexed_sentence)

    print('Vocabulary Size :', i)
    return indexed_sentences,word2idx





def bigram_model(sentences,V,start_idx,end_idx,smoothing=1):
    bigram_prob = np.ones((V, V)) * smoothing

    for sentence in sentences:
        for i in range(len(sentence)):

            if i == 0:
                bigram_prob[start_idx,sentence[i]] = bigram_prob[start_idx,sentence[i]]+1

            else :
                bigram_prob[sentence[i-1],sentence[i]]=bigram_prob[sentence[i-1],sentence[i]]+1

            if i == len(sentence) - 1:
                # final word
                bigram_prob[sentence[i], end_idx] += 1

    bigram_prob = bigram_prob / bigram_prob.sum(axis=1,keepdims=True)
    return bigram_prob


if __name__ == '__main__':

    get_sentences()

    indexed_sentences,word2idx=get_sentences_with_word2idx_limit_vocab(n_vocab=200)

    V=len(word2idx)

    print ('Final Vocabulary size', V)

    start_idx = word2idx['START']
    print(start_idx)
    end_idx = word2idx['END']
    print(end_idx)

    bigram_prob=bigram_model(indexed_sentences,V,start_idx,end_idx,smoothing=0.1)


    def get_score(sentence):
        score = 0

        for i in range(len(sentence)):

            if i == 0:
                score = score + np.log(bigram_prob[start_idx,sentence[i]])

            else:
                score = score + np.log(bigram_prob[sentence[i-1],sentence[i]])
            if i == len(sentence)-1:
                score=score+np.log(bigram_prob[sentence[-1],end_idx])
        score=score/len(sentence)+1
        return score


    idx2word=((v, k) for k,v in iteritems(word2idx))

    def get_words(sentence):
        return ' '.join(idx2word[i] for i in sentence)

    sample_probs = np.ones(V)
    sample_probs[start_idx] = 0
    sample_probs[end_idx] = 0
    sample_probs /= sample_probs.sum()




       # logistic regression code
    def softmax(a):
        _a=a-a.max()
        return (np.exp(_a)/(np.exp(_a)).sum(axis=1,keepdims=True))

    def smoothed_loss(x, decay=0.99):
        y = np.zeros(len(x))
        last = 0
        for t in range(len(x)):
            z = decay * last + (1 - decay) * x[t]
            y[t] = z / (1 - decay ** (t + 1))
            last = z
        return y
    #  neural network approach
    print ('Neural network Approach')
    epochs = 5
    lr = 1e-1
    t0 = datetime.datetime.now()
    print('Current time :', t0)
    print('Neural Network Approach')
    D=10
    W1=np.random.rand(V,D)/np.sqrt(V)
    W2=np.random.rand(D,V)/np.sqrt(D)
    losses=[]
    j=0
    for epoch in range(epochs):
        random.shuffle(indexed_sentences)
        for sentence in indexed_sentences:
            sentence = [start_idx]+sentence+[end_idx]
            n=len(sentence)
            inputs=np.zeros((n-1,V))
            targets=np.zeros((n-1,V))
            inputs[np.arange(n-1),sentence[:n-1]]=1
            targets[np.arange(n-1),sentence[1:]]=1
            hidden=np.tanh(inputs.dot(W1))
            predictions=softmax(hidden.dot(W2))
            # doing a gradient descent step
            W2= W2 - lr*hidden.T.dot(predictions-targets)
            dhidden = (predictions-targets).dot(W2.T)*(1-hidden*hidden)
            W1 = W1 - lr*inputs.T.dot(dhidden)
            loss = -np.sum(targets*np.log(predictions))/(n-1)
            losses.append(loss)
            print('Training Epoch :',epoch,'iteration: ', j , 'loss: ', loss)
            if j==0:
                print ('Stop')
            j=j+1

    plt.plot(smoothed_loss(losses))
    plt.title('by 2 Layer Neural networks')
    plt.show()


























































































