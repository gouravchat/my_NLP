# -*- coding: utf-8 -*-
"""
Created on Thu Sep 20 18:55:06 2018

@author: GHA4KOR
"""

import numpy as np

from  sklearn.ensemble import RandomForestClassifier

from sklearn.linear_model import LogisticRegression

from sklearn import svm

from sklearn.neighbors import KNeighborsClassifier


import pandas as pd


class Glove_vectorizer:
    def __init__(self):
        # copy constructor
        word2vec={};
        embedding=[];
        idx2word=[];
        with open('C:\\Users\\gha4kor\\Documents\\machine_learning_examples-master\\nlp_class2\\large_files\\glove.6B.50d.txt',encoding='utf-8') as f:
                
            for line in f:
                values=line.split()
                word=values[0];
                features=values[1:];
                feature_array=np.asarray(features,dtype='float32')
                word2vec[word]=feature_array
                embedding.append(feature_array)
                idx2word.append(word)
            print('Found %s word vectors.' % len(word2vec))
        self.word2vec=word2vec
        self.embedding=np.array(embedding)
        self.word2idx={v:k for k,v in enumerate(idx2word)}
        self.V,self.D=self.embedding.shape  # number of rows and numbe rof coloumns 
        
        
        
        
    def fit (self,data):
        pass
        
        
    def transform(self,data):
            
            X=np.zeros((len(data),self.D)) #input vector
            n=0
            empty_count=0
            
            for sentence in data:
                words=sentence.lower().split()
                vecs=[]
                for word in words:
                    
                    if word in self.word2vec:
                        vec=self.word2vec[word]
                        vecs.append(vec)
                    
                if len(vecs)>0:
                    vecs=np.array(vecs)
                    X[n]=vecs.mean(axis=0)
                    
                else:
                    empty_count=empty_count+1
                    
                n=n+1

            print("Number of samples with no words found: %s / %s" % (empty_count, len(data)))
            return X
        
        
    def fit_transform(self,data):
            self.fit(data)
            return self.transform(data)
        
        
        
        
        
# main function starts 
        
#cd C:\Users\gha4kor\Documents\machine_learning_examples-master\nlp_class2\large_files
train = pd.read_csv('C:\\Users\\gha4kor\\Documents\\machine_learning_examples-master\\nlp_class2\\large_files\\r8-train-all-terms.txt',header=None,sep='\t')
test = pd.read_csv('C:\\Users\\gha4kor\\Documents\\machine_learning_examples-master\\nlp_class2\\large_files\\r8-test-all-terms.txt',header=None,sep='\t')
train.columns = ['label', 'content']
test.columns = ['label', 'content']

vectorizer=Glove_vectorizer()
X_train=vectorizer.fit_transform(train.content)
Y_train=train.label
#
X_test=vectorizer.fit_transform(test.content)
Y_test=test.label
#
model=RandomForestClassifier(n_estimators=200)
model.fit(X_train,Y_train)
print("train score",model.score(X_train,Y_train))
print("test score",model.score(X_test,Y_test))



# testing with logistic regression

print("Logistic regression running")
logisticRegr = LogisticRegression()
logisticRegr.fit(X_train, Y_train)
score = logisticRegr.score(X_train, Y_train)
print("LR_train_score:",score)

score = logisticRegr.score(X_test, Y_test)
print("LR_test_score:",score)

# using support vector machine

print("Support vector machine running")
clf=svm.LinearSVC()
clf.fit(X_train,Y_train)
score=clf.score(X_train,Y_train)
print("SVM train score",score)
score=clf.score(X_test,Y_test)
print("SVM test score",score)

# nearest neightbour classifier
print("Decision tree running ")
clf=KNeighborsClassifier()
clf.fit(X_train,Y_train)
score=clf.score(X_test,Y_test)
print("Decision tree test score: ",score)

























        
        
        
        
        
        
        
                    
                        
                    
            
            
        
        
            
        

            
            
            
            
            
            
        
        
        