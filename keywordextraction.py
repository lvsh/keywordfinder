#!/usr/bin/env python
"""
Contains functions for keyword extraction using a classifer trained on the Crowd500 dataset [Marujo2012]
"""

import os
import re
import random
import numpy as np
import pickle

import string
import nltk
from nltk.corpus import stopwords
stoplist = stopwords.words('english')

from gensim import corpora, models, similarities
from collections import defaultdict

from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import cross_val_score

from features import *


##################################################################
# functions to get train/test set and extract features from text #
##################################################################

def get_crowdd500_data(set_type):
  """
  Returns documents and keywords in either train or test sets of Crowd500 [Marujo2012]
  """
  path = 'data/500N-KPCrowd-v1.1/CorpusAndCrowdsourcingAnnotations/' + set_type + '/'
  files = [f[:-4] for f in os.listdir(path) if re.search('\.key',f)]

  documents = []
  all_keywords = []

  if set_type=='test':
    documents = pickle.load(open(path + 'scraped_testdata.pkl','rb')) # scraped webpages in test set
    skip_these = [3,7,14,19,26,27,32,33,43,45] # these webpages no longer exist, cannot find source text

  for file_idx in xrange(len(files)):
    if set_type=='train':

      # original text
      f = open(path + files[file_idx] + '.txt','r')
      text = f.read()
      f.close()  

      # encoding issues in Crowd500  
      try:
        text = text.encode('utf-8')
        sentences = nltk.sent_tokenize(text.lower())        
      except:
        text = text.decode('utf-8')
        sentences = nltk.sent_tokenize(text.lower())   
      
      documents.append(text)

      # keywords
      keywords = []
      with open(path + files[file_idx] + '.key','r') as f:
        for line in f:
          keywords.append(line.strip('\n'))            
      keywords = [remove_punctuation(k.lower()) for k in keywords]
      all_keywords.append(keywords)

    else:
      if file_idx not in skip_these:
        keywords = []
        with open(path + files[file_idx] + '.key','r') as f:
          for line in f:
            keywords.append(line.strip('\n'))            
        keywords = [remove_punctuation(k.lower()) for k in keywords]
        all_keywords.append(keywords)
  
  return {'documents':documents, 'keywords':all_keywords}


def to_tfidf(documents):
  """
  Returns documents transformed to tf-idf vector space
  """
  texts = [[remove_punctuation(word) for word in document.lower().split() if word not in stoplist]
    for document in documents]
  dictionary = corpora.Dictionary(texts)
  corpus = [dictionary.doc2bow(text) for text in texts]
  tfidf = models.TfidfModel(corpus,normalize=True)
  corpus_tfidf = tfidf[corpus]    

  return {'dictionary':dictionary, 'corpus':corpus_tfidf, 'tfidf_model': tfidf}


def get_features_labels(data,corpus,dictionary,verbose):
  """
  Returns matrices X containing features and Y containing labels.
  Labels are 0 (not a keyword) and 1 (keyword).  
  """
  num_docs = len(data['documents'])
  
  for doc_idx in xrange(num_docs):
    text = data['documents'][doc_idx]
    keywords = data['keywords'][doc_idx]
    corpus_entry = corpus[doc_idx]
 
    # as keyword classification operates at the level of single word,
    # we define any word that occurs in a keyword phrase as a keyword
    separate_keywords = []
    for k in keywords: 
      separate_keywords.extend(remove_punctuation(k.lower()).split())

    # collect positive (keyword) and negative (non-keyword) examples
    positive_examples = separate_keywords
    num_positive = len(positive_examples)

    all_words = [remove_punctuation(w) for w in text.lower().split()]
    negative_examples = [w for w in all_words if (w not in positive_examples) and (w not in stoplist)]
    if len(negative_examples)>num_positive:
      negative_examples = random.sample(negative_examples,num_positive)
    num_negative = len(negative_examples)

    # balance the number of positive and negative examples
    if num_positive < num_negative:
      candidate_keywords = positive_examples + random.sample(negative_examples,num_positive)
      labels = np.array([1]*num_positive + [0]*num_positive)
    elif num_positive > num_negative:
      candidate_keywords = random.sample(positive_examples,num_negative) + negative_examples
      labels = np.array([1]*num_negative + [0]*num_negative)
    else:
      candidate_keywords = positive_examples + negative_examples
      labels = np.array([1]*num_positive + [0]*num_negative)

    # assemble labels
    if doc_idx==0:
      all_labels = labels
    else:
      all_labels = np.concatenate((all_labels,labels))

    # assemble features
    feature_set = extract_features(text,candidate_keywords,corpus_entry,dictionary)
    if doc_idx==0:
      all_features = feature_set['features']
    else:
      all_features = np.vstack((all_features,feature_set['features']))

    if verbose:
      print 'get_features_labels: extracted %d samples from document %d of %d' % (len(labels),doc_idx+1,num_docs)
  
  return {'features':all_features, 'labels':all_labels}


###########################################
# functions to perform keyword extraction #
###########################################

def get_keywordclassifier(preload):
  """
  Returns a keyword classifier trained and tested on dataset derived from Crowd500 [Marujo2012]
  """  
  if preload==1:
    train_XY = pickle.load(open('saved/trainXY_crowd500.pkl','rb'))
    test_XY = pickle.load(open('saved/testXY_crowd500.pkl','rb'))    
    model = pickle.load(open('saved/logisticregression_crowd500.pkl','rb'))    
  else:
    # get training data from crowd500 corpus
    traindata = get_crowdd500_data('train')
    tx_traindata = to_tfidf(traindata['documents'])
    train_XY = get_features_labels(traindata,tx_traindata['corpus'],tx_traindata['dictionary'],1)
    pickle.dump(train_XY, open('saved/trainXY_crowd500.pkl','wb'))    
  
    # get test data from crowd500 corpus
    testdata = get_crowdd500_data('test')

    # use tf-idf dictionary learned on training data to transform test data
    dictionary = tx_traindata['dictionary']
    tfidf = tx_traindata['tfidf_model']
    texts = [[remove_punctuation(word) for word in document.lower().split() if word not in stoplist]
              for document in testdata['documents']]
    corpus = [dictionary.doc2bow(text) for text in texts]
    corpus_tfidf = tfidf[corpus]    
    tx_testdata = {'dictionary':dictionary, 'corpus':corpus_tfidf, 'tfidf_model': tfidf}

    test_XY = get_features_labels(testdata,tx_testdata['corpus'],tx_testdata['dictionary'],1)
    pickle.dump(test_XY, open('saved/testXY_crowd500.pkl','wb'))

    # train model for keyword classification 
    model = LogisticRegression()
    model = model.fit(train_XY['features'],train_XY['labels'])
    pickle.dump(model, open('saved/logisticregression_crowd500.pkl','wb'))    

  # show performance of classifier
  in_sample_acc = model.score(train_XY['features'],train_XY['labels'])
  out_sample_acc = model.score(test_XY['features'],test_XY['labels'])
  print '-----------------------------------------------------------------------------------------'
  print 'Using logistic regression model for keyword classification (0 = non-keyword, 1 = keyword)'
  print 'Trained and tested on dataset derived from Crowd500 [Marujo2012]'
  print 'Number of features = %d, Number of training samples = %d, Number of test samples %d' % (train_XY['features'].shape[1],train_XY['features'].shape[0],test_XY['features'].shape[0])
  print 'In-sample accuracy: %.4f, Out-of-sample accuracy: %.4f, Chance: 0.5' % (in_sample_acc,out_sample_acc)
  print '-----------------------------------------------------------------------------------------'
  
  return {'model': model, 'train_XY':train_XY, 'test_XY':test_XY}


def generate_candidates(text):
  """
  Returns candidate words that occur in named entities, noun phrases, or top trigrams
  """
  num_trigrams = 5
  named_entities = get_namedentities(text)
  noun_phrases = get_nounphrases(text)
  top_trigrams = get_trigrams(text,num_trigrams)
  return list(set.union(set(named_entities),set(noun_phrases),set(top_trigrams)))


def extract_keywords(text,keyword_classifier,top_k,preload):
  """
  Returns top k keywords using specified keyword classifier
  """
  # pre-processing to enable tf-idf representation
  if preload==1:
    preprocessing = pickle.load(open('saved/tfidf_preprocessing.pkl','rb'))
    dictionary = preprocessing['dictionary']
    tfidf = preprocessing['tfidf_model'] 
  else:
    traindata = get_crowdd500_data('train')
    tx_traindata = to_tfidf(traindata['documents'])
    dictionary = tx_traindata['dictionary']
    tfidf = tx_traindata['tfidf_model']
    pickle.dump({'dictionary': dictionary, 'tfidf_model':tfidf},open('saved/tfidf_preprocessing.pkl','wb'))

  text_processed = [remove_punctuation(word) for word in text.lower().split() if word not in stoplist]
  corpus = [dictionary.doc2bow(text_processed)]
  corpus_entry = tfidf[corpus][0]    

  # generate canddiate keywords
  candidate_keywords = generate_candidates(text)
  if len(candidate_keywords) < top_k:
    candidate_keywords = text_processed   

  # select from candidate keywords 
  feature_set = extract_features(text,candidate_keywords,corpus_entry,dictionary)
  predicted_prob = keyword_classifier.predict_proba(feature_set['features'])
  this_column = np.where(keyword_classifier.classes_==1)[0][0]
  sorted_indices = [i[0] for i in sorted(enumerate(predicted_prob[:,this_column]),key = lambda x:x[1],reverse = True)]
  chosen_keywords = [candidate_keywords[j] for j in sorted_indices[:top_k]]    
  
  # predicted_labels = keyword_classifier.predict(feature_set['features'])
  # chosen_keywords = [candidate_keywords[j] for j in xrange(len(candidate_keywords)) if predicted_labels[j]==1]
  # if len(chosen_keywords) > top_k:
  #   chosen_keywords = random.sample(chosen_keywords,top_k)

  return chosen_keywords


######################################################
# function to evaluate success of keyword extraction #
######################################################

def evaluate_keywords(proposed,groundtruth):
  proposed_set = set(proposed)
  true_set = set(groundtruth)
  
  true_positives = len(proposed_set.intersection(true_set))
  if len(proposed_set)==0:
    precision = 0
  else:
    precision = true_positives/float(len(proposed_set))
      
  if len(true_set)==0:
    recall = 0
  else:
    recall = true_positives/float(len(true_set))
  
  if precision + recall > 0:
    f1 = 2*precision*recall/float(precision + recall)
  else:
    f1 = 0

  return (precision, recall, f1)