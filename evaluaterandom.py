#!/usr/bin/env python
"""
Evaluate chance performance on held out set of Crowd500 dataset [Marujo2012]
"""

import pickle
import numpy as np
from keywordextraction import get_crowdd500_data,evaluate_keywords
import random

def main():
  top_k = 15
  verbose = 1

  # evaluate on Crowd500 test set
  testdata = get_crowdd500_data('test')
  num_docs = len(testdata['documents'])
  
  performance_data = np.zeros((num_docs,3))
  for doc_idx in xrange(num_docs):
    given_text = testdata['documents'][doc_idx]
    words = [w for w in given_text.lower().split()]
    words = list(set(words))

    true_keyphrases = testdata['keywords'][doc_idx]    
    true_keywords = []
    for phrase in true_keyphrases:
      true_keywords.extend(phrase.lower().split())

    suggested_keywords = random.sample(words,top_k)

    (precision,recall,f1score) = evaluate_keywords(suggested_keywords,true_keywords)
    performance_data[doc_idx,0] = precision
    performance_data[doc_idx,1] = recall
    performance_data[doc_idx,2] = f1score
    if verbose == 1:
      print 'Document %d of %d: f1-score for top-%d keywords extracted by model = %.4f' % (doc_idx+1,num_docs,top_k,f1score)

  print '----------------------------------------------------------------'
  print 'Evaluation of random word extraction on Crowd500 test set'
  print 'Number of documents = %d, keywords extracted per document = %d' % (num_docs,top_k)
  print 'Precision: Mean = %.4f, SEM = %.4f' % (np.mean(performance_data[:,0]),np.std(performance_data[:,0])/float(np.sqrt(num_docs)))
  print 'Recall: Mean = %.4f, SEM = %.4f' % (np.mean(performance_data[:,1]),np.std(performance_data[:,1])/float(np.sqrt(num_docs)))
  print 'F-1 score: Mean = %.4f, SEM = %.4f' % (np.mean(performance_data[:,2]),np.std(performance_data[:,2])/float(np.sqrt(num_docs)))
  print '----------------------------------------------------------------'

  pickle.dump(performance_data,open('saved/random_evaluation.pkl','wb'))
  return


if __name__ == '__main__':
  main()