#!/usr/bin/env python
# coding:utf-8
"""
An example of how to use the keyword extraction model 
"""
__author__ = "Lavanya Sharan"


import sys
from keywordextraction import *


def main():
  if len(sys.argv)==1:
    raise ValueError('Must specify input text file.')	
  else:
		f = open(sys.argv[1],'r')
		text = f.read()
		f.close()

  # load keyword classifier
  preload = 1
  classifier_type = 'logistic'
  keyword_classifier = get_keywordclassifier(preload,classifier_type)['model']

  # extract top k keywords
  top_k = 15
  keywords = extract_keywords(text,keyword_classifier,top_k,preload)  
  print 'ORIGINAL TEXT:\n%s\nTOP-%d KEYWORDS returned by model: %s\n' % (text,top_k,', '.join(keywords))

  # evaluate performance for inspec example
  if sys.argv[1]=='inspec.txt':
    true_keywords = []
    with open('inspec.key','r') as f:
      for line in f:
        true_keywords.extend(line.strip('\n').split())
    true_keywords = [remove_punctuation(kw.lower()) for kw in true_keywords]

    (precision,recall,f1score) = evaluate_keywords(keywords,true_keywords)
    print 'MANUALLY SELECTED KEYWORDS:\n%s' % ', '.join(true_keywords)
    print '\nModel achieves %.4f precision, %.4f recall, and %.4f f1 score.' % (precision,recall,f1score)


if __name__ == '__main__':
	main()
