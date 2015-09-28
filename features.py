#!/usr/bin/env python
"""
Contains functions to extract features for keyword classification
"""

import re
import string
import nltk
from nltk.tag import StanfordNERTagger
from nltk.collocations import *
from gensim import corpora, models, similarities
from collections import defaultdict
import wikiwords
import numpy as np


def remove_punctuation(text):
  """
  Returns text free of punctuation marks
  """
  exclude = set(string.punctuation)
  return ''.join([ch for ch in text if ch not in exclude])


def get_namedentities(text):
  """
  Returns named entities in text using StanfordNERTagger
  """
  st = StanfordNERTagger('utils/english.conll.4class.caseless.distsim.crf.ser.gz','utils/stanford-ner.jar')   
  ner_tagged = st.tag(text.lower().split())     
  
  named_entities = []
  if len(ner_tagged) > 0:
    for n in ner_tagged:
      if n[1]!='O':
        named_entities.append(remove_punctuation(n[0]))

  named_entities = [n for n in named_entities if n] 
  return named_entities


def get_nounphrases(text):
  """
  Returns noun phrases in text
  """
  grammar = r""" 
    NBAR:
        {<NN.*|JJ>*<NN.*>}  

    NP:
        {<NBAR>}
        {<NBAR><IN><NBAR>}   # from Alex Bowe's nltk tutorial
  """    
  chunker = nltk.RegexpParser(grammar)
  sentences = nltk.sent_tokenize(text.lower())
  sentences = [nltk.word_tokenize(sent) for sent in sentences]
  sentences = [nltk.pos_tag(sent) for sent in sentences]

  noun_phrases = []
  for sent in sentences:
    tree = chunker.parse(sent)
    for subtree in tree.subtrees():
      if subtree.label() == 'NP': 
        noun_phrases.extend([w[0] for w in subtree.leaves()])

  noun_phrases = [remove_punctuation(nphrase) for nphrase in noun_phrases]
  noun_phrases = [n for n in noun_phrases if n]    
  return noun_phrases


def get_trigrams(text,num_trigrams):
  """
  Return all members of most frequent trigrams
  """
  trigram_measures = nltk.collocations.TrigramAssocMeasures()
  finder = TrigramCollocationFinder.from_words(text.lower().split())
  finder.apply_freq_filter(1) # ignore trigrams that occur only once
  top_ngrams = finder.nbest(trigram_measures.pmi,num_trigrams)
  
  ngrams = []
  for ng in top_ngrams:
      ngrams.extend(list(ng))    

  ngrams = [remove_punctuation(n) for n in list(set(ngrams))]
  ngrams = [n for n in ngrams if n]
  return ngrams


def get_binaryfeature(words,selected_words):
  """
  Returns a 0/1 encoding indicating membership in the set of selected words 
  """
  feature = map(lambda x: 1 if x else 0, [(w in selected_words) for w in words])
  return feature


def get_termfrequency(text,candidate_keywords):
  """
  Returns normalized term frequency for given keywords in text
  """
  words = [remove_punctuation(w) for w in text.lower().split()]
  words_str = ' '.join(words)
  return [len(re.findall(re.escape(c),words_str))/float(len(words)) for c in candidate_keywords]


def get_tfidf(candidate_keywords,corpus_entry,dictionary):
  """
  Returns tf-idf scores for keywords using a tf-idf transformation of 
  the text containing keywords
  """
  weights = []
  if corpus_entry:
    for candidate in candidate_keywords:
      if candidate in dictionary.token2id:
        tfidf_score = [w[1] for w in corpus_entry if w[0]==dictionary.token2id[candidate]]
        if len(tfidf_score)>0:
            weights.append(tfidf_score[0])
        else:
            weights.append(0)
      else:
        weights.append(0)
  else:
    weights = [0]*len(candidate_keywords)
      
  return weights  


def get_length(candidate_keywords):
  """
  Returns number of characters in each keyword
  """
  return [len(c) for c in candidate_keywords]


def get_position(text,candidate_keywords):
  """
  Returns first occurence of each keyword in text
  """
  words = [remove_punctuation(w) for w in text.lower().split()]  
  position = []
  for candidate in candidate_keywords:
    occurences = [pos for pos,w in enumerate(words) if w == candidate]
    if len(occurences)>0:
      position.append(occurences[0])
    else:
      position.append(0)
          
  return position  


def get_spread(text,candidate_keywords):
  """
  Returns the spread of each keyword in text. Spread is defined
  as the number of words between the first and last occurence of
  a keyword divided by the total number of words in text
  """
  words = [remove_punctuation(w) for w in text.lower().split()]  
  spread = []
  for candidate in candidate_keywords:
    occurences = [pos for pos,w in enumerate(words) if w == candidate]
    if len(occurences)>0:
      spread.append((occurences[-1]-occurences[0])/float(len(words)))
    else:
      spread.append(0)
          
  return spread


def get_capitalized(text,candidate_keywords):
  """
  Returns a 0/1 encoding indicating if any occurence of keyword included 
  capitalization
  """
  words_original = [remove_punctuation(w) for w in text.split()]
  words_lower = [remove_punctuation(w) for w in text.lower().split()]
  
  caps = []
  for candidate in candidate_keywords:
    occurences = [pos for pos,w in enumerate(words_lower) if w == candidate]
    if len(occurences)>0:
      any_caps = sum([1 for o in occurences if words_original[o]!=words_lower[o]])
      if any_caps>0:
        caps.append(1)
      else:
        caps.append(0)
    else:
      caps.append(0)
  
  return caps


def get_wikifrequencies(candidate_keywords):
  """
  Return absolute word frequency for each keyword in Wikipedia
  """
  return [wikiwords.freq(w) for w in candidate_keywords]


def extract_features(text,candidate_keywords,corpus_entry,dictionary):
  """
  Returns features for each candidate keyword using: 
  (i) the original text the keywords were derived from
  (ii) tf-idf transformation of original text
  """

  # setup 
  num_features = 10
  num_trigrams = 5

  # identify name entities, noun phrases, and ngrams
  named_entities = get_namedentities(text)
  noun_phrases = get_nounphrases(text)
  top_trigrams = get_trigrams(text,num_trigrams)


  # features 0-2: is the word in a named entity, noun phrase, or ngram?
  ne_feature = np.array(get_binaryfeature(candidate_keywords,named_entities))
  np_feature = np.array(get_binaryfeature(candidate_keywords,noun_phrases))
  ng_feature = np.array(get_binaryfeature(candidate_keywords,top_trigrams))

  # feature 3: term frequency
  tf_feature = np.array(get_termfrequency(text,candidate_keywords))

  # feature 4: tf-idf score
  tfidf_feature = np.array(get_tfidf(candidate_keywords,corpus_entry,dictionary))

  # feature 5: term length
  length_feature = np.array(get_length(candidate_keywords))

  # feature 6: first occurence of term in text
  position_feature = np.array(get_position(text,candidate_keywords))

  # feature 7: spread of term occurrences in text
  spread_feature = np.array(get_spread(text,candidate_keywords))

  # feature 8: capitalized?
  caps_feature = np.array(get_capitalized(text,candidate_keywords))

  # feature 9: frequency of occurence in wikipedia
  wiki_feature = np.array(get_wikifrequencies(candidate_keywords))

  # collect features
  features = np.zeros((len(candidate_keywords),num_features))
  features[:,0] = ne_feature
  features[:,1] = np_feature
  features[:,2] = ng_feature
  features[:,3] = tf_feature
  features[:,4] = tfidf_feature
  features[:,5] = length_feature
  features[:,6] = position_feature
  features[:,7] = spread_feature
  features[:,8] = caps_feature
  features[:,9] = wiki_feature

  feature_names = ['Named Entity','Noun Phrase','N-gram','Term Freq','TF-IDF','Term Length','First Occurence','Spread','Capitalized','Wikipedia frequency']
  return {'features': features, 'names': feature_names}
