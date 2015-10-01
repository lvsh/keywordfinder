Automatic keyword extraction
============================

As an [Insight Data Science Fellow](http://insightdatascience.com/), I completed a 3-week project that involved building a keyword extraction algorithm. Given a block of text as input, the algorithm selects keywords that describe what the text is about. Keywords are useful, compact descriptions of the original text, and they are widely used in information retrieval applications. 

#### Background

For this project, I partnered with [URX](http://urx.com/), a San Francisco-based startup in the mobile advertising space. URX matches advertisers and content providers, in a context-specific way. For example, if the content consists of a news article about hip-hop music, URX will serve ads for hip-hop albums on Spotify, a music streaming service. URX accomplishes this matching by extracting keywords from a content page, using those keywords to search a database of advertisers, and then serving the best matching ad.   

In my project, I focused on the keyword extraction step, and I built a prototype keyword extractor for URX. The deliverables were: (i) an algorithm for keyword extraction; and (ii) Python scripts to implement the algorithm. To learn about the algorithm I developed, check out the [project page](http://people.csail.mit.edu/lavanya/keywordfinder). 

#### Running the code

To get started, run the example:

```
python example.py inspec.txt
```

To evaluate the algorithm on [Crowd500](https://github.com/snkim/AutomaticKeyphraseExtraction) dataset from [Marujo et al., 2012](http://www.lrec-conf.org/proceedings/lrec2012/pdf/672_Paper.pdf), run:

```
python evaluatemodel.py 
```
Note that the algorithm is trained on the train set of Crowd500, but it is evaluated only on the test set of Crowd500. 

For comparison, I provide two baselines: random and [AlchemyAPI](http://www.alchemyapi.com/). The random baseline selects words at random from the given text, whereas the AlchemyAPI baseline consists of keywords returned by the Alchemy analytics engine. The top-15 keyword evaluation methodology is similar to that of [Jean-Louis et al., 2014](http://azouaq.athabascau.ca/publications/Conferences,%20Workshops,%20Books/%5BC28%5D_PRICAI_2014.pdf).

```
python evaluaterandom.py 
python evaluatealchemy.py 
```

My algorithm (f1 score = 23.95) outperforms AlchemyAPI (f1 score = 21.19), and beats the random baseline quite easily (f1 = 8.41). It is worth noting that my algorithm was trained on the Crowd500 train set, whereas the Alchemy keyword extractor (presumably) was not. Additionally, Alchemy excels at returning keyphrases rather than keywords, which this benchmark does not assess. 



