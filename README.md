Automatic keyword extraction from text
=======================================

As an [Insight Data Science Fellow](http://insightdatascience.com/), I completed a 3-week project that involved building a keyword extraction algorithm. Given a block of text as input, the algorithm selects keywords that describe what the text is about. Keywords are useful, compact descriptions of the original text, and they are widely used in information retrieval. 

### Background

For this project, I partnered with [URX](http://urx.com/), a San Francisco-based startup in the mobile advertising space. URX matches advertisers and content providers, in a context-specific way. For example, if the content consists of a news article about hip-hop music, URX serves an ad for Spotify, a music streaming service. URX accomplishes this matching by: (1) extracting keywords from a content page; (2) using those keywords to index into a database of advertisers; and (3) serving the best matching ad.   

In my project, I focused on the keyword extraction step, and I built a prototype keyword extractor for URX.  


### Running the code

To get started, run the example:

```
python example.py inspec.txt
```
