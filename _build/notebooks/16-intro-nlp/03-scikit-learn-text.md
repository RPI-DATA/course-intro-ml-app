---
interact_link: content/notebooks/16-intro-nlp/03-scikit-learn-text.ipynb
kernel_name: python3
has_widgets: false
title: 'Scikit Learn Text'
prev_page:
  url: /notebooks/16-intro-nlp/02-corpus-simple.html
  title: 'Corpus Simple'
next_page:
  url: /notebooks/16-intro-nlp/04-what-cooking-python.html
  title: 'What's Cooking Python'
comment: "***PROGRAMMATICALLY GENERATED, DO NOT EDIT. SEE ORIGINAL FILES IN /content***"
---


[![AnalyticsDojo](https://github.com/rpi-techfundamentals/spring2019-materials/blob/master/fig/final-logo.png?raw=1)](http://rpi.analyticsdojo.com)
<center><h1>Methods - Text Feature Extraction with Bag-of-Words Using Scikit Learn</h1></center>
<center><h3><a href = 'http://rpi.analyticsdojo.com'>rpi.analyticsdojo.com</a></h3></center>



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```%matplotlib inline
import matplotlib.pyplot as plt
import numpy as np

```
</div>

</div>



# Methods - Text Feature Extraction with Bag-of-Words Using Scikit Learn




In many tasks, like in the classical spam detection, your input data is text.
Free text with variables length is very far from the fixed length numeric representation that we need to do machine learning with scikit-learn.
However, there is an easy and effective way to go from text data to a numeric representation using the so-called bag-of-words model, which provides a data structure that is compatible with the machine learning aglorithms in scikit-learn.



Let's assume that each sample in your dataset is represented as one string, which could be just a sentence, an email, or a whole news article or book. To represent the sample, we first split the string into a list of tokens, which correspond to (somewhat normalized) words. A simple way to do this to just split by whitespace, and then lowercase the word. 

Then, we build a vocabulary of all tokens (lowercased words) that appear in our whole dataset. This is usually a very large vocabulary.
Finally, looking at our single sample, we could show how often each word in the vocabulary appears.
We represent our string by a vector, where each entry is how often a given word in the vocabulary appears in the string.

As each sample will only contain very few words, most entries will be zero, leading to a very high-dimensional but sparse representation.

The method is called "bag-of-words," as the order of the words is lost entirely.



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```X = ["Mr. Green killed Colonel Mustard in the study with the candlestick. \
Mr. Green is not a very nice fellow.",
     "Professor Plum has a green plant in his study.",
    "Miss Scarlett watered Professor Plum's green plant while he was away \
from his office last week."]


```
</div>

</div>



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```len(X)

```
</div>

</div>



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer()
vectorizer.fit(X)

```
</div>

</div>



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```vectorizer.vocabulary_

```
</div>

</div>



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```X_bag_of_words = vectorizer.transform(X)

```
</div>

</div>



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```X_bag_of_words.shape

```
</div>

</div>



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```X_bag_of_words

```
</div>

</div>



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```X_bag_of_words.toarray()

```
</div>

</div>



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```vectorizer.get_feature_names()

```
</div>

</div>



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```vectorizer.inverse_transform(X_bag_of_words)

```
</div>

</div>



# tf-idf Encoding
A useful transformation that is often applied to the bag-of-word encoding is the so-called term-frequency inverse-document-frequency (tf-idf) scaling, which is a non-linear transformation of the word counts.

The tf-idf encoding rescales words that are common to have less weight:



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```from sklearn.feature_extraction.text import TfidfVectorizer
tfidf_vectorizer = TfidfVectorizer()
tfidf_vectorizer.fit(X)

```
</div>

</div>



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```import numpy as np
np.set_printoptions(precision=2)

print(tfidf_vectorizer.transform(X).toarray())

```
</div>

</div>



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```tfidf_vectorizer.get_feature_names()

```
</div>

</div>



tf-idfs are a way to represent documents as feature vectors. tf-idfs can be understood as a modification of the raw term frequencies (`tf`); the `tf` is the count of how often a particular word occurs in a given document. The concept behind the tf-idf is to downweight terms proportionally to the number of documents in which they occur. Here, the idea is that terms that occur in many different documents are likely unimportant or don't contain any useful information for Natural Language Processing tasks such as document classification. If you are interested in the mathematical details and equations, see this [external IPython Notebook](http://nbviewer.jupyter.org/github/rasbt/pattern_classification/blob/master/machine_learning/scikit-learn/tfidf_scikit-learn.ipynb) that walks you through the computation.



# Bigrams and N-Grams

In the example illustrated in the figure at the beginning of this notebook, we used the so-called 1-gram (unigram) tokenization: Each token represents a single element with regard to the splittling criterion. 

Entirely discarding word order is not always a good idea, as composite phrases often have specific meaning, and modifiers like "not" can invert the meaning of words.

A simple way to include some word order are n-grams, which don't only look at a single token, but at all pairs of neighborhing tokens. For example, in 2-gram (bigram) tokenization, we would group words together with an overlap of one word; in 3-gram (trigram) splits we would create an overlap two words, and so forth:

- original text: "this is how you get ants"
- 1-gram: "this", "is", "how", "you", "get", "ants"
- 2-gram: "this is", "is how", "how you", "you get", "get ants"
- 3-gram: "this is how", "is how you", "how you get", "you get ants"

Which "n" we choose for "n-gram" tokenization to obtain the optimal performance in our predictive model depends on the learning algorithm, dataset, and task. Or in other words, we have consider "n" in "n-grams" as a tuning parameters, and in later notebooks, we will see how we deal with these.

Now, let's create a bag of words model of bigrams using scikit-learn's `CountVectorizer`:



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```# look at sequences of tokens of minimum length 2 and maximum length 2
bigram_vectorizer = CountVectorizer(ngram_range=(2, 2))
bigram_vectorizer.fit(X)

```
</div>

</div>



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```bigram_vectorizer.get_feature_names()

```
</div>

</div>



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```bigram_vectorizer.transform(X).toarray()

```
</div>

</div>



Often we want to include unigrams (single tokens) AND bigrams, wich we can do by passing the following tuple as an argument to the `ngram_range` parameter of the `CountVectorizer` function:



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```gram_vectorizer = CountVectorizer(ngram_range=(1, 2))
gram_vectorizer.fit(X)

```
</div>

</div>



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```gram_vectorizer.get_feature_names()

```
</div>

</div>



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```gram_vectorizer.transform(X).toarray()

```
</div>

</div>



Character n-grams
=================

Sometimes it is also helpful not only to look at words, but to consider single characters instead.   
That is particularly useful if we have very noisy data and want to identify the language, or if we want to predict something about a single word.
We can simply look at characters instead of words by setting ``analyzer="char"``.
Looking at single characters is usually not very informative, but looking at longer n-grams of characters could be:



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```X

```
</div>

</div>



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```char_vectorizer = CountVectorizer(ngram_range=(2, 2), analyzer="char")
char_vectorizer.fit(X)

```
</div>

</div>



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```print(char_vectorizer.get_feature_names())

```
</div>

</div>

