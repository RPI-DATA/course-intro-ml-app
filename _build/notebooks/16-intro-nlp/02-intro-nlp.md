---
interact_link: content/notebooks/16-intro-nlp/02-intro-nlp.ipynb
kernel_name: python3
has_widgets: false
title: 'Overview of NLP'
prev_page:
  url: /notebooks/16-intro-nlp/06-sentiment.html
  title: 'Sentiment'
next_page:
  url: /notebooks/16-intro-nlp/07-fastai-imdb.html
  title: 'FAST.ai NLP'
comment: "***PROGRAMMATICALLY GENERATED, DO NOT EDIT. SEE ORIGINAL FILES IN /content***"
---


## Introduction to Natural Language Processing

In this workbook, at a high-level we will learn about text tokenization; text normalization such as lowercasing, stemming; part-of-speech tagging; Named entity recognition; Sentiment analysis; Topic modeling; Word embeddings







<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```####PLEASE EXECUTE THESE COMMANDS BEFORE PROCEEDING####

import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')

```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">
{:.output_stream}
```
[nltk_data] Downloading package punkt to /root/nltk_data...
[nltk_data]   Package punkt is already up-to-date!
[nltk_data] Downloading package averaged_perceptron_tagger to
[nltk_data]     /root/nltk_data...
[nltk_data]   Package averaged_perceptron_tagger is already up-to-
[nltk_data]       date!
[nltk_data] Downloading package stopwords to /root/nltk_data...
[nltk_data]   Unzipping corpora/stopwords.zip.
```
</div>
</div>
<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">


{:.output_data_text}
```
True
```


</div>
</div>
</div>



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```#Tokenization -- Text into word tokens; Paragraphs into sentences;
from nltk.tokenize import sent_tokenize 
  
text = "Hello everyone. Welcome to Intro to Machine Learning Applications. We are now learning important basics of NLP."
sent_tokenize(text) 



```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">


{:.output_data_text}
```
['Hello everyone.',
 'Welcome to Intro to Machine Learning Applications.',
 'We are now learning important basics of NLP.']
```


</div>
</div>
</div>



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```import nltk.data 
  
german_tokenizer = nltk.data.load('tokenizers/punkt/PY3/german.pickle') 
  
text = 'Wie geht es Ihnen? Mir geht es gut.'
german_tokenizer.tokenize(text) 


```
</div>

</div>



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```from nltk.tokenize import word_tokenize 
  
text = "Hello everyone. Welcome to Intro to Machine Learning Applications. We are now learning important basics of NLP."
word_tokenize(text) 



```
</div>

</div>



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```from nltk.tokenize import TreebankWordTokenizer 
  
tokenizer = TreebankWordTokenizer() 
tokenizer.tokenize(text) 


```
</div>

</div>



###n-grams vs tokens

##### n-grams are contiguous sequences of n-items in a sentence. N can be 1, 2 or any other positive integers, although usually we do not consider very large N because those n-grams rarely appears in many different places.

##### Tokens do not have any conditions on contiguity



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```#Using pure python

import re

def generate_ngrams(text, n):
    # Convert to lowercases
    text = text.lower()
    
    # Replace all none alphanumeric characters with spaces
    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
    
    # Break sentence in the token, remove empty tokens
    tokens = [token for token in text.split(" ") if token != ""]
    
    # Use the zip function to help us generate n-grams
    # Concatentate the tokens into ngrams and return
    ngrams = zip(*[tokens[i:] for i in range(n)])
    return [" ".join(ngram) for ngram in ngrams]

text = "Hello everyone. Welcome to Intro to Machine Learning Applications. We are now learning important basics of NLP."
print(text)
generate_ngrams(text, n=2)

```
</div>

</div>



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```#Using NLTK import ngrams

import re
from nltk.util import ngrams

text = text.lower()
text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
tokens = [token for token in text.split(" ") if token != ""]
output = list(ngrams(tokens, 3))
print(output)

```
</div>

</div>



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```#Text Normalization

#Lowercasing
text = "Hello everyone. Welcome to Intro to Machine Learning Applications. We are now learning important basics of NLP."
lowert = text.lower()
uppert = text.upper()

print(lowert)
print(uppert)


```
</div>

</div>



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```#Text Normalization
#stemming
#Porter stemmer is a famous stemming approach

from nltk.stem import PorterStemmer 
from nltk.tokenize import word_tokenize 
   
ps = PorterStemmer() 
  
# choose some words to be stemmed 
words = ["hike", "hikes", "hiked", "hiking", "hikers", "hiker"] 
  
for w in words: 
    print(w, " : ", ps.stem(w)) 



```
</div>

</div>



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```from nltk.stem import PorterStemmer 
from nltk.tokenize import word_tokenize 
import re
   
ps = PorterStemmer() 
text = "Hello everyone. Welcome to Intro to Machine Learning Applications. We are now learning important basics of NLP."
print(text)


#Tokenize and stem the words
text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
tokens = [token for token in text.split(" ") if token != ""]

i=0
while i<len(tokens):
  tokens[i]=ps.stem(tokens[i])
  i=i+1

#merge all the tokens to form a long text sequence 
text2 = ' '.join(tokens) 

print(text2)

```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">
{:.output_stream}
```
Hello everyone. Welcome to Intro to Machine Learning Applications. We are now learning important basics of NLP.
hello everyon welcom to intro to machin learn applic We are now learn import basic of nlp
```
</div>
</div>
</div>



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import word_tokenize 
import re
   
ss = SnowballStemmer("english")
text = "Hello everyone. Welcome to Intro to Machine Learning Applications. We are now learning important basics of NLP."
print(text)


#Tokenize and stem the words
text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
tokens = [token for token in text.split(" ") if token != ""]

i=0
while i<len(tokens):
  tokens[i]=ss.stem(tokens[i])
  i=i+1

#merge all the tokens to form a long text sequence 
text2 = ' '.join(tokens) 

print(text2)

```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">
{:.output_stream}
```
Hello everyone. Welcome to Intro to Machine Learning Applications. We are now learning important basics of NLP.
hello everyon welcom to intro to machin learn applic we are now learn import basic of nlp
```
</div>
</div>
</div>



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```#Stopwords removal 

from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize 

text = "Hello everyone. Welcome to Intro to Machine Learning Applications. We are now learning important basics of NLP."

stop_words = set(stopwords.words('english')) 
word_tokens = word_tokenize(text) 
  
filtered_sentence = [w for w in word_tokens if not w in stop_words] 
  
filtered_sentence = [] 
  
for w in word_tokens: 
    if w not in stop_words: 
        filtered_sentence.append(w) 
  
print(word_tokens) 
print(filtered_sentence) 

text2 = ' '.join(filtered_sentence)

```
</div>

</div>



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```#Part-of-Speech tagging

import nltk
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag

text = 'GitHub is a development platform inspired by the way you work. From open source to business, you can host and review code, manage projects, and build software alongside 40 million developers.'

def preprocess(sent):
    sent = nltk.word_tokenize(sent)
    sent = nltk.pos_tag(sent)
    return sent

sent = preprocess(text)
print(sent)


```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">
{:.output_stream}
```
[('GitHub', 'NNP'), ('is', 'VBZ'), ('a', 'DT'), ('development', 'NN'), ('platform', 'NN'), ('inspired', 'VBN'), ('by', 'IN'), ('the', 'DT'), ('way', 'NN'), ('you', 'PRP'), ('work', 'VBP'), ('.', '.'), ('From', 'IN'), ('open', 'JJ'), ('source', 'NN'), ('to', 'TO'), ('business', 'NN'), (',', ','), ('you', 'PRP'), ('can', 'MD'), ('host', 'VB'), ('and', 'CC'), ('review', 'VB'), ('code', 'NN'), (',', ','), ('manage', 'NN'), ('projects', 'NNS'), (',', ','), ('and', 'CC'), ('build', 'VB'), ('software', 'NN'), ('alongside', 'RB'), ('40', 'CD'), ('million', 'CD'), ('developers', 'NNS'), ('.', '.')]
```
</div>
</div>
</div>



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```#Named entity recognition

#spaCy is an NLP Framework -- easy to use and having ability to use neural networks

import en_core_web_sm
nlp = en_core_web_sm.load()

text = 'GitHub is a development platform inspired by the way you work. From open source to business, you can host and review code, manage projects, and build software alongside 40 million developers.'

doc = nlp(text)
print(doc.ents)
print([(X.text, X.label_) for X in doc.ents])

```
</div>

</div>



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```#Sentiment analysis

```
</div>

</div>



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```#Topic modeling

```
</div>

</div>



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```#Word embeddings


```
</div>

</div>



#Class exercise

#### 1. Read a file from its URL
#### 2. Extract the text and tokenize it meaningfully into words.
#### 3. Print the entire text combined after tokenization.
#### 4. Perform stemming using both porter and snowball stemmers. Which one works the best? Why?
#### 5. Remove stopwords
#### 6. Identify the top-10 unigrams based on their frequency.




<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```
#Load the file first
!wget https://www.dropbox.com/s/o8lxi6yrezmt5em/reviews.txt


```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">
{:.output_stream}
```
--2019-11-04 17:16:22--  https://www.dropbox.com/s/o8lxi6yrezmt5em/reviews.txt
Resolving www.dropbox.com (www.dropbox.com)... 162.125.9.1, 2620:100:601b:1::a27d:801
Connecting to www.dropbox.com (www.dropbox.com)|162.125.9.1|:443... connected.
HTTP request sent, awaiting response... 301 Moved Permanently
Location: /s/raw/o8lxi6yrezmt5em/reviews.txt [following]
--2019-11-04 17:16:23--  https://www.dropbox.com/s/raw/o8lxi6yrezmt5em/reviews.txt
Reusing existing connection to www.dropbox.com:443.
HTTP request sent, awaiting response... 302 Found
Location: https://ucb753980f94c903b140fb69cb47.dl.dropboxusercontent.com/cd/0/inline/AruGnazr2R1e797TKXdu6chwkg102fB893qSsoT5EeI2_mAFsj2rCinxKGPdm-HpQjOZqWQ21tvsPDpyA7PBxc7QxoDCWKG45GDwN1gZw3C7RlMLoxb8D9NG9IqmJ25IXJc/file# [following]
--2019-11-04 17:16:23--  https://ucb753980f94c903b140fb69cb47.dl.dropboxusercontent.com/cd/0/inline/AruGnazr2R1e797TKXdu6chwkg102fB893qSsoT5EeI2_mAFsj2rCinxKGPdm-HpQjOZqWQ21tvsPDpyA7PBxc7QxoDCWKG45GDwN1gZw3C7RlMLoxb8D9NG9IqmJ25IXJc/file
Resolving ucb753980f94c903b140fb69cb47.dl.dropboxusercontent.com (ucb753980f94c903b140fb69cb47.dl.dropboxusercontent.com)... 162.125.9.6, 2620:100:601f:6::a27d:906
Connecting to ucb753980f94c903b140fb69cb47.dl.dropboxusercontent.com (ucb753980f94c903b140fb69cb47.dl.dropboxusercontent.com)|162.125.9.6|:443... connected.
HTTP request sent, awaiting response... 200 OK
Length: 3851 (3.8K) [text/plain]
Saving to: ‘reviews.txt’

reviews.txt         100%[===================>]   3.76K  --.-KB/s    in 0s      

2019-11-04 17:16:24 (328 MB/s) - ‘reviews.txt’ saved [3851/3851]

```
</div>
</div>
</div>

