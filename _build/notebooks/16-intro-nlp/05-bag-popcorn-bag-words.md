---
interact_link: content/notebooks/16-intro-nlp/05-bag-popcorn-bag-words.ipynb
kernel_name: python3
has_widgets: false
title: 'Bag of Popcorn Bag of Words'
prev_page:
  url: /notebooks/16-intro-nlp/04-what-cooking-python.html
  title: 'What's Cooking Python'
next_page:
  url: /notebooks/16-intro-nlp/06-sentiment.html
  title: 'Sentiment'
comment: "***PROGRAMMATICALLY GENERATED, DO NOT EDIT. SEE ORIGINAL FILES IN /content***"
---


[![AnalyticsDojo](https://github.com/rpi-techfundamentals/spring2019-materials/blob/master/fig/final-logo.png?raw=1)](http://rpi.analyticsdojo.com)
<center><h1> Bag of Words</h1></center>
<center><h3><a href = 'http://rpi.analyticsdojo.com'>rpi.analyticsdojo.com</a></h3></center>



This is adopted from: [Bag of Words Meets Bags of Popcorn](https://www.kaggle.com/c/word2vec-nlp-tutorial/details/part-1-for-beginners-bag-of-words)
[https://github.com/wendykan/DeepLearningMovies](https://github.com/wendykan/DeepLearningMovies)




<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
import nltk
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
from nltk.corpus import stopwords

```
</div>

</div>



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
!wget https://github.com/rpi-techfundamentals/spring2019-materials/raw/master/input/labeledTrainData.tsv
!wget https://github.com/rpi-techfundamentals/spring2019-materials/raw/master/input/unlabeledTrainData.tsv
!wget https://github.com/rpi-techfundamentals/spring2019-materials/raw/master/input/testData.tsv

```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">
{:.output_stream}
```
--2019-03-14 00:57:13--  https://github.com/rpi-techfundamentals/spring2019-materials/raw/master/input/labeledTrainData.tsv
Resolving github.com (github.com)... 192.30.253.113, 192.30.253.112
Connecting to github.com (github.com)|192.30.253.113|:443... connected.
HTTP request sent, awaiting response... 302 Found
Location: https://raw.githubusercontent.com/rpi-techfundamentals/spring2019-materials/master/input/labeledTrainData.tsv [following]
--2019-03-14 00:57:13--  https://raw.githubusercontent.com/rpi-techfundamentals/spring2019-materials/master/input/labeledTrainData.tsv
Resolving raw.githubusercontent.com (raw.githubusercontent.com)... 151.101.0.133, 151.101.64.133, 151.101.128.133, ...
Connecting to raw.githubusercontent.com (raw.githubusercontent.com)|151.101.0.133|:443... connected.
HTTP request sent, awaiting response... 200 OK
Length: 33556378 (32M) [text/plain]
Saving to: ‘labeledTrainData.tsv.1’

labeledTrainData.ts 100%[===================>]  32.00M   191MB/s    in 0.2s    

2019-03-14 00:57:14 (191 MB/s) - ‘labeledTrainData.tsv.1’ saved [33556378/33556378]

--2019-03-14 00:57:15--  https://github.com/rpi-techfundamentals/spring2019-materials/raw/master/input/unlabeledTrainData.tsv
Resolving github.com (github.com)... 192.30.253.113, 192.30.253.112
Connecting to github.com (github.com)|192.30.253.113|:443... connected.
HTTP request sent, awaiting response... 302 Found
Location: https://raw.githubusercontent.com/rpi-techfundamentals/spring2019-materials/master/input/unlabeledTrainData.tsv [following]
--2019-03-14 00:57:15--  https://raw.githubusercontent.com/rpi-techfundamentals/spring2019-materials/master/input/unlabeledTrainData.tsv
Resolving raw.githubusercontent.com (raw.githubusercontent.com)... 151.101.0.133, 151.101.64.133, 151.101.128.133, ...
Connecting to raw.githubusercontent.com (raw.githubusercontent.com)|151.101.0.133|:443... connected.
HTTP request sent, awaiting response... 200 OK
Length: 67281491 (64M) [text/plain]
Saving to: ‘unlabeledTrainData.tsv.1’

unlabeledTrainData. 100%[===================>]  64.16M   234MB/s    in 0.3s    

2019-03-14 00:57:15 (234 MB/s) - ‘unlabeledTrainData.tsv.1’ saved [67281491/67281491]

--2019-03-14 00:57:16--  https://github.com/rpi-techfundamentals/spring2019-materials/raw/master/input/testData.tsv
Resolving github.com (github.com)... 192.30.253.113, 192.30.253.112
Connecting to github.com (github.com)|192.30.253.113|:443... connected.
HTTP request sent, awaiting response... 302 Found
Location: https://raw.githubusercontent.com/rpi-techfundamentals/spring2019-materials/master/input/testData.tsv [following]
--2019-03-14 00:57:17--  https://raw.githubusercontent.com/rpi-techfundamentals/spring2019-materials/master/input/testData.tsv
Resolving raw.githubusercontent.com (raw.githubusercontent.com)... 151.101.0.133, 151.101.64.133, 151.101.128.133, ...
Connecting to raw.githubusercontent.com (raw.githubusercontent.com)|151.101.0.133|:443... connected.
HTTP request sent, awaiting response... 200 OK
Length: 32724746 (31M) [text/plain]
Saving to: ‘testData.tsv.1’

testData.tsv.1      100%[===================>]  31.21M   147MB/s    in 0.2s    

2019-03-14 00:57:17 (147 MB/s) - ‘testData.tsv.1’ saved [32724746/32724746]

```
</div>
</div>
</div>



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
train = pd.read_csv('labeledTrainData.tsv', header=0, \
                    delimiter="\t", quoting=3)
unlabeled_train= pd.read_csv('unlabeledTrainData.tsv', header=0, \
                    delimiter="\t", quoting=3)
test = pd.read_csv('testData.tsv', header=0, \
                    delimiter="\t", quoting=3)

```
</div>

</div>



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
import os
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np

```
</div>

</div>



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
print(train.columns.values, test.columns.values)

```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">
{:.output_stream}
```
['id' 'sentiment' 'review'] ['id' 'review']
```
</div>
</div>
</div>



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
train.head()

```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">



<div markdown="0" class="output output_html">
<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>sentiment</th>
      <th>review</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>"5814_8"</td>
      <td>1</td>
      <td>"With all this stuff going down at the moment ...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>"2381_9"</td>
      <td>1</td>
      <td>"\"The Classic War of the Worlds\" by Timothy ...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>"7759_3"</td>
      <td>0</td>
      <td>"The film starts with a manager (Nicholas Bell...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>"3630_4"</td>
      <td>0</td>
      <td>"It must be assumed that those who praised thi...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>"9495_8"</td>
      <td>1</td>
      <td>"Superbly trashy and wondrously unpretentious ...</td>
    </tr>
  </tbody>
</table>
</div>
</div>


</div>
</div>
</div>



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
print('The train shape is: ', train.shape)
print('The train shape is: ', test.shape)

```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">
{:.output_stream}
```
The train shape is:  (25000, 3)
The train shape is:  (25000, 2)
```
</div>
</div>
</div>



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
print('The first review is:')
print(train["review"][0])


```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">
{:.output_stream}
```
The first review is:
"With all this stuff going down at the moment with MJ i've started listening to his music, watching the odd documentary here and there, watched The Wiz and watched Moonwalker again. Maybe i just want to get a certain insight into this guy who i thought was really cool in the eighties just to maybe make up my mind whether he is guilty or innocent. Moonwalker is part biography, part feature film which i remember going to see at the cinema when it was originally released. Some of it has subtle messages about MJ's feeling towards the press and also the obvious message of drugs are bad m'kay.<br /><br />Visually impressive but of course this is all about Michael Jackson so unless you remotely like MJ in anyway then you are going to hate this and find it boring. Some may call MJ an egotist for consenting to the making of this movie BUT MJ and most of his fans would say that he made it for the fans which if true is really nice of him.<br /><br />The actual feature film bit when it finally starts is only on for 20 minutes or so excluding the Smooth Criminal sequence and Joe Pesci is convincing as a psychopathic all powerful drug lord. Why he wants MJ dead so bad is beyond me. Because MJ overheard his plans? Nah, Joe Pesci's character ranted that he wanted people to know it is he who is supplying drugs etc so i dunno, maybe he just hates MJ's music.<br /><br />Lots of cool things in this like MJ turning into a car and a robot and the whole Speed Demon sequence. Also, the director must have had the patience of a saint when it came to filming the kiddy Bad sequence as usually directors hate working with one kid let alone a whole bunch of them performing a complex dance scene.<br /><br />Bottom line, this movie is for people who like MJ on one level or another (which i think is most people). If not, then stay away. It does try and give off a wholesome message and ironically MJ's bestest buddy in this movie is a girl! Michael Jackson is truly one of the most talented people ever to grace this planet but is he guilty? Well, with all the attention i've gave this subject....hmmm well i don't know because people can be different behind closed doors, i know this for a fact. He is either an extremely nice but stupid guy or one of the most sickest liars. I hope he is not the latter."
```
</div>
</div>
</div>



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
# Import BeautifulSoup into your workspace
from bs4 import BeautifulSoup             

# Initialize the BeautifulSoup object on a single movie review     
example1 = BeautifulSoup(train["review"][0], "html.parser" )  
print(example1.get_text())

```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">
{:.output_stream}
```
"With all this stuff going down at the moment with MJ i've started listening to his music, watching the odd documentary here and there, watched The Wiz and watched Moonwalker again. Maybe i just want to get a certain insight into this guy who i thought was really cool in the eighties just to maybe make up my mind whether he is guilty or innocent. Moonwalker is part biography, part feature film which i remember going to see at the cinema when it was originally released. Some of it has subtle messages about MJ's feeling towards the press and also the obvious message of drugs are bad m'kay.Visually impressive but of course this is all about Michael Jackson so unless you remotely like MJ in anyway then you are going to hate this and find it boring. Some may call MJ an egotist for consenting to the making of this movie BUT MJ and most of his fans would say that he made it for the fans which if true is really nice of him.The actual feature film bit when it finally starts is only on for 20 minutes or so excluding the Smooth Criminal sequence and Joe Pesci is convincing as a psychopathic all powerful drug lord. Why he wants MJ dead so bad is beyond me. Because MJ overheard his plans? Nah, Joe Pesci's character ranted that he wanted people to know it is he who is supplying drugs etc so i dunno, maybe he just hates MJ's music.Lots of cool things in this like MJ turning into a car and a robot and the whole Speed Demon sequence. Also, the director must have had the patience of a saint when it came to filming the kiddy Bad sequence as usually directors hate working with one kid let alone a whole bunch of them performing a complex dance scene.Bottom line, this movie is for people who like MJ on one level or another (which i think is most people). If not, then stay away. It does try and give off a wholesome message and ironically MJ's bestest buddy in this movie is a girl! Michael Jackson is truly one of the most talented people ever to grace this planet but is he guilty? Well, with all the attention i've gave this subject....hmmm well i don't know because people can be different behind closed doors, i know this for a fact. He is either an extremely nice but stupid guy or one of the most sickest liars. I hope he is not the latter."
```
</div>
</div>
</div>



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
import re
# Use regular expressions to do a find-and-replace
letters_only = re.sub("[^a-zA-Z]",           # The pattern to search for
                      " ",                   # The pattern to replace it with
                      example1.get_text() )  # The text to search
print (letters_only)

```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">
{:.output_stream}
```
 With all this stuff going down at the moment with MJ i ve started listening to his music  watching the odd documentary here and there  watched The Wiz and watched Moonwalker again  Maybe i just want to get a certain insight into this guy who i thought was really cool in the eighties just to maybe make up my mind whether he is guilty or innocent  Moonwalker is part biography  part feature film which i remember going to see at the cinema when it was originally released  Some of it has subtle messages about MJ s feeling towards the press and also the obvious message of drugs are bad m kay Visually impressive but of course this is all about Michael Jackson so unless you remotely like MJ in anyway then you are going to hate this and find it boring  Some may call MJ an egotist for consenting to the making of this movie BUT MJ and most of his fans would say that he made it for the fans which if true is really nice of him The actual feature film bit when it finally starts is only on for    minutes or so excluding the Smooth Criminal sequence and Joe Pesci is convincing as a psychopathic all powerful drug lord  Why he wants MJ dead so bad is beyond me  Because MJ overheard his plans  Nah  Joe Pesci s character ranted that he wanted people to know it is he who is supplying drugs etc so i dunno  maybe he just hates MJ s music Lots of cool things in this like MJ turning into a car and a robot and the whole Speed Demon sequence  Also  the director must have had the patience of a saint when it came to filming the kiddy Bad sequence as usually directors hate working with one kid let alone a whole bunch of them performing a complex dance scene Bottom line  this movie is for people who like MJ on one level or another  which i think is most people   If not  then stay away  It does try and give off a wholesome message and ironically MJ s bestest buddy in this movie is a girl  Michael Jackson is truly one of the most talented people ever to grace this planet but is he guilty  Well  with all the attention i ve gave this subject    hmmm well i don t know because people can be different behind closed doors  i know this for a fact  He is either an extremely nice but stupid guy or one of the most sickest liars  I hope he is not the latter  
```
</div>
</div>
</div>



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
lower_case = letters_only.lower()        # Convert to lower case
words = lower_case.split()               # Split into words


```
</div>

</div>



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
# Enter Download then stopwords.
nltk.download('stopwords')

```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">
{:.output_stream}
```
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
```python
print (stopwords.words("english"))

```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">
{:.output_stream}
```
['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"]
```
</div>
</div>
</div>



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
# Remove stop words from "words"
words = [w for w in words if not w in stopwords.words("english")]
print (words)

```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">
{:.output_stream}
```
['stuff', 'going', 'moment', 'mj', 'started', 'listening', 'music', 'watching', 'odd', 'documentary', 'watched', 'wiz', 'watched', 'moonwalker', 'maybe', 'want', 'get', 'certain', 'insight', 'guy', 'thought', 'really', 'cool', 'eighties', 'maybe', 'make', 'mind', 'whether', 'guilty', 'innocent', 'moonwalker', 'part', 'biography', 'part', 'feature', 'film', 'remember', 'going', 'see', 'cinema', 'originally', 'released', 'subtle', 'messages', 'mj', 'feeling', 'towards', 'press', 'also', 'obvious', 'message', 'drugs', 'bad', 'kay', 'visually', 'impressive', 'course', 'michael', 'jackson', 'unless', 'remotely', 'like', 'mj', 'anyway', 'going', 'hate', 'find', 'boring', 'may', 'call', 'mj', 'egotist', 'consenting', 'making', 'movie', 'mj', 'fans', 'would', 'say', 'made', 'fans', 'true', 'really', 'nice', 'actual', 'feature', 'film', 'bit', 'finally', 'starts', 'minutes', 'excluding', 'smooth', 'criminal', 'sequence', 'joe', 'pesci', 'convincing', 'psychopathic', 'powerful', 'drug', 'lord', 'wants', 'mj', 'dead', 'bad', 'beyond', 'mj', 'overheard', 'plans', 'nah', 'joe', 'pesci', 'character', 'ranted', 'wanted', 'people', 'know', 'supplying', 'drugs', 'etc', 'dunno', 'maybe', 'hates', 'mj', 'music', 'lots', 'cool', 'things', 'like', 'mj', 'turning', 'car', 'robot', 'whole', 'speed', 'demon', 'sequence', 'also', 'director', 'must', 'patience', 'saint', 'came', 'filming', 'kiddy', 'bad', 'sequence', 'usually', 'directors', 'hate', 'working', 'one', 'kid', 'let', 'alone', 'whole', 'bunch', 'performing', 'complex', 'dance', 'scene', 'bottom', 'line', 'movie', 'people', 'like', 'mj', 'one', 'level', 'another', 'think', 'people', 'stay', 'away', 'try', 'give', 'wholesome', 'message', 'ironically', 'mj', 'bestest', 'buddy', 'movie', 'girl', 'michael', 'jackson', 'truly', 'one', 'talented', 'people', 'ever', 'grace', 'planet', 'guilty', 'well', 'attention', 'gave', 'subject', 'hmmm', 'well', 'know', 'people', 'different', 'behind', 'closed', 'doors', 'know', 'fact', 'either', 'extremely', 'nice', 'stupid', 'guy', 'one', 'sickest', 'liars', 'hope', 'latter']
```
</div>
</div>
</div>



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
#Now we are going to do our first class
class KaggleWord2VecUtility(object):
    """KaggleWord2VecUtility is a utility class for processing raw HTML text into segments for further learning"""

    @staticmethod
    def review_to_wordlist( review, remove_stopwords=False ):
        # Function to convert a document to a sequence of words,
        # optionally removing stop words.  Returns a list of words.
        #
        # 1. Remove HTML
        review_text = BeautifulSoup(review,"html.parser"  ).get_text()
        #
        # 2. Remove non-letters
        review_text = re.sub("[^a-zA-Z]"," ", review_text)
        #
        # 3. Convert words to lower case and split them
        words = review_text.lower().split()
        #
        # 4. Optionally remove stop words (false by default)
        if remove_stopwords:
            stops = set(stopwords.words("english"))
            words = [w for w in words if not w in stops]
        #
        # 5. Return a list of words
        return(words)

    # Define a function to split a review into parsed sentences
    @staticmethod
    def review_to_sentences( review, tokenizer, remove_stopwords=False ):
        # Function to split a review into parsed sentences. Returns a
        # list of sentences, where each sentence is a list of words
        #
        # 1. Use the NLTK tokenizer to split the paragraph into sentences
        raw_sentences = tokenizer.tokenize(review.strip())
        #
        # 2. Loop over each sentence
        sentences = []
        for raw_sentence in raw_sentences:
            # If a sentence is empty, skip it
            if len(raw_sentence) > 0:
                # Otherwise, call review_to_wordlist to get a list of words
                sentences.append( KaggleWord2VecUtility.review_to_wordlist( raw_sentence, remove_stopwords ))
        #
        # Return the list of sentences (each sentence is a list of words,
        # so this returns a list of lists
        return sentences



```
</div>

</div>



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
clean_review_word = KaggleWord2VecUtility.review_to_wordlist \
        ( train["review"][0], True )
clean_review_sentence = KaggleWord2VecUtility.review_to_wordlist \
        ( train["review"][0], True )


```
</div>

</div>



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
# Get the number of reviews based on the dataframe column size
num_reviews = train["review"].size





```
</div>

</div>



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
print ("Cleaning and parsing the training set movie reviews...\n")
clean_train_reviews = []
for i in range( 0, len(train["review"])):
    if( (i+1)%1000 == 0 ):
        print ("Review %d of %d\n" % ( i+1, num_reviews )  )  
    clean_train_reviews.append(" ".join(KaggleWord2VecUtility.review_to_wordlist(train["review"][i], True)))

```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">
{:.output_stream}
```
Cleaning and parsing the training set movie reviews...

Review 1000 of 25000

Review 2000 of 25000

Review 3000 of 25000

Review 4000 of 25000

Review 5000 of 25000

Review 6000 of 25000

Review 7000 of 25000

Review 8000 of 25000

Review 9000 of 25000

Review 10000 of 25000

Review 11000 of 25000

Review 12000 of 25000

Review 13000 of 25000

Review 14000 of 25000

Review 15000 of 25000

Review 16000 of 25000

Review 17000 of 25000

Review 18000 of 25000

Review 19000 of 25000

Review 20000 of 25000

Review 21000 of 25000

Review 22000 of 25000

Review 23000 of 25000

Review 24000 of 25000

Review 25000 of 25000

```
</div>
</div>
</div>



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
clean_train_reviews[0:5]

```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">


{:.output_data_text}
```
['stuff going moment mj started listening music watching odd documentary watched wiz watched moonwalker maybe want get certain insight guy thought really cool eighties maybe make mind whether guilty innocent moonwalker part biography part feature film remember going see cinema originally released subtle messages mj feeling towards press also obvious message drugs bad kay visually impressive course michael jackson unless remotely like mj anyway going hate find boring may call mj egotist consenting making movie mj fans would say made fans true really nice actual feature film bit finally starts minutes excluding smooth criminal sequence joe pesci convincing psychopathic powerful drug lord wants mj dead bad beyond mj overheard plans nah joe pesci character ranted wanted people know supplying drugs etc dunno maybe hates mj music lots cool things like mj turning car robot whole speed demon sequence also director must patience saint came filming kiddy bad sequence usually directors hate working one kid let alone whole bunch performing complex dance scene bottom line movie people like mj one level another think people stay away try give wholesome message ironically mj bestest buddy movie girl michael jackson truly one talented people ever grace planet guilty well attention gave subject hmmm well know people different behind closed doors know fact either extremely nice stupid guy one sickest liars hope latter',
 'classic war worlds timothy hines entertaining film obviously goes great effort lengths faithfully recreate h g wells classic book mr hines succeeds watched film appreciated fact standard predictable hollywood fare comes every year e g spielberg version tom cruise slightest resemblance book obviously everyone looks different things movie envision amateur critics look criticize everything others rate movie important bases like entertained people never agree critics enjoyed effort mr hines put faithful h g wells classic novel found entertaining made easy overlook critics perceive shortcomings',
 'film starts manager nicholas bell giving welcome investors robert carradine primal park secret project mutating primal animal using fossilized dna like jurassik park scientists resurrect one nature fearsome predators sabretooth tiger smilodon scientific ambition turns deadly however high voltage fence opened creature escape begins savagely stalking prey human visitors tourists scientific meanwhile youngsters enter restricted area security center attacked pack large pre historical animals deadlier bigger addition security agent stacy haiduk mate brian wimmer fight hardly carnivorous smilodons sabretooths course real star stars astounding terrifyingly though convincing giant animals savagely stalking prey group run afoul fight one nature fearsome predators furthermore third sabretooth dangerous slow stalks victims movie delivers goods lots blood gore beheading hair raising chills full scares sabretooths appear mediocre special effects story provides exciting stirring entertainment results quite boring giant animals majority made computer generator seem totally lousy middling performances though players reacting appropriately becoming food actors give vigorously physical performances dodging beasts running bound leaps dangling walls packs ridiculous final deadly scene small kids realistic gory violent attack scenes films sabretooths smilodon following sabretooth james r hickox vanessa angel david keith john rhys davies much better bc roland emmerich steven strait cliff curtis camilla belle motion picture filled bloody moments badly directed george miller originality takes many elements previous films miller australian director usually working television tidal wave journey center earth many others occasionally cinema man snowy river zeus roxanne robinson crusoe rating average bottom barrel',
 'must assumed praised film greatest filmed opera ever read somewhere either care opera care wagner care anything except desire appear cultured either representation wagner swan song movie strikes unmitigated disaster leaden reading score matched tricksy lugubrious realisation text questionable people ideas opera matter play especially one shakespeare allowed anywhere near theatre film studio syberberg fashionably without smallest justification wagner text decided parsifal bisexual integration title character latter stages transmutes kind beatnik babe though one continues sing high tenor actors film singers get double dose armin jordan conductor seen face heard voice amfortas also appears monstrously double exposure kind batonzilla conductor ate monsalvat playing good friday music way transcendant loveliness nature represented scattering shopworn flaccid crocuses stuck ill laid turf expedient baffles theatre sometimes piece imperfections thoughts think syberberg splice parsifal gurnemanz mountain pasture lush provided julie andrews sound music sound hard endure high voices trumpets particular possessing aural glare adds another sort fatigue impatience uninspired conducting paralytic unfolding ritual someone another review mentioned bayreuth recording knappertsbusch though tempi often slow jordan altogether lacks sense pulse feeling ebb flow music half century orchestral sound set modern pressings still superior film',
 'superbly trashy wondrously unpretentious exploitation hooray pre credits opening sequences somewhat give false impression dealing serious harrowing drama need fear barely ten minutes later necks nonsensical chainsaw battles rough fist fights lurid dialogs gratuitous nudity bo ingrid two orphaned siblings unusually close even slightly perverted relationship imagine playfully ripping towel covers sister naked body stare unshaven genitals several whole minutes well bo sister judging dubbed laughter mind sick dude anyway kids fled russia parents nasty soldiers brutally slaughtered mommy daddy friendly smuggler took custody however even raised trained bo ingrid expert smugglers actual plot lifts years later facing ultimate quest mythical incredibly valuable white fire diamond coincidentally found mine things life ever made little sense plot narrative structure white fire sure lot fun watch time clue beating cause bet actors understood even less whatever violence magnificently grotesque every single plot twist pleasingly retarded script goes totally bonkers beyond repair suddenly reveal reason bo needs replacement ingrid fred williamson enters scene big cigar mouth sleazy black fingers local prostitutes bo principal opponent italian chick big breasts hideous accent preposterous catchy theme song plays least dozen times throughout film obligatory falling love montage loads attractions god brilliant experience original french title translates life survive uniquely appropriate makes much sense rest movie none']
```


</div>
</div>
</div>



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
print ("Creating the bag of words...\n")
from sklearn.feature_extraction.text import CountVectorizer
# Initialize the "CountVectorizer" object, which is scikit-learn's
# bag of words tool.  
vectorizer = CountVectorizer(analyzer = "word",   \
                             tokenizer = None,    \
                             preprocessor = None, \
                             stop_words = None,   \
                             max_features = 5000) 



```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">
{:.output_stream}
```
Creating the bag of words...

```
</div>
</div>
</div>



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
train_data_features = vectorizer.fit_transform(clean_train_reviews)

```
</div>

</div>



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
train_data_features = train_data_features.toarray()

```
</div>

</div>



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
print ("Training the random forest (this may take a while)...")

# Initialize a Random Forest classifier with 100 trees
forest = RandomForestClassifier(n_estimators = 100)
  

```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">
{:.output_stream}
```
Training the random forest (this may take a while)...
```
</div>
</div>
</div>



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
# Fit the forest to the training set, using the bag of words as
# features and the sentiment labels as the response variable
#
# This may take a few minutes to run
forest = forest.fit( train_data_features, train["sentiment"] )

```
</div>

</div>



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
# Create an empty list and append the clean reviews one by one
clean_test_reviews = []

print ("Cleaning and parsing the test set movie reviews...\n")
for i in range(0,len(test["review"])):
    clean_test_reviews.append(" ".join(KaggleWord2VecUtility.review_to_wordlist(test["review"][i], True)))


```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">
{:.output_stream}
```
Cleaning and parsing the test set movie reviews...

```
</div>
</div>
</div>



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python

# Get a bag of words for the test set, and convert to a numpy array
test_data_features = vectorizer.transform(clean_test_reviews)
test_data_features = test_data_features.toarray()


```
</div>

</div>



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
# Use the random forest to make sentiment label predictions
print ("Predicting test labels...\n")
result = forest.predict(test_data_features)

# Copy the results to a pandas dataframe with an "id" column and
# a "sentiment" column
output = pd.DataFrame( data={"id":test["id"], "sentiment":result} )

# Use pandas to write the comma-separated output file
output.to_csv('Bag_of_Words_model.csv', index=False, quoting=3)
print ("Wrote results to Bag_of_Words_model.csv")

```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">
{:.output_stream}
```
Predicting test labels...

Wrote results to Bag_of_Words_model.csv
```
</div>
</div>
</div>



### Word2Vec




<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
#!pip install gensim

```
</div>

</div>



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
import pandas as pd
import os
from nltk.corpus import stopwords
import nltk.data
import logging
import numpy as np  # Make sure that numpy is imported
from gensim.models import Word2Vec
from sklearn.ensemble import RandomForestClassifier

```
</div>

</div>



"In lexical analysis, tokenization is the process of breaking a stream of text up into words, phrases, symbols, or other meaningful elements called tokens. The list of tokens becomes input for further processing such as parsing or text mining." -[Wikipedia](https://en.wikipedia.org/wiki/Tokenization_(lexical_analysis)

Punkt is a specific tokenizer.
[http://www.nltk.org/_modules/nltk/tokenize/punkt.html](http://www.nltk.org/_modules/nltk/tokenize/punkt.html) 



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
# download punkt
nltk.download('punkt')

```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">
{:.output_stream}
```
[nltk_data] Downloading package punkt to /root/nltk_data...
[nltk_data]   Unzipping tokenizers/punkt.zip.
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
```python
#What is a tokenizer 
# http://www.nltk.org/_modules/nltk/tokenize/punkt.html 
tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

```
</div>

</div>



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
# ****** Split the labeled and unlabeled training sets into clean sentences
# Note this will take a while and produce some warnings. 
sentences = []  # Initialize an empty list of sentences

print ("Parsing sentences from training set")
for review in train["review"]:
    sentences += KaggleWord2VecUtility.review_to_sentences(review, tokenizer)


```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">
{:.output_stream}
```
Parsing sentences from training set
```
</div>
</div>
</div>



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
    print ("Parsing sentences from unlabeled set")
    for review in unlabeled_train["review"]:
        sentences += KaggleWord2VecUtility.review_to_sentences(review, tokenizer)

```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">
{:.output_stream}
```
Parsing sentences from unlabeled set
```
</div>
</div>
</div>



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
# ****** Define functions to create average word vectors
#

def makeFeatureVec(words, model, num_features):
    # Function to average all of the word vectors in a given
    # paragraph
    #
    # Pre-initialize an empty numpy array (for speed)
    featureVec = np.zeros((num_features,),dtype="float32")
    #
    nwords = 0.
    #
    # Index2word is a list that contains the names of the words in
    # the model's vocabulary. Convert it to a set, for speed
    index2word_set = set(model.wv.index2word)
    #
    # Loop over each word in the review and, if it is in the model's
    # vocaublary, add its feature vector to the total
    for word in words:
        if word in index2word_set:
            nwords = nwords + 1.
            featureVec = np.add(featureVec,model[word])
    #
    # Divide the result by the number of words to get the average
    featureVec = np.divide(featureVec,nwords)
    return featureVec


def getAvgFeatureVecs(reviews, model, num_features):
    # Given a set of reviews (each one a list of words), calculate
    # the average feature vector for each one and return a 2D numpy array
    #
    # Initialize a counter
    counter = 0.
    #
    # Preallocate a 2D numpy array, for speed
    reviewFeatureVecs = np.zeros((len(reviews),num_features),dtype="float32")
    #
    # Loop through the reviews
    for review in reviews:
    # Print a status message every 1000th review
       if counter%1000. == 0.:
            print ("Review %d of %d" % (counter, len(reviews)))
               # Call the function (defined above) that makes average feature vectors
            reviewFeatureVecs[counter] = makeFeatureVec(review, model, num_features)
            # Increment the counter
            counter = counter + 1.
    return reviewFeatureVecs


def getCleanReviews(reviews):
    clean_reviews = []
    for review in reviews["review"]:
        clean_reviews.append( KaggleWord2VecUtility.review_to_wordlist( review, remove_stopwords=True ))
    return clean_reviews

```
</div>

</div>



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',level=logging.INFO)

# Set values for various parameters
num_features = 300    # Word vector dimensionality
min_word_count = 40   # Minimum word count
num_workers = 4       # Number of threads to run in parallel
context = 10          # Context window size
downsampling = 1e-3   # Downsample setting for frequent words

```
</div>

</div>



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
# Initialize and train the model (this will take some time)
print ("Training Word2Vec model...")
model = Word2Vec(sentences, workers=num_workers, \
            size=num_features, min_count = min_word_count, \
            window = context, sample = downsampling, seed=1)

```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">
{:.output_stream}
```
Training Word2Vec model...
```
</div>
</div>
</div>



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
# If you don't plan to train the model any further, calling
# init_sims will make the model much more memory-efficient.
model.init_sims(replace=True)



```
</div>

</div>



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
# It can be helpful to create a meaningful model name and
# save the model for later use. You can load it later using Word2Vec.load()
model_name = "300features_40minwords_10context"
model.save(model_name)

model.doesnt_match("man woman child kitchen".split())



```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">


{:.output_data_text}
```
'kitchen'
```


</div>
</div>
</div>



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
model.doesnt_match("france england germany soccer".split())


```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">


{:.output_data_text}
```
'soccer'
```


</div>
</div>
</div>



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
model.most_similar("soccer")

```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">


{:.output_data_text}
```
[('football', 0.8039368391036987),
 ('basketball', 0.6595849990844727),
 ('poker', 0.6242333054542542),
 ('baseball', 0.6185872554779053),
 ('coach', 0.6010462641716003),
 ('sports', 0.5769123435020447),
 ('hockey', 0.574334442615509),
 ('shaolin', 0.5512605905532837),
 ('champions', 0.5357030034065247),
 ('wrestling', 0.5321739912033081)]
```


</div>
</div>
</div>



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
model.most_similar("man")

```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">


{:.output_data_text}
```
[('woman', 0.6294258832931519),
 ('lad', 0.6104426383972168),
 ('lady', 0.5893934965133667),
 ('monk', 0.5500040650367737),
 ('farmer', 0.5429803729057312),
 ('guy', 0.521837592124939),
 ('person', 0.5184544324874878),
 ('chap', 0.5139154195785522),
 ('millionaire', 0.5120488405227661),
 ('politician', 0.5048956871032715)]
```


</div>
</div>
</div>



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
model["computer"]

```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">


{:.output_data_text}
```
array([-4.79339510e-02, -2.53810696e-02, -9.83078852e-02,  7.23835602e-02,
       -2.23066527e-02,  5.72536923e-02,  3.17422301e-02, -6.21821284e-02,
       -3.82531472e-02, -2.85035223e-02,  6.96288198e-02, -8.61958042e-03,
       -5.25614843e-02,  8.91598314e-02,  1.33388013e-01, -4.59663719e-02,
        5.79963811e-02,  4.03241627e-02,  1.37066737e-01, -1.39122888e-01,
        2.03789044e-02, -4.67378348e-02,  1.71173755e-02, -6.14754260e-02,
       -8.26872066e-02,  4.17799037e-03, -6.88192695e-02, -4.71851118e-02,
        5.73772416e-02, -5.46796173e-02, -3.86515111e-02,  9.65327471e-02,
        2.05815546e-02,  2.76276264e-02, -4.25591022e-02,  2.51196809e-02,
        3.82702723e-02,  4.93922504e-03,  1.12461247e-01, -1.01153374e-01,
        3.35556641e-02,  5.55540156e-03, -3.79679240e-02, -2.24771872e-02,
        8.73093233e-02, -5.04509695e-02, -4.05530483e-02,  6.14347458e-02,
        8.23021308e-02,  5.65582402e-02,  2.69791781e-05,  4.77958284e-03,
        7.75242895e-02,  1.00004807e-01,  5.92902536e-03, -1.05087563e-01,
       -3.61691602e-02, -3.15997313e-04,  2.78723650e-02,  1.86387897e-02,
       -1.70327332e-02,  7.38388335e-04, -5.49275465e-02, -5.79706319e-02,
        5.06388247e-02,  1.26366448e-02, -7.05252960e-02,  1.17502294e-01,
        9.84748304e-02, -1.39781563e-02, -3.13345194e-02, -6.53982442e-03,
       -5.12062805e-03,  1.38693685e-02,  3.74527611e-02,  2.43657101e-02,
        3.43408324e-02,  8.02342147e-02, -1.71090271e-02, -7.04095000e-03,
        1.04522845e-02,  3.43474708e-02, -1.40156727e-02,  6.84063807e-02,
        3.38155553e-02,  1.56406701e-01,  1.79476216e-02, -4.58403230e-02,
       -4.15017596e-03,  2.82820929e-02,  5.71620092e-02, -1.77074708e-02,
       -3.98685411e-02, -1.47484854e-01, -7.26067871e-02,  6.60759658e-02,
       -1.58032253e-02, -6.50163293e-02,  4.24350947e-02,  1.12647265e-01,
        4.10237350e-02,  2.98280325e-02,  3.11948135e-02, -7.24988058e-02,
        1.41753890e-02, -3.46501283e-02,  1.00943372e-02, -4.47276309e-02,
       -2.15441603e-02, -1.69435084e-01, -2.18652375e-02,  6.05334900e-02,
        9.29217041e-02, -4.98547107e-02,  1.08486228e-02, -1.16886824e-01,
       -7.03361072e-03, -6.75052553e-02,  5.19460291e-02, -3.68494578e-02,
        3.72104347e-02,  4.01992016e-02, -3.65689546e-02,  5.73715456e-02,
       -8.58063698e-02, -1.69679038e-02,  1.78947430e-02, -5.47890067e-02,
       -1.79170370e-02,  6.18545935e-02,  1.40716946e-02, -5.53099699e-02,
        3.28098610e-02, -8.61988366e-02, -1.38060516e-02, -7.77973756e-02,
        3.01322360e-02, -2.97213811e-02, -5.12061007e-02, -7.44357333e-02,
       -1.18710482e-02,  5.09540029e-02,  5.79989143e-02, -7.21113309e-02,
        1.41676925e-02,  1.35432437e-01, -6.59376010e-03,  4.38096523e-02,
       -6.01608269e-02,  2.00556521e-03,  1.74984355e-02,  4.49043103e-02,
        1.09420130e-02,  8.18869006e-03,  1.52458772e-02, -2.65226234e-02,
       -2.01061089e-02, -8.11586305e-02,  9.00103152e-02,  5.94500527e-02,
        2.89643300e-03,  2.56725634e-03, -1.78116038e-02,  1.90644264e-02,
       -2.26962790e-02,  3.18668643e-03, -7.71036968e-02,  1.93123269e-04,
        2.58924272e-02,  3.18960845e-02, -8.47435892e-02,  3.89174446e-02,
       -9.14918333e-02, -3.47692855e-02, -1.05178788e-01, -7.81329945e-02,
       -2.56739929e-02, -2.41342392e-02, -5.45198135e-02, -3.73307355e-02,
        2.69567072e-02,  4.54399176e-03, -4.00420465e-02,  5.33767864e-02,
       -6.76332489e-02,  3.17393392e-02,  5.90901449e-02,  9.45589691e-02,
        3.21563855e-02,  2.33184472e-02,  6.14998117e-03,  4.27653221e-03,
       -5.73230572e-02, -5.30507602e-03, -8.49941671e-02, -2.04346944e-02,
        9.48881656e-02,  5.88757657e-02, -1.11293055e-01,  1.90836471e-02,
       -4.70569022e-02, -4.22467291e-02,  3.86966486e-03, -9.70810875e-02,
       -7.43101314e-02,  5.24649993e-02, -3.29410285e-02, -2.38991026e-02,
       -3.36377881e-02,  5.44135012e-02, -9.94117185e-02,  2.49974765e-02,
       -4.42564599e-02,  8.74478966e-02, -3.85684483e-02,  3.34127881e-02,
        8.24790299e-02,  5.33424094e-02,  1.31256990e-02,  2.96878815e-02,
        2.22183373e-02, -2.11739577e-02,  8.04685652e-02, -3.71479578e-02,
        7.73914915e-04, -4.12550308e-02,  5.00466563e-02,  2.51460876e-02,
        3.03647742e-02, -1.26310617e-01, -1.64576340e-02,  6.23791106e-02,
        5.53251542e-02, -7.69763142e-02, -6.70548202e-03, -6.63869753e-02,
        1.24717746e-02, -1.68166086e-01,  1.68424156e-02, -6.28778115e-02,
        9.77337211e-02,  4.71105427e-02,  8.92058201e-03, -1.37098841e-02,
        3.06127220e-02,  1.73426419e-02, -2.89770309e-02, -1.85693940e-03,
       -1.15272313e-04, -3.52074020e-02, -3.21245864e-02, -8.34794790e-02,
        2.75801215e-02, -5.00087887e-02,  4.92503829e-02, -9.92316827e-02,
       -2.22996939e-02,  1.18917022e-02,  3.88455503e-02,  1.20371632e-01,
        5.02688885e-02, -1.04167230e-01,  4.91058379e-02, -1.05557941e-01,
       -1.98254809e-02, -4.02186513e-02,  7.50040542e-03, -1.05268778e-02,
       -7.07997382e-03,  9.48763639e-02,  4.84618731e-02, -3.02425772e-02,
        6.50335848e-02, -3.24018374e-02, -3.04284208e-02, -9.73176360e-02,
        6.17954507e-03,  6.41165003e-02, -6.53638914e-02,  3.88511792e-02,
        1.19865581e-01,  6.39352277e-02,  3.71861272e-02, -6.28246441e-02,
       -9.44986590e-04,  4.55985926e-02,  4.20346893e-02, -5.09408861e-02,
        5.30342124e-02,  2.55930256e-02,  1.57046262e-02,  1.00914791e-01,
       -5.51233701e-02, -5.27993068e-02, -1.68002993e-02, -6.69082999e-02,
       -1.73713006e-02,  7.25108683e-02, -7.00484961e-02,  3.51085626e-02],
      dtype=float32)
```


</div>
</div>
</div>



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
model.most_similar("car")

```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">


{:.output_data_text}
```
[('truck', 0.7517951726913452),
 ('jeep', 0.6925300359725952),
 ('bus', 0.6857529282569885),
 ('train', 0.6580698490142822),
 ('bike', 0.6402906179428101),
 ('plane', 0.6360883712768555),
 ('boat', 0.6104196310043335),
 ('helicopter', 0.6086692214012146),
 ('cars', 0.5979301929473877),
 ('chevy', 0.5919524431228638)]
```


</div>
</div>
</div>

