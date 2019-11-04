---
interact_link: content/notebooks/16-intro-nlp/04-what-cooking-python.ipynb
kernel_name: python3
has_widgets: false
title: 'What's Cooking Python'
prev_page:
  url: /notebooks/16-intro-nlp/03-scikit-learn-text.html
  title: 'Scikit Learn Text'
next_page:
  url: /notebooks/16-intro-nlp/05-bag-popcorn-bag-words.html
  title: 'Bag of Popcorn Bag of Words'
comment: "***PROGRAMMATICALLY GENERATED, DO NOT EDIT. SEE ORIGINAL FILES IN /content***"
---


[![AnalyticsDojo](https://github.com/rpi-techfundamentals/spring2019-materials/blob/master/fig/final-logo.png?raw=1)](http://rpi.analyticsdojo.com)
<center><h1> What's Cooking  in Python</h1></center>
<center><h3><a href = 'http://rpi.analyticsdojo.com'>rpi.analyticsdojo.com</a></h3></center>




This was adopted from. 
https://www.kaggle.com/manuelatadvice/whats-cooking/noname/code



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```#This imports a bunch of packages.  
import pandas as pd
import matplotlib.pyplot as plt
import nltk
from nltk.stem import WordNetLemmatizer
from collections import Counter
import json
from nltk.stem import WordNetLemmatizer
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
#from sklearn import grid_search



```
</div>

</div>



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```
#If you import the codes locally, this seems to cause some issues.  
import json
from urllib.request import urlopen

urltrain= 'https://raw.githubusercontent.com/RPI-Analytics/MGMT6963-2015/master/data/whatscooking/whatscookingtrain.json'
urltest = 'https://raw.githubusercontent.com/RPI-Analytics/MGMT6963-2015/master/data/whatscooking/whatscookingtest.json'


train = pd.read_json(urlopen(urltrain))
test = pd.read_json(urlopen(urltest))

```
</div>

</div>



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```#First we want to see the most popular cuisine for the naive model. 
train.groupby('cuisine').size()

```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">


{:.output_data_text}
```
cuisine
brazilian        467
british          804
cajun_creole    1546
chinese         2673
filipino         755
french          2646
greek           1175
indian          3003
irish            667
italian         7838
jamaican         526
japanese        1423
korean           830
mexican         6438
moroccan         821
russian          489
southern_us     4320
spanish          989
thai            1539
vietnamese       825
dtype: int64
```


</div>
</div>
</div>



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```#Here we write the most popular selection.  This is the baseline by which we will judge other models. 
test['cuisine']='italian'

```
</div>

</div>



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```#THis is a much more simple version that selects out the columns ID and cuisinte
submission=test[['id' ,  'cuisine' ]]
#This is a more complex method I showed that gives same.
#submission=pd.DataFrame(test.ix[:,['id' ,  'cuisine' ]])

```
</div>

</div>



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```#This outputs the file.
submission.to_csv("1_cookingSubmission.csv",index=False)
from google.colab import files
files.download('1_cookingSubmission.csv')


```
</div>

</div>



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```#So it seems there is some data we need to use the NLTK leemmatizer.  
stemmer = WordNetLemmatizer()
nltk.download('wordnet')

 

```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">
{:.output_stream}
```
[nltk_data] Downloading package wordnet to /root/nltk_data...
[nltk_data]   Unzipping corpora/wordnet.zip.
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
```train

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
      <th>cuisine</th>
      <th>ingredients</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>10259</td>
      <td>greek</td>
      <td>[romaine lettuce, black olives, grape tomatoes...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>25693</td>
      <td>southern_us</td>
      <td>[plain flour, ground pepper, salt, tomatoes, g...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>20130</td>
      <td>filipino</td>
      <td>[eggs, pepper, salt, mayonaise, cooking oil, g...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>22213</td>
      <td>indian</td>
      <td>[water, vegetable oil, wheat, salt]</td>
    </tr>
    <tr>
      <th>4</th>
      <td>13162</td>
      <td>indian</td>
      <td>[black pepper, shallots, cornflour, cayenne pe...</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>39769</th>
      <td>29109</td>
      <td>irish</td>
      <td>[light brown sugar, granulated sugar, butter, ...</td>
    </tr>
    <tr>
      <th>39770</th>
      <td>11462</td>
      <td>italian</td>
      <td>[KRAFT Zesty Italian Dressing, purple onion, b...</td>
    </tr>
    <tr>
      <th>39771</th>
      <td>2238</td>
      <td>irish</td>
      <td>[eggs, citrus fruit, raisins, sourdough starte...</td>
    </tr>
    <tr>
      <th>39772</th>
      <td>41882</td>
      <td>chinese</td>
      <td>[boneless chicken skinless thigh, minced garli...</td>
    </tr>
    <tr>
      <th>39773</th>
      <td>2362</td>
      <td>mexican</td>
      <td>[green chile, jalapeno chilies, onions, ground...</td>
    </tr>
  </tbody>
</table>
<p>39774 rows × 3 columns</p>
</div>
</div>


</div>
</div>
</div>



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```#We see this in a Python Solution. 
train['ingredients_clean_string1'] = [','.join(z).strip() for z in train['ingredients']] 

#We also know that we can do something similar though a Lambda function. 
strip = lambda x: ' , '.join(x).strip() 
#Finally, we call the function for name
train['ingredients_clean_string2'] = train['ingredients'].map(strip)

#Now that we used the lambda function, we can reuse this for the test dataset. 
test['ingredients_clean_string1'] = test['ingredients'].map(strip)
 


```
</div>

</div>



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```#We see this in one of the solutions.  We can reconstruct it in a way that makes it abit easier to follow, but I found when doing that it took forever.  

#To interpret this, read from right to left. 
train['ingredients_string1'] = [' '.join([WordNetLemmatizer().lemmatize(re.sub('[^A-Za-z]', ' ', line)) for line in lists]).strip() for lists in train['ingredients']]       
test['ingredients_string1'] = [' '.join([WordNetLemmatizer().lemmatize(re.sub('[^A-Za-z]', ' ', line)) for line in lists]).strip() for lists in test['ingredients']]       




```
</div>

</div>



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```train['ingredients_string1']

```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">


{:.output_data_text}
```
0        romaine lettuce black olives grape tomatoes ga...
1        plain flour ground pepper salt tomato ground b...
2        egg pepper salt mayonaise cooking oil green ch...
3                           water vegetable oil wheat salt
4        black pepper shallot cornflour cayenne pepper ...
                               ...                        
39769    light brown sugar granulated sugar butter warm...
39770    KRAFT Zesty Italian Dressing purple onion broc...
39771    egg citrus fruit raisin sourdough starter flou...
39772    boneless chicken skinless thigh minced garlic ...
39773    green chile jalapeno chilies onion ground blac...
Name: ingredients_string1, Length: 39774, dtype: object
```


</div>
</div>
</div>



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```ingredients = train['ingredients'].apply(lambda x:','.join(x))
ingredients

```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">


{:.output_data_text}
```
0        romaine lettuce,black olives,grape tomatoes,ga...
1        plain flour,ground pepper,salt,tomatoes,ground...
2        eggs,pepper,salt,mayonaise,cooking oil,green c...
3                           water,vegetable oil,wheat,salt
4        black pepper,shallots,cornflour,cayenne pepper...
                               ...                        
39769    light brown sugar,granulated sugar,butter,warm...
39770    KRAFT Zesty Italian Dressing,purple onion,broc...
39771    eggs,citrus fruit,raisins,sourdough starter,fl...
39772    boneless chicken skinless thigh,minced garlic,...
39773    green chile,jalapeno chilies,onions,ground bla...
Name: ingredients, Length: 39774, dtype: object
```


</div>
</div>
</div>



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```#Now we will create a corpus.
corpustr = train['ingredients_string1']
corpusts = test['ingredients_string1']
corpustr

```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">


{:.output_data_text}
```
0        romaine lettuce black olives grape tomatoes ga...
1        plain flour ground pepper salt tomato ground b...
2        egg pepper salt mayonaise cooking oil green ch...
3                           water vegetable oil wheat salt
4        black pepper shallot cornflour cayenne pepper ...
                               ...                        
39769    light brown sugar granulated sugar butter warm...
39770    KRAFT Zesty Italian Dressing purple onion broc...
39771    egg citrus fruit raisin sourdough starter flou...
39772    boneless chicken skinless thigh minced garlic ...
39773    green chile jalapeno chilies onion ground blac...
Name: ingredients_string1, Length: 39774, dtype: object
```


</div>
</div>
</div>



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```#http://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html
#You could develop an understanding based on each.  
vectorizertr = TfidfVectorizer(stop_words='english',
                             ngram_range = ( 1 , 1 ),analyzer="word", 
                             max_df = .57 , binary=False , token_pattern=r'\w+' , sublinear_tf=False)
vectorizerts = TfidfVectorizer(stop_words='english')

```
</div>

</div>



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```#Note that this doesn't work with the #todense option.  
tfidftr=vectorizertr.fit_transform(corpustr)
predictors_tr = tfidftr

```
</div>

</div>



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```#Note that this doesn't work with the #todense option.  This creates a matrix of predictors from the corpus. 
tfidfts=vectorizertr.transform(corpusts)
predictors_ts= tfidfts

```
</div>

</div>



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```#This is target variable.  
targets_tr = train['cuisine']


```
</div>

</div>



## Logistic Regression and Regularization.

- Regularlization can help us with the large matrix by adding a penalty for each parameter. 
- Finding out how much regularization via grid search (search across hyperparameters.)

http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html

```C : float, default: 1.0
Inverse of regularization strength; must be a positive float. Like in support vector machines, smaller values specify stronger regularization.```



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```#Logistic Regression. 
parameters = {'C':[1, 10]}
#clf = LinearSVC()
clf = LogisticRegression()



```
</div>

</div>



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```predictors_tr

```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">


{:.output_data_text}
```
<39774x2963 sparse matrix of type '<class 'numpy.float64'>'
	with 727921 stored elements in Compressed Sparse Row format>
```


</div>
</div>
</div>



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```from sklearn.model_selection import GridSearchCV
#This uses that associated paramters to search a grid space. 
classifier = GridSearchCV(clf, parameters)
classifier=classifier.fit(predictors_tr,targets_tr)



```
</div>

</div>



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```#This predicts the outcome for the test set. 
predictions=classifier.predict(predictors_ts)

```
</div>

</div>



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```#This adds it to the resulting dataframe. 
test['cuisine'] = predictions

```
</div>

</div>



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```#This creates the submision dataframe
submission2=test[['id' ,  'cuisine' ]]

```
</div>

</div>



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```#This outputs the file.
submission2.to_csv("2_logisticSubmission.csv",index=False)
from google.colab import files
files.download('2_logisticSubmission.csv')

```
</div>

</div>



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```from sklearn.ensemble import RandomForestClassifier 



```
</div>

</div>



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```# Create the random forest object which will include all the parameters
# for the fit
forest = RandomForestClassifier(n_estimators = 10)



```
</div>

</div>



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```# Fit the training data to the Survived labels and create the decision trees
forest = forest.fit(predictors_tr,targets_tr)



```
</div>

</div>



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```# Take the same decision trees and run it on the test data
predictions = forest.predict(predictors_ts)

```
</div>

</div>



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```#This adds it to the resulting dataframe. 
test['cuisine'] = predictions

```
</div>

</div>



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```#This creates the submision dataframe
submission3=test[['id' ,  'cuisine' ]]
submission3.to_csv("3_random_submission.csv",index=False)

```
</div>

</div>



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```from google.colab import files
files.download('3_random_submission.csv')

```
</div>

</div>



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```ingredients = train['ingredients'].apply(lambda x:','.join(x))
ingredients
train

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
      <th>cuisine</th>
      <th>ingredients</th>
      <th>ingredients_clean_string1</th>
      <th>ingredients_clean_string2</th>
      <th>ingredients_string1</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>10259</td>
      <td>greek</td>
      <td>[romaine lettuce, black olives, grape tomatoes...</td>
      <td>romaine lettuce,black olives,grape tomatoes,ga...</td>
      <td>romaine lettuce , black olives , grape tomatoe...</td>
      <td>romaine lettuce black olives grape tomatoes ga...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>25693</td>
      <td>southern_us</td>
      <td>[plain flour, ground pepper, salt, tomatoes, g...</td>
      <td>plain flour,ground pepper,salt,tomatoes,ground...</td>
      <td>plain flour , ground pepper , salt , tomatoes ...</td>
      <td>plain flour ground pepper salt tomato ground b...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>20130</td>
      <td>filipino</td>
      <td>[eggs, pepper, salt, mayonaise, cooking oil, g...</td>
      <td>eggs,pepper,salt,mayonaise,cooking oil,green c...</td>
      <td>eggs , pepper , salt , mayonaise , cooking oil...</td>
      <td>egg pepper salt mayonaise cooking oil green ch...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>22213</td>
      <td>indian</td>
      <td>[water, vegetable oil, wheat, salt]</td>
      <td>water,vegetable oil,wheat,salt</td>
      <td>water , vegetable oil , wheat , salt</td>
      <td>water vegetable oil wheat salt</td>
    </tr>
    <tr>
      <th>4</th>
      <td>13162</td>
      <td>indian</td>
      <td>[black pepper, shallots, cornflour, cayenne pe...</td>
      <td>black pepper,shallots,cornflour,cayenne pepper...</td>
      <td>black pepper , shallots , cornflour , cayenne ...</td>
      <td>black pepper shallot cornflour cayenne pepper ...</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>39769</th>
      <td>29109</td>
      <td>irish</td>
      <td>[light brown sugar, granulated sugar, butter, ...</td>
      <td>light brown sugar,granulated sugar,butter,warm...</td>
      <td>light brown sugar , granulated sugar , butter ...</td>
      <td>light brown sugar granulated sugar butter warm...</td>
    </tr>
    <tr>
      <th>39770</th>
      <td>11462</td>
      <td>italian</td>
      <td>[KRAFT Zesty Italian Dressing, purple onion, b...</td>
      <td>KRAFT Zesty Italian Dressing,purple onion,broc...</td>
      <td>KRAFT Zesty Italian Dressing , purple onion , ...</td>
      <td>KRAFT Zesty Italian Dressing purple onion broc...</td>
    </tr>
    <tr>
      <th>39771</th>
      <td>2238</td>
      <td>irish</td>
      <td>[eggs, citrus fruit, raisins, sourdough starte...</td>
      <td>eggs,citrus fruit,raisins,sourdough starter,fl...</td>
      <td>eggs , citrus fruit , raisins , sourdough star...</td>
      <td>egg citrus fruit raisin sourdough starter flou...</td>
    </tr>
    <tr>
      <th>39772</th>
      <td>41882</td>
      <td>chinese</td>
      <td>[boneless chicken skinless thigh, minced garli...</td>
      <td>boneless chicken skinless thigh,minced garlic,...</td>
      <td>boneless chicken skinless thigh , minced garli...</td>
      <td>boneless chicken skinless thigh minced garlic ...</td>
    </tr>
    <tr>
      <th>39773</th>
      <td>2362</td>
      <td>mexican</td>
      <td>[green chile, jalapeno chilies, onions, ground...</td>
      <td>green chile,jalapeno chilies,onions,ground bla...</td>
      <td>green chile , jalapeno chilies , onions , grou...</td>
      <td>green chile jalapeno chilies onion ground blac...</td>
    </tr>
  </tbody>
</table>
<p>39774 rows × 6 columns</p>
</div>
</div>


</div>
</div>
</div>

