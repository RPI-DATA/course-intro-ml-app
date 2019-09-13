---
interact_link: content/notebooks/04-python/05-intro-kaggle-baseline.ipynb
kernel_name: python3
has_widgets: false
title: 'Kaggle Baseline'
prev_page:
  url: /notebooks/04-python/04-intro-python-groupby.html
  title: 'Groupby and Pivot Tables'
next_page:
  url: /notebooks/06-viz-api-scraper/01-intro-api-twitter.html
  title: 'Twitter'
comment: "***PROGRAMMATICALLY GENERATED, DO NOT EDIT. SEE ORIGINAL FILES IN /content***"
---


[![AnalyticsDojo](https://github.com/rpi-techfundamentals/spring2019-materials/blob/master/fig/final-logo.png?raw=1)](http://rpi.analyticsdojo.com)
<center><h1>Introduction to Python - Kaggle Baseline</h1></center>
<center><h3><a href = 'http://rpi.analyticsdojo.com'>rpi.analyticsdojo.com</a></h3></center>






## Running Code using Kaggle Notebooks
- Kaggle utilizes Docker to create a fully functional environment for hosting competitions in data science.
- You could download/run this locally or view the [published version](https://www.kaggle.com/analyticsdojo/titanic-baseline-models-analyticsdojo-python/editnb) and `fork` it. 
- Kaggle has created an incredible resource for learning analytics.  You can view a number of *toy* examples that can be used to understand data science and also compete in real problems faced by top companies. 



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```!wget https://raw.githubusercontent.com/rpi-techfundamentals/spring2019-materials/master/input/train.csv
!wget https://raw.githubusercontent.com/rpi-techfundamentals/spring2019-materials/master/input/test.csv

```
</div>

</div>



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```import numpy as np 
import pandas as pd 

# Input data files are available in the "../input/" directory.
# Let's input them into a Pandas DataFrame
train = pd.read_csv("train.csv")
test  = pd.read_csv("test.csv")

```
</div>

</div>



## `train` and `test` set on Kaggle
- The `train` file contains a wide variety of information that might be useful in understanding whether they survived or not. It also includes a record as to whether they survived or not.
- The `test` file contains all of the columns of the first file except whether they survived. Our goal is to predict whether the individuals survived.



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```train.head()

```
</div>

</div>



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```test.head()

```
</div>

</div>



## Baseline Models: No Survivors
- The Titanic problem is one of classification, and often the simplest baseline of all 0/1 is an appropriate baseline.
- Think of the baseline as the simplest model you can think of that can be used to lend intuition on how your model is working. 
- Even if you aren't familiar with the history of the tragedy, by checking out the [Wikipedia Page](https://en.wikipedia.org/wiki/RMS_Titanic) we can quickly see that the majority of people (68%) died.
- As a result, our baseline model will be for no survivors.



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```test["Survived"] = 0

```
</div>

</div>



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```submission = test.loc[:,["PassengerId", "Survived"]]

```
</div>

</div>



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```submission.head()

```
</div>

</div>



## Write to CSV
The code below will write your dataframe to a CSV. 



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```submission.to_csv('everyone_dies.csv', index=False)

```
</div>

</div>



## Download from Colab
Working on colab requires you to download a file via a google specific package.  



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```from google.colab import files
files.download('everyone_dies.csv')

```
</div>

</div>



## The First Rule of Shipwrecks
- You may have seen it in a movie or read it in a novel, but [women and children first](https://en.wikipedia.org/wiki/Women_and_children_first) has at it's roots something that could provide our first model.
- Now let's recode the `Survived` column based on whether was a man or a woman.  
- We are using conditionals to *select* rows of interest (for example, where test['Sex'] == 'male') and recoding appropriate columns.



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```#Here we can code it as Survived, but if we do so we will overwrite our other prediction. 
#Instead, let's code it as PredGender

test.loc[test['Sex'] == 'male', 'PredGender'] = 0
test.loc[test['Sex'] == 'female', 'PredGender'] = 1
#test.PredGender.astype(int)
test

```
</div>

</div>



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```submission = test.loc[:,['PassengerId', 'PredGender']]
# But we have to change the column name.
# Option 1: submission.columns = ['PassengerId', 'Survived']
# Option 2: Rename command.
submission.rename(columns={'PredGender': 'Survived'}, inplace=True)

```
</div>

</div>



## Writeout and then Download your File
Try your first submission to Kaggle!



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```submission.to_csv('women_survive.csv', index=False)


```
</div>

</div>



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```from google.colab import files
files.download('women_survive.csv')

```
</div>

</div>

