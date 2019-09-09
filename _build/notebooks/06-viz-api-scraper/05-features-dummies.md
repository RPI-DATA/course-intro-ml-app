---
interact_link: content/notebooks/06-viz-api-scraper/05-features-dummies.ipynb
kernel_name: python3
has_widgets: false
title: 'Feature Dummies'
prev_page:
  url: /notebooks/06-viz-api-scraper/04-strings-regular-expressions.html
  title: 'Strings - Regular Expressions'
next_page:
  url: /notebooks/06-viz-api-scraper/ALT-visualization-python-matplotlib.html
  title: 'Matplotlib'
comment: "***PROGRAMMATICALLY GENERATED, DO NOT EDIT. SEE ORIGINAL FILES IN /content***"
---



[![AnalyticsDojo](https://github.com/rpi-techfundamentals/spring2019-materials/blob/master/fig/final-logo.png?raw=1)](http://rpi.analyticsdojo.com)
<center><h1>Introduction to Feature Creation & Dummy Variables</h1></center>
<center><h3><a href = 'http://rpi.analyticsdojo.com'>rpi.analyticsdojo.com</a></h3></center>





<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```!wget https://raw.githubusercontent.com/rpi-techfundamentals/spring2019-materials/master/input/train.csv
!wget https://raw.githubusercontent.com/rpi-techfundamentals/spring2019-materials/master/input/test.csv

```
</div>

</div>



# Feature Extraction



Here we will talk about an important piece of machine learning: the extraction of
quantitative features from data.  By the end of this section you will

- Know how features are extracted from real-world data.
- See an example of extracting numerical features from textual data

In addition, we will go over several basic tools within scikit-learn which can be used to accomplish the above tasks.



## What Are Features?



### Numerical Features



Recall that data in scikit-learn is expected to be in two-dimensional arrays, of size
**n_samples** $\times$ **n_features**.

Previously, we looked at the iris dataset, which has 150 samples and 4 features



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```from sklearn.datasets import load_iris
iris = load_iris()
print(iris.data.shape)

```
</div>

</div>



These features are:

- sepal length in cm
- sepal width in cm
- petal length in cm
- petal width in cm

Numerical features such as these are pretty straightforward: each sample contains a list
of floating-point numbers corresponding to the features



### Categorical Features



What if you have categorical features?  For example, imagine there is data on the color of each
iris:

    color in [red, blue, purple]

You might be tempted to assign numbers to these features, i.e. *red=1, blue=2, purple=3*
but in general **this is a bad idea**.  Estimators tend to operate under the assumption that
numerical features lie on some continuous scale, so, for example, 1 and 2 are more alike
than 1 and 3, and this is often not the case for categorical features.

In fact, the example above is a subcategory of "categorical" features, namely, "nominal" features. Nominal features don't imply an order, whereas "ordinal" features are categorical features that do imply an order. An example of ordinal features would be T-shirt sizes, e.g., XL > L > M > S. 

One work-around for parsing nominal features into a format that prevents the classification algorithm from asserting an order is the so-called one-hot encoding representation. Here, we give each category its own dimension.  

The enriched iris feature set would hence be in this case:

- sepal length in cm
- sepal width in cm
- petal length in cm
- petal width in cm
- color=purple (1.0 or 0.0)
- color=blue (1.0 or 0.0)
- color=red (1.0 or 0.0)

Note that using many of these categorical features may result in data which is better
represented as a **sparse matrix**, as we'll see with the text classification example
below.



### Derived Features



Another common feature type are **derived features**, where some pre-processing step is
applied to the data to generate features that are somehow more informative.  Derived
features may be based in **feature extraction** and **dimensionality reduction** (such as PCA or manifold learning),
may be linear or nonlinear combinations of features (such as in polynomial regression),
or may be some more sophisticated transform of the features.



### Combining Numerical and Categorical Features



As an example of how to work with both categorical and numerical data, we will perform survival predicition for the passengers of the HMS Titanic.




<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```import os
import pandas as pd
titanic = pd.read_csv('train.csv')
print(titanic.columns)

```
</div>

</div>



Here is a broad description of the keys and what they mean:

```
pclass          Passenger Class
                (1 = 1st; 2 = 2nd; 3 = 3rd)
survival        Survival
                (0 = No; 1 = Yes)
name            Name
sex             Sex
age             Age
sibsp           Number of Siblings/Spouses Aboard
parch           Number of Parents/Children Aboard
ticket          Ticket Number
fare            Passenger Fare
cabin           Cabin
embarked        Port of Embarkation
                (C = Cherbourg; Q = Queenstown; S = Southampton)
boat            Lifeboat
body            Body Identification Number
home.dest       Home/Destination
```

In general, it looks like `name`, `sex`, `cabin`, `embarked`, `boat`, `body`, and `homedest` may be candidates for categorical features, while the rest appear to be numerical features. We can also look at the first couple of rows in the dataset to get a better understanding:



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```titanic.head()

```
</div>

</div>



We clearly want to discard the "boat" and "body" columns for any classification into survived vs not survived as they already contain this information. The name is unique to each person (probably) and also non-informative. For a first try, we will use "pclass", "sibsp", "parch", "fare" and "embarked" as our features:



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```labels = titanic.Survived.values
features = titanic[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']]

```
</div>

</div>



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```features.head()

```
</div>

</div>



The data now contains only useful features, but they are not in a format that the machine learning algorithms can understand. We need to transform the strings "male" and "female" into binary variables that indicate the gender, and similarly for "embarked".
We can do that using the pandas ``get_dummies`` function:



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```featuremodel=pd.get_dummies(features)
featuremodel

```
</div>

</div>



Notice that this includes N dummy variables.  When we are modeling we will need N-1 categorical variables. 



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```pd.get_dummies(features, drop_first=True).head()

```
</div>

</div>



This transformation successfully encoded the string columns. However, one might argue that the class is also a categorical variable. We can explicitly list the columns to encode using the ``columns`` parameter, and include ``pclass``:



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```features_dummies = pd.get_dummies(features, columns=['Pclass', 'Sex', 'Embarked'], drop_first=True)
features_dummies

```
</div>

</div>



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```#Transform from Pandas to numpy with .values
data = features_dummies.values

```
</div>

</div>



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```data

```
</div>

</div>



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```type(data)

```
</div>

</div>

