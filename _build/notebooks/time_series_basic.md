---
redirect_from:
  - "/notebooks/time-series-basic"
interact_link: content/C:\Users\sjgar\Documents\GitHub\jupyter-book\content\notebooks/time_series_basic.ipynb
kernel_name: python3
has_widgets: false
title: 'Time Series Analysis'
prev_page:
  url: /notebooks/PCA
  title: 'Principal Component Analysis'
next_page:
  url: /team/index
  title: 'Our Team'
comment: "***PROGRAMMATICALLY GENERATED, DO NOT EDIT. SEE ORIGINAL FILES IN /content***"
---


<head>
    <title>Time Series Analysis</title>
</head>



[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/RPI-DATA/tutorials-intro/blob/master/website/Forecasting%20the%20Rossmann%20Store%20Sales.ipynb)
<img src="https://raw.githubusercontent.com/RPI-DATA/website/master/static/images/rpilogo.png" alt="RPI LOGO">



# Time Series Analysis
- - - - - - - -- - - - - - - -- - - - - - - - - - - 
A **_time series_** is a series of data points indexed (or listed or graphed) in time order. Time series analysis pertains to methods extracting meaningful statistics from time series data. This is commonly used for forecasting and other models.



# Learning Objectives
- - - - - - - - - - - - - -- - - - - - - - - - -
1. Understand the uses of Time Series Analysis
2. Understand the pros and cons of various TSA methods, including differentiating between linear and non-linear methods.
3. Apply the facebook prophet model and analyze the results on given rossman store data.



# Sections
- - - -- - - - - - - - - - - - - - -- - - - -
1. [ Problem Description](#probdes)
2. Exploratory Data Analysis
  1. [Training Data](#edat)
  2. [Store Data](#edasd)
3. [Moving Average Model](#mam)
4. [Facebook Prophet Model](#fpm)
5. [Conclusion](#c)
6. [References](#r)



<a id="probdes"></a>
    
# Problem Description
- - - - -- -- - - - -   - - - - 
We will use the rossman store sales database for this notebook. Following is the description of Data from the website:

"Rossmann operates over 3,000 drug stores in 7 European countries. Currently, Rossmann store managers are tasked with predicting their daily sales for up to six weeks in advance. Store sales are influenced by many factors, including promotions, competition, school and state holidays, seasonality, and locality. With thousands of individual managers predicting sales based on their unique circumstances, the accuracy of results can be quite varied."



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
# Library Imports
import numpy as np
import pandas as pd
import matplotlib
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
from scipy.stats import skew
from scipy.stats.stats import pearsonr
from math import sqrt
from sklearn.metrics import mean_squared_error
import fbprophet

# matplotlib parameters
matplotlib.rcParams['axes.labelsize'] = 14
matplotlib.rcParams['xtick.labelsize'] = 12
matplotlib.rcParams['ytick.labelsize'] = 12
matplotlib.rcParams['text.color'] = 'k'

%config InlineBackend.figure_format = 'retina' 
%matplotlib inline

```
</div>

</div>



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
# Data Reading
train = pd.read_csv('https://raw.githubusercontent.com/RPI-DATA/tutorials-intro/master/rossmann-store-sales/rossmann-store-sales/train.csv', parse_dates = True, low_memory = False, index_col = 'Date')
store = pd.read_csv('https://raw.githubusercontent.com/RPI-DATA/tutorials-intro/master/rossmann-store-sales/rossmann-store-sales/store.csv', low_memory = False)

```
</div>

</div>



<a id="edat"></a>

# Exploratory Data Analysis (Train)
- - - - - - - - - --  - - - - - - -  - - - - - - - - --  - - -- - - -
We start by seeing what our data conists of. We want to see which variables are continuous vs which are categorical. After exploring some of the data, we see that we can create a feature. Number of sales divided by customers could give us a good metric to measure average sales per customer. We can also make an assumption that if we have missing values in this column that we have 0 customers. Since customers drive sales, we elect to remove all of these values.

Notice the order in which the data is listed. It is ordered from most recent date to oldest date. This may cause a problem when we look to develop our model.



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
      <th>Store</th>
      <th>DayOfWeek</th>
      <th>Sales</th>
      <th>Customers</th>
      <th>Open</th>
      <th>Promo</th>
      <th>StateHoliday</th>
      <th>SchoolHoliday</th>
    </tr>
    <tr>
      <th>Date</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2015-07-31</th>
      <td>1</td>
      <td>5</td>
      <td>5263</td>
      <td>555</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2015-07-31</th>
      <td>2</td>
      <td>5</td>
      <td>6064</td>
      <td>625</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2015-07-31</th>
      <td>3</td>
      <td>5</td>
      <td>8314</td>
      <td>821</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2015-07-31</th>
      <td>4</td>
      <td>5</td>
      <td>13995</td>
      <td>1498</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2015-07-31</th>
      <td>5</td>
      <td>5</td>
      <td>4822</td>
      <td>559</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
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
      <th>Store</th>
      <th>DayOfWeek</th>
      <th>Sales</th>
      <th>Customers</th>
      <th>Open</th>
      <th>Promo</th>
      <th>StateHoliday</th>
      <th>SchoolHoliday</th>
    </tr>
    <tr>
      <th>Date</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2015-07-31</th>
      <td>1</td>
      <td>5</td>
      <td>5263</td>
      <td>555</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2015-07-31</th>
      <td>2</td>
      <td>5</td>
      <td>6064</td>
      <td>625</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2015-07-31</th>
      <td>3</td>
      <td>5</td>
      <td>8314</td>
      <td>821</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2015-07-31</th>
      <td>4</td>
      <td>5</td>
      <td>13995</td>
      <td>1498</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2015-07-31</th>
      <td>5</td>
      <td>5</td>
      <td>4822</td>
      <td>559</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
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
train.shape

```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">


{:.output_data_text}
```
(1017209, 8)
```


</div>
</div>
</div>



After that we will use the amazing `.describe()` function which can provide us most of the statistic elements. 



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
train.describe()

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
      <th>Store</th>
      <th>DayOfWeek</th>
      <th>Sales</th>
      <th>Customers</th>
      <th>Open</th>
      <th>Promo</th>
      <th>SchoolHoliday</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>1.017209e+06</td>
      <td>1.017209e+06</td>
      <td>1.017209e+06</td>
      <td>1.017209e+06</td>
      <td>1.017209e+06</td>
      <td>1.017209e+06</td>
      <td>1.017209e+06</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>5.584297e+02</td>
      <td>3.998341e+00</td>
      <td>5.773819e+03</td>
      <td>6.331459e+02</td>
      <td>8.301067e-01</td>
      <td>3.815145e-01</td>
      <td>1.786467e-01</td>
    </tr>
    <tr>
      <th>std</th>
      <td>3.219087e+02</td>
      <td>1.997391e+00</td>
      <td>3.849926e+03</td>
      <td>4.644117e+02</td>
      <td>3.755392e-01</td>
      <td>4.857586e-01</td>
      <td>3.830564e-01</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1.000000e+00</td>
      <td>1.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>2.800000e+02</td>
      <td>2.000000e+00</td>
      <td>3.727000e+03</td>
      <td>4.050000e+02</td>
      <td>1.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>5.580000e+02</td>
      <td>4.000000e+00</td>
      <td>5.744000e+03</td>
      <td>6.090000e+02</td>
      <td>1.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>8.380000e+02</td>
      <td>6.000000e+00</td>
      <td>7.856000e+03</td>
      <td>8.370000e+02</td>
      <td>1.000000e+00</td>
      <td>1.000000e+00</td>
      <td>0.000000e+00</td>
    </tr>
    <tr>
      <th>max</th>
      <td>1.115000e+03</td>
      <td>7.000000e+00</td>
      <td>4.155100e+04</td>
      <td>7.388000e+03</td>
      <td>1.000000e+00</td>
      <td>1.000000e+00</td>
      <td>1.000000e+00</td>
    </tr>
  </tbody>
</table>
</div>
</div>


</div>
</div>
</div>



We will check all the missing elements over here.



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
missing = train.isnull().sum()
missing.sort_values(ascending=False)

```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">


{:.output_data_text}
```
SchoolHoliday    0
StateHoliday     0
Promo            0
Open             0
Customers        0
Sales            0
DayOfWeek        0
Store            0
dtype: int64
```


</div>
</div>
</div>



Next, we create a new metric to see average sales per customer.



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
train['SalesPerCustomer'] = train['Sales']/train['Customers']
train['SalesPerCustomer'].head()

```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">


{:.output_data_text}
```
Date
2015-07-31     9.482883
2015-07-31     9.702400
2015-07-31    10.126675
2015-07-31     9.342457
2015-07-31     8.626118
Name: SalesPerCustomer, dtype: float64
```


</div>
</div>
</div>



We are going to Check if there are any missing values with our new metric and drop them. Either the customers or the sales should be zero to give us a null SalesPerCustomer. 



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
missing = train.isnull().sum()
missing.sort_values(ascending=False)

```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">


{:.output_data_text}
```
SalesPerCustomer    172869
SchoolHoliday            0
StateHoliday             0
Promo                    0
Open                     0
Customers                0
Sales                    0
DayOfWeek                0
Store                    0
dtype: int64
```


</div>
</div>
</div>



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
train.dropna().head()

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
      <th>Store</th>
      <th>DayOfWeek</th>
      <th>Sales</th>
      <th>Customers</th>
      <th>Open</th>
      <th>Promo</th>
      <th>StateHoliday</th>
      <th>SchoolHoliday</th>
      <th>SalesPerCustomer</th>
    </tr>
    <tr>
      <th>Date</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2015-07-31</th>
      <td>1</td>
      <td>5</td>
      <td>5263</td>
      <td>555</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>9.482883</td>
    </tr>
    <tr>
      <th>2015-07-31</th>
      <td>2</td>
      <td>5</td>
      <td>6064</td>
      <td>625</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>9.702400</td>
    </tr>
    <tr>
      <th>2015-07-31</th>
      <td>3</td>
      <td>5</td>
      <td>8314</td>
      <td>821</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>10.126675</td>
    </tr>
    <tr>
      <th>2015-07-31</th>
      <td>4</td>
      <td>5</td>
      <td>13995</td>
      <td>1498</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>9.342457</td>
    </tr>
    <tr>
      <th>2015-07-31</th>
      <td>5</td>
      <td>5</td>
      <td>4822</td>
      <td>559</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>8.626118</td>
    </tr>
  </tbody>
</table>
</div>
</div>


</div>
</div>
</div>



<a id="edasd"></a>
# Exploratory Data Analysis (Store Data)
- - - - - -- - - - - - - - - - - - - -  - - - - - - - - - -
We do the same as we did for our training set. Exploring the data, we see that there are only 3 missing values in CompetitionDistance. Because this is such a small amount, we elect to replace these with the mean of the column. The other missing values are all dependent on Promo2. Because these missing values are because Promo2 is equal to 0, we can replace these nulls with 0.



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
store.head()

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
      <th>Store</th>
      <th>StoreType</th>
      <th>Assortment</th>
      <th>CompetitionDistance</th>
      <th>CompetitionOpenSinceMonth</th>
      <th>CompetitionOpenSinceYear</th>
      <th>Promo2</th>
      <th>Promo2SinceWeek</th>
      <th>Promo2SinceYear</th>
      <th>PromoInterval</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>c</td>
      <td>a</td>
      <td>1270.0</td>
      <td>9.0</td>
      <td>2008.0</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>a</td>
      <td>a</td>
      <td>570.0</td>
      <td>11.0</td>
      <td>2007.0</td>
      <td>1</td>
      <td>13.0</td>
      <td>2010.0</td>
      <td>Jan,Apr,Jul,Oct</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>a</td>
      <td>a</td>
      <td>14130.0</td>
      <td>12.0</td>
      <td>2006.0</td>
      <td>1</td>
      <td>14.0</td>
      <td>2011.0</td>
      <td>Jan,Apr,Jul,Oct</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>c</td>
      <td>c</td>
      <td>620.0</td>
      <td>9.0</td>
      <td>2009.0</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>a</td>
      <td>a</td>
      <td>29910.0</td>
      <td>4.0</td>
      <td>2015.0</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
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
store.shape

```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">


{:.output_data_text}
```
(1115, 10)
```


</div>
</div>
</div>



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
store.isnull().sum()

```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">


{:.output_data_text}
```
Store                          0
StoreType                      0
Assortment                     0
CompetitionDistance            3
CompetitionOpenSinceMonth    354
CompetitionOpenSinceYear     354
Promo2                         0
Promo2SinceWeek              544
Promo2SinceYear              544
PromoInterval                544
dtype: int64
```


</div>
</div>
</div>



Since there are only 3 missing values from this, we fill with the average from the column



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
store['CompetitionDistance'].fillna(store['CompetitionDistance'].mean(), inplace = True)

```
</div>

</div>



The rows that do not have any Promo2 we can fill the rest of the values with 0



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
store.fillna(0, inplace = True)

```
</div>

</div>



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
store.head()

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
      <th>Store</th>
      <th>StoreType</th>
      <th>Assortment</th>
      <th>CompetitionDistance</th>
      <th>CompetitionOpenSinceMonth</th>
      <th>CompetitionOpenSinceYear</th>
      <th>Promo2</th>
      <th>Promo2SinceWeek</th>
      <th>Promo2SinceYear</th>
      <th>PromoInterval</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>c</td>
      <td>a</td>
      <td>1270.0</td>
      <td>9.0</td>
      <td>2008.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>a</td>
      <td>a</td>
      <td>570.0</td>
      <td>11.0</td>
      <td>2007.0</td>
      <td>1</td>
      <td>13.0</td>
      <td>2010.0</td>
      <td>Jan,Apr,Jul,Oct</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>a</td>
      <td>a</td>
      <td>14130.0</td>
      <td>12.0</td>
      <td>2006.0</td>
      <td>1</td>
      <td>14.0</td>
      <td>2011.0</td>
      <td>Jan,Apr,Jul,Oct</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>c</td>
      <td>c</td>
      <td>620.0</td>
      <td>9.0</td>
      <td>2009.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>a</td>
      <td>a</td>
      <td>29910.0</td>
      <td>4.0</td>
      <td>2015.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>
</div>


</div>
</div>
</div>



Join the data together using an inner join so only the data that is in both data set is joined



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
train = train.merge(right=store, on='Store', how='left')

```
</div>

</div>



<a id="mam"></a>
# Moving-Average Model (Naive Model)
- - - - - - - - - - - - - - -
We are going to be using a moving average model for the stock prediction of GM for our baseline model. The moving average model will take the average of different "windows" of time to come up with its forecast

We reload the data because now we have a sense of how we want to maniplate it for our model. After doing the same data manipulation as before, we start to look at the trend of our sales. 



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
train = pd.read_csv("https://raw.githubusercontent.com/RPI-DATA/tutorials-intro/master/rossmann-store-sales/rossmann-store-sales/train.csv", parse_dates = True, low_memory = False, index_col = 'Date')
train = train.sort_index(ascending = True)

```
</div>

</div>



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
train['SalesPerCustomer'] = train['Sales']/train['Customers']
train['SalesPerCustomer'].head()

```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">


{:.output_data_text}
```
Date
2013-01-01   NaN
2013-01-01   NaN
2013-01-01   NaN
2013-01-01   NaN
2013-01-01   NaN
Name: SalesPerCustomer, dtype: float64
```


</div>
</div>
</div>



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
train = train.dropna()

```
</div>

</div>



Here, we are simply graphing the sales that we have. As you can see, there are a tremendous amount of sales to the point where our graph just looks like a blue shading. However, we can get a sense of how our sales are distributed.



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
plt.plot(train.index, train['Sales'])
plt.title('Rossmann Sales')
plt.ylabel('Price ($)');
plt.show()

```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">

{:.output_png}
![png](C%3A/Users/sjgar/Documents/GitHub/jupyter-book/_build/images/notebooks/time_series_basic_37_0.png)

</div>
</div>
</div>



To clean up our graph, we want to form a new column which only accounts for the year of the sales. 



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
train['Year'] = train.index.year

# Take Dates from index and move to Date column 
train.reset_index(level=0, inplace = True)
train['sales'] = 0


```
</div>

</div>



Split the data into a train and test set. We use an 80/20 split. Then, we look to start are modein. 

test_store stands for the forecasting part. 



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
train_store=train[0:675472] 
test_store=train[675472:]

```
</div>

</div>



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
train_store.Date = pd.to_datetime(train_store.Date, format="%Y-%m-%d")
train_store.index = train_store.Date
test_store.Date = pd.to_datetime(test_store.Date, format="%Y-%m-%d")
test_store.index = test_store.Date

train_store = train_store.resample('D').mean()
train_store = train_store.interpolate(method='linear')

test_store = test_store.resample('D').mean()
test_store = test_store.interpolate(method='linear')

```
</div>

</div>



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
train_store.Sales.plot(figsize=(25,10), title='daily sales', fontsize=25)
test_store.Sales.plot()

```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">


{:.output_data_text}
```
<matplotlib.axes._subplots.AxesSubplot at 0x1a1c3f5b00>
```


</div>
</div>
<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">

{:.output_png}
![png](C%3A/Users/sjgar/Documents/GitHub/jupyter-book/_build/images/notebooks/time_series_basic_43_1.png)

</div>
</div>
</div>



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
y_hat_avg_moving = test_store.copy()
y_hat_avg_moving['moving_avg_forcast'] = train_store.Sales.rolling(90).mean().iloc[-1]
plt.figure(figsize=(25,10))
plt.plot(train_store['Sales'], label='Train')
plt.plot(test_store['Sales'], label='Test')
plt.plot(y_hat_avg_moving['moving_avg_forcast'], label='Moving Forecast')
plt.legend(loc='best')
plt.title('Moving Average Forecast')

```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">


{:.output_data_text}
```
Text(0.5, 1.0, 'Moving Average Forecast')
```


</div>
</div>
<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">

{:.output_png}
![png](C%3A/Users/sjgar/Documents/GitHub/jupyter-book/_build/images/notebooks/time_series_basic_44_1.png)

</div>
</div>
</div>



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
rms_avg_rolling = sqrt(mean_squared_error(test_store.Sales, y_hat_avg_moving.moving_avg_forcast))
print('ROLLING AVERAGE',rms_avg_rolling)

```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">
{:.output_stream}
```
ROLLING AVERAGE 1915.886219620586
```
</div>
</div>
</div>



The rolling average for our model is 1,915.88. This prediction seems to be very consistent in hitting the average of the future sales. This naive model definitely looks like a solid model, however, it is not the best one. 



<a id="fpm"></a>
# Facebook Prophet Model
- - - - - - - -  - - - - -- - - -
The Facebook Prophet package is designed to analyze time series data with daily observations, which can display patterns on different time scales. Prophet is optimized for business tasks with the following characteristics:

hourly, daily, or weekly observations with at least a few months (preferably a year) of history
strong multiple "human-scale" seasonalities: day of week and time of year
important holidays that occur at irregular intervals that are known in advance (e.g. the Super Bowl)
a reasonable number of missing observations or large outliers
historical trend changes, for instance due to product launches or logging changes
trends that are non-linear growth curves, where a trend hits a natural limit or saturates
inspired by https://research.fb.com/prophet-forecasting-at-scale/



According to the "facebook research" website, there is four main component inside the facebook prophet model.

** A piecewise linear or logistic growth trend. 

** A yearly seasonal component modeled using Fourier series. 

** A weekly seasonal component using dummy variables.

** A user-provided list of important holidays.

The method of combing different models into one makes the facebook prophet model much more precise and flexible.



First of all, we will dealt with the data as same as the naive model



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
train = pd.read_csv("https://raw.githubusercontent.com/RPI-DATA/tutorials-intro/master/rossmann-store-sales/rossmann-store-sales/train.csv", parse_dates = True, low_memory = False)

```
</div>

</div>



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
train['SalesPerCustomer'] = train['Sales']/train['Customers']
train['SalesPerCustomer'].head()

train = train.dropna()

```
</div>

</div>



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
sales = train[train.Store == 1].loc[:, ['Date', 'Sales']]

# reverse to the order: from 2013 to 2015
sales = sales.sort_index(ascending = False)

sales['Date'] = pd.DatetimeIndex(sales['Date'])
sales.dtypes

```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">


{:.output_data_text}
```
Date     datetime64[ns]
Sales             int64
dtype: object
```


</div>
</div>
</div>



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
sales = sales.rename(columns = {'Date': 'ds',
                                'Sales': 'y'})

```
</div>

</div>



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
sales.head()

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
      <th>ds</th>
      <th>y</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1014980</th>
      <td>2013-01-02</td>
      <td>5530</td>
    </tr>
    <tr>
      <th>1013865</th>
      <td>2013-01-03</td>
      <td>4327</td>
    </tr>
    <tr>
      <th>1012750</th>
      <td>2013-01-04</td>
      <td>4486</td>
    </tr>
    <tr>
      <th>1011635</th>
      <td>2013-01-05</td>
      <td>4997</td>
    </tr>
    <tr>
      <th>1009405</th>
      <td>2013-01-07</td>
      <td>7176</td>
    </tr>
  </tbody>
</table>
</div>
</div>


</div>
</div>
</div>



Now, despite of the naive model, we will apply our sales set to the face book model by using the function `fbprophet`. `fbprophet.Prophet` can change the value of "changepoint_prior_scale" to 0.05 to achieve a better fit or to 0.15 to control how sensitive the trend is.



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
sales_prophet = fbprophet.Prophet(changepoint_prior_scale=0.05, daily_seasonality=True)
sales_prophet.fit(sales)

```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">


{:.output_data_text}
```
<fbprophet.forecaster.Prophet at 0x1a1c952da0>
```


</div>
</div>
</div>



We will figure out the best forecasting by changing the value of changepoints.

If we find that our model is is fitting too closely to our training data (overfitting), our data will not be able to generalize new data.

If our model is not fitting closely enough to our training data (underfitting), our data has too much bias.

Underfitting: increase changepoint to allow more flexibility Overfitting: decrease changepoint to limit flexibili



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
# Try 4 different changepoints
for changepoint in [0.001, 0.05, 0.1, 0.5]:
    model = fbprophet.Prophet(daily_seasonality=False, changepoint_prior_scale=changepoint)
    model.fit(sales)
    
    future = model.make_future_dataframe(periods=365, freq='D')
    future = model.predict(future)
    
    sales[changepoint] = future['yhat']

```
</div>

</div>



We can now create the plot under all of the situation.



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
# Create the plot
plt.figure(figsize=(10, 8))

# Actual observations
plt.plot(sales['ds'], sales['y'], 'ko', label = 'Observations')
colors = {0.001: 'b', 0.05: 'r', 0.1: 'grey', 0.5: 'gold'}

# Plot each of the changepoint predictions
for changepoint in [0.001, 0.05, 0.1, 0.5]:
    plt.plot(sales['ds'], sales[changepoint], color = colors[changepoint], label = '%.3f prior scale' % changepoint)
    
plt.legend(prop={'size': 14})
plt.xlabel('Date'); plt.ylabel('Rossmann Sales'); plt.title('Rossmann Effect of Changepoint Prior Scale');

```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">

{:.output_png}
![png](C%3A/Users/sjgar/Documents/GitHub/jupyter-book/_build/images/notebooks/time_series_basic_60_0.png)

</div>
</div>
</div>



Predictions for 6 Weeks
In order to make forecasts, we need to create a future dataframe. We need to specify the amount of future periods to predict and the frequency of our prediction.

Periods: 6 Weeks 

Frequency: Daily



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
# Make a future dataframe for 6weeks
sales_forecast = sales_prophet.make_future_dataframe(periods=6*7, freq='D')
# Make predictions
sales_forecast = sales_prophet.predict(sales_forecast)

```
</div>

</div>



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
sales_prophet.plot(sales_forecast, xlabel = 'Date', ylabel = 'Sales')
plt.title('Rossmann Sales');

```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">

{:.output_png}
![png](C%3A/Users/sjgar/Documents/GitHub/jupyter-book/_build/images/notebooks/time_series_basic_63_0.png)

</div>
</div>
</div>



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
sales_prophet.changepoints[:10]

```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">


{:.output_data_text}
```
25    2013-01-31
50    2013-03-01
75    2013-04-02
100   2013-05-02
125   2013-06-04
150   2013-07-03
174   2013-07-31
199   2013-08-29
224   2013-09-27
249   2013-10-28
Name: ds, dtype: datetime64[ns]
```


</div>
</div>
</div>



We have listed out the most significant changepoints in our data. This is representing when the time series growth rate significantly changes.



<a id="c"></a>
# Conclusion
- -- - - - - - -  - - 
In this notebook, we made 2 different math model for the rossmann store sales dataset to forecast the future sales. Moving-average model brings us a basic understand of how the math model works, while facebook prophet model calculates the best solid result. Those math model will give us both of the rolling average and test model. 



<a id="r"></a>
# References
- - - -  - -  - - - - - - - - -
The dataset is the rossmann store sales dataset from kaggle:
https://www.kaggle.com/c/rossmann-store-sales

Facebook Prophet Documentation:
https://research.fb.com/prophet-forecasting-at-scale/





# Contributers

* nickespo21
* Linghao Dong
* Jose Figueroa



[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/RPI-DATA/tutorials-intro/blob/master/website/Forecasting%20the%20Rossmann%20Store%20Sales.ipynb)

