---
interact_link: content/notebooks/10-intro-r/04-intro-r-dataframes.ipynb
kernel_name: ir
has_widgets: false
title: 'Dataframes'
prev_page:
  url: /notebooks/10-intro-r/03-intro-r-datastructures.html
  title: 'Data Structures'
next_page:
  url: /notebooks/10-intro-r/05-intro-r-functions.html
  title: 'Functions'
comment: "***PROGRAMMATICALLY GENERATED, DO NOT EDIT. SEE ORIGINAL FILES IN /content***"
---


[![AnalyticsDojo](https://github.com/rpi-techfundamentals/spring2019-materials/blob/master/fig/final-logo.png?raw=1)](http://rpi.analyticsdojo.com)
<center><h1>Introduction to R - DataFrames</h1></center>
<center><h3><a href = 'http://rpi.analyticsdojo.com'>rpi.analyticsdojo.com</a></h3></center>





## Introduction to R DataFrames
- Data frames are combinations of vectors of the same length, but can be of different types.
- It is a special type of list.  
- Data frames are what is used for standard rectangular (record by field) datasets, similar to a spreadsheet
- Data frames are a functionality that both sets R aside from some languages (e.g., Matlab) and provides functionality similar to some statistical packages (e.g., Stata, SAS) and Python's Pandas Packages.




<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```R
frame=read.csv(file="../../input/iris.csv", header=TRUE,sep=",")
class(frame)
head(frame) #The first few rows.
tail(frame) #The last few rows.
str(frame) #The Structure.



```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">

<div markdown="0" class="output output_html">
'data.frame'
</div>

</div>
</div>
<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">

<div markdown="0" class="output output_html">
<table>
<thead><tr><th scope=col>sepal_length</th><th scope=col>sepal_width</th><th scope=col>petal_length</th><th scope=col>petal_width</th><th scope=col>species</th></tr></thead>
<tbody>
	<tr><td>5.1   </td><td>3.5   </td><td>1.4   </td><td>0.2   </td><td>setosa</td></tr>
	<tr><td>4.9   </td><td>3.0   </td><td>1.4   </td><td>0.2   </td><td>setosa</td></tr>
	<tr><td>4.7   </td><td>3.2   </td><td>1.3   </td><td>0.2   </td><td>setosa</td></tr>
	<tr><td>4.6   </td><td>3.1   </td><td>1.5   </td><td>0.2   </td><td>setosa</td></tr>
	<tr><td>5.0   </td><td>3.6   </td><td>1.4   </td><td>0.2   </td><td>setosa</td></tr>
	<tr><td>5.4   </td><td>3.9   </td><td>1.7   </td><td>0.4   </td><td>setosa</td></tr>
</tbody>
</table>

</div>

</div>
</div>
<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">

<div markdown="0" class="output output_html">
<table>
<thead><tr><th></th><th scope=col>sepal_length</th><th scope=col>sepal_width</th><th scope=col>petal_length</th><th scope=col>petal_width</th><th scope=col>species</th></tr></thead>
<tbody>
	<tr><th scope=row>145</th><td>6.7      </td><td>3.3      </td><td>5.7      </td><td>2.5      </td><td>virginica</td></tr>
	<tr><th scope=row>146</th><td>6.7      </td><td>3.0      </td><td>5.2      </td><td>2.3      </td><td>virginica</td></tr>
	<tr><th scope=row>147</th><td>6.3      </td><td>2.5      </td><td>5.0      </td><td>1.9      </td><td>virginica</td></tr>
	<tr><th scope=row>148</th><td>6.5      </td><td>3.0      </td><td>5.2      </td><td>2.0      </td><td>virginica</td></tr>
	<tr><th scope=row>149</th><td>6.2      </td><td>3.4      </td><td>5.4      </td><td>2.3      </td><td>virginica</td></tr>
	<tr><th scope=row>150</th><td>5.9      </td><td>3.0      </td><td>5.1      </td><td>1.8      </td><td>virginica</td></tr>
</tbody>
</table>

</div>

</div>
</div>
<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">
{:.output_stream}
```
'data.frame':	150 obs. of  5 variables:
 $ sepal_length: num  5.1 4.9 4.7 4.6 5 5.4 4.6 5 4.4 4.9 ...
 $ sepal_width : num  3.5 3 3.2 3.1 3.6 3.9 3.4 3.4 2.9 3.1 ...
 $ petal_length: num  1.4 1.4 1.3 1.5 1.4 1.7 1.4 1.5 1.4 1.5 ...
 $ petal_width : num  0.2 0.2 0.2 0.2 0.2 0.4 0.3 0.2 0.2 0.1 ...
 $ species     : Factor w/ 3 levels "setosa","versicolor",..: 1 1 1 1 1 1 1 1 1 1 ...
```
</div>
</div>
</div>



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```R
dim(frame) #Results in rows x columns
nrow(frame)  #The number of Rows
names(frame) #Provides the names
length(frame) #The number of columns
summary(frame) #Provides summary statistics.
is.matrix(frame) #Yields False because it has different types.  
is.list(frame) #Yields True
class(frame$sepal_length)
class(frame$species)
levels(frame$species)

```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">

<div markdown="0" class="output output_html">
<ol class=list-inline>
	<li>150</li>
	<li>5</li>
</ol>

</div>

</div>
</div>
<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">

<div markdown="0" class="output output_html">
150
</div>

</div>
</div>
<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">

<div markdown="0" class="output output_html">
<ol class=list-inline>
	<li>'sepal_length'</li>
	<li>'sepal_width'</li>
	<li>'petal_length'</li>
	<li>'petal_width'</li>
	<li>'species'</li>
</ol>

</div>

</div>
</div>
<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">

<div markdown="0" class="output output_html">
5
</div>

</div>
</div>
<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">
{:.output_data_text}
```
  sepal_length    sepal_width     petal_length    petal_width   
 Min.   :4.300   Min.   :2.000   Min.   :1.000   Min.   :0.100  
 1st Qu.:5.100   1st Qu.:2.800   1st Qu.:1.600   1st Qu.:0.300  
 Median :5.800   Median :3.000   Median :4.350   Median :1.300  
 Mean   :5.843   Mean   :3.054   Mean   :3.759   Mean   :1.199  
 3rd Qu.:6.400   3rd Qu.:3.300   3rd Qu.:5.100   3rd Qu.:1.800  
 Max.   :7.900   Max.   :4.400   Max.   :6.900   Max.   :2.500  
       species  
 setosa    :50  
 versicolor:50  
 virginica :50  
                
                
                
```

</div>
</div>
<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">

<div markdown="0" class="output output_html">
FALSE
</div>

</div>
</div>
<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">

<div markdown="0" class="output output_html">
TRUE
</div>

</div>
</div>
<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">

<div markdown="0" class="output output_html">
'numeric'
</div>

</div>
</div>
<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">

<div markdown="0" class="output output_html">
'factor'
</div>

</div>
</div>
<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">

<div markdown="0" class="output output_html">
<ol class=list-inline>
	<li>'setosa'</li>
	<li>'versicolor'</li>
	<li>'virginica'</li>
</ol>

</div>

</div>
</div>
</div>



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```R
frame[c("species","sepal_width")]

```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">

<div markdown="0" class="output output_html">
<table>
<thead><tr><th scope=col>species</th><th scope=col>sepal_width</th></tr></thead>
<tbody>
	<tr><td>setosa</td><td>3.5   </td></tr>
	<tr><td>setosa</td><td>3.0   </td></tr>
	<tr><td>setosa</td><td>3.2   </td></tr>
	<tr><td>setosa</td><td>3.1   </td></tr>
	<tr><td>setosa</td><td>3.6   </td></tr>
	<tr><td>setosa</td><td>3.9   </td></tr>
	<tr><td>setosa</td><td>3.4   </td></tr>
	<tr><td>setosa</td><td>3.4   </td></tr>
	<tr><td>setosa</td><td>2.9   </td></tr>
	<tr><td>setosa</td><td>3.1   </td></tr>
	<tr><td>setosa</td><td>3.7   </td></tr>
	<tr><td>setosa</td><td>3.4   </td></tr>
	<tr><td>setosa</td><td>3.0   </td></tr>
	<tr><td>setosa</td><td>3.0   </td></tr>
	<tr><td>setosa</td><td>4.0   </td></tr>
	<tr><td>setosa</td><td>4.4   </td></tr>
	<tr><td>setosa</td><td>3.9   </td></tr>
	<tr><td>setosa</td><td>3.5   </td></tr>
	<tr><td>setosa</td><td>3.8   </td></tr>
	<tr><td>setosa</td><td>3.8   </td></tr>
	<tr><td>setosa</td><td>3.4   </td></tr>
	<tr><td>setosa</td><td>3.7   </td></tr>
	<tr><td>setosa</td><td>3.6   </td></tr>
	<tr><td>setosa</td><td>3.3   </td></tr>
	<tr><td>setosa</td><td>3.4   </td></tr>
	<tr><td>setosa</td><td>3.0   </td></tr>
	<tr><td>setosa</td><td>3.4   </td></tr>
	<tr><td>setosa</td><td>3.5   </td></tr>
	<tr><td>setosa</td><td>3.4   </td></tr>
	<tr><td>setosa</td><td>3.2   </td></tr>
	<tr><td>⋮</td><td>⋮</td></tr>
	<tr><td>virginica</td><td>3.2      </td></tr>
	<tr><td>virginica</td><td>2.8      </td></tr>
	<tr><td>virginica</td><td>2.8      </td></tr>
	<tr><td>virginica</td><td>2.7      </td></tr>
	<tr><td>virginica</td><td>3.3      </td></tr>
	<tr><td>virginica</td><td>3.2      </td></tr>
	<tr><td>virginica</td><td>2.8      </td></tr>
	<tr><td>virginica</td><td>3.0      </td></tr>
	<tr><td>virginica</td><td>2.8      </td></tr>
	<tr><td>virginica</td><td>3.0      </td></tr>
	<tr><td>virginica</td><td>2.8      </td></tr>
	<tr><td>virginica</td><td>3.8      </td></tr>
	<tr><td>virginica</td><td>2.8      </td></tr>
	<tr><td>virginica</td><td>2.8      </td></tr>
	<tr><td>virginica</td><td>2.6      </td></tr>
	<tr><td>virginica</td><td>3.0      </td></tr>
	<tr><td>virginica</td><td>3.4      </td></tr>
	<tr><td>virginica</td><td>3.1      </td></tr>
	<tr><td>virginica</td><td>3.0      </td></tr>
	<tr><td>virginica</td><td>3.1      </td></tr>
	<tr><td>virginica</td><td>3.1      </td></tr>
	<tr><td>virginica</td><td>3.1      </td></tr>
	<tr><td>virginica</td><td>2.7      </td></tr>
	<tr><td>virginica</td><td>3.2      </td></tr>
	<tr><td>virginica</td><td>3.3      </td></tr>
	<tr><td>virginica</td><td>3.0      </td></tr>
	<tr><td>virginica</td><td>2.5      </td></tr>
	<tr><td>virginica</td><td>3.0      </td></tr>
	<tr><td>virginica</td><td>3.4      </td></tr>
	<tr><td>virginica</td><td>3.0      </td></tr>
</tbody>
</table>

</div>

</div>
</div>
</div>



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```R
frame['petals']<-0
frame$petals2<-0
head(frame)

```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">

<div markdown="0" class="output output_html">
<table>
<thead><tr><th scope=col>sepal_length</th><th scope=col>sepal_width</th><th scope=col>petal_length</th><th scope=col>petal_width</th><th scope=col>species</th><th scope=col>petals</th><th scope=col>petals2</th></tr></thead>
<tbody>
	<tr><td>5.1   </td><td>3.5   </td><td>1.4   </td><td>0.2   </td><td>setosa</td><td>0     </td><td>0     </td></tr>
	<tr><td>4.9   </td><td>3.0   </td><td>1.4   </td><td>0.2   </td><td>setosa</td><td>0     </td><td>0     </td></tr>
	<tr><td>4.7   </td><td>3.2   </td><td>1.3   </td><td>0.2   </td><td>setosa</td><td>0     </td><td>0     </td></tr>
	<tr><td>4.6   </td><td>3.1   </td><td>1.5   </td><td>0.2   </td><td>setosa</td><td>0     </td><td>0     </td></tr>
	<tr><td>5.0   </td><td>3.6   </td><td>1.4   </td><td>0.2   </td><td>setosa</td><td>0     </td><td>0     </td></tr>
	<tr><td>5.4   </td><td>3.9   </td><td>1.7   </td><td>0.4   </td><td>setosa</td><td>0     </td><td>0     </td></tr>
</tbody>
</table>

</div>

</div>
</div>
</div>



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```R
mean.sepalLenth.setosa<-mean(frame[,'sepal_length'])

```
</div>

</div>



## Slicing a Dataframe by Column
- Remember the syntax of `df[rows,columns]` 
- Using `dataframe$column` provides one way of selecting a column. 
- We can also specify the index position: `dataframe[,columnIndex]`
- We can also specify the column name: `dataframe[,columnsName]`



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```R
sepal_length1<-frame$sepal_length #Using Dollar Sign and the column name.
sepal_length2<- frame[,1]  #Using the Index Location
sepal_length3<- frame[,'sepal_length']
sepal_length4<- frame[,c('sepal_length','sepal_width')]

sepal_length1[1:5]  #Print the first 5  
sepal_length2[1:5]
sepal_length3[1:5]


```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">

<div markdown="0" class="output output_html">
<ol class=list-inline>
	<li>5.1</li>
	<li>4.9</li>
	<li>4.7</li>
	<li>4.6</li>
	<li>5</li>
</ol>

</div>

</div>
</div>
<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">

<div markdown="0" class="output output_html">
<ol class=list-inline>
	<li>5.1</li>
	<li>4.9</li>
	<li>4.7</li>
	<li>4.6</li>
	<li>5</li>
</ol>

</div>

</div>
</div>
<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">

<div markdown="0" class="output output_html">
<ol class=list-inline>
	<li>5.1</li>
	<li>4.9</li>
	<li>4.7</li>
	<li>4.6</li>
	<li>5</li>
</ol>

</div>

</div>
</div>
</div>



## Selecting Rows
- We can select rows from a dataframe using index position: `dataframe[rowIndex,columnIndex]`. 
- Use `c(row1, row2, row3)` to select out specific rows. 



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```R
frame2<-frame[1:20,]   
frame3<-frame[c(1,5,6),] #This selects out specific rows
nrow(frame2)
nrow(frame3)
frame3

```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">

<div markdown="0" class="output output_html">
20
</div>

</div>
</div>
<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">

<div markdown="0" class="output output_html">
3
</div>

</div>
</div>
<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">

<div markdown="0" class="output output_html">
<table>
<thead><tr><th></th><th scope=col>sepal_length</th><th scope=col>sepal_width</th><th scope=col>petal_length</th><th scope=col>petal_width</th><th scope=col>species</th><th scope=col>petals</th><th scope=col>petals2</th></tr></thead>
<tbody>
	<tr><th scope=row>1</th><td>5.1   </td><td>3.5   </td><td>1.4   </td><td>0.2   </td><td>setosa</td><td>0     </td><td>0     </td></tr>
	<tr><th scope=row>5</th><td>5.0   </td><td>3.6   </td><td>1.4   </td><td>0.2   </td><td>setosa</td><td>0     </td><td>0     </td></tr>
	<tr><th scope=row>6</th><td>5.4   </td><td>3.9   </td><td>1.7   </td><td>0.4   </td><td>setosa</td><td>0     </td><td>0     </td></tr>
</tbody>
</table>

</div>

</div>
</div>
</div>



## Conditional Statements and Dataframes with Subset
- We can select subsets of a dataframe by putting an equality in the row or subset. 
- Subset is also a dataframe. 
- Can optionally select columns with the `select = c(col1, col2)`



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```R
setosa.df <- subset(frame, species == 'setosa')

head(setosa.df)
class(setosa.df)
nrow(setosa.df)
mean.sepalLenth.setosa<-mean(setosa.df$sepal_length) #This creates a new vector
mean.sepalLenth.setosa
setosa.df.highseptalLength <- subset(setosa.df, sepal_length > mean.sepalLenth.setosa)
nrow(setosa.df.highseptalLength)
head(setosa.df.highseptalLength)
setosa.dfhighseptalLength2 <- subset(setosa.df, sepal_length > mean.sepalLenth.setosa, select = c(sepal_length, species))
head(setosa.dfhighseptalLength2)

```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">

<div markdown="0" class="output output_html">
<table>
<thead><tr><th scope=col>sepal_length</th><th scope=col>sepal_width</th><th scope=col>petal_length</th><th scope=col>petal_width</th><th scope=col>species</th><th scope=col>petals</th><th scope=col>petals2</th></tr></thead>
<tbody>
	<tr><td>5.1   </td><td>3.5   </td><td>1.4   </td><td>0.2   </td><td>setosa</td><td>0     </td><td>0     </td></tr>
	<tr><td>4.9   </td><td>3.0   </td><td>1.4   </td><td>0.2   </td><td>setosa</td><td>0     </td><td>0     </td></tr>
	<tr><td>4.7   </td><td>3.2   </td><td>1.3   </td><td>0.2   </td><td>setosa</td><td>0     </td><td>0     </td></tr>
	<tr><td>4.6   </td><td>3.1   </td><td>1.5   </td><td>0.2   </td><td>setosa</td><td>0     </td><td>0     </td></tr>
	<tr><td>5.0   </td><td>3.6   </td><td>1.4   </td><td>0.2   </td><td>setosa</td><td>0     </td><td>0     </td></tr>
	<tr><td>5.4   </td><td>3.9   </td><td>1.7   </td><td>0.4   </td><td>setosa</td><td>0     </td><td>0     </td></tr>
</tbody>
</table>

</div>

</div>
</div>
<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">

<div markdown="0" class="output output_html">
'data.frame'
</div>

</div>
</div>
<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">

<div markdown="0" class="output output_html">
50
</div>

</div>
</div>
<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">

<div markdown="0" class="output output_html">
5.006
</div>

</div>
</div>
<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">

<div markdown="0" class="output output_html">
22
</div>

</div>
</div>
<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">

<div markdown="0" class="output output_html">
<table>
<thead><tr><th></th><th scope=col>sepal_length</th><th scope=col>sepal_width</th><th scope=col>petal_length</th><th scope=col>petal_width</th><th scope=col>species</th><th scope=col>petals</th><th scope=col>petals2</th></tr></thead>
<tbody>
	<tr><th scope=row>1</th><td>5.1   </td><td>3.5   </td><td>1.4   </td><td>0.2   </td><td>setosa</td><td>0     </td><td>0     </td></tr>
	<tr><th scope=row>6</th><td>5.4   </td><td>3.9   </td><td>1.7   </td><td>0.4   </td><td>setosa</td><td>0     </td><td>0     </td></tr>
	<tr><th scope=row>11</th><td>5.4   </td><td>3.7   </td><td>1.5   </td><td>0.2   </td><td>setosa</td><td>0     </td><td>0     </td></tr>
	<tr><th scope=row>15</th><td>5.8   </td><td>4.0   </td><td>1.2   </td><td>0.2   </td><td>setosa</td><td>0     </td><td>0     </td></tr>
	<tr><th scope=row>16</th><td>5.7   </td><td>4.4   </td><td>1.5   </td><td>0.4   </td><td>setosa</td><td>0     </td><td>0     </td></tr>
	<tr><th scope=row>17</th><td>5.4   </td><td>3.9   </td><td>1.3   </td><td>0.4   </td><td>setosa</td><td>0     </td><td>0     </td></tr>
</tbody>
</table>

</div>

</div>
</div>
<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">

<div markdown="0" class="output output_html">
<table>
<thead><tr><th></th><th scope=col>sepal_length</th><th scope=col>species</th></tr></thead>
<tbody>
	<tr><th scope=row>1</th><td>5.1   </td><td>setosa</td></tr>
	<tr><th scope=row>6</th><td>5.4   </td><td>setosa</td></tr>
	<tr><th scope=row>11</th><td>5.4   </td><td>setosa</td></tr>
	<tr><th scope=row>15</th><td>5.8   </td><td>setosa</td></tr>
	<tr><th scope=row>16</th><td>5.7   </td><td>setosa</td></tr>
	<tr><th scope=row>17</th><td>5.4   </td><td>setosa</td></tr>
</tbody>
</table>

</div>

</div>
</div>
</div>



## Subsetting Rows Using Indices
- Just like pandas, we are using conditional statements to specify specific rows. 
- See [here](http://www.ats.ucla.edu/stat/r/faq/subset_R.htm) for good coverage and examples. 



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```R
setosa.df <- frame[frame$species == "setosa",]
head(setosa.df)
class(setosa.df)
nrow(setosa.df)
mean.sepalLenth.setosa<-mean(setosa.df$sepal_length) #This creates a new vector
mean.sepalLenth.setosa
setosa.df.highseptalLength <- setosa.df[setosa.df$sepal_length > mean.sepalLenth.setosa,]
nrow(setosa.df.highseptalLength)
head(setosa.df.highseptalLength)

```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">

<div markdown="0" class="output output_html">
<table>
<thead><tr><th></th><th scope=col>sepal_length</th><th scope=col>sepal_width</th><th scope=col>petal_length</th><th scope=col>petal_width</th><th scope=col>species</th><th scope=col>petals</th></tr></thead>
<tbody>
	<tr><th scope=row>1</th><td>5.1   </td><td>3.5   </td><td>1.4   </td><td>0.2   </td><td>setosa</td><td>0     </td></tr>
	<tr><th scope=row>2</th><td>4.9   </td><td>3     </td><td>1.4   </td><td>0.2   </td><td>setosa</td><td>0     </td></tr>
	<tr><th scope=row>3</th><td>4.7   </td><td>3.2   </td><td>1.3   </td><td>0.2   </td><td>setosa</td><td>0     </td></tr>
	<tr><th scope=row>4</th><td>4.6   </td><td>3.1   </td><td>1.5   </td><td>0.2   </td><td>setosa</td><td>0     </td></tr>
	<tr><th scope=row>5</th><td>5     </td><td>3.6   </td><td>1.4   </td><td>0.2   </td><td>setosa</td><td>0     </td></tr>
	<tr><th scope=row>6</th><td>5.4   </td><td>3.9   </td><td>1.7   </td><td>0.4   </td><td>setosa</td><td>0     </td></tr>
</tbody>
</table>

</div>

</div>
</div>
<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">

<div markdown="0" class="output output_html">
'data.frame'
</div>

</div>
</div>
<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">

<div markdown="0" class="output output_html">
50
</div>

</div>
</div>
<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">

<div markdown="0" class="output output_html">
5.006
</div>

</div>
</div>
<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">

<div markdown="0" class="output output_html">
22
</div>

</div>
</div>
<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">

<div markdown="0" class="output output_html">
<table>
<thead><tr><th></th><th scope=col>sepal_length</th><th scope=col>sepal_width</th><th scope=col>petal_length</th><th scope=col>petal_width</th><th scope=col>species</th><th scope=col>petals</th></tr></thead>
<tbody>
	<tr><th scope=row>1</th><td>5.1   </td><td>3.5   </td><td>1.4   </td><td>0.2   </td><td>setosa</td><td>0     </td></tr>
	<tr><th scope=row>6</th><td>5.4   </td><td>3.9   </td><td>1.7   </td><td>0.4   </td><td>setosa</td><td>0     </td></tr>
	<tr><th scope=row>11</th><td>5.4   </td><td>3.7   </td><td>1.5   </td><td>0.2   </td><td>setosa</td><td>0     </td></tr>
	<tr><th scope=row>15</th><td>5.8   </td><td>4     </td><td>1.2   </td><td>0.2   </td><td>setosa</td><td>0     </td></tr>
	<tr><th scope=row>16</th><td>5.7   </td><td>4.4   </td><td>1.5   </td><td>0.4   </td><td>setosa</td><td>0     </td></tr>
	<tr><th scope=row>17</th><td>5.4   </td><td>3.9   </td><td>1.3   </td><td>0.4   </td><td>setosa</td><td>0     </td></tr>
</tbody>
</table>

</div>

</div>
</div>
</div>



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```R
specific.df <- frame[frame$sepal_length %in% c(5.1,5.8),]
head(specific.df)


```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">

<div markdown="0" class="output output_html">
<table>
<thead><tr><th></th><th scope=col>sepal_length</th><th scope=col>sepal_width</th><th scope=col>petal_length</th><th scope=col>petal_width</th><th scope=col>species</th><th scope=col>petals</th></tr></thead>
<tbody>
	<tr><th scope=row>1</th><td>5.1   </td><td>3.5   </td><td>1.4   </td><td>0.2   </td><td>setosa</td><td>0     </td></tr>
	<tr><th scope=row>15</th><td>5.8   </td><td>4.0   </td><td>1.2   </td><td>0.2   </td><td>setosa</td><td>0     </td></tr>
	<tr><th scope=row>18</th><td>5.1   </td><td>3.5   </td><td>1.4   </td><td>0.3   </td><td>setosa</td><td>0     </td></tr>
	<tr><th scope=row>20</th><td>5.1   </td><td>3.8   </td><td>1.5   </td><td>0.3   </td><td>setosa</td><td>0     </td></tr>
	<tr><th scope=row>22</th><td>5.1   </td><td>3.7   </td><td>1.5   </td><td>0.4   </td><td>setosa</td><td>0     </td></tr>
	<tr><th scope=row>24</th><td>5.1   </td><td>3.3   </td><td>1.7   </td><td>0.5   </td><td>setosa</td><td>0     </td></tr>
</tbody>
</table>

</div>

</div>
</div>
</div>



## Basics

1. Load the Titanic train.csv data into an R data frame.
2. Calculate the number of rows in the data frame.
3. Calcuated general descriptive statistics for the data frame.
4. Slice the data frame into 2 parts, selecting the first half of the rows. 
5. Select just the columns passangerID and whether they survivied or not. 



## CREDITS

Copyright [AnalyticsDojo](http://rpi.analyticsdojo.com) 2016
This work is licensed under the [Creative Commons Attribution 4.0 International](https://creativecommons.org/licenses/by/4.0/) license agreement.
Adopted from [Berkley R Bootcamp](https://github.com/berkeley-scf/r-bootcamp-2016).


