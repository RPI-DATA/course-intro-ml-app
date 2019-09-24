---
interact_link: content/notebooks/10-intro-r/03-intro-r-datastructures.ipynb
kernel_name: ir
has_widgets: false
title: 'Data Structures'
prev_page:
  url: /notebooks/10-intro-r/02-intro-r-localfile.html
  title: 'Local Files'
next_page:
  url: /notebooks/10-intro-r/04-intro-r-dataframes.html
  title: 'Dataframes'
comment: "***PROGRAMMATICALLY GENERATED, DO NOT EDIT. SEE ORIGINAL FILES IN /content***"
---


[![AnalyticsDojo](https://github.com/rpi-techfundamentals/spring2019-materials/blob/master/fig/final-logo.png?raw=1)](http://rpi.analyticsdojo.com)
<center><h1>Introduction to R - Datastructures</h1></center>
<center><h3><a href = "http://rpi.analyticsdojo.com">rpi.analyticsdojo.com</a></h3></center>





# Overview
Common to R and Python
- Vectors
- Opearations on Numeric and String Variables
- Lists






## Vectors in R
- The most basic form of an R object is a vector. 
- In fact, individual (scalar) values (variables) are vectors of length one.
- An R vector is a single set of values in a particular order of the **same type**. 
- We can concatenate values into a vector with c(): `ages<-c(18,19,18,23)`
- Comparable Python objects include Panda Series and single dimensional numpy array. 
- While Python arrays start at 0, R arrays start at index position 1. 



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```R

ages<-c(18,19,18,23)
ages
ages[1]
ages[2:4]



```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">

<div markdown="0" class="output output_html">
<ol class=list-inline>
	<li>18</li>
	<li>19</li>
	<li>18</li>
	<li>23</li>
</ol>

</div>

</div>
</div>
<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">

<div markdown="0" class="output output_html">
18
</div>

</div>
</div>
<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">

<div markdown="0" class="output output_html">
<ol class=list-inline>
	<li>19</li>
	<li>18</li>
	<li>23</li>
</ol>

</div>

</div>
</div>
</div>



## Vectors Type in R
- Items in a vector must be of the same type. 
- *Character.* These are the clear character vectors. (Typically use quotes to add to these vectors.)
- *Numeric.* Numbers in a set. Note there is not a different type.
- *Boolean.* TRUE or FALSE values in a set.
- *Factor.* A situation in which there is a select set of options. Things such as states or zip codes. These are typically things which are related to dummy variables, a topic we will discuss later.
- Determine the data type by using the `str` command: `str(teachers)`



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```R
names<-c("Sally", "Jason", "Bob", "Susy") #Text
female<-c(TRUE, FALSE, FALSE, TRUE)  #While Python uses True and False, R uses TRUE and FALSE.
teachers<-c("Smith", "Johnson", "Johnson", "Smith")
teachers.f<-factor(teachers)
grades<-c(20, 15, 13, 19) #25 points possible
gradesdec<-c(20.32, 15.32, 13.12, 19.32) #25 points possible

str(names)
str(female)
str(teachers)  
str(teachers.f) 
str(grades)    #Note that the grades and gradesdec are both numeric.
str(gradesdec) #Note that the grades and gradesdec are both numeric.


```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">
{:.output_stream}
```
 chr [1:4] "Sally" "Jason" "Bob" "Susy"
 logi [1:4] TRUE FALSE FALSE TRUE
 chr [1:4] "Smith" "Johnson" "Johnson" "Smith"
 Factor w/ 2 levels "Johnson","Smith": 2 1 1 2
 num [1:4] 20 15 13 19
 num [1:4] 20.3 15.3 13.1 19.3
```
</div>
</div>
</div>



## Strings in R
- Lot's of different types of operations we can perform on Strings. 



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```R
chars <- c('hi', 'hallo', "mother's", 'father\'s', "He said, \'hi\'" )
length(chars)
nchar(chars)
paste("bill", "clinton", sep = " ")  # paste together a set of strings
paste(chars, collapse = ' ')  # paste together things from a vector

strlist<-strsplit("This is the Analytics Dojo", split = " ") #This taks a string ant splits to a list
strlist
substring(chars, 2, 3) #this takes the 2nd-3rd character from the sentance above 
chars2 <- chars
substring(chars2, 2, 3) <- "ZZ"  #this takes the 2nd-3rd character from the sentance above 
chars2

```
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

<div markdown="0" class="output output_html">
<ol class=list-inline>
	<li>2</li>
	<li>5</li>
	<li>8</li>
	<li>8</li>
	<li>13</li>
</ol>

</div>

</div>
</div>
<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">

<div markdown="0" class="output output_html">
'bill clinton'
</div>

</div>
</div>
<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">

<div markdown="0" class="output output_html">
'hi hallo mother\'s father\'s He said, \'hi\''
</div>

</div>
</div>
<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">

<div markdown="0" class="output output_html">
<ol>
	<li><ol class=list-inline>
	<li>'This'</li>
	<li>'is'</li>
	<li>'the'</li>
	<li>'Analytics'</li>
	<li>'Dojo'</li>
</ol>
</li>
</ol>

</div>

</div>
</div>
<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">

<div markdown="0" class="output output_html">
<ol class=list-inline>
	<li>'i'</li>
	<li>'al'</li>
	<li>'ot'</li>
	<li>'at'</li>
	<li>'e '</li>
</ol>

</div>

</div>
</div>
<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">

<div markdown="0" class="output output_html">
<ol class=list-inline>
	<li>'hZ'</li>
	<li>'hZZlo'</li>
	<li>'mZZher\'s'</li>
	<li>'fZZher\'s'</li>
	<li>'HZZsaid, \'hi\''</li>
</ol>

</div>

</div>
</div>
</div>



## Factors in R
- A factor is a special data type in R used for categorical data. In some cases it works like magic and in others it is incredibly frustrating.





<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```R
class(teachers.f) # What order are the factors in?
levels(teachers.f)  # note alternate way to get the variable
summary(teachers.f) #gives the count for each level. 


```
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
	<li>'Johnson'</li>
	<li>'Smith'</li>
</ol>

</div>

</div>
</div>
<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">

<div markdown="0" class="output output_html">
<dl class=dl-horizontal>
	<dt>Johnson</dt>
		<dd>2</dd>
	<dt>Smith</dt>
		<dd>2</dd>
</dl>

</div>

</div>
</div>
</div>



## Creating Vectors in R
- Concatenate fields to a vector: `nums <- c(1.1, 3, -5.7)`
- Generate random values from normal distribution with `devs <- rnorm(5)`
- idevs <- sample(ints, 100, replace = TRUE)




<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```R
# numeric vector
nums <- c(1.1, 3, -5.7)
devs <- rnorm(5)
devs

# integer vector
ints <- c(1L, 5L, -3L) # force storage as integer not decimal number
# "L" is for "long integer" (historical)

idevs <- sample(ints, 100, replace = TRUE)

# character vector
chars <- c("hi", "hallo", "mother's", "father\'s", 
   "She said", "hi", "He said, \'hi\'" )
chars
cat(chars, sep = "\n")

# logical vector
bools <- c(TRUE, FALSE, TRUE)
bools

```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">

<div markdown="0" class="output output_html">
<ol class=list-inline>
	<li>0.620478857748794</li>
	<li>0.355719819931768</li>
	<li>-0.482420730604138</li>
	<li>1.9607784989951</li>
	<li>-1.2218590305962</li>
</ol>

</div>

</div>
</div>
<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">

<div markdown="0" class="output output_html">
<ol class=list-inline>
	<li>'hi'</li>
	<li>'hallo'</li>
	<li>'mother\'s'</li>
	<li>'father\'s'</li>
	<li>'She said'</li>
	<li>'hi'</li>
	<li>'He said, \'hi\''</li>
</ol>

</div>

</div>
</div>
<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">
{:.output_stream}
```
hi
hallo
mother's
father's
She said
hi
He said, 'hi'
```
</div>
</div>
<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">

<div markdown="0" class="output output_html">
<ol class=list-inline>
	<li>TRUE</li>
	<li>FALSE</li>
	<li>TRUE</li>
</ol>

</div>

</div>
</div>
</div>



## Variable Type
- In R when we write `b = 30` this means the value of `30` is assigned to the `b` object. 
- R is a [dynamically typed](https://pythonconquerstheuniverse.wordpress.com/2009/10/03/static-vs-dynamic-typing-of-programming-languages/).
- Unlike some languages, we don"t have to declare the type of a variable before using it. 
- Variable type can also change with the reassignment of a variable. 
- We can query the class a value using the `class` function.
- The `str` function gives additional details for complex objects like dataframes.





<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```R
a <- 1L
print (c("The value of a is ", a))
print (c("The value of a is ", a), quote=FALSE)
class(a)
str(a)

a <- 2.5
print (c("Now the value of a is ", a),quote=FALSE)
class(a)
str(a)

a <- "hello there"
print (c("Now the value of a is ", a ),quote=FALSE)
class(a)
str(a)

```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">
{:.output_stream}
```
[1] "The value of a is " "1"                 
[1] The value of a is  1                 
```
</div>
</div>
<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">

<div markdown="0" class="output output_html">
'integer'
</div>

</div>
</div>
<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">
{:.output_stream}
```
 int 1
[1] Now the value of a is  2.5                   
```
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
{:.output_stream}
```
 num 2.5
[1] Now the value of a is  hello there           
```
</div>
</div>
<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">

<div markdown="0" class="output output_html">
'character'
</div>

</div>
</div>
<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">
{:.output_stream}
```
 chr "hello there"
```
</div>
</div>
</div>



## Converting Values Between Types

- We can convert values between different types.
- To convert to string use the `as.character` function.
- To convert to numeric use the `as.integer` function.
- To convert to an integer use the `as.integer` function.
- To convert to a boolean use the `as.logical` function.




<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```R
#This is a way of specifying a long integer.
a <- 1L
a
class(a)
str(a)
a<-as.character(a)
a
class(a)
str(a)
a<-as.numeric(a)
a
class(a)
str(a)
a<-as.logical(a)
a
class(a)
str(a)


```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">

<div markdown="0" class="output output_html">
1
</div>

</div>
</div>
<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">

<div markdown="0" class="output output_html">
'integer'
</div>

</div>
</div>
<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">
{:.output_stream}
```
 int 1
```
</div>
</div>
<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">

<div markdown="0" class="output output_html">
'1'
</div>

</div>
</div>
<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">

<div markdown="0" class="output output_html">
'character'
</div>

</div>
</div>
<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">
{:.output_stream}
```
 chr "1"
```
</div>
</div>
<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">

<div markdown="0" class="output output_html">
1
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
{:.output_stream}
```
 num 1
```
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
'logical'
</div>

</div>
</div>
<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">
{:.output_stream}
```
 logi TRUE
```
</div>
</div>
</div>



## Quotes
- Double Quotes are preferred in R, though both will work as long as they aren't mixed. 



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```R
#Double Quotes are preferred in R, though both will work as long as they aren't mixed. 
a <- "hello"
class(a)
str(a)
a <- 'hello'
class(a)
str(a)

```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">

<div markdown="0" class="output output_html">
'character'
</div>

</div>
</div>
<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">
{:.output_stream}
```
 chr "hello"
```
</div>
</div>
<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">

<div markdown="0" class="output output_html">
'character'
</div>

</div>
</div>
<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">
{:.output_stream}
```
 chr "hello"
```
</div>
</div>
</div>



# Null Values

- Since it was designed by statisticians, R handles missing values very well relative to other languages.
- `NA` is a missing value




<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```R
#Notice nothing is printed.
a<-NA
a
vec <- rnorm(12)    #This creates a vector with randomly distributed values
vec[c(3, 5)] <- NA  #This sets values 3 and 5 as NA
vec                 #This prints the Vector
sum(vec)            #What is the Sum of a vector that has NA?  
sum(vec, na.rm = TRUE)   #This Sums the vector with the NA removed. 
is.na(vec)          #This returns a vector of whether specific values are equal to NA.

```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">

<div markdown="0" class="output output_html">
&lt;NA&gt;
</div>

</div>
</div>
<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">

<div markdown="0" class="output output_html">
<ol class=list-inline>
	<li>-1.33644546389391</li>
	<li>1.87421154996928</li>
	<li>&lt;NA&gt;</li>
	<li>-0.217346245734894</li>
	<li>&lt;NA&gt;</li>
	<li>0.435770349019708</li>
	<li>-1.14025525433378</li>
	<li>-0.48345946330215</li>
	<li>-0.900282592359427</li>
	<li>-0.61861874592141</li>
	<li>1.04707474251708</li>
	<li>-1.50789510144605</li>
</ol>

</div>

</div>
</div>
<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">

<div markdown="0" class="output output_html">
&lt;NA&gt;
</div>

</div>
</div>
<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">

<div markdown="0" class="output output_html">
-2.84724622548555
</div>

</div>
</div>
<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">

<div markdown="0" class="output output_html">
<ol class=list-inline>
	<li>FALSE</li>
	<li>FALSE</li>
	<li>TRUE</li>
	<li>FALSE</li>
	<li>TRUE</li>
	<li>FALSE</li>
	<li>FALSE</li>
	<li>FALSE</li>
	<li>FALSE</li>
	<li>FALSE</li>
	<li>FALSE</li>
	<li>FALSE</li>
</ol>

</div>

</div>
</div>
</div>



## Logical/Boolean Vectors
- Here we can see that summing and averaging boolean vectors treats `TRUE=1 & FALSE=0`



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```R
answers <- c(TRUE, TRUE, FALSE, FALSE)
update <- c(TRUE, FALSE, TRUE, FALSE)

# Here we see that True coul
sum(answers)
mean(answers)
total<-answers + update
total
class(total)

```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">

<div markdown="0" class="output output_html">
2
</div>

</div>
</div>
<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">

<div markdown="0" class="output output_html">
0.5
</div>

</div>
</div>
<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">

<div markdown="0" class="output output_html">
<ol class=list-inline>
	<li>2</li>
	<li>1</li>
	<li>1</li>
	<li>0</li>
</ol>

</div>

</div>
</div>
<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">

<div markdown="0" class="output output_html">
'integer'
</div>

</div>
</div>
</div>



## R Calculations
- R can act as a basic calculator.



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```R
2 + 2 # add numbers
2 * pi # multiply by a constant
7 + runif(1) # add a random number
3^4 # powers
sqrt(4^4) # functions
log(10)
log(100, base = 10)
23 %/% 2 
23 %% 2

# scientific notation
5000000000 * 1000
5e9 * 1e3

```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">

<div markdown="0" class="output output_html">
4
</div>

</div>
</div>
<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">

<div markdown="0" class="output output_html">
6.28318530717959
</div>

</div>
</div>
<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">

<div markdown="0" class="output output_html">
7.38707033358514
</div>

</div>
</div>
<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">

<div markdown="0" class="output output_html">
81
</div>

</div>
</div>
<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">

<div markdown="0" class="output output_html">
16
</div>

</div>
</div>
<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">

<div markdown="0" class="output output_html">
2.30258509299405
</div>

</div>
</div>
<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">

<div markdown="0" class="output output_html">
2
</div>

</div>
</div>
<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">

<div markdown="0" class="output output_html">
11
</div>

</div>
</div>
<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">

<div markdown="0" class="output output_html">
1
</div>

</div>
</div>
<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">

<div markdown="0" class="output output_html">
5e+12
</div>

</div>
</div>
<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">

<div markdown="0" class="output output_html">
5e+12
</div>

</div>
</div>
</div>



## Operations on Vectors
- R can be used as a basic calculator.
- We can do calculations on vectors easily. 
- Direct operations are much faster easier than looping.




<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```R
#vals <- rnorm(10)
#squared2vals <- vals^2
#sum_squared2vals <- sum(chi2vals)
#ount_squared2vals<-length(squared2vals)  
#vals
#squared2vals
#sum_df1000
#count_squared2vals


```
</div>

</div>



## R is a Functional Language

- Operations are carried out with functions. Functions take objects as inputs and return objects as outputs. 
- An analysis can be considered a pipeline of function calls, with output from a function used later in a subsequent operation as input to another function.
- Functions themselves are objects. 
- We can get help on functions with help(lm) or ?lm



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```R
vals <- rnorm(10)
median(vals)
class(median)
median(vals, na.rm = TRUE)
mean(vals, na.rm = TRUE)
help(lm)
?lm
?log



```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">

<div markdown="0" class="output output_html">
0.644546626183949
</div>

</div>
</div>
<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">

<div markdown="0" class="output output_html">
'function'
</div>

</div>
</div>
<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">

<div markdown="0" class="output output_html">
0.644546626183949
</div>

</div>
</div>
<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">

<div markdown="0" class="output output_html">
0.631539617203971
</div>

</div>
</div>
<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">

<div markdown="0" class="output output_html">

<table width="100%" summary="page for lm {stats}"><tr><td>lm {stats}</td><td style="text-align: right;">R Documentation</td></tr></table>

<h2>Fitting Linear Models</h2>

<h3>Description</h3>

<p><code>lm</code> is used to fit linear models.
It can be used to carry out regression,
single stratum analysis of variance and
analysis of covariance (although <code>aov</code> may provide a more
convenient interface for these).
</p>


<h3>Usage</h3>

<pre>
lm(formula, data, subset, weights, na.action,
   method = "qr", model = TRUE, x = FALSE, y = FALSE, qr = TRUE,
   singular.ok = TRUE, contrasts = NULL, offset, ...)
</pre>


<h3>Arguments</h3>

<table summary="R argblock">
<tr valign="top"><td><code>formula</code></td>
<td>
<p>an object of class <code>"formula"</code> (or one that
can be coerced to that class): a symbolic description of the
model to be fitted.  The details of model specification are given
under &lsquo;Details&rsquo;.</p>
</td></tr>
<tr valign="top"><td><code>data</code></td>
<td>
<p>an optional data frame, list or environment (or object
coercible by <code>as.data.frame</code> to a data frame) containing
the variables in the model.  If not found in <code>data</code>, the
variables are taken from <code>environment(formula)</code>,
typically the environment from which <code>lm</code> is called.</p>
</td></tr>
<tr valign="top"><td><code>subset</code></td>
<td>
<p>an optional vector specifying a subset of observations
to be used in the fitting process.</p>
</td></tr>
<tr valign="top"><td><code>weights</code></td>
<td>
<p>an optional vector of weights to be used in the fitting
process.  Should be <code>NULL</code> or a numeric vector.
If non-NULL, weighted least squares is used with weights
<code>weights</code> (that is, minimizing <code>sum(w*e^2)</code>); otherwise
ordinary least squares is used.  See also &lsquo;Details&rsquo;,</p>
</td></tr>
<tr valign="top"><td><code>na.action</code></td>
<td>
<p>a function which indicates what should happen
when the data contain <code>NA</code>s.  The default is set by
the <code>na.action</code> setting of <code>options</code>, and is
<code>na.fail</code> if that is unset.  The &lsquo;factory-fresh&rsquo;
default is <code>na.omit</code>.  Another possible value is
<code>NULL</code>, no action.  Value <code>na.exclude</code> can be useful.</p>
</td></tr>
<tr valign="top"><td><code>method</code></td>
<td>
<p>the method to be used; for fitting, currently only
<code>method = "qr"</code> is supported; <code>method = "model.frame"</code> returns
the model frame (the same as with <code>model = TRUE</code>, see below).</p>
</td></tr>
<tr valign="top"><td><code>model, x, y, qr</code></td>
<td>
<p>logicals.  If <code>TRUE</code> the corresponding
components of the fit (the model frame, the model matrix, the
response, the QR decomposition) are returned.
</p>
</td></tr>
<tr valign="top"><td><code>singular.ok</code></td>
<td>
<p>logical. If <code>FALSE</code> (the default in S but
not in <span style="font-family: Courier New, Courier; color: #666666;"><b>R</b></span>) a singular fit is an error.</p>
</td></tr>
<tr valign="top"><td><code>contrasts</code></td>
<td>
<p>an optional list. See the <code>contrasts.arg</code>
of <code>model.matrix.default</code>.</p>
</td></tr>
<tr valign="top"><td><code>offset</code></td>
<td>
<p>this can be used to specify an <em>a priori</em> known
component to be included in the linear predictor during fitting.
This should be <code>NULL</code> or a numeric vector of length equal to
the number of cases.  One or more <code>offset</code> terms can be
included in the formula instead or as well, and if more than one are
specified their sum is used.  See <code>model.offset</code>.</p>
</td></tr>
<tr valign="top"><td><code>...</code></td>
<td>
<p>additional arguments to be passed to the low level
regression fitting functions (see below).</p>
</td></tr>
</table>


<h3>Details</h3>

<p>Models for <code>lm</code> are specified symbolically.  A typical model has
the form <code>response ~ terms</code> where <code>response</code> is the (numeric)
response vector and <code>terms</code> is a series of terms which specifies a
linear predictor for <code>response</code>.  A terms specification of the form
<code>first + second</code> indicates all the terms in <code>first</code> together
with all the terms in <code>second</code> with duplicates removed.  A
specification of the form <code>first:second</code> indicates the set of
terms obtained by taking the interactions of all terms in <code>first</code>
with all terms in <code>second</code>.  The specification <code>first*second</code>
indicates the <em>cross</em> of <code>first</code> and <code>second</code>.  This is
the same as <code>first + second + first:second</code>.
</p>
<p>If the formula includes an <code>offset</code>, this is evaluated and
subtracted from the response.
</p>
<p>If <code>response</code> is a matrix a linear model is fitted separately by
least-squares to each column of the matrix.
</p>
<p>See <code>model.matrix</code> for some further details.  The terms in
the formula will be re-ordered so that main effects come first,
followed by the interactions, all second-order, all third-order and so
on: to avoid this pass a <code>terms</code> object as the formula (see
<code>aov</code> and <code>demo(glm.vr)</code> for an example).
</p>
<p>A formula has an implied intercept term.  To remove this use either
<code>y ~ x - 1</code> or <code>y ~ 0 + x</code>.  See <code>formula</code> for
more details of allowed formulae.
</p>
<p>Non-<code>NULL</code> <code>weights</code> can be used to indicate that
different observations have different variances (with the values in
<code>weights</code> being inversely proportional to the variances); or
equivalently, when the elements of <code>weights</code> are positive
integers <i>w_i</i>, that each response <i>y_i</i> is the mean of
<i>w_i</i> unit-weight observations (including the case that there
are <i>w_i</i> observations equal to <i>y_i</i> and the data have been
summarized). However, in the latter case, notice that within-group
variation is not used.  Therefore, the sigma estimate and residual
degrees of freedom may be suboptimal; in the case of replication
weights, even wrong. Hence, standard errors and analysis of variance
tables should be treated with care.
</p>
<p><code>lm</code> calls the lower level functions <code>lm.fit</code>, etc,
see below, for the actual numerical computations.  For programming
only, you may consider doing likewise.
</p>
<p>All of <code>weights</code>, <code>subset</code> and <code>offset</code> are evaluated
in the same way as variables in <code>formula</code>, that is first in
<code>data</code> and then in the environment of <code>formula</code>.
</p>


<h3>Value</h3>

<p><code>lm</code> returns an object of <code>class</code> <code>"lm"</code> or for
multiple responses of class <code>c("mlm", "lm")</code>.
</p>
<p>The functions <code>summary</code> and <code>anova</code> are used to
obtain and print a summary and analysis of variance table of the
results.  The generic accessor functions <code>coefficients</code>,
<code>effects</code>, <code>fitted.values</code> and <code>residuals</code> extract
various useful features of the value returned by <code>lm</code>.
</p>
<p>An object of class <code>"lm"</code> is a list containing at least the
following components:
</p>
<table summary="R valueblock">
<tr valign="top"><td><code>coefficients</code></td>
<td>
<p>a named vector of coefficients</p>
</td></tr>
<tr valign="top"><td><code>residuals</code></td>
<td>
<p>the residuals, that is response minus fitted values.</p>
</td></tr>
<tr valign="top"><td><code>fitted.values</code></td>
<td>
<p>the fitted mean values.</p>
</td></tr>
<tr valign="top"><td><code>rank</code></td>
<td>
<p>the numeric rank of the fitted linear model.</p>
</td></tr>
<tr valign="top"><td><code>weights</code></td>
<td>
<p>(only for weighted fits) the specified weights.</p>
</td></tr>
<tr valign="top"><td><code>df.residual</code></td>
<td>
<p>the residual degrees of freedom.</p>
</td></tr>
<tr valign="top"><td><code>call</code></td>
<td>
<p>the matched call.</p>
</td></tr>
<tr valign="top"><td><code>terms</code></td>
<td>
<p>the <code>terms</code> object used.</p>
</td></tr>
<tr valign="top"><td><code>contrasts</code></td>
<td>
<p>(only where relevant) the contrasts used.</p>
</td></tr>
<tr valign="top"><td><code>xlevels</code></td>
<td>
<p>(only where relevant) a record of the levels of the
factors used in fitting.</p>
</td></tr>
<tr valign="top"><td><code>offset</code></td>
<td>
<p>the offset used (missing if none were used).</p>
</td></tr>
<tr valign="top"><td><code>y</code></td>
<td>
<p>if requested, the response used.</p>
</td></tr>
<tr valign="top"><td><code>x</code></td>
<td>
<p>if requested, the model matrix used.</p>
</td></tr>
<tr valign="top"><td><code>model</code></td>
<td>
<p>if requested (the default), the model frame used.</p>
</td></tr>
<tr valign="top"><td><code>na.action</code></td>
<td>
<p>(where relevant) information returned by
<code>model.frame</code> on the special handling of <code>NA</code>s.</p>
</td></tr>
</table>
<p>In addition, non-null fits will have components <code>assign</code>,
<code>effects</code> and (unless not requested) <code>qr</code> relating to the linear
fit, for use by extractor functions such as <code>summary</code> and
<code>effects</code>.
</p>


<h3>Using time series</h3>

<p>Considerable care is needed when using <code>lm</code> with time series.
</p>
<p>Unless <code>na.action = NULL</code>, the time series attributes are
stripped from the variables before the regression is done.  (This is
necessary as omitting <code>NA</code>s would invalidate the time series
attributes, and if <code>NA</code>s are omitted in the middle of the series
the result would no longer be a regular time series.)
</p>
<p>Even if the time series attributes are retained, they are not used to
line up series, so that the time shift of a lagged or differenced
regressor would be ignored.  It is good practice to prepare a
<code>data</code> argument by <code>ts.intersect(..., dframe = TRUE)</code>,
then apply a suitable <code>na.action</code> to that data frame and call
<code>lm</code> with <code>na.action = NULL</code> so that residuals and fitted
values are time series.
</p>


<h3>Note</h3>

<p>Offsets specified by <code>offset</code> will not be included in predictions
by <code>predict.lm</code>, whereas those specified by an offset term
in the formula will be.
</p>


<h3>Author(s)</h3>

<p>The design was inspired by the S function of the same name described
in Chambers (1992).  The implementation of model formula by Ross Ihaka
was based on Wilkinson &amp; Rogers (1973).
</p>


<h3>References</h3>

<p>Chambers, J. M. (1992)
<em>Linear models.</em>
Chapter 4 of <em>Statistical Models in S</em>
eds J. M. Chambers and T. J. Hastie, Wadsworth &amp; Brooks/Cole.
</p>
<p>Wilkinson, G. N. and Rogers, C. E. (1973).
Symbolic descriptions of factorial models for analysis of variance.
<em>Applied Statistics</em>, <b>22</b>, 392&ndash;399.
doi: <a href="http://doi.org/10.2307/2346786">10.2307/2346786</a>.
</p>


<h3>See Also</h3>

<p><code>summary.lm</code> for summaries and <code>anova.lm</code> for
the ANOVA table; <code>aov</code> for a different interface.
</p>
<p>The generic functions <code>coef</code>, <code>effects</code>,
<code>residuals</code>, <code>fitted</code>, <code>vcov</code>.
</p>
<p><code>predict.lm</code> (via <code>predict</code>) for prediction,
including confidence and prediction intervals;
<code>confint</code> for confidence intervals of <em>parameters</em>.
</p>
<p><code>lm.influence</code> for regression diagnostics, and
<code>glm</code> for <b>generalized</b> linear models.
</p>
<p>The underlying low level functions,
<code>lm.fit</code> for plain, and <code>lm.wfit</code> for weighted
regression fitting.
</p>
<p>More <code>lm()</code> examples are available e.g., in
<code>anscombe</code>, <code>attitude</code>, <code>freeny</code>,
<code>LifeCycleSavings</code>, <code>longley</code>,
<code>stackloss</code>, <code>swiss</code>.
</p>
<p><code>biglm</code> in package <a href="https://CRAN.R-project.org/package=biglm"><span class="pkg">biglm</span></a> for an alternative
way to fit linear models to large datasets (especially those with many
cases).
</p>


<h3>Examples</h3>

<pre>
require(graphics)

## Annette Dobson (1990) "An Introduction to Generalized Linear Models".
## Page 9: Plant Weight Data.
ctl &lt;- c(4.17,5.58,5.18,6.11,4.50,4.61,5.17,4.53,5.33,5.14)
trt &lt;- c(4.81,4.17,4.41,3.59,5.87,3.83,6.03,4.89,4.32,4.69)
group &lt;- gl(2, 10, 20, labels = c("Ctl","Trt"))
weight &lt;- c(ctl, trt)
lm.D9 &lt;- lm(weight ~ group)
lm.D90 &lt;- lm(weight ~ group - 1) # omitting intercept

anova(lm.D9)
summary(lm.D90)

opar &lt;- par(mfrow = c(2,2), oma = c(0, 0, 1.1, 0))
plot(lm.D9, las = 1)      # Residuals, Fitted, ...
par(opar)

### less simple examples in "See Also" above
</pre>

<hr /><div style="text-align: center;">[Package <em>stats</em> version 3.5.1 ]</div>
</div>

</div>
</div>
</div>



## Matrix
- Multiple column vector
- Matrix is useful for linear algebra
- Matrix must be **all of the same type**
- Could relatively easily do regression calculations using undlerlying matrix



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```R
#This is setup a matrix(vector, nrow,ncol)
mat <- matrix(rnorm(12), nrow = 3, ncol = 4)
mat
# This is setup a matrix(vector, rows)
A <- matrix(1:12, 3)
B <- matrix(1:12, 4)
C <- matrix(seq(4,36, by = 4), 3)
A
B
C

```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">

<div markdown="0" class="output output_html">
<table>
<tbody>
	<tr><td>-0.83526885</td><td> 0.06464287</td><td> 0.39680157</td><td> 0.89122461</td></tr>
	<tr><td> 2.0589154</td><td>-0.4442069</td><td> 0.4423272</td><td> 0.1642597</td></tr>
	<tr><td> 1.330993</td><td>-2.328589</td><td>-1.856013</td><td> 2.048782</td></tr>
</tbody>
</table>

</div>

</div>
</div>
<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">

<div markdown="0" class="output output_html">
<table>
<tbody>
	<tr><td> 1</td><td> 4</td><td> 7</td><td>10</td></tr>
	<tr><td> 2</td><td> 5</td><td> 8</td><td>11</td></tr>
	<tr><td> 3</td><td> 6</td><td> 9</td><td>12</td></tr>
</tbody>
</table>

</div>

</div>
</div>
<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">

<div markdown="0" class="output output_html">
<table>
<tbody>
	<tr><td>1</td><td>5</td><td>9</td></tr>
	<tr><td> 2</td><td> 6</td><td>10</td></tr>
	<tr><td> 3</td><td> 7</td><td>11</td></tr>
	<tr><td> 4</td><td> 8</td><td>12</td></tr>
</tbody>
</table>

</div>

</div>
</div>
<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">

<div markdown="0" class="output output_html">
<table>
<tbody>
	<tr><td> 4</td><td>16</td><td>28</td></tr>
	<tr><td> 8</td><td>20</td><td>32</td></tr>
	<tr><td>12</td><td>24</td><td>36</td></tr>
</tbody>
</table>

</div>

</div>
</div>
</div>



## Slicing Vectors and Matrixs
- Can use matrix[rows,columns] with specificating of row/column index, name, range. 




<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```R
vec <- rnorm(12)
mat <- matrix(vec, 4, 3)
rownames(mat) <- letters[1:4] #This assigns a row name
vec
mat



```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">

<div markdown="0" class="output output_html">
<ol class=list-inline>
	<li>0.235847096846066</li>
	<li>-1.22658917279729</li>
	<li>-1.18402683885278</li>
	<li>1.50615064907639</li>
	<li>-0.206182106051588</li>
	<li>-0.412355878955171</li>
	<li>-0.0799934537639763</li>
	<li>-1.72561819147371</li>
	<li>0.76499317236754</li>
	<li>1.25314418417645</li>
	<li>-1.12655889886209</li>
	<li>-2.45245615420083</li>
</ol>

</div>

</div>
</div>
<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">

<div markdown="0" class="output output_html">
<table>
<tbody>
	<tr><th scope=row>a</th><td> 0.2358471 </td><td>-0.20618211</td><td> 0.7649932 </td></tr>
	<tr><th scope=row>b</th><td>-1.2265892 </td><td>-0.41235588</td><td> 1.2531442 </td></tr>
	<tr><th scope=row>c</th><td>-1.1840268 </td><td>-0.07999345</td><td>-1.1265589 </td></tr>
	<tr><th scope=row>d</th><td> 1.5061506 </td><td>-1.72561819</td><td>-2.4524562 </td></tr>
</tbody>
</table>

</div>

</div>
</div>
</div>



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```R
#Slicing Vector
vec[c(3, 5, 8:10)] # This gives position 3, 5, and 8-10

```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">

<div markdown="0" class="output output_html">
<ol class=list-inline>
	<li>-1.18402683885278</li>
	<li>-0.206182106051588</li>
	<li>-1.72561819147371</li>
	<li>0.76499317236754</li>
	<li>1.25314418417645</li>
</ol>

</div>

</div>
</div>
</div>



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```R
# matrix[rows,columns]  leaving blank means all columns/rows
mat[c('a', 'd'), ]
mat[c(1,4), ]
mat[c(1,4), 1:2]
mat[c(1,4), c(1,3)] #Notice when providing a list we surround with c
mat[, 1:2]          #When providing a range we use a colon.

```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">

<div markdown="0" class="output output_html">
<table>
<tbody>
	<tr><th scope=row>a</th><td>0.2358471 </td><td>-0.2061821</td><td> 0.7649932</td></tr>
	<tr><th scope=row>d</th><td>1.5061506 </td><td>-1.7256182</td><td>-2.4524562</td></tr>
</tbody>
</table>

</div>

</div>
</div>
<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">

<div markdown="0" class="output output_html">
<table>
<tbody>
	<tr><th scope=row>a</th><td>0.2358471 </td><td>-0.2061821</td><td> 0.7649932</td></tr>
	<tr><th scope=row>d</th><td>1.5061506 </td><td>-1.7256182</td><td>-2.4524562</td></tr>
</tbody>
</table>

</div>

</div>
</div>
<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">

<div markdown="0" class="output output_html">
<table>
<tbody>
	<tr><th scope=row>a</th><td>0.2358471 </td><td>-0.2061821</td></tr>
	<tr><th scope=row>d</th><td>1.5061506 </td><td>-1.7256182</td></tr>
</tbody>
</table>

</div>

</div>
</div>
<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">

<div markdown="0" class="output output_html">
<table>
<tbody>
	<tr><th scope=row>a</th><td>0.2358471 </td><td> 0.7649932</td></tr>
	<tr><th scope=row>d</th><td>1.5061506 </td><td>-2.4524562</td></tr>
</tbody>
</table>

</div>

</div>
</div>
<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">

<div markdown="0" class="output output_html">
<table>
<tbody>
	<tr><th scope=row>a</th><td> 0.2358471 </td><td>-0.20618211</td></tr>
	<tr><th scope=row>b</th><td>-1.2265892 </td><td>-0.41235588</td></tr>
	<tr><th scope=row>c</th><td>-1.1840268 </td><td>-0.07999345</td></tr>
	<tr><th scope=row>d</th><td> 1.5061506 </td><td>-1.72561819</td></tr>
</tbody>
</table>

</div>

</div>
</div>
</div>



## Lists
- Collections of disparate or complicated objects.
- Can be of multiple different types. 
- Here we assign individual values with the list with `=`.
- Slice the list with the index position or the name. 



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```R
#myList <- list(stuff = 3, mat = matrix(1:4, nrow = 2), moreStuff = c('china', 'japan'), list(5, 'bear'))
myList<-list(stuff=3,mat = matrix(1:4, nrow = 2),vector=c(1,2,3,4),morestuff=c("Albany","New York", "San Francisco"))
myList

#
myList['stuff']
myList[2]
myList[2:3]
myList[c(1,4)]


```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">

<div markdown="0" class="output output_html">
<dl>
	<dt>$stuff</dt>
		<dd>3</dd>
	<dt>$mat</dt>
		<dd><table>
<tbody>
	<tr><td>1</td><td>3</td></tr>
	<tr><td>2</td><td>4</td></tr>
</tbody>
</table>
</dd>
	<dt>$vector</dt>
		<dd><ol class=list-inline>
	<li>1</li>
	<li>2</li>
	<li>3</li>
	<li>4</li>
</ol>
</dd>
	<dt>$morestuff</dt>
		<dd><ol class=list-inline>
	<li>'Albany'</li>
	<li>'New York'</li>
	<li>'San Francisco'</li>
</ol>
</dd>
</dl>

</div>

</div>
</div>
<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">

<div markdown="0" class="output output_html">
<strong>$stuff</strong> = 3
</div>

</div>
</div>
<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">

<div markdown="0" class="output output_html">
<strong>$mat</strong> = <table>
<tbody>
	<tr><td>1</td><td>3</td></tr>
	<tr><td>2</td><td>4</td></tr>
</tbody>
</table>

</div>

</div>
</div>
<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">

<div markdown="0" class="output output_html">
<dl>
	<dt>$mat</dt>
		<dd><table>
<tbody>
	<tr><td>1</td><td>3</td></tr>
	<tr><td>2</td><td>4</td></tr>
</tbody>
</table>
</dd>
	<dt>$vector</dt>
		<dd><ol class=list-inline>
	<li>1</li>
	<li>2</li>
	<li>3</li>
	<li>4</li>
</ol>
</dd>
</dl>

</div>

</div>
</div>
<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">

<div markdown="0" class="output output_html">
<dl>
	<dt>$stuff</dt>
		<dd>3</dd>
	<dt>$morestuff</dt>
		<dd><ol class=list-inline>
	<li>'Albany'</li>
	<li>'New York'</li>
	<li>'San Francisco'</li>
</ol>
</dd>
</dl>

</div>

</div>
</div>
</div>



## CREDITS

Copyright [AnalyticsDojo](http://rpi.analyticsdojo.com) 2016
This work is licensed under the [Creative Commons Attribution 4.0 International](https://creativecommons.org/licenses/by/4.0/) license agreement.

Adopted from the [Berkley R Bootcamp](https://github.com/berkeley-scf/r-bootcamp-2016).



