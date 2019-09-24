---
interact_link: content/notebooks/10-intro-r/06-intro-r-conditionals-loops.ipynb
kernel_name: ir
has_widgets: false
title: 'Conditional-Loops'
prev_page:
  url: /notebooks/10-intro-r/05-intro-r-functions.html
  title: 'Functions'
next_page:
  url: /notebooks/10-intro-r/07-intro-r-merge-agg-fun.html
  title: 'Aggregation and Merge'
comment: "***PROGRAMMATICALLY GENERATED, DO NOT EDIT. SEE ORIGINAL FILES IN /content***"
---


[![AnalyticsDojo](https://github.com/rpi-techfundamentals/spring2019-materials/blob/master/fig/final-logo.png?raw=1)](http://rpi.analyticsdojo.com)
<center><h1>Introduction to R - Conditional Statements and Loops </h1></center>
<center><h3><a href = 'http://rpi.analyticsdojo.com'>rpi.analyticsdojo.com</a></h3></center>



## Overview
- What are conditional statements? Why do we need them?
- If statements in R
- Why, Why not Loops?
- Loops in R



# What are conditional statements? Why do we need them?



## `if` Statements
- Enables logical branching and recoding of data.
- BUT, `if statements` can result in long code branches, repeated code.
- Best to keep if statements short.



## Conditional Statements
- `if` statemenet enable logic. 
- `else` gives what to do if other conditions are not met.
- The else if function is achieved through nesting a if statement within an else function. 



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```R
#How long did the homework take? 
#Imagine this is the hours for each assignment. 
hours<- c(1,3,4,3)
#This is the experience of the individual, which can be high, medium, or low.
experience<-'low'
#experience<- 'high'

#toy = 'old'
if(experience=='high'){
exp.hours <- hours/2       
} else {
  if(experience=='low'){
     exp.hours <- hours * 2    
  } else {
    exp.hours <- hours      
  }
}
#Notice how this adjusted 
print(exp.hours)


```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">
{:.output_stream}
```
[1] 2 6 8 6
```
</div>
</div>
</div>



## R Logit and Conditions
- `<`     less than
- `<=`    less than or equal to
- `>`     greater than
- `>=`    greater than or equal to
- `==`    exactly equal to
- `!=`    not equal to
- `!x`    This corresponsds to *not x.* 
- `x & y`  This cooresponds to `and`. *(This does and element by element comparson.)*
- `x | y`  This cooresponds to `or`. *(This does and element by element comparson.)*



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```R
#simple
x<-FALSE
y<-FALSE
if (!x){
print("X is False")
}else{
print("X is True")
}




```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">
{:.output_stream}
```
[1] "X is False"
```
</div>
</div>
</div>



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```R
x<-TRUE
y<-TRUE
if((x==TRUE)|(y==TRUE)){
print("Either X or Y is True")
}

```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">
{:.output_stream}
```
[1] "Either X or Y is True"
```
</div>
</div>
</div>



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```R
if((x==TRUE)&(y==TRUE)){
print("X and Y are both True")
}

```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">
{:.output_stream}
```
[1] "X and Y are both True"
```
</div>
</div>
</div>



## Conditionals and `ifelse`
- `ifelse(*conditional*, True, False)` can be used to recode variables. 
- `ifelse` can be nested.
- Use the cut function as an alternative for more than 3 categroies. 
- This can be really useful when 



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```R
# create 2 age categories 
age<-c(18,15, 25,30)
agecat <- ifelse(age > 18, c("adult"), c("child"))
agecat

```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">

<div markdown="0" class="output output_html">
<ol class=list-inline>
	<li>'child'</li>
	<li>'child'</li>
	<li>'adult'</li>
	<li>'adult'</li>
</ol>

</div>

</div>
</div>
</div>



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```R
df=read.csv(file="../../input/iris.csv", header=TRUE,sep=",")

#Let's say we want to categorize sepalLenth as short/long or short/medium/long.
sl.med<-median(df$sepal_length)
sl.sd<-sd(df$sepal_length)
sl.max<-max(df$sepal_length)

df$agecat2 <- ifelse(df$sepal_length > sl.med, c("long"), c("short"))
df$agecat3 <- ifelse(df$sepal_length > (sl.med+sl.sd), c("long"), 
        ifelse(df$sepal_length < (sl.med-sl.sd), c("short"), c("medium")))


#This sets the different cuts for the categories. 
cuts<-c(0,sl.med-sl.sd,sl.med+sl.sd,sl.max)
cutlabels<-c("short", "medium", "long") 

df$agecat3altcut<-cut(df$sepal_length, breaks=cuts, labels=cutlabels)
df[,c(1,6,7,8)]

```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">

<div markdown="0" class="output output_html">
<table>
<thead><tr><th scope=col>sepal_length</th><th scope=col>agecat2</th><th scope=col>agecat3</th><th scope=col>agecat3altcut</th></tr></thead>
<tbody>
	<tr><td>5.1   </td><td>short </td><td>medium</td><td>medium</td></tr>
	<tr><td>4.9   </td><td>short </td><td>short </td><td>short </td></tr>
	<tr><td>4.7   </td><td>short </td><td>short </td><td>short </td></tr>
	<tr><td>4.6   </td><td>short </td><td>short </td><td>short </td></tr>
	<tr><td>5.0   </td><td>short </td><td>medium</td><td>medium</td></tr>
	<tr><td>5.4   </td><td>short </td><td>medium</td><td>medium</td></tr>
	<tr><td>4.6   </td><td>short </td><td>short </td><td>short </td></tr>
	<tr><td>5.0   </td><td>short </td><td>medium</td><td>medium</td></tr>
	<tr><td>4.4   </td><td>short </td><td>short </td><td>short </td></tr>
	<tr><td>4.9   </td><td>short </td><td>short </td><td>short </td></tr>
	<tr><td>5.4   </td><td>short </td><td>medium</td><td>medium</td></tr>
	<tr><td>4.8   </td><td>short </td><td>short </td><td>short </td></tr>
	<tr><td>4.8   </td><td>short </td><td>short </td><td>short </td></tr>
	<tr><td>4.3   </td><td>short </td><td>short </td><td>short </td></tr>
	<tr><td>5.8   </td><td>short </td><td>medium</td><td>medium</td></tr>
	<tr><td>5.7   </td><td>short </td><td>medium</td><td>medium</td></tr>
	<tr><td>5.4   </td><td>short </td><td>medium</td><td>medium</td></tr>
	<tr><td>5.1   </td><td>short </td><td>medium</td><td>medium</td></tr>
	<tr><td>5.7   </td><td>short </td><td>medium</td><td>medium</td></tr>
	<tr><td>5.1   </td><td>short </td><td>medium</td><td>medium</td></tr>
	<tr><td>5.4   </td><td>short </td><td>medium</td><td>medium</td></tr>
	<tr><td>5.1   </td><td>short </td><td>medium</td><td>medium</td></tr>
	<tr><td>4.6   </td><td>short </td><td>short </td><td>short </td></tr>
	<tr><td>5.1   </td><td>short </td><td>medium</td><td>medium</td></tr>
	<tr><td>4.8   </td><td>short </td><td>short </td><td>short </td></tr>
	<tr><td>5.0   </td><td>short </td><td>medium</td><td>medium</td></tr>
	<tr><td>5.0   </td><td>short </td><td>medium</td><td>medium</td></tr>
	<tr><td>5.2   </td><td>short </td><td>medium</td><td>medium</td></tr>
	<tr><td>5.2   </td><td>short </td><td>medium</td><td>medium</td></tr>
	<tr><td>4.7   </td><td>short </td><td>short </td><td>short </td></tr>
	<tr><td>⋮</td><td>⋮</td><td>⋮</td><td>⋮</td></tr>
	<tr><td>6.9   </td><td>long  </td><td>long  </td><td>long  </td></tr>
	<tr><td>5.6   </td><td>short </td><td>medium</td><td>medium</td></tr>
	<tr><td>7.7   </td><td>long  </td><td>long  </td><td>long  </td></tr>
	<tr><td>6.3   </td><td>long  </td><td>medium</td><td>medium</td></tr>
	<tr><td>6.7   </td><td>long  </td><td>long  </td><td>long  </td></tr>
	<tr><td>7.2   </td><td>long  </td><td>long  </td><td>long  </td></tr>
	<tr><td>6.2   </td><td>long  </td><td>medium</td><td>medium</td></tr>
	<tr><td>6.1   </td><td>long  </td><td>medium</td><td>medium</td></tr>
	<tr><td>6.4   </td><td>long  </td><td>medium</td><td>medium</td></tr>
	<tr><td>7.2   </td><td>long  </td><td>long  </td><td>long  </td></tr>
	<tr><td>7.4   </td><td>long  </td><td>long  </td><td>long  </td></tr>
	<tr><td>7.9   </td><td>long  </td><td>long  </td><td>long  </td></tr>
	<tr><td>6.4   </td><td>long  </td><td>medium</td><td>medium</td></tr>
	<tr><td>6.3   </td><td>long  </td><td>medium</td><td>medium</td></tr>
	<tr><td>6.1   </td><td>long  </td><td>medium</td><td>medium</td></tr>
	<tr><td>7.7   </td><td>long  </td><td>long  </td><td>long  </td></tr>
	<tr><td>6.3   </td><td>long  </td><td>medium</td><td>medium</td></tr>
	<tr><td>6.4   </td><td>long  </td><td>medium</td><td>medium</td></tr>
	<tr><td>6.0   </td><td>long  </td><td>medium</td><td>medium</td></tr>
	<tr><td>6.9   </td><td>long  </td><td>long  </td><td>long  </td></tr>
	<tr><td>6.7   </td><td>long  </td><td>long  </td><td>long  </td></tr>
	<tr><td>6.9   </td><td>long  </td><td>long  </td><td>long  </td></tr>
	<tr><td>5.8   </td><td>short </td><td>medium</td><td>medium</td></tr>
	<tr><td>6.8   </td><td>long  </td><td>long  </td><td>long  </td></tr>
	<tr><td>6.7   </td><td>long  </td><td>long  </td><td>long  </td></tr>
	<tr><td>6.7   </td><td>long  </td><td>long  </td><td>long  </td></tr>
	<tr><td>6.3   </td><td>long  </td><td>medium</td><td>medium</td></tr>
	<tr><td>6.5   </td><td>long  </td><td>medium</td><td>medium</td></tr>
	<tr><td>6.2   </td><td>long  </td><td>medium</td><td>medium</td></tr>
	<tr><td>5.9   </td><td>long  </td><td>medium</td><td>medium</td></tr>
</tbody>
</table>

</div>

</div>
</div>
</div>



# Why, Why Not Loops?



## Why, Why Not Loops?
- Iterate over arrays or lists easily. 
- BUT, in many cases for loops don't scale well and are slower than alternate methods involving functions. 
- BUT, don't worry about prematurely optimizing code.
- Often if you are doing a loop, there is a function that is faster.  You might not care for small data applications.
- Here is a basic example of `For`/`While` loop.



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```R
sum<-0
avgs <- numeric (8)
for (i in 1:8){
    print (i)
    sum<-sum+i  
}
print(sum)
for (i in 1:8) print (i)


```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">
{:.output_stream}
```
[1] 1
[1] 2
[1] 3
[1] 4
[1] 5
[1] 6
[1] 7
[1] 8
[1] 36
[1] 1
[1] 2
[1] 3
[1] 4
[1] 5
[1] 6
[1] 7
[1] 8
```
</div>
</div>
</div>



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```R
for (i in 1:8) print (i)

```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">
{:.output_stream}
```
[1] 1
[1] 2
[1] 3
[1] 4
[1] 5
[1] 6
[1] 7
[1] 8
```
</div>
</div>
</div>



## While Loop
- Performs a loop while a conditional is TRUE.
- Doesn't auto-increment.



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```R
#This produces the same.
i<-1
sum<-0
x<-TRUE
while (x) {
  print (i)
  sum<-sum+i
  i<-i+1
  if (i>8){x<-FALSE}
}  
print(sum)

```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">
{:.output_stream}
```
[1] 1
[1] 2
[1] 3
[1] 4
[1] 5
[1] 6
[1] 7
[1] 8
[1] 36
```
</div>
</div>
</div>



## For Loops can be Nested



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```R
#Nexting Example,
x=c(0,1,2)
y=c("a","b","c")
#Nested for loops
for (a in x){
    for (b in y){
        print(c(a,b), quote = FALSE)
}}

```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">
{:.output_stream}
```
[1] 0 a
[1] 0 b
[1] 0 c
[1] 1 a
[1] 1 b
[1] 1 c
[1] 2 a
[1] 2 b
[1] 2 c
```
</div>
</div>
</div>



Copyright [AnalyticsDojo](http://rpi.analyticsdojo.com) 2016.
This work is licensed under the [Creative Commons Attribution 4.0 International](https://creativecommons.org/licenses/by/4.0/) license agreement.


