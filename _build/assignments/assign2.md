---
interact_link: content/assignments/assign2.ipynb
kernel_name: python3
has_widgets: false
title: 'Assignment 2'
prev_page:
  url: /assignments/assign1.html
  title: 'Assignment 1'
next_page:
  url: /grading.html
  title: 'Grading'
comment: "***PROGRAMMATICALLY GENERATED, DO NOT EDIT. SEE ORIGINAL FILES IN /content***"
---


## Introduction to Python Exercises

Before you turn this problem in, make sure everything runs as expected. First, restart the kernel (in the menubar, select Kernel > Restart) and then run all cells (in the menubar, select Cell > Run All).  You can speak with others regarding the assignment but all work must be your own. 


### This is a 30 point assignment graded from answers to questions and automated tests that should be run at the bottom. Be sure to clearly label all of your answers and commit final tests at the end.


**You may find it useful to go through the notebooks from the course materials when doing these exercises.**



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
name = "Jason Kuruzovich"
collaborators = "Alyssa Hacker"  #You can speak with others regarding the assignment, but all typed work must be your own.

```
</div>

</div>



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
tests = "https://github.com/rpi-techfundamentals/spring2019-materials/raw/master/03-python/hm-02/tests.zip"
ok="https://raw.githubusercontent.com/rpi-techfundamentals/spring2019-materials/master/03-python/hm-02/hm02.ok"
!pip install git+https://github.com/data-8/Gofer-Grader && wget  $ok && wget  $tests && unzip -o tests.zip

```
</div>

</div>



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
#This loads the grading software.  If you later get a "failed to import OK.py" you need to rerun.
from client.api.notebook import Notebook
ok = Notebook('hm02.ok')
_ = ok.auth(inline=True)

```
</div>

</div>



**If you attempt to fake passing the tests you will receive a 0 on the assignment and it will be considered an ethical violation.**



## Exercise-Packages

This creates an Numpy array. Numpy is a common package that we will use to work with arrays. You can read more about Numpy [here](http://www.numpy.org/). 

```
a = np.array([2,3,4])
print(a)
```

To get this to work, you will have to make sure that the numpy(np) package is installed. 




<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
q1_question= """
(1) Verify that Numpy is installed. How did you know?  
Describe how you would you install it if it wasn't installed?
"""
#You must assign your answer to q1. 
q1_answer="""
Enter your answer here.
"""

```
</div>

</div>



(2) Fix the cell below so that `a` is a `numpy` array.



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
#Fix this code of q2. 
a = [5,6,7,8]
print(a, type(a))


```
</div>

</div>



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
_ = ok.grade('q02')

```
</div>

</div>



(3) Create a numpy array `b` with the values `12, 13, 14, 15`.





<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
#<insert q3 code here>



```
</div>

</div>



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
_ = ok.grade('q03')

```
</div>

</div>



## Exercise - Operations on Variables






<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
q4_question= """
(4) Describe what happens when you multiply an integer times a boolean? 
What is the resulting type? Provide examples.
"""
#You must assign your answer to q4_answer. 
q4_answer="""
Enter your answer here.
"""

```
</div>

</div>



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
q5_question= """
(5) Describe happens when you try to multiply an integer value times a null?
"""
#You must assign your answer to q5_answer. 
q5_answer="""
Enter your answer here.
"""


```
</div>

</div>



(6) Take 5 to the power of 4 and assign it to a variable `c`. Then transform the variable `c` to a type `float`. 



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
#<insert q6 code here>

```
</div>

</div>



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
_ = ok.grade('q06')

```
</div>

</div>



## Exercise-Lists
Hint: [This link is useful.](https://docs.python.org/3/tutorial/datastructures.html#more-on-lists) as is the process of tab completion (using tab to find available methods of an object).

(7) Create a list `elist1` with the following values `1,2,3,4,5`.<br>




<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
#<insert q7 code here>


```
</div>

</div>



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
_ = ok.grade('q07')

```
</div>

</div>



(8) Create a new list `elist2` by first creating a copy of `elist1` and then reversing the order.

*HINT, remember there is a specific function to copy a list.* 




<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
#<insert q8 code here>


```
</div>

</div>



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
_ = ok.grade('q08')

```
</div>

</div>



(9) Create a new list `elist3` by first creating a copy of `elist1` and then adding `7, 8, 9` to the end. (Hint: Search for a different function if appending doesn't work.) 



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
#<insert q9 code here>


```
</div>

</div>



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
_ = ok.grade('q09')

```
</div>

</div>



(10) Create a new list `elist4` by first creating a copy of `elist3` and then insert `6` between `5` and `7`.



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
#<insert q10 code here>


```
</div>

</div>



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
_ = ok.grade('q10')

```
</div>

</div>



## Exercise-Sets/Dictionary

This [link to documentation on sets](https://docs.python.org/3/tutorial/datastructures.html#sets) may be useful.

(11) Create a set `eset1` with the following values (1,2,3,4,5).




<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
#<insert q11 code here>


```
</div>

</div>



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
_ = ok.grade('q11')

```
</div>

</div>




(12) Create a new set `eset2` the following values (1,3,6).





<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
#<insert q12 code here>


```
</div>

</div>



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
_ = ok.grade('q12')

```
</div>

</div>



(13) Create a new set `eset3` that is `eset1-eset2`.





<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
#<insert q13 code here>

```
</div>

</div>



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
_ = ok.grade('q13')

```
</div>

</div>



(14) Create a new set `eset4` that is the union of `eset1+eset2`.




<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
#<insert q14 code here>

```
</div>

</div>



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
_ = ok.grade('q14')

```
</div>

</div>




(15) Create a new set `eset5` that includes values that are in both `eset1` and `eset2` (intersection).





<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
#<insert q15 code here>

```
</div>

</div>



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
_ = ok.grade('q15')

```
</div>

</div>



(16) Create a new dict `edict1` with the following keys and associated values: st1=45; st2=32; st3=40; st4=31.

*Hint: There is a good section on dictionaries [here](https://docs.python.org/3/tutorial/datastructures.html#dictionaries).




<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
#<insert q16 code here>


```
</div>

</div>



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
_ = ok.grade('q16')

```
</div>

</div>




(17) Create a new variable `key1` from edict1 where the *key* is `st3`.



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
#<insert q17 code here>

```
</div>

</div>



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
_ = ok.grade('q17')

```
</div>

</div>



# Exercise-Numpy Array

(18) Create a new numpy array `nparray1` that is 3x3 and all the number 3 (should be integer type).





<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
#<insert q18 code here>


```
</div>

</div>



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
_ = ok.grade('q18')

```
</div>

</div>




(19) Create a new variable `nparray1sum` that sums all of column 0.




<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
#<insert q19 code here>

```
</div>

</div>



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
_ = ok.grade('q19')

```
</div>

</div>



(20) Create a new variable `nparray1mean` that takes the average of column 0.




<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
#<insert q20 code here>


```
</div>

</div>



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
_ = ok.grade('q20')

```
</div>

</div>



(21) Create a new numpy array `nparray2` that selects only column 1 of `nparray1` (all rows).




<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
#<insert q21 code here>


```
</div>

</div>



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
_ = ok.grade('q21')

```
</div>

</div>



(22) Create a new numpy array `nparray3` that is equal to `nparray1` times `2` (you should not alter `nparray1`).




<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
#<insert q22 code here>


```
</div>

</div>



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
_ = ok.grade('q22')

```
</div>

</div>



(23) Create a new numpy array nparray4 that is a verticle stack of `nparray1` and `nparray3`.



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
#<insert q23 code here>


```
</div>

</div>



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
_ = ok.grade('q23')

```
</div>

</div>



## Exercise-Pandas

For these you will need to import the iris dataset. You should find the file `iris.csv` in the main directory.  

While we showed 2 ways of importing a csv, you should use the `read_csv` method of Pandas to load the csv into a dataframe called `df`. 




<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
!wget https://raw.githubusercontent.com/rpi-techfundamentals/spring2019-materials/master/input/iris.csv

```
</div>

</div>



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
#Load iris.csv into a Pandas dataframe here.
#Check out the first few rows with the head command. 


```
</div>

</div>



(24) Create a variable `df_rows` that includes the number of rows in the `df` dataframe.



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
#<insert q24 code here>

```
</div>

</div>



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
_ = ok.grade('q24')

```
</div>

</div>



(25) Create a new dataframe `df_train` that includes the first half of the `df` dataframe. Create a new dataframe `df_test` that includes the second half. 



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
#<insert q25 code here>


```
</div>

</div>



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
_ = ok.grade('q25')

```
</div>

</div>



(26) Create a new Pandas Series `sepal_length` from the `sepal_length` column of the df dataframe.



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
#<insert q26 code here>


```
</div>

</div>



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
_ = ok.grade('q26')

```
</div>

</div>




(24) Using, the Iris dataset, find the mean of the `sepal_length` series in our sample and assign it to the `sepal_length_mean` variable. You should round the result to 3 digits after the decimal. 

```
#Round example
a=99.9999999999999
#For example, the following will round a to 2 digits.
b = round(a,2)   

```



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
#<insert q27 code here>


```
</div>

</div>



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
_ = ok.grade('q27')

```
</div>

</div>



## Run these Tests before Submission
This is a collection of all of the tests from the exercises above which will be used for grading.  



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
import os
_ = [ok.grade(q[:-3]) for q in os.listdir("tests") if q.startswith('q')]

```
</div>

</div>



## Export as a PDF and submit via LMS.

