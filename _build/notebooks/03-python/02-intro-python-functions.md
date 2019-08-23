---
interact_link: content/notebooks/03-python/02-intro-python-functions.ipynb
kernel_name: python3
has_widgets: false
title: 'Functions'
prev_page:
  url: /notebooks/03-python/01-intro-python-conditionals-loops.html
  title: 'Conditional-Loops'
next_page:
  url: /notebooks/03-python/03-intro-python-null-values.html
  title: 'Null Values'
comment: "***PROGRAMMATICALLY GENERATED, DO NOT EDIT. SEE ORIGINAL FILES IN /content***"
---


[![AnalyticsDojo](https://github.com/rpi-techfundamentals/spring2019-materials/blob/master/fig/final-logo.png?raw=1)](http://rpi.analyticsdojo.com)
<center><h1>Introduction to Python - Functions</h1></center>
<center><h3><a href = 'http://rpi.analyticsdojo.com'>rpi.analyticsdojo.com</a></h3></center>





## Overview
- Why functions?
- (Need to know) Indentation in Python
- Predefined functions
- Custom functions
- Lambda functions
- Exercises



# Why Functions? What are they for?



## Why Functions?
- Code reuse. 
- Abstract away complexity. 
- Simple, efficient robust code.
- Specific functional programming languages like Lisp & Haskell built around *functional programming*, which enforces great practices.
- Read more about functional programming in Python [here](http://www.ibm.com/developerworks/library/l-prog/).



## Predefined Functions
- We have used predefined functions in earlier exercises
- Python has predefined functions for embedded data structures like variables and lists.
- Packages like Numpy and Pandas include functions for working with those data
- We can string functions together ex. `b = np.arange(15).reshape(3, 5)`



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```#Simple Rounding Functions
a=3.14
a=round(a)
print(a)

```
</div>

</div>



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```#We can string functions together
import numpy as np
b = np.arange(15).reshape(3, 5)
print(b)

```
</div>

</div>



## Custom Functions in Python
- The statement `def` introduces a function definition, which is followed by the *function name* and the *list of pramaters that are passed to the function* as well as a colon to end the line.
- The second line with three quotes (""") is called a *docstring* which can be used to automatically generate documentation.
- Values used in intermediate calculations within a function are *local variables* to the function and not global variables.
- The `return` statement returns a value to a passed value.  
- Execution of the function requires passing parameters. For example, `squared(5)` as shown below.




<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```testvalue=0
# Defining New Functions

def squared(x):
    """Function to square a value if passed another value."""  #This is a docstring
    testvalue=10 #Note this is not stored outside of the function but is a local variable which is not returned.
    x=x**2
    return x 
print (squared(5), testvalue)

```
</div>

</div>



## Functional Programming

- Functions are, like everything else in Python, object.
- Functions can passed around just like any other value.
- That means we can do really cool things, like pass a function to a function. 







<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```print(squared)

```
</div>

</div>



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```y = squared
print (y)

```
</div>

</div>



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```print (y(5))

```
</div>

</div>



## Applying a Function to Each Element of a Collection with Map

- We can apply a function to each element of a collection using the built-in function `map()`.
- Provided using the form *map(Function, Sequence)*.
- This will work with any collection: dict, list, set, and tuple. (Tuples are immutable data structures similar to lists that we won't really be using). 




<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```exampleList=[1, 2, 3, 4]
list(map(squared, exampleList)) #While Python 2 returned a list, Python 3 will return a map object. 

```
</div>

</div>



## Applying a Function to Each Element of a Collection with  List Comprehensions
- Because this is such a common operation, Python has a special syntax to do the same thing, called a list comprehension.
- Can change whether the result is a set or list by changing brackets.
- If we want a set instead of a list we can use a set comprehension or dictionary comprehension



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```
#List Comprehensions
list1=[squared(i) for i in [1, 2, 3, 4]]
print(list1)

```
</div>

</div>



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```{squared(i) for i in [1, 2, 3, 4]}

```
</div>

</div>



## Applying a Function to Subset of a Collection with  List Comprehensions
- List Comprehensions can be nested
- The `if` statemen can be added to make the function apply to only some. 



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```lista= [(x, y) for x in ['a','b','c'] for y in ['c','d','e'] if x != y]
print(lista)

```
</div>

</div>



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```#This is a comparable for loop. 
listb=[]
for x in ['a','b','c']:
    for y in ['c','d','e']:
        if x != y:
            listb.append((x, y))
        
print(listb)       

```
</div>

</div>



## Cartesian product using List Comprehensions
![Cartesian Product](https://upload.wikimedia.org/wikipedia/commons/4/4e/Cartesian_Product_qtl1.svg)
By Quartl (Own work) [GFDL (http://www.gnu.org/copyleft/fdl.html) or CC BY-SA 3.0 (http://creativecommons.org/licenses/by-sa/3.0)], via Wikimedia Commons

<br>
- The [Cartesian product](https://en.wikipedia.org/wiki/Cartesian_product) of two collections $X = A \times B$ can be expressed by using multiple `for` statements in a comprehension.
-This can be thoguht of as nested for loop. 




<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```A = {'x', 'y', 'z'}
B = {1, 2, 3}
{(a,b) for a in A for b in B}

```
</div>

</div>



# Cartesian products with other collections

- The syntax for Cartesian products can be used with any collection type.




<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```first_names = ('Steve', 'John', 'Peter')
surnames = ('Smith', 'Doe')

[(first_name, surname) for first_name in first_names for surname in surnames]

```
</div>

</div>



## Anonymous Functions: _lambda expressions_

- While we can create named functions with `def`, we can also write _anonymous_ functions that do not necessarily have a name.
- They are called _lambda expressions_ (after the $\lambda-$calculus).
- Useful if creating a function that is only going to be used once. 
- Previously we created a `squared` function. This creates an anonymous lambda function.



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```y=map(lambda x: x ** 2, [1, 2, 3, 4])
print(list(y))

```
</div>

</div>



## `lambda` : Filtering LIst

- We can filter a list by applying a _predicate_ to each element of the list.

- A *predicate* is a function which takes a single argument, and returns a boolean value.

- We will be using a similar procedure to filter DataFrames.




<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```lista=[-5, 2, 3, -10, 0, 1]
listb=list(filter(lambda x: x > 0, lista))
print(listb)



```
</div>

</div>



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```#Here we could rereate an equavalent filter. 

def postive_numbers(listz):
    return [i for i in listz if i>0]  #Here we are using list comprehension.
postive_numbers(lista)

```
</div>

</div>



## Example Filter String
We can use both `filter()` and `map()` on other collections such as strings or sets.



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```list(filter(lambda x: x != ' ', 'hello world'))

```
</div>

</div>



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```list(filter(lambda x: x > 0, {-5, 2, 3, -10, 0, 1}))

```
</div>

</div>



## Filtering using a List Comprehension

- Again, because this is such a common operation, we can use simpler syntax to say the same thing.

- We can express a filter using a list-comprehension by using the keyword `if`:



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```data = [-5, 2, 3, -10, 0, 1]
[x for x in data if x > 0]

```
</div>

</div>



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```#We can also filter and then map in the same expression:
from numpy import sqrt
[sqrt(x) for x in data if x > 0]

```
</div>

</div>



## Big Data

- The `map()` and `reduce()` functions form the basis of the map-reduce programming model.

- [Map-reduce](https://en.wikipedia.org/wiki/MapReduce) is the basis of modern highly-distributed large-scale computing frameworks.

- It is used in BigTable, Hadoop and Apache Spark. 




### Modules
- Functions are great, but you might not want to repeat same function in different programs.
- To faciliate code reuse, Python `modules` can be imported.  
- Must have modeules placed somewhere they are in the `PYTHONPATH` (a list of directory names, with the same syntax as the shell variable PATH). 
- Review the fibo.py file, which contains a function for a Fibonacci series



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```# A Fibonacci series is a series of numbers in which each number ( Fibonacci number ) is the sum of the two preceding numbers.
import fibo
fibo.fib(1000)

```
</div>

</div>



### Working with Modules in Jupyter Notebooks
- If a model is loaded, re-running the import command won't update it. 
- Instead, we need to reload it. 
- Alternately, one could restart the kernal.



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```def get5():
    return 5

```
</div>

</div>



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```!wget https://raw.githubusercontent.com/rpi-techfundamentals/spring2019-materials/master/03-python/fibo.py

```
</div>

</div>



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```#@title
import importlib
import fibo
importlib.reload(fibo)  # 
fibo.get5()

```
</div>

</div>




Copyright [AnalyticsDojo](http://rpi.analyticsdojo.com) 2016.
This work is licensed under the [Creative Commons Attribution 4.0 International](https://creativecommons.org/licenses/by/4.0/) license agreement.
Adopted from [materials](https://github.com/phelps-sg/python-bigdata) Copyright [Steve Phelps](http://sphelps.net) 2014

