---
interact_link: content/notebooks/02-intro-python/03-intro-python-numpy.ipynb
kernel_name: python3
has_widgets: false
title: 'Numpy'
prev_page:
  url: /notebooks/02-intro-python/02-intro-python-datastructures.html
  title: 'Basic Data Structures'
next_page:
  url: /notebooks/02-intro-python/04-intro-python-pandas.html
  title: 'Pandas'
comment: "***PROGRAMMATICALLY GENERATED, DO NOT EDIT. SEE ORIGINAL FILES IN /content***"
---


[![AnalyticsDojo](https://github.com/rpi-techfundamentals/spring2019-materials/blob/master/fig/final-logo.png?raw=1)](http://rpi.analyticsdojo.com)
<center><h1>Introduction to Python - Numpy</h1></center>
<center><h3><a href = 'http://rpi.analyticsdojo.com'>rpi.analyticsdojo.com</a></h3></center>




## Overview of Numpy

- Numpy is a package that provides additional functionality often useful working with arrays for data science. 
- Typically Numpy is imported as `np`.
- `np.array()` will cast a list (or other collection) as a numpy array.
- You can slice an array in the same way yo can slice a list.




<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```import numpy as np
a = np.array([0, 1, 2, 3, 4, 5, 6])
print('A is of type:', type(a))
print('Print the entire array:', a)
print('Print the first value:', a[0])
print('Print the first three value:', a[0:3])
print('Print from second value till end  of list:', a[2:])
print('Print the last value of a numpy array:', a[-1])
print('Print up till the 2nd to last value:', a[:-2]) 


```
</div>

</div>



## Arrays and Functions
- A really powerful aspect of arrays is the capaiblity to do calculations over arrays.
- Numpy has a number of functions possible listed [here](http://docs.scipy.org/doc/numpy/reference/routines.math.html).
- Often it is possible to do calculations directly or via np functions, as shown below. 



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```import numpy as np
a = np.array([1, 2, 3, 4, 5, 6])
b1=10*a
b2=np.multiply(10,a)
c1=a+b1
c2=np.add(a,b1) #This is an alternate way of adding 
d=np.log(a)
e=np.sqrt(a)
f=a**2  #This squares the value. 

np.square([-1j, 1])
print('Print the entire array a:', a)
print('Print the entire array b1:', b1)
print('Print the entire array b2:', b2)
print('Print the entire array b3:', c1)
print('Print the entire array c2:', c2)
print('Print the entire array d:', d)
print('Print the entire array e:', e)
print('Print the entire array f:', f)

```
</div>

</div>



## Creating and Manipulating Numpy Arrays
- The arrange function will generate an array. 
- Reshape changes the structure of the array to n rows and m columns.
    `a=a.reshape(n, m)`
-`ones` will create an array with all ones and `zeros` with all zeros.
- Reshaping can get it in the appropriate structure, but make sure that the size fits the appropriate dimensions.



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```import numpy as np
a = np.arange(15) 
print(a)
a2 = np.arange( 0, 15, 1 ) #Alternate specification with np.arrange(start, end, step)
print(a2)
a=a.reshape(3, 5)
print(a)
b= np.ones(shape=(3, 5), dtype=float)
print(b)
c= np.zeros(shape=(3, 5), dtype=int)
print(c)
d= np.full((3, 5), 4, dtype=int)
print(d)
e= np.arange( 0, 1.5, .1 ).reshape(3,5)  #String together creations and reshaping. Also can use decimals.
print(e)

```
</div>

</div>



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```e= np.arange( 0, 1.5, .1 ).reshape(3,5) 

```
</div>

</div>



## Generating Random Numpy Data
- This is often useful, and we will be using it to demonstrate some initial techniques.
- Often you want random but repeatable results, so that for example a test could have a consistent average on a random array. For this we need to set a seed. You only have to do this once.





<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```
np.random.seed([2335])
a = np.random.uniform(50, 150, 10)  #Between 50-150, generate 10 variables from uniform
b = np.random.standard_normal(10)   #With mean 0 and standard deviation 1 
print(a)
print(b)



```
</div>

</div>



## Combining Numpy Arrays
- `concatenate` will string a list of numpy arrays together `np.concatenate([a,b])`
- `vstack` will stack numpy arrays 
- Defaults: start =0, end =last and step is 1.
- To print the entire array, leave start/stop/step blank`a[::]`



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```a = np.arange(5)
b=np.concatenate([a,a])
c=np.vstack([a,a])
d=np.hstack([c,c])
print('a:',a,'\nb:',b,'\nc:',c,'\nd:',d)


```
</div>

</div>



## Slicing Single Dimension Numpy Arrays
- Slicing arrays includes  three numbers `a[start:stop:step]` but not all are required.
- Defaults: start =0, end =last and step is 1.
- To print the entire array, leave start/stop/step blank`a[::]`




<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```e= np.arange( 0, 15, 1 ) 
print(e)
#[start:end:step]


print("This is the start, end, and step:",e[2:9:3]) 
print("Print every other:",e[::2]) 
print("Print starting at 2 and ending at 9, default step 1:",e[2:9]) 
print("Print all:",e[::])
print("Print all:",e[:]) 
print("Print all:",e) 

```
</div>

</div>



## Numpy Arrays From External Datasets
- We can take a list from an external dataset and change it to an numpy array. 



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```#First let's download some data. 
!wget https://raw.githubusercontent.com/rpi-techfundamentals/spring2019-materials/master/input/iris.csv

```
</div>

</div>



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```import csv
csv_file_object = csv.reader(open('iris.csv', newline=''), delimiter=',')

data=[]
header = next(csv_file_object) #
for row in csv_file_object:  
    data.append(row)  # add each row to the 
data = np.array(data)
print(data)

```
</div>

</div>



## Slicing 2 Dimensional Numpy Arrays
- We can slice arrays with `array[row, column]` were row and column each include the (start:stop:step) like in arrays
- We can sepecify the type with the `.astype(np.float_)`
- For a full list of Numpy types, see [documentation](http://docs.scipy.org/doc/numpy-1.10.1/user/basics.types.html)
- If we create a one dimensional array from 2 dimensional numpy array, it will also be a numpy array of same type.




<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```#We can slice the array several different ways and generate new variables.

irisdata=data[0::,0:4:].astype(np.float_)  #This will select only the first 4 columns and change the type to float
irisdata=data[:,0:4].astype(np.float_)
iristype=data[0::,4:5:] # This will select only the type. 
print(irisdata,'\n',iristype)

```
</div>

</div>



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```#This can be used to select column 1 and assign to new variable. 
#This will sum up column 1
newvariable=irisdata[::,0:1:]

#This will sum up column 0
final=irisdata[::,0:1:].sum()

type(newvariable)
#print(newvariable)
print(final)

```
</div>

</div>



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```#This will take the mean of column 1
print('mean:', irisdata[::,0:1:].mean())

```
</div>

</div>



## CREDITS


Copyright [AnalyticsDojo](http://rpi.analyticsdojo.com) 2016
This work is licensed under the [Creative Commons Attribution 4.0 International](https://creativecommons.org/licenses/by/4.0/) license agreement.



