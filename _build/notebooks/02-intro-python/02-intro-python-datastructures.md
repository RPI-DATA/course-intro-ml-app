---
interact_link: content/notebooks/02-intro-python/02-intro-python-datastructures.ipynb
kernel_name: python3
has_widgets: false
title: 'Basic Data Structures'
prev_page:
  url: /notebooks/02-intro-python/01-intro-python-overview.html
  title: 'Python Overview'
next_page:
  url: /notebooks/02-intro-python/03-intro-python-numpy.html
  title: 'Numpy'
comment: "***PROGRAMMATICALLY GENERATED, DO NOT EDIT. SEE ORIGINAL FILES IN /content***"
---


[![AnalyticsDojo](https://github.com/rpi-techfundamentals/spring2019-materials/blob/master/fig/final-logo.png?raw=1)](http://rpi.analyticsdojo.com)
<center><h1>Introduction to Python - Datastructures</h1></center>
<center><h3><a href = 'http://rpi.analyticsdojo.com'>rpi.analyticsdojo.com</a></h3></center>





# Overview
Common to R and Python
- Variables
- Opearations on Numeric and String Variables
- Lists

Python Only
- Dictionaries
- Sets





## Variables
- Single value
- Strings, Integer, Floats and boolean are the most common types of variables.
- Remember, under the covers they are all objects.
- Multiple variables can be output with the `print()` statement. 
- `\t` can be used to add a tab while `\n` can input a new line.



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
a = '#pythonrules' # string
b = 30              # integer
c = True            # boolean

#This prints (1) only the variables, (2) with labels, (3) including tabs, and (4) with new lines.
print('1:', a,  b, c)
print('2:','String:', a, 'Integer:', b, 'Boolean:', c)
print('3:','String:', a, '\tInteger:', b, '\tBoolean:', c)
print('4a:','String:', a, '\n4b: Integer:', b, '\n4c: Boolean:', c)
print(a+str(b))

```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">
{:.output_stream}
```
1: #pythonrules 30 True
2: String: #pythonrules Integer: 30 Boolean: True
3: String: #pythonrules 	Integer: 30 	Boolean: True
4a: String: #pythonrules 
4b: Integer: 30 
4c: Boolean: True
#pythonrules30
```
</div>
</div>
</div>



## Variable Type (continued)
- In Python when we write `b = 30` this means the value of `30` is assigned to the `b` object. 
- Python is a [dynamically typed](https://pythonconquerstheuniverse.wordpress.com/2009/10/03/static-vs-dynamic-typing-of-programming-languages/).
- Unlike some languages, we don't have to declare the type of a variable before using it. 
- Variable type can also change with the reassignment of a variable. 





<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
a = 1
print ('The value of a is ', a,  'and type ', type(a) )

a = 2.5
print ('Now the value of a is ', a,  'and type ', type(a) )

a = 'hello there'
print ('Now the value of a is ', a,  'and of type ', type(a) )

```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">
{:.output_stream}
```
The value of a is  1 and type  <class 'int'>
Now the value of a is  2.5 and type  <class 'float'>
Now the value of a is  hello there and of type  <class 'str'>
```
</div>
</div>
</div>



## Variable Type (continued)

- _Variables_ themselves do not have a fixed type.
- It is only the values that they refer to that have an associated _type_.
- This means that the type referred to by a variable can change as more statements are interpreted.
- If we combine types incorrectly we get an error. 



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
#We can't add 5 to a 
b = 'string variable'
c=b+5
c

```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">


{:.output_data_text}
```
'string variable5'
```


</div>
</div>
</div>



# The `type` Function

- We can query the type of a value using the `type` function.
- Variables can be reassigned to a different type. 
- There are integer, floating point, and complex number [numeric types](https://docs.python.org/3/library/stdtypes.html#numeric-types-int-float-complex).
- Boolean is a special type of integer.



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
a = 1
type(a)

```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">


{:.output_data_text}
```
int
```


</div>
</div>
</div>



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
a = 'hello'
type(a)

```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">


{:.output_data_text}
```
str
```


</div>
</div>
</div>



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
a=2.5
type(a)

```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">


{:.output_data_text}
```
float
```


</div>
</div>
</div>



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
a=True
type(a)

```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">


{:.output_data_text}
```
bool
```


</div>
</div>
</div>



# Converting Values Between Types

- We can convert values between different types.
- To convert to string use the `str()` function.
- To convert to floating-point use the `float()` function.
- To convert to an integer use the `int()` function.
- To convert to a boolean use the `bool()` function.




<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
a = 1
print(a, type(a))

a = str(a)
print (a, type(a))

a = float(a)
print (a, type(a))

a = int(a)
print (a, type(a))

```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">
{:.output_stream}
```
1 <class 'int'>
1 <class 'str'>
1.0 <class 'float'>
1 <class 'int'>
```
</div>
</div>
</div>



# Converting Values Between Types (Continued)
- To convert to a boolean use the `bool()` function.
- `bool` can work with a String type that is `True` or `False`
- `bool` can work with an integer type that is `1` for `True` or `0` for `False`



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
b = 'True'
print (b, type(b))

b = bool(b)
print (b, type(b))

c = 1
c= bool(c)
print (c, type(c))

d = 0
d= bool(d)
print (d, type(d))

```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">
{:.output_stream}
```
True <class 'str'>
True <class 'bool'>
True <class 'bool'>
False <class 'bool'>
```
</div>
</div>
</div>



# Null Values

- Sometimes we represent "no data" or "not applicable".  
- In Python we use the special value `None`.
- This corresponds to `NA` in R of `Null` in Java/SQL.
- When we print the value `None` is printed. 
- If we enter the variable, no result is printed out.




<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
a = None
print(a)


```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">
{:.output_stream}
```
None
```
</div>
</div>
</div>



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
#Notice nothing is printed.
a

```
</div>

</div>



## Operations on Numeric Variables
- Python can be used as a basic calculator.
- Check out this associated [tutorial](https://docs.python.org/3/tutorial/introduction.html#using-python-as-a-calculator).



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
print('Addition:', 53 + 5)
print('Multiplication:', 53 * 5)
print('Subtraction:', 53 - 5)
print('Division', 53 / 5 )
print('Floor Division (discards the fractional part)', 53 // 5 )
print('Floor Division (returns the remainder)', 53 % 5 )
print('Exponents:', 5 ** 2 )

```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">
{:.output_stream}
```
Addition: 58
Multiplication: 265
Subtraction: 48
Division 10.6
Floor Division (discards the fractional part) 10
Floor Division (returns the remainder) 3
Exponents: 25
```
</div>
</div>
</div>



## Operations on String Variables
- Just as we can do numeric operations, we can also do operations on strings.
- Concatentate Strings 
- A *backslash* is used as an escape variable.
- More info on this [tutorial](https://docs.python.org/3/tutorial/introduction.html#using-python-as-a-calculator).



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
a='Start'
b='End'
tab='\t'
newline='\n'
c='can\'t'  #Note that we have to use the Escape character '\' to inclue a apostrophe '  in the key.
cb="can't"
continueline = 'This is the first line. \
This is the second line, but we have included a line continuation character: \\'
#Note that to print the continueline character we have to list 2 (\\)
#Note that to print the continueline character we have to list 2 (\\)

contin2= """
This is the second line, but we have included a line continuation character: 
#Note that to print the continueline character we have to list 2 
#Note that to print the continueline character we have to list 2


"""



print('Concatenation:', a+b )
print('Tab:', a+tab+b )
print('Newline:', a+newline+b )
print('Apostrophe:', c )
print('Apostrophe:', cb )
print('Continue line:', continueline )
print(contin2)

```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">
{:.output_stream}
```
Concatenation: StartEnd
Tab: Start	End
Newline: Start
End
Apostrophe: can't
Apostrophe: can't
Continue line: This is the first line. This is the second line, but we have included a line continuation character: \

This is the second line, but we have included a line continuation character: 
#Note that to print the continueline character we have to list 2 
#Note that to print the continueline character we have to list 2



```
</div>
</div>
</div>



# Calling Functions on Variables

- We can call functions in a conventional way using round brackets
- Python has a wide variety of [built in functions](https://docs.python.org/3/library/functions.html),



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
a=abs(-98.45)
print('abs() takes the absolute value:', a )
a=round(a)
print('round() rounds to nearest integer:', a )
character=chr(a)
print('chr(98) returns the string representing a character whose Unicode code point is associated with the integer:',character) 

```
</div>

</div>



## Exercise - Operations on Variables

1. What happens when you multiply a number times a boolean? What is the resulting type? 
2. What happens when you try to multiply an integer value times a null?
3. Take 5 to the power of 4. 





## Lists
- Lists can be used to contain a sequence of values of any type. 
- You can do operations on lists.
- The list values start at 0 and that the first value of a list can be printed using `a[0]`
- Lists can be *sliced* or *indexed* using the start and end value `a[start:end]`
- Lists are *mutable datastructures*, meaning that they can be changed (added to).



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
#Set the value of the list
a = [1, 2, 'three', 'four', 5.0]

print('Print the entire array:', a)
print('Print the first value:', a[0])
print('Print the first three value:', a[0:3])
print('Print from second value till end  of list:', a[2:])
print('Print the last value of a list:', a[-1])
print('Print up till the 2nd to last value:', a[:-2]) 
type(a)

```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">
{:.output_stream}
```
Print the entire array: [1, 2, 'three', 'four', 5.0]
Print the first value: 1
Print the first three value: [1, 2, 'three']
Print from second value till end  of list: ['three', 'four', 5.0]
Print the last value of a list: 5.0
Print up till the 2nd to last value: [1, 2, 'three']
```
</div>
</div>
<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">


{:.output_data_text}
```
list
```


</div>
</div>
</div>



## Lists
- Lists can be nested, where there are lists of lists.
- The elements of a nested list is specified after the first list when slicing `c[0][0]`



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
a = [1, 2, 'three', 'four', 5.0]
b = [6, 'seven', 8, 'nine']
c = [a, b]

print('This is a list with 2 lists in it:', c)
print('This is the first list:', c[0])
print('This is the first element of the second list:', c[1][0])

```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">
{:.output_stream}
```
This is a list with 2 lists in it: [[1, 2, 'three', 'four', 5.0], [6, 'seven', 8, 'nine']]
This is the first list: [1, 2, 'three', 'four', 5.0]
This is the first element of the second list: 6
```
</div>
</div>
</div>



## Lists
- Lists can added to with the `append` method or your can directly assign location in list.
- You can identify the length of a list with `len(a)`
- [More fuctions on lists](https://docs.python.org/3/tutorial/datastructures.html) include `pop()` `insert()` etc.
- If you set a `lista = listb` this list will not be a copy but instead be the same list, where if you modify one it will modify both.
- To create a copy of a list, you can use `lista=listb[:]`



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
b = [6, 'seven', 8, 'nine']
b.append(10)
print('We added 10 to b:', b)
print('the length of b is now:', len(b))

b[len(b):] = ['Eleven',12]
print('We added 11 to b:', b)



```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">
{:.output_stream}
```
We added 10 to b: [6, 'seven', 8, 'nine', 10]
the length of b is now: 5
We added 11 to b: [6, 'seven', 8, 'nine', 10, 'Eleven', 12]
```
</div>
</div>
</div>



## List

- If you set a lista = listb this list will not be a copy but instead be the same list, where if you modify one it will modify both.
- To create a copy of a list, you can use `lista=listb[:]` or `lista=listb.copy()`



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
listb=[1,2,3,4]
listb1=[1,2,3,4]
listb2=[1,2,3,4]
#This assigns one variable to another, linking them
lista=listb
#This creates a copy
lista1=listb1[:]
lista2=listb2.copy() # This does the same thing.
#This deletes the third item in the array.
lista.pop(3)
lista1.pop(3)
lista2.pop(3)
#Notice how when we pop lista, listb is also impacted.
print(lista, listb)
#Notice how when we use a copy, listb1 is not impacted. 
print(lista1, listb1)
print(lista2, listb2)

```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">
{:.output_stream}
```
[1, 2, 3] [1, 2, 3]
[1, 2, 3] [1, 2, 3, 4]
[1, 2, 3] [1, 2, 3, 4]
```
</div>
</div>
</div>



## Exercise-Lists
Hint: [This list of functions on lists is useful.](https://docs.python.org/3/tutorial/datastructures.html)

1. Create a list `elists1` with the following values (1,2,3,4,5).
2. Create a new list `elists2` by first creating a copy of `elist1` and then reversing the order.
3. Create a new list `elists3` by first creating a copy of `elist1` and then adding 7 8 9 to the end. *(Hint: Search for a different function if appending doesn't work.)*
4. Create a new list `elists4` by first creating a copy of `elist3` and then insert 6 between 5 and 7. 






# Sets

- Lists can contain duplicate values.
- A set, in contrast, contains no duplicates.
- Sets can be created from lists using the `set()` function.
- Alternatively we can write a set literal using the `{` and `}` brackets.






<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
#This creates a set from a list. 
X = set([1, 2, 3, 3, 4])

print(X, type(X))

```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">
{:.output_stream}
```
{1, 2, 3, 4} <class 'set'>
```
</div>
</div>
</div>



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
X = {1, 2, 3, 4, 4}
print(X, type(X))

```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">
{:.output_stream}
```
{1, 2, 3, 4} <class 'set'>
```
</div>
</div>
</div>



# Sets are Mutable

- Sets are mutable like lists (meaning we can change them)
- Duplicates are automatically removed



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
X = {1, 2, 3, 4}
X.add(0)
X.add(5)
print(X)
X.add(5)
print(X)

```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">
{:.output_stream}
```
{0, 1, 2, 3, 4, 5}
{0, 1, 2, 3, 4, 5}
```
</div>
</div>
</div>



# Sets are Unordered

- Sets do not have an order.
- Therefore we cannot index or slice them.





<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
X[0]

```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">
{:.output_traceback_line}
```

    ---------------------------------------------------------------------------

    TypeError                                 Traceback (most recent call last)

    <ipython-input-30-19c40ecbd036> in <module>()
    ----> 1 X[0]
    

    TypeError: 'set' object does not support indexing


```
</div>
</div>
</div>



## Operations on Sets

- Union: $X \cup Y$ combines two sets




<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
X = {1, 2, 3, 4}
Y = {4, 5, 6}
X.union(Y)

```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">


{:.output_data_text}
```
{1, 2, 3, 4, 5, 6}
```


</div>
</div>
</div>



## Operations on Sets
- Intersection: $X \cap Y$:



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
X = {1, 2, 3, 4}
Y = {3, 4, 5}
X.intersection(Y)

```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">


{:.output_data_text}
```
{3, 4}
```


</div>
</div>
</div>



## Operations on Sets
- Difference $X - Y$:




<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
X = {1, 2, 3, 4}
Y = {3, 4, 5}
X - Y

```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">


{:.output_data_text}
```
{1, 2}
```


</div>
</div>
</div>



## Dictionaries
- You can think of dictionaries as arrays that help you assocaite a `key` with a `value`.
- Dictionaries can be specified with `{key: value, key: value}`
- Dictionaries can be specified with dict([('key', value), ('key', value)])
- Key's and values can be either string or numeric. 
- Dictionaries are mutable, (can be changed) `adict['g'] = 41`



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
adict1 = {'a' : 0, 'b' : 1, 'c' : 2}
adict2 = dict([(1, 'a'), (2, 'b'), (3, 'c')])
print(adict1,adict2, '\n', type(adict1),type(adict2), '\n',adict1['b'],adict2[2])

```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">
{:.output_stream}
```
{'c': 2, 'b': 1, 'a': 0} {1: 'a', 2: 'b', 3: 'c'} 
 <class 'dict'> <class 'dict'> 
 1 b
```
</div>
</div>
</div>



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
adict2['g']=1234

```
</div>

</div>



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
adict2['g']

```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">


{:.output_data_text}
```
1234
```


</div>
</div>
</div>



## Exercise-Sets/Dictionary

1. Create a set `eset1` with the following values (1,2,3,4,5).
2. Create a new set `eset2` the following values (1,3,6).
3. Create a new set `eset3` that is `eset1-eset2`.
4. Create a new set `eset4` that is the union of `eset1+eset2`.
5. Create a new set `eset5` that includes values that are in both `eset1` and `eset2` (intersection).
6. Create a new dict `edict1` with the following keys and associated values: st1=45; st2=32; st3=40; st4=31.
7. Create a new variable edict2 from edict 1 where the key is st3.
 






## CREDITS


Copyright [AnalyticsDojo](http://rpi.analyticsdojo.com) 2016
This work is licensed under the [Creative Commons Attribution 4.0 International](https://creativecommons.org/licenses/by/4.0/) license agreement.

This work has been adopted from the [origional version](https://github.com/phelps-sg/python-bigdata):
Copyright [Steve Phelps](http://sphelps.net) 2014



