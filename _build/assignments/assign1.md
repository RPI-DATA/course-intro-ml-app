---
interact_link: content/assignments/assign1.ipynb
kernel_name: python3
has_widgets: false
title: 'Assignment 1'
prev_page:
  url: /assignments/index.html
  title: 'Assignments'
next_page:
  url: /assignments/assign2.html
  title: 'Assignment 2'
comment: "***PROGRAMMATICALLY GENERATED, DO NOT EDIT. SEE ORIGINAL FILES IN /content***"
---


## Homework 01 - Instructions

- The goal for this lab will be to collect some data from you and get you some experience in working with Github and Jupyter Notebooks. We are going to also learn the basics of loading data into Python, updating Python objects, and writing out files.

Here you will learn:
- Importing data from a number of different file types (.csv, .json, Parquet).
- Updating lists, dataframes, and JSON objects. 
- Outputting data as comma delimited, tab delimited, JSON, Parquet 





### Download some Data
First, we are going do townload some files.  These files have my contact info in various structures.  



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```!wget https://github.com/rpi-techfundamentals/spring2019-materials/raw/master/02-intro-python/hm-01/in.zip && unzip -o in.zip

```
</div>

</div>



### Making a Directory
First we are going to make a directory called out that has the updated data structures.  



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```!mkdir out

```
</div>

</div>



### Importing a CSV File to a Python List
- A list is a rather simple Python data structure. While it isn't very common to import a CSV to a list, we are going to start off there.   
- Luckily, someone has written a `csv` package that does the majority of the heavy lifting. 
- `import csv` imports all methods in the csv package.  
- We are going to import the values from the csv into a specific type of Python data structure, a `list`. Declaring `listcsv=[]` initializes the objects as a list and makes available the `append` method.  
- By using the `with open` syntax shown below, we don't have to open and then close the data structure. 



## CSV
- Comma delimited files are a common way of transmitting data. 
- Data for different columns is separated by a comma.
- It is possible to open a csv in different ways.  



### Importing a CSV File
- The `open` command specifies the file name and what we want to do with it, here `r` stand for read.
- The `csvreader` is an object which iterates on the file. 
- We then read each `row` using a `for` loop. 



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```#This is an example of how to import a CSV into a Python list. 
import csv  #This imports the CSV package.
listcsv=[] #This initializes a list data structure.

with open('in/name.csv', 'r') as data_file:   #The "with" incorporates an open and close of file. 
    csvreader = csv.reader(data_file, delimiter=',')
    for row in csvreader:
        print("each row of the reader imported as a:", type(row), row,"\n")
        listcsv.append(row)


```
</div>

</div>



### Printing out the List
- Just enter the name of the list in order to get the contents of the list. 



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```listcsv

```
</div>

</div>



### Updating List Values
- `listcsv` is a 2 dimensional list, with the first number indicating the row and the second number indicating the column.
- Objects start numbering at 0, so that in this case the header-row is 0. 

**Go ahead and reassign the values of the list to include your first and last name**




<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```listcsv[1][0] = "sam"   #row/column numbers start at 0
listcsv[1][1] = "smith"    #row/column numbers start at 0



```
</div>

</div>



### Let's print it out to make sure that it worked.



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```listcsv

```
</div>

</div>



**In the cell below replace the values for 'email-rpi', 'email-other', 'github-userid', 'slack-userid' just like you did for first and last name.**



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```listcsv[1][3] = "sam@rpi"   #row/column numbers start at 0
listcsv[1][4] = "san@gmail"    #row/column numbers start at 0

```
</div>

</div>



### Let's print it out to make sure that it worked.



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```listcsv

```
</div>

</div>



### Outputing a CSV File from a List
- It is very common to write to a CSV file.  Here we are doing it using a list. 
- Here, notice we are doing just the same thing as reading. However, we are doing it by opening the file with a `w`. 



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```#Here we are going to save as a tab delimited file
with open("out/name.csv", 'w' ) as outfile:
    writer = csv.writer(outfile, delimiter=',')
    writer.writerows(listcsv)

```
</div>

</div>



### Writing a Tab Delimited File from a List
- Here we are able to output the file as a tab delimited file.  
- A tab delimed file utilizes '\t' as the delimiter.
- Notice that this code is about the same, with just the filename and the delimiter changed 



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```#Here we are going to save as a tab delimited file
with open("out/name.txt", 'w') as outfile:
    writer = csv.writer(outfile, delimiter='\t')
    writer.writerows(listcsv)

```
</div>

</div>



### Importing CSV into a Pandas Dataframe
- Data structured like CSV's are extremely common.
- The most common data structure you will import them into is a Pandas dataframe. 
- Pandas will give access to many useful methods for working with data.  
- `pandas` is often imported as the abbreviated `pd`.
- Typing the object name of a pandas dataframe (here `dfcsv`) gives a *pretty printed* version of the table.



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```# This will load the local name.csv file into a Pandas dataframe.  We will work with these a lot in the future.
import pandas as pd # This line imports the pandas package. 
dfcsv = pd.read_csv('in/name.csv')
dfcsv

```
</div>

</div>



### Updating a Pandas Dataframe
- Pandas figured out that you have columns and even knows the rows. Notice the Pandas Magic! 
- We will utilize 2 different methods of updating a Pandas dataframe. 
- The `loc` method uses an index of the `[rownum, colname]`
- The `iloc` method uses an indes of the `[rownum, colnum]`

**Again update the first 2 values. Here we have added the dfcsf to the end of the cell, printing the dataframe. **



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```# Updating  
dfcsv.loc[0, 'first-name'] = 'sam'
dfcsv.loc[0, 'last-name'] = 'smith'
dfcsv

```
</div>

</div>



**Update the remainder of the columns.** 



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```#Enter your code to update the remaining columns. 

dfcsv.loc[0, 'email-rpi'] = 'sam@rpi'
dfcsv.loc[0, 'email-other'] = 'smith'

## Notice how we can just as easily write the file with Pandas. 
dfcsv.to_csv('out/nameloc.csv')
dfcsv #this prints the dataframe

```
</div>

</div>



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```#Reloading Data
# This will load the local name.csv file into a Pandas dataframe.  We will work with these a lot in the future.
dfcsv = pd.read_csv('in/name.csv')
dfcsv

```
</div>

</div>



**Again update the first 2 values. Here we have added the dfcsv to the end of the cell, printing the dataframe. **



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```# This just utilizes the integer position. 
dfcsv.iloc[0, 0] =  'your first name'
dfcsv.iloc[0, 1] = 'your last name'
dfcsv

```
</div>

</div>



**Update the remainder of the columns.** 



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```#Enter your code to update the remaining columns using iloc. 



## Notice how we can just as easily write the file with Pandas. 
dfcsv.to_csv('out/nameiloc.csv')


```
</div>

</div>



### Loading a JSON File
- Javascript object notation is another file format you may frequently see, particularlly from APIs.  
- This enables multipled layers of nesting, something that could take multiple files in a CSV or relational tables.
- Our JSON is imported as a `dictionary`, which is another internal type of Python data structure.
- *Pretty Printing* (pprint) will keep structure. 




<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```import json   #This imports the JSON
from pprint import pprint  #This will print the file in a nested way. 

with open('in/name.json') as data_file:   #The "with" incorporates an open and close of file.   
    datajson = json.load(data_file)

print("data is a python object of type: ", type(datajson),"\n")
pprint(datajson) #Pretty printing (pprint) makes it easier to see the nesting of the files. 


```
</div>

</div>



### Updating a Dictionary
- Here we are going to again update the first two rows and then print the results. 



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```#Here is how you update the dictionary: 
#We are indicating that we want the first student, and from there we list which "key" for 
#the dictionary we want (i.e., 'first-name').

datajson['student'][0]['first-name']  = 'sam '
datajson['student'][0]['last-name']  = 'smith'
pprint(datajson)

```
</div>

</div>



**Update the remainder of the columns below.** 



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```#Enter your code to update the remaining dictionary values. 


#This writes the file
with open('out/name.json', 'w') as data_file:   #The "with" incorporates an open and close of file.   
    json.dump(datajson, data_file)
pprint(datajson)

```
</div>

</div>



## Parquet Files
- CSV files are great for humans to read and understand.  
- For "big data" though, it isn't a great long term storage option (inefficient/slow).
- Parquet is a type columnar storage format.  It makes dealing with lots of columns fast. 
- [fastparquet](https://fastparquet.readthedocs.io) is a Python package for dealing with Parquet files. 
- Apache Spark also natively reads Parquet Files. 
- If you are running this locally you may need to install fastparquet.  This is a package. Look [here](https://anaconda.org/conda-forge/fastparquet) for instructions on installing the fastparquet package. 



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```!pip install fastparquet

```
</div>

</div>



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```from fastparquet import ParquetFile
pf = ParquetFile('in/name.parq')
dfparq = pf.to_pandas()
dfparq

```
</div>

</div>



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```#Update the dfparq dataframe with your info (Hint: You updated a dataframe earlier).
dfparq

```
</div>

</div>



## Create a Parquet file.
- This is a binary file, so it can't be read by a text editor. 



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```# We can similarly easily write to a .parq file. 
from fastparquet import write
write('out/name.parq', dfparq)
dfparq

```
</div>

</div>



## This is All
- Unfortunately 



## Homework Rubric
The following Rubric will be used to grade homwork. 
q1. Updated `out/name.csv` file with your information  (1 pt).<br>
q2. Updated `out/name.txt` file with your information  (1 pt).<br>
q3. Updated `out/name.csv` file with your information  (1 pt).<br>
q4. Updated `out/name.json`  file with your information  (1 pt).<br>
q5. Updated `out/name.parq` parquet file (1 pt)<br>
q6. Review the [pandas documentation](https://pandas.pydata.org) and give a 3 sentence summary of what the Pandas Package is for. (3 pt) <br>
q7.  Read through the [fastparquet documentation](https://fastparquet.readthedocs.io) and give a 2 sentence summary of what the Pandas Package is for. (2 pt)





<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```#Notice how we can use 3 quotes. 
q6 = """q6.  


"""

```
</div>

</div>



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```q7 = """q7.
Enter answer here. 


"""

```
</div>

</div>



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```#q8 I work throug this entire homework step by step and did not copy the file from somone else. Change to True if you did. 

q8= "False"

```
</div>

</div>



### Generate Answers.txt
This will output the questions to a text file. 



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```answers= [q6,q7,q8]
with open('out/answers.txt', 'w') as outfile:   #The "with" incorporates an open and close of file. 
    outfile.write("\n".join(answers))



```
</div>

</div>



### SUBMIT YOUR ANSWERS BY PUSHING TO THE HOMEWORK REPOSITORY. (DON'T FORGET PDF OF LAB01




This work is licensed under the [Creative Commons Attribution 4.0 International](https://creativecommons.org/licenses/by/4.0/) license agreement.

