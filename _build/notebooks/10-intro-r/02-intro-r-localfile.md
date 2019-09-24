---
interact_link: content/notebooks/10-intro-r/02-intro-r-localfile.ipynb
kernel_name: ir
has_widgets: false
title: 'Local Files'
prev_page:
  url: /notebooks/10-intro-r/01-intro-r-overview.html
  title: 'Introduction to R'
next_page:
  url: /notebooks/10-intro-r/03-intro-r-datastructures.html
  title: 'Data Structures'
comment: "***PROGRAMMATICALLY GENERATED, DO NOT EDIT. SEE ORIGINAL FILES IN /content***"
---


[![AnalyticsDojo](https://github.com/rpi-techfundamentals/spring2019-materials/blob/master/fig/final-logo.png?raw=1)](http://rpi.analyticsdojo.com)
<center><h1>Introduction to R - Test Local Jupyter Notebook</h1></center>
<center><h3><a href = 'http://rpi.analyticsdojo.com'>rpi.analyticsdojo.com</a></h3></center>



## Test Notebook

The goal of this notebook is to simply test the R environment and to show how to interact with a local data file. Let's first check the R version. 




<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```R
version

```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">
{:.output_data_text}
```
               _                           
platform       x86_64-apple-darwin14.5.0   
arch           x86_64                      
os             darwin14.5.0                
system         x86_64, darwin14.5.0        
status                                     
major          3                           
minor          5.1                         
year           2018                        
month          07                          
day            02                          
svn rev        74947                       
language       R                           
version.string R version 3.5.1 (2018-07-02)
nickname       Feather Spray               
```

</div>
</div>
</div>



## Reading a Local CSV File
- The `read.csv` command can accepted a variety of delimited files.
- Relative references are indicated with `..` to indcate going up a directory.
- Windows: - use either \ or / to indicate directories
- setwd('C:\\Users\\Your_username\\Desktop\\r-bootcamp-2016')
- setwd('..\\r-bootcamp-2016')



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```R
# This will load the local iris.csv file into an R dataframe.  We will work with these a lot in the future.
#This is refered to a relative reference.  This is Relative to the current working directory. 
frame=read.csv(file="../../input/iris.csv", header=TRUE, sep=",")

```
</div>

</div>



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```R
# This will print out the dataframe.
frame

```
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
	<tr><td>4.6   </td><td>3.4   </td><td>1.4   </td><td>0.3   </td><td>setosa</td></tr>
	<tr><td>5.0   </td><td>3.4   </td><td>1.5   </td><td>0.2   </td><td>setosa</td></tr>
	<tr><td>4.4   </td><td>2.9   </td><td>1.4   </td><td>0.2   </td><td>setosa</td></tr>
	<tr><td>4.9   </td><td>3.1   </td><td>1.5   </td><td>0.1   </td><td>setosa</td></tr>
	<tr><td>5.4   </td><td>3.7   </td><td>1.5   </td><td>0.2   </td><td>setosa</td></tr>
	<tr><td>4.8   </td><td>3.4   </td><td>1.6   </td><td>0.2   </td><td>setosa</td></tr>
	<tr><td>4.8   </td><td>3.0   </td><td>1.4   </td><td>0.1   </td><td>setosa</td></tr>
	<tr><td>4.3   </td><td>3.0   </td><td>1.1   </td><td>0.1   </td><td>setosa</td></tr>
	<tr><td>5.8   </td><td>4.0   </td><td>1.2   </td><td>0.2   </td><td>setosa</td></tr>
	<tr><td>5.7   </td><td>4.4   </td><td>1.5   </td><td>0.4   </td><td>setosa</td></tr>
	<tr><td>5.4   </td><td>3.9   </td><td>1.3   </td><td>0.4   </td><td>setosa</td></tr>
	<tr><td>5.1   </td><td>3.5   </td><td>1.4   </td><td>0.3   </td><td>setosa</td></tr>
	<tr><td>5.7   </td><td>3.8   </td><td>1.7   </td><td>0.3   </td><td>setosa</td></tr>
	<tr><td>5.1   </td><td>3.8   </td><td>1.5   </td><td>0.3   </td><td>setosa</td></tr>
	<tr><td>5.4   </td><td>3.4   </td><td>1.7   </td><td>0.2   </td><td>setosa</td></tr>
	<tr><td>5.1   </td><td>3.7   </td><td>1.5   </td><td>0.4   </td><td>setosa</td></tr>
	<tr><td>4.6   </td><td>3.6   </td><td>1.0   </td><td>0.2   </td><td>setosa</td></tr>
	<tr><td>5.1   </td><td>3.3   </td><td>1.7   </td><td>0.5   </td><td>setosa</td></tr>
	<tr><td>4.8   </td><td>3.4   </td><td>1.9   </td><td>0.2   </td><td>setosa</td></tr>
	<tr><td>5.0   </td><td>3.0   </td><td>1.6   </td><td>0.2   </td><td>setosa</td></tr>
	<tr><td>5.0   </td><td>3.4   </td><td>1.6   </td><td>0.4   </td><td>setosa</td></tr>
	<tr><td>5.2   </td><td>3.5   </td><td>1.5   </td><td>0.2   </td><td>setosa</td></tr>
	<tr><td>5.2   </td><td>3.4   </td><td>1.4   </td><td>0.2   </td><td>setosa</td></tr>
	<tr><td>4.7   </td><td>3.2   </td><td>1.6   </td><td>0.2   </td><td>setosa</td></tr>
	<tr><td>⋮</td><td>⋮</td><td>⋮</td><td>⋮</td><td>⋮</td></tr>
	<tr><td>6.9      </td><td>3.2      </td><td>5.7      </td><td>2.3      </td><td>virginica</td></tr>
	<tr><td>5.6      </td><td>2.8      </td><td>4.9      </td><td>2.0      </td><td>virginica</td></tr>
	<tr><td>7.7      </td><td>2.8      </td><td>6.7      </td><td>2.0      </td><td>virginica</td></tr>
	<tr><td>6.3      </td><td>2.7      </td><td>4.9      </td><td>1.8      </td><td>virginica</td></tr>
	<tr><td>6.7      </td><td>3.3      </td><td>5.7      </td><td>2.1      </td><td>virginica</td></tr>
	<tr><td>7.2      </td><td>3.2      </td><td>6.0      </td><td>1.8      </td><td>virginica</td></tr>
	<tr><td>6.2      </td><td>2.8      </td><td>4.8      </td><td>1.8      </td><td>virginica</td></tr>
	<tr><td>6.1      </td><td>3.0      </td><td>4.9      </td><td>1.8      </td><td>virginica</td></tr>
	<tr><td>6.4      </td><td>2.8      </td><td>5.6      </td><td>2.1      </td><td>virginica</td></tr>
	<tr><td>7.2      </td><td>3.0      </td><td>5.8      </td><td>1.6      </td><td>virginica</td></tr>
	<tr><td>7.4      </td><td>2.8      </td><td>6.1      </td><td>1.9      </td><td>virginica</td></tr>
	<tr><td>7.9      </td><td>3.8      </td><td>6.4      </td><td>2.0      </td><td>virginica</td></tr>
	<tr><td>6.4      </td><td>2.8      </td><td>5.6      </td><td>2.2      </td><td>virginica</td></tr>
	<tr><td>6.3      </td><td>2.8      </td><td>5.1      </td><td>1.5      </td><td>virginica</td></tr>
	<tr><td>6.1      </td><td>2.6      </td><td>5.6      </td><td>1.4      </td><td>virginica</td></tr>
	<tr><td>7.7      </td><td>3.0      </td><td>6.1      </td><td>2.3      </td><td>virginica</td></tr>
	<tr><td>6.3      </td><td>3.4      </td><td>5.6      </td><td>2.4      </td><td>virginica</td></tr>
	<tr><td>6.4      </td><td>3.1      </td><td>5.5      </td><td>1.8      </td><td>virginica</td></tr>
	<tr><td>6.0      </td><td>3.0      </td><td>4.8      </td><td>1.8      </td><td>virginica</td></tr>
	<tr><td>6.9      </td><td>3.1      </td><td>5.4      </td><td>2.1      </td><td>virginica</td></tr>
	<tr><td>6.7      </td><td>3.1      </td><td>5.6      </td><td>2.4      </td><td>virginica</td></tr>
	<tr><td>6.9      </td><td>3.1      </td><td>5.1      </td><td>2.3      </td><td>virginica</td></tr>
	<tr><td>5.8      </td><td>2.7      </td><td>5.1      </td><td>1.9      </td><td>virginica</td></tr>
	<tr><td>6.8      </td><td>3.2      </td><td>5.9      </td><td>2.3      </td><td>virginica</td></tr>
	<tr><td>6.7      </td><td>3.3      </td><td>5.7      </td><td>2.5      </td><td>virginica</td></tr>
	<tr><td>6.7      </td><td>3.0      </td><td>5.2      </td><td>2.3      </td><td>virginica</td></tr>
	<tr><td>6.3      </td><td>2.5      </td><td>5.0      </td><td>1.9      </td><td>virginica</td></tr>
	<tr><td>6.5      </td><td>3.0      </td><td>5.2      </td><td>2.0      </td><td>virginica</td></tr>
	<tr><td>6.2      </td><td>3.4      </td><td>5.4      </td><td>2.3      </td><td>virginica</td></tr>
	<tr><td>5.9      </td><td>3.0      </td><td>5.1      </td><td>1.8      </td><td>virginica</td></tr>
</tbody>
</table>

</div>

</div>
</div>
</div>



## Writing data out from R

Here you have a number of options. 

1) You can write out R objects to an R Data file, as we've seen, using `save()` and `save.image()`.
2) You can use `write.csv()` and `write.table()` to write data frames/matrices to flat text files with delimiters such as comma and tab.
3) You can use `write()` to write out matrices in a simple flat text format.
4) You can use `cat()` to write to a file, while controlling the formatting to a fine degree.
5) You can write out in the various file formats mentioned on the previous slide



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```R
#Writing Dataframe to a file. 
write.csv(frame, file= "iris2.csv")

#Kaggle won't want the rownames. 
write.csv(frame, file = "iris3.csv",row.names=FALSE)

```
</div>

</div>



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```R
setwd('/Users/jasonkuruzovich/githubdesktop/0_class/techfundamentals-spring2018-materials/classes/')

```
</div>

</div>



## The Working Directory
- To read and write from R, you need to have a firm grasp of where in the computer's filesystem you are reading and writing from.
- It is common to set the working directory, and then just list the specific file without a path.
- Windows: - use either \\ or / to indicate directories
- `setwd('C:\\Users\\Your_username\\Desktop\\r-bootcamp-2016')`
- `setwd('..\\r-bootcamp-2016')`



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```R
#Whhile this example is Docker, should work similarly for Mac or Windows Based Machines. 
setwd("/home/jovyan/techfundamentals-fall2017-materials/classes/05-intro-r")
getwd()  # what directory will R look in?
setwd("/home/jovyan/techfundamentals-fall2017-materials/classes/input") # change the working directory
getwd() 
setwd("/home/jovyan/techfundamentals-fall2017-materials/classes/05-intro-r")
setwd('../input') # This is an alternate way of moving to data.
getwd()

```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">

<div markdown="0" class="output output_html">
'/home/jovyan/techfundamentals-fall2017-materials/classes/05-intro-r'
</div>

</div>
</div>
<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">

<div markdown="0" class="output output_html">
'/home/jovyan/techfundamentals-fall2017-materials/classes/input'
</div>

</div>
</div>
<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">

<div markdown="0" class="output output_html">
'/home/jovyan/techfundamentals-fall2017-materials/classes/input'
</div>

</div>
</div>
</div>



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```R
#Notice path isn't listed. We don't have to list the path if we have set the working directory

frame=read.csv(file="iris.csv", header=TRUE,sep=",")

```
</div>

</div>



## Exercises

### Basics

1) Make sure you are able to install packages from CRAN. E.g., try to install *lmtest*.

2) Figure out what your current working directory is.

### Using the ideas

3) Put the *data/iris.csv* file in some other directory. Use `setwd()` to set your working directory to be that directory. Read the file in using `read.csv()`.  Now use `setwd()` to point to a different directory. Write the data frame out to a file without any row names and without quotes on the character strings.




### CREDITS

Copyright [AnalyticsDojo](http://rpi.analyticsdojo.com) 2016.
This work is licensed under the [Creative Commons Attribution 4.0 International](https://creativecommons.org/licenses/by/4.0/) license agreement.
Work adopted from [Berkley R Bootcamp](https://github.com/berkeley-scf/r-bootcamp-2016).

