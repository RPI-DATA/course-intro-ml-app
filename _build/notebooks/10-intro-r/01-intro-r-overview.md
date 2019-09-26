---
interact_link: content/notebooks/10-intro-r/01-intro-r-overview.ipynb
kernel_name: ir
has_widgets: false
title: 'Introduction to R'
prev_page:
  url: /notebooks/assignments/05-starter.html
  title: 'Assignment 5'
next_page:
  url: /notebooks/10-intro-r/02-intro-r-localfile.html
  title: 'Local Files'
comment: "***PROGRAMMATICALLY GENERATED, DO NOT EDIT. SEE ORIGINAL FILES IN /content***"
---


[![AnalyticsDojo](https://github.com/rpi-techfundamentals/spring2019-materials/blob/master/fig/final-logo.png?raw=1)](http://rpi.analyticsdojo.com)
<center><h1>Introduction to R - Overview and Packages</h1></center>
<center><h3><a href = 'http://rpi.analyticsdojo.com'>rpi.analyticsdojo.com</a></h3></center>





# “The best thing about R is that it was developed by statisticians. The worst thing about R is that… it was developed by statisticians.”
##                                         - Bo Cowgill, Google



## Overview
- Language Features and Use Cases  
- R 
- R Studio
- R and Packages



## What is R?

- R is an Open Source (and freely available) environment for statistical computing and graphics.
- It is a full-featured programming language, in particular a scripting language (with similarities to Matlab and Python).
- It can be run interactively or as a batch/background job.
- R is being actively developed with ongoing updates/new releases.
- R has a variety of built-in as well as community-provided packages that extend its functionality with code and data; see CRAN for the thousands of add-on packages.
- It is freely-available and modifiable (Available for Windows, Mac OS X, and Linux).



## Modes of Using R

- From the command line in a Linux/Mac terminal window
- Using the Windows/Mac GUIs
- Using the RStudio GUI, an 'integrated development environment'
- Running an R script in the background on a Linux/Mac machine (Windows?)
- RStudio



## Why R?

- R is widely used (statisticians, scientists, social scientists) and has the widest statistical functionality of any software
- Users add functionality via packages all the time
- R is free and available on all major platforms
- As a scripting language, R is very powerful, flexible, and easy to use
- As a scripting language, R allows for reproducibility and automating tasks
- As a language, R can do essentially anything
- Wide usage helps to improve quality and reduce bugs
- R can interact with other software, databases, the operating system, the web, etc.
- R is built on C and can call user-written and external C code and packages (in particular, see the *Rcpp* R package)




## Why Not R?

- Other software is better than R at various tasks
    
i.e., [Python](http://imgs.xkcd.com/comics/python.png) is very good for text manipulation, interacting with the operating system, and as a glue for tying together various applications/software in a workflow-* R can be much slower than compiled languages (but is often quite fast with good coding practices!)
- R's packages are only as good as the person who wrote them; no explicit quality control
- R is a sprawling and unstandardized ecosystem
- Google has a [recommended style](https://google.github.io/styleguide/Rguide.xml) guide that should be taken into account. 



## R and Jupyter
- R commands can be executed in a Jupyter Notebook just by play at the end of a cell.
- Blocks of cells or even the entire notebook can be executed by clicking on the *Cell* above.
- The Kernal is responsible for interpreting the code, and the current kernal is listed on the top right of the notebook. 
- While Jupyter started as a Python project, there are now a variety of Kernals for different programming languages including R, Scala, and SAS. 
- Read more about Jupyter in the documentation [here](http://jupyter.readthedocs.io/en/latest/).
- If a variable isn't assigned it will be provided as output. 




<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```R
test<-5
test

```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">

<div markdown="0" class="output output_html">
5
</div>

</div>
</div>
</div>



## R and RStudio
- Powerful IDE for R
- Integrated usage of git
- Integrated GUI based package management
- Integrated GIT
- Solid enterprise infrastructure, owned by Microsoft




### R and Conda
- Conda can quickly and easily install R. We love conda! 
- This is a rather long install, so may do it from command line. 
`!conda install r-essentials`



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```R
#conda install -y r-essentials

```
</div>

</div>



## R as calculator



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```R
# R as a calculator
2 * pi # multiply by a constant
7 + runif(1) # add a random number
3^4 # powers
sqrt(4^4) # functions
log(10)
log(100, base = 10)
23 %/% 2 
23 %% 2

```
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
7.7227991654072
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
</div>



## Assignment of Values
- Don't used the equal sign.  
- Do use `<-`




<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```R
#Assignment DON"T use =
val <- 3
val
print(val)

Val <- 7 # case-sensitive!
print(c(val, Val))


```
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
{:.output_stream}
```
[1] 3
[1] 3 7
```
</div>
</div>
</div>



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```R
# This will gnerate numbers from 1 to 6. 
mySeq <- 1:6
mySeq

#Notice how this worrks like arrange in Python.
myOtherSeq <- seq(1.1, 11.1, by = 2)
myOtherSeq
length(myOtherSeq)

fours <- rep(4, 6)
fours

# This is a comment: here is an example of non-numeric data
depts <- c('espm', 'pmb', 'stats')
depts


```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">

<div markdown="0" class="output output_html">
<ol class=list-inline>
	<li>1</li>
	<li>2</li>
	<li>3</li>
	<li>4</li>
	<li>5</li>
	<li>6</li>
</ol>

</div>

</div>
</div>
<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">

<div markdown="0" class="output output_html">
<ol class=list-inline>
	<li>1.1</li>
	<li>3.1</li>
	<li>5.1</li>
	<li>7.1</li>
	<li>9.1</li>
	<li>11.1</li>
</ol>

</div>

</div>
</div>
<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">

<div markdown="0" class="output output_html">
6
</div>

</div>
</div>
<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">

<div markdown="0" class="output output_html">
<ol class=list-inline>
	<li>4</li>
	<li>4</li>
	<li>4</li>
	<li>4</li>
	<li>4</li>
	<li>4</li>
</ol>

</div>

</div>
</div>
<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">

<div markdown="0" class="output output_html">
<ol class=list-inline>
	<li>'espm'</li>
	<li>'pmb'</li>
	<li>'stats'</li>
</ol>

</div>

</div>
</div>
</div>



## R and Packages (R's killer app)
- There are a tremendous number of packages available which extend the core capabilities of the R language.
-  "Currently, the CRAN package repository features 9233 available packages." (from https://cran.r-project.org/web/packages/)
- [Packages on CRAN](https://cran.r-project.org/web/packages/). Also see the [CRAN Task Views](https://cran.r-project.org/web/views/). 
- Packages may be source or compiled binary files.
- Installing source packages which contain C/C++/Fortran code requires that compilers and related tools be installed. 
- Binary packages are platform-specific (Windows/Mac). This can cause problems in a large class and is a great reason to work in a standardized environment like the cloud or on Docker/VM.
- If you want to sound like an R expert, make sure to call them *packages* and not *libraries*. A *library* is the location in the directory structure where the packages are installed/stored.




## Using R packages
1. Install the package on your machine
2. Load the package



## Installing Packages in RStudio
- Only needs to be done one time on machine. 
- To install a package, in RStudio, just do `Packages->Install Packages`.
- Option to specify the source repository: `install.packages('chron', repos='http://cran.us.r-project.org')`
- Option to install multiple packages: `install.packages(c("pkg1", "pkg2"))`
- You can install dependencies with: `install.packages("chron", dependencies = TRUE)`
- If binary files aren't available for your OS, install from source: `install.packages(path_to_file, repos = NULL, type="source")`
- R installs are not done from the terminal (no `!`).
    



## Install Packages with Conda
This will only work from the Anaconda Prompt or terminal. 

`!conda install -y -c r r-yaml`



## Loading Packages
- R packages must be imported before using. 
- Packages only have to be imported once in a notebook (not in every cell). 
- The `library(chron)` would import the chron package, provided it has beeen installed. 
- See all packages loaded with `search()`
- search()    # see packages currently loaded



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```R
search() # see packages currently loaded
library("yaml") # Load the package chron
search()    # see packages currently loaded


```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">

<div markdown="0" class="output output_html">
<ol class=list-inline>
	<li>'.GlobalEnv'</li>
	<li>'jupyter:irkernel'</li>
	<li>'jupyter:irkernel'</li>
	<li>'package:stats'</li>
	<li>'package:graphics'</li>
	<li>'package:grDevices'</li>
	<li>'package:utils'</li>
	<li>'package:datasets'</li>
	<li>'package:methods'</li>
	<li>'Autoloads'</li>
	<li>'package:base'</li>
</ol>

</div>

</div>
</div>
<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">

<div markdown="0" class="output output_html">
<ol class=list-inline>
	<li>'.GlobalEnv'</li>
	<li>'package:yaml'</li>
	<li>'jupyter:irkernel'</li>
	<li>'jupyter:irkernel'</li>
	<li>'package:stats'</li>
	<li>'package:graphics'</li>
	<li>'package:grDevices'</li>
	<li>'package:utils'</li>
	<li>'package:datasets'</li>
	<li>'package:methods'</li>
	<li>'Autoloads'</li>
	<li>'package:base'</li>
</ol>

</div>

</div>
</div>
</div>



## CREDITS


Copyright [AnalyticsDojo](http://rpi.analyticsdojo.com) 2016.
This work is licensed under the [Creative Commons Attribution 4.0 International](https://creativecommons.org/licenses/by/4.0/) license agreement.
This work is adopted from the Berkley R Bootcamp.  

