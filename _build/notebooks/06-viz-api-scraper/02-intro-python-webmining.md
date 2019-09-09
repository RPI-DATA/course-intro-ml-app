---
interact_link: content/notebooks/06-viz-api-scraper/02-intro-python-webmining.ipynb
kernel_name: python3
has_widgets: false
title: 'Web Mining'
prev_page:
  url: /notebooks/06-viz-api-scraper/01-intro-api-twitter.html
  title: 'Twitter'
next_page:
  url: /notebooks/06-viz-api-scraper/03-visualization-python-seaborn.html
  title: 'Visualizations - Seaborn'
comment: "***PROGRAMMATICALLY GENERATED, DO NOT EDIT. SEE ORIGINAL FILES IN /content***"
---


[![AnalyticsDojo](https://github.com/rpi-techfundamentals/spring2019-materials/blob/master/fig/final-logo.png?raw=1)](http://rpi.analyticsdojo.com)
<center><h1>Introduction to Python - Web Mining</h1></center>
<center><h3><a href = 'http://rpi.analyticsdojo.com'>rpi.analyticsdojo.com</a></h3></center>



## This tutorial is directly from the the BeautifulSoup documentation.
[https://www.crummy.com/software/BeautifulSoup/bs4/doc/]

### Before you begin
If running locally you need to make sure that you have beautifulsoup4 installed. 
`conda install beautifulsoup4` 

It should already be installed on colab. 




# All html documents have structure.  Here, we can see a basic html page.



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```html_doc = """
<html><head><title>The Dormouse's story</title></head>
<body>
<p class="title"><b>The Dormouse's story</b></p>

<p class="story">Once upon a time there were three little sisters; and their names were
<a href="http://example.com/elsie" class="sister" id="link1">Elsie</a>,
<a href="http://example.com/lacie" class="sister" id="link2">Lacie</a> and
<a href="http://example.com/tillie" class="sister" id="link3">Tillie</a>;
and they lived at the bottom of a well.</p>

<p class="story">...</p>
"""

```
</div>

</div>



<html><head><title>The Dormouse's story</title></head>
<body>
<p class="title"><b>The Dormouse's story</b></p>
<p class="story">Once upon a time there were three little sisters; and their names were
<a class="sister" href="http://example.com/elsie" id="link1">Elsie</a>,
<a class="sister" href="http://example.com/lacie" id="link2">Lacie</a> and
<a class="sister" href="http://example.com/tillie" id="link3">Tillie</a>;
and they lived at the bottom of a well.</p>
<p class="story">...</p>
</body></html>



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```from bs4 import BeautifulSoup
import requests
soup = BeautifulSoup(html_doc, 'html.parser')

print(soup.prettify())


```
</div>

</div>



### A Retreived Beautiful Soup Object
- Can be parsed via dot notation to travers down the hierarchy by *class name*, *tag name*, *tag type*, etc.





<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```soup

```
</div>

</div>



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```#Select the title class.
soup.title
 


```
</div>

</div>



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```#Name of the tag.
soup.title.name




```
</div>

</div>



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```#String contence inside the tag
soup.title.string




```
</div>

</div>



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```#Parent in hierarchy.
soup.title.parent.name




```
</div>

</div>



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```#List the first p tag.
soup.p




```
</div>

</div>



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```#List the class of the first p tag.
soup.p['class']




```
</div>

</div>



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```#List the class of the first a tag.
soup.a




```
</div>

</div>



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```#List all a tags.
soup.find_all('a')



```
</div>

</div>



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```
soup.find(id="link3")


```
</div>

</div>



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```#The Robots.txt listing who is allowed.
response = requests.get("https://en.wikipedia.org/robots.txt")
txt = response.text
print(txt)

```
</div>

</div>



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```response = requests.get("https://www.rpi.edu")
txt = response.text
soup = BeautifulSoup(txt, 'html.parser')

print(soup.prettify())

```
</div>

</div>



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```soup.find_all('a')

```
</div>

</div>



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```# Experiment with selecting your own website.  Selecting out a url. 

response = requests.get("enter url here")
txt = response.text
soup = BeautifulSoup(txt, 'html.parser')

print(soup.prettify())

```
</div>

</div>



#For more info, see
[https://github.com/stanfordjournalism/search-script-scrape](https://github.com/stanfordjournalism/search-script-scrape) 

