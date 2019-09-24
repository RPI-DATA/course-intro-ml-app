---
interact_link: content/notebooks/10-intro-r/07-intro-r-merge-agg-fun.ipynb
kernel_name: ir
has_widgets: false
title: 'Aggregation and Merge'
prev_page:
  url: /notebooks/10-intro-r/06-intro-r-conditionals-loops.html
  title: 'Conditional-Loops'
next_page:
  url: /notebooks/10-intro-r/08-intro-r-tidyverse.html
  title: 'Tidyvere'
comment: "***PROGRAMMATICALLY GENERATED, DO NOT EDIT. SEE ORIGINAL FILES IN /content***"
---


[![AnalyticsDojo](https://github.com/rpi-techfundamentals/spring2019-materials/blob/master/fig/final-logo.png?raw=1)](http://rpi.analyticsdojo.com)
<center><h1>Introduction to R - Merging and Aggregating Data</h1></center>
<center><h3><a href = 'http://rpi.analyticsdojo.com'>rpi.analyticsdojo.com</a></h3></center>




## Overview
- Merging Dataframes 
- Aggregating Dataframes
- Advanced Functions




## Merging Data Frame with Vector
- Can combine vector with data frame in multiple ways. 
- `data.frame(a,b)` where a & b can be vectors, matrices, or data frames. 



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```R
#Below is the sample data we will be creating 2 dataframes  
key=(1:10)

#Here we are passing the row names and column names as a list. 
m<- data.frame(matrix(rnorm(40, mean=20, sd=5), nrow=10, ncol=4, dimnames=list((1:10),c("a","b","c","d"))))
m2<- data.frame(matrix(rnorm(40, mean=1000, sd=5), nrow=10, ncol=4, dimnames=list((1:10),c("e","f","g","h"))))

#This is one way of combining a vector with a dataframe. 
df<-  data.frame(key,m)
df2<- data.frame(key,m2)

#This is another way way of combining a vector with a dataframe. 
dfb<-  cbind(key,m)
df2b<- cbind(key,m2)

df
df2
dfb
df2b


```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">

<div markdown="0" class="output output_html">
<table>
<thead><tr><th scope=col>key</th><th scope=col>a</th><th scope=col>b</th><th scope=col>c</th><th scope=col>d</th></tr></thead>
<tbody>
	<tr><td> 1       </td><td>16.19873 </td><td>14.41495 </td><td>27.73225 </td><td>15.564688</td></tr>
	<tr><td> 2       </td><td>27.76677 </td><td>18.54772 </td><td>21.06688 </td><td>15.697810</td></tr>
	<tr><td> 3       </td><td>11.68592 </td><td>14.91207 </td><td>22.82086 </td><td>15.666790</td></tr>
	<tr><td> 4       </td><td>16.39982 </td><td>28.30284 </td><td>10.97550 </td><td>19.083633</td></tr>
	<tr><td> 5       </td><td>22.16232 </td><td>16.82574 </td><td>14.28676 </td><td>20.162797</td></tr>
	<tr><td> 6       </td><td>17.17425 </td><td>14.36932 </td><td>18.55487 </td><td>13.498498</td></tr>
	<tr><td> 7       </td><td>20.15380 </td><td>18.00987 </td><td>15.99028 </td><td>14.325000</td></tr>
	<tr><td> 8       </td><td>20.68866 </td><td>12.83505 </td><td>25.24119 </td><td>24.538494</td></tr>
	<tr><td> 9       </td><td>18.84664 </td><td>24.01079 </td><td>12.69775 </td><td> 8.095156</td></tr>
	<tr><td>10       </td><td>16.29913 </td><td>21.51270 </td><td>15.14676 </td><td>23.722103</td></tr>
</tbody>
</table>

</div>

</div>
</div>
<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">

<div markdown="0" class="output output_html">
<table>
<thead><tr><th scope=col>key</th><th scope=col>e</th><th scope=col>f</th><th scope=col>g</th><th scope=col>h</th></tr></thead>
<tbody>
	<tr><td> 1       </td><td>1004.1240</td><td> 997.4379</td><td> 997.5697</td><td>1000.8540</td></tr>
	<tr><td> 2       </td><td>1002.6933</td><td> 998.4041</td><td>1009.1720</td><td>1010.4120</td></tr>
	<tr><td> 3       </td><td> 995.9138</td><td>1001.0959</td><td>1004.6025</td><td>1002.5405</td></tr>
	<tr><td> 4       </td><td> 999.5493</td><td> 998.8054</td><td>1003.9649</td><td>1000.0133</td></tr>
	<tr><td> 5       </td><td>1007.2373</td><td>1006.2580</td><td>1000.1882</td><td> 992.9980</td></tr>
	<tr><td> 6       </td><td>1000.2068</td><td> 994.7482</td><td> 998.2876</td><td>1002.7093</td></tr>
	<tr><td> 7       </td><td> 999.1622</td><td> 998.6231</td><td> 998.7175</td><td> 998.0497</td></tr>
	<tr><td> 8       </td><td>1003.1263</td><td>1002.7279</td><td>1004.1623</td><td>1000.5204</td></tr>
	<tr><td> 9       </td><td>1003.1548</td><td> 994.2030</td><td>1002.1614</td><td> 999.3726</td></tr>
	<tr><td>10       </td><td>1006.8744</td><td>1004.2677</td><td> 998.8720</td><td> 993.5726</td></tr>
</tbody>
</table>

</div>

</div>
</div>
<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">

<div markdown="0" class="output output_html">
<table>
<thead><tr><th scope=col>key</th><th scope=col>a</th><th scope=col>b</th><th scope=col>c</th><th scope=col>d</th></tr></thead>
<tbody>
	<tr><td> 1       </td><td>16.19873 </td><td>14.41495 </td><td>27.73225 </td><td>15.564688</td></tr>
	<tr><td> 2       </td><td>27.76677 </td><td>18.54772 </td><td>21.06688 </td><td>15.697810</td></tr>
	<tr><td> 3       </td><td>11.68592 </td><td>14.91207 </td><td>22.82086 </td><td>15.666790</td></tr>
	<tr><td> 4       </td><td>16.39982 </td><td>28.30284 </td><td>10.97550 </td><td>19.083633</td></tr>
	<tr><td> 5       </td><td>22.16232 </td><td>16.82574 </td><td>14.28676 </td><td>20.162797</td></tr>
	<tr><td> 6       </td><td>17.17425 </td><td>14.36932 </td><td>18.55487 </td><td>13.498498</td></tr>
	<tr><td> 7       </td><td>20.15380 </td><td>18.00987 </td><td>15.99028 </td><td>14.325000</td></tr>
	<tr><td> 8       </td><td>20.68866 </td><td>12.83505 </td><td>25.24119 </td><td>24.538494</td></tr>
	<tr><td> 9       </td><td>18.84664 </td><td>24.01079 </td><td>12.69775 </td><td> 8.095156</td></tr>
	<tr><td>10       </td><td>16.29913 </td><td>21.51270 </td><td>15.14676 </td><td>23.722103</td></tr>
</tbody>
</table>

</div>

</div>
</div>
<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">

<div markdown="0" class="output output_html">
<table>
<thead><tr><th scope=col>key</th><th scope=col>e</th><th scope=col>f</th><th scope=col>g</th><th scope=col>h</th></tr></thead>
<tbody>
	<tr><td> 1       </td><td>1004.1240</td><td> 997.4379</td><td> 997.5697</td><td>1000.8540</td></tr>
	<tr><td> 2       </td><td>1002.6933</td><td> 998.4041</td><td>1009.1720</td><td>1010.4120</td></tr>
	<tr><td> 3       </td><td> 995.9138</td><td>1001.0959</td><td>1004.6025</td><td>1002.5405</td></tr>
	<tr><td> 4       </td><td> 999.5493</td><td> 998.8054</td><td>1003.9649</td><td>1000.0133</td></tr>
	<tr><td> 5       </td><td>1007.2373</td><td>1006.2580</td><td>1000.1882</td><td> 992.9980</td></tr>
	<tr><td> 6       </td><td>1000.2068</td><td> 994.7482</td><td> 998.2876</td><td>1002.7093</td></tr>
	<tr><td> 7       </td><td> 999.1622</td><td> 998.6231</td><td> 998.7175</td><td> 998.0497</td></tr>
	<tr><td> 8       </td><td>1003.1263</td><td>1002.7279</td><td>1004.1623</td><td>1000.5204</td></tr>
	<tr><td> 9       </td><td>1003.1548</td><td> 994.2030</td><td>1002.1614</td><td> 999.3726</td></tr>
	<tr><td>10       </td><td>1006.8744</td><td>1004.2677</td><td> 998.8720</td><td> 993.5726</td></tr>
</tbody>
</table>

</div>

</div>
</div>
</div>



## Merging Columns of Data Frame with another Data Frame
- Can combine data frame in multiple ways. 
- `merge(a,b,by="key")` where a & b are dataframes with the same keys.
- `cbind(a,b)` where a & b are dataframes with the same number of rows.



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```R
# This manages the merge by an associated key.
df3 <- merge(df,df2,by="key")
# This just does a "column bind" 
df4<- cbind(df,df2)
df5<- data.frame(df,df2)
df3
df4
df5

```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">

<div markdown="0" class="output output_html">
<table>
<thead><tr><th scope=col>key</th><th scope=col>a</th><th scope=col>b</th><th scope=col>c</th><th scope=col>d</th><th scope=col>e</th><th scope=col>f</th><th scope=col>g</th><th scope=col>h</th></tr></thead>
<tbody>
	<tr><td> 1       </td><td>16.278799</td><td>23.004297</td><td>19.262524</td><td>22.11648 </td><td>1004.4714</td><td> 995.4055</td><td>1001.5156</td><td>1004.8862</td></tr>
	<tr><td> 2       </td><td>19.287252</td><td>19.229253</td><td>18.817575</td><td>13.67939 </td><td>1000.7399</td><td>1004.6248</td><td> 995.8950</td><td>1008.5242</td></tr>
	<tr><td> 3       </td><td>18.833623</td><td>16.142004</td><td>18.454224</td><td>18.29241 </td><td> 994.3794</td><td>1002.2578</td><td> 998.6004</td><td> 999.7609</td></tr>
	<tr><td> 4       </td><td>23.780847</td><td>10.934207</td><td>13.448540</td><td>17.83936 </td><td>1001.4795</td><td>1009.0885</td><td>1002.5866</td><td> 998.8287</td></tr>
	<tr><td> 5       </td><td>13.982935</td><td>16.924402</td><td>19.037475</td><td>20.19748 </td><td> 993.9745</td><td> 999.9868</td><td>1001.0336</td><td> 987.3751</td></tr>
	<tr><td> 6       </td><td>15.534589</td><td>23.437320</td><td> 6.795926</td><td>20.19305 </td><td> 996.1284</td><td>1008.8440</td><td>1005.5196</td><td>1003.6926</td></tr>
	<tr><td> 7       </td><td>16.660076</td><td>18.315077</td><td>32.107139</td><td>23.35534 </td><td> 994.5026</td><td>1004.9990</td><td>1004.0972</td><td>1005.6532</td></tr>
	<tr><td> 8       </td><td>19.447799</td><td>18.278384</td><td>11.823108</td><td>13.09162 </td><td>1007.8858</td><td> 993.8745</td><td>1005.1093</td><td> 996.8686</td></tr>
	<tr><td> 9       </td><td> 9.225069</td><td>24.925796</td><td>13.868021</td><td>17.06181 </td><td> 997.6026</td><td>1001.1045</td><td> 991.7969</td><td>1000.5898</td></tr>
	<tr><td>10       </td><td>25.809451</td><td> 7.492747</td><td>18.483003</td><td>24.99244 </td><td> 995.6190</td><td>1010.2642</td><td> 998.6192</td><td> 998.8618</td></tr>
</tbody>
</table>

</div>

</div>
</div>
<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">

<div markdown="0" class="output output_html">
<table>
<thead><tr><th scope=col>key</th><th scope=col>a</th><th scope=col>b</th><th scope=col>c</th><th scope=col>d</th><th scope=col>key</th><th scope=col>e</th><th scope=col>f</th><th scope=col>g</th><th scope=col>h</th></tr></thead>
<tbody>
	<tr><td> 1       </td><td>16.278799</td><td>23.004297</td><td>19.262524</td><td>22.11648 </td><td> 1       </td><td>1004.4714</td><td> 995.4055</td><td>1001.5156</td><td>1004.8862</td></tr>
	<tr><td> 2       </td><td>19.287252</td><td>19.229253</td><td>18.817575</td><td>13.67939 </td><td> 2       </td><td>1000.7399</td><td>1004.6248</td><td> 995.8950</td><td>1008.5242</td></tr>
	<tr><td> 3       </td><td>18.833623</td><td>16.142004</td><td>18.454224</td><td>18.29241 </td><td> 3       </td><td> 994.3794</td><td>1002.2578</td><td> 998.6004</td><td> 999.7609</td></tr>
	<tr><td> 4       </td><td>23.780847</td><td>10.934207</td><td>13.448540</td><td>17.83936 </td><td> 4       </td><td>1001.4795</td><td>1009.0885</td><td>1002.5866</td><td> 998.8287</td></tr>
	<tr><td> 5       </td><td>13.982935</td><td>16.924402</td><td>19.037475</td><td>20.19748 </td><td> 5       </td><td> 993.9745</td><td> 999.9868</td><td>1001.0336</td><td> 987.3751</td></tr>
	<tr><td> 6       </td><td>15.534589</td><td>23.437320</td><td> 6.795926</td><td>20.19305 </td><td> 6       </td><td> 996.1284</td><td>1008.8440</td><td>1005.5196</td><td>1003.6926</td></tr>
	<tr><td> 7       </td><td>16.660076</td><td>18.315077</td><td>32.107139</td><td>23.35534 </td><td> 7       </td><td> 994.5026</td><td>1004.9990</td><td>1004.0972</td><td>1005.6532</td></tr>
	<tr><td> 8       </td><td>19.447799</td><td>18.278384</td><td>11.823108</td><td>13.09162 </td><td> 8       </td><td>1007.8858</td><td> 993.8745</td><td>1005.1093</td><td> 996.8686</td></tr>
	<tr><td> 9       </td><td> 9.225069</td><td>24.925796</td><td>13.868021</td><td>17.06181 </td><td> 9       </td><td> 997.6026</td><td>1001.1045</td><td> 991.7969</td><td>1000.5898</td></tr>
	<tr><td>10       </td><td>25.809451</td><td> 7.492747</td><td>18.483003</td><td>24.99244 </td><td>10       </td><td> 995.6190</td><td>1010.2642</td><td> 998.6192</td><td> 998.8618</td></tr>
</tbody>
</table>

</div>

</div>
</div>
<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">

<div markdown="0" class="output output_html">
<table>
<thead><tr><th scope=col>key</th><th scope=col>a</th><th scope=col>b</th><th scope=col>c</th><th scope=col>d</th><th scope=col>key.1</th><th scope=col>e</th><th scope=col>f</th><th scope=col>g</th><th scope=col>h</th></tr></thead>
<tbody>
	<tr><td> 1       </td><td>16.278799</td><td>23.004297</td><td>19.262524</td><td>22.11648 </td><td> 1       </td><td>1004.4714</td><td> 995.4055</td><td>1001.5156</td><td>1004.8862</td></tr>
	<tr><td> 2       </td><td>19.287252</td><td>19.229253</td><td>18.817575</td><td>13.67939 </td><td> 2       </td><td>1000.7399</td><td>1004.6248</td><td> 995.8950</td><td>1008.5242</td></tr>
	<tr><td> 3       </td><td>18.833623</td><td>16.142004</td><td>18.454224</td><td>18.29241 </td><td> 3       </td><td> 994.3794</td><td>1002.2578</td><td> 998.6004</td><td> 999.7609</td></tr>
	<tr><td> 4       </td><td>23.780847</td><td>10.934207</td><td>13.448540</td><td>17.83936 </td><td> 4       </td><td>1001.4795</td><td>1009.0885</td><td>1002.5866</td><td> 998.8287</td></tr>
	<tr><td> 5       </td><td>13.982935</td><td>16.924402</td><td>19.037475</td><td>20.19748 </td><td> 5       </td><td> 993.9745</td><td> 999.9868</td><td>1001.0336</td><td> 987.3751</td></tr>
	<tr><td> 6       </td><td>15.534589</td><td>23.437320</td><td> 6.795926</td><td>20.19305 </td><td> 6       </td><td> 996.1284</td><td>1008.8440</td><td>1005.5196</td><td>1003.6926</td></tr>
	<tr><td> 7       </td><td>16.660076</td><td>18.315077</td><td>32.107139</td><td>23.35534 </td><td> 7       </td><td> 994.5026</td><td>1004.9990</td><td>1004.0972</td><td>1005.6532</td></tr>
	<tr><td> 8       </td><td>19.447799</td><td>18.278384</td><td>11.823108</td><td>13.09162 </td><td> 8       </td><td>1007.8858</td><td> 993.8745</td><td>1005.1093</td><td> 996.8686</td></tr>
	<tr><td> 9       </td><td> 9.225069</td><td>24.925796</td><td>13.868021</td><td>17.06181 </td><td> 9       </td><td> 997.6026</td><td>1001.1045</td><td> 991.7969</td><td>1000.5898</td></tr>
	<tr><td>10       </td><td>25.809451</td><td> 7.492747</td><td>18.483003</td><td>24.99244 </td><td>10       </td><td> 995.6190</td><td>1010.2642</td><td> 998.6192</td><td> 998.8618</td></tr>
</tbody>
</table>

</div>

</div>
</div>
</div>



## Merging Rows of Data Frame with another Data Frame
- `rbind(a,b)` combines rows of data frames of a and b.
- `rbind(a,b, make.row.names=FALSE)` this will reset the index.



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```R
#Here we can combine rows with rbind. 
df5<-df
#The make Row
df6<-rbind(df,df5)
df6
df7<-rbind(df,df5, make.row.names=FALSE)
df7

```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">

<div markdown="0" class="output output_html">
<table>
<thead><tr><th></th><th scope=col>key</th><th scope=col>a</th><th scope=col>b</th><th scope=col>c</th><th scope=col>d</th></tr></thead>
<tbody>
	<tr><th scope=row>1</th><td> 1       </td><td>16.19873 </td><td>14.41495 </td><td>27.73225 </td><td>15.564688</td></tr>
	<tr><th scope=row>2</th><td> 2       </td><td>27.76677 </td><td>18.54772 </td><td>21.06688 </td><td>15.697810</td></tr>
	<tr><th scope=row>3</th><td> 3       </td><td>11.68592 </td><td>14.91207 </td><td>22.82086 </td><td>15.666790</td></tr>
	<tr><th scope=row>4</th><td> 4       </td><td>16.39982 </td><td>28.30284 </td><td>10.97550 </td><td>19.083633</td></tr>
	<tr><th scope=row>5</th><td> 5       </td><td>22.16232 </td><td>16.82574 </td><td>14.28676 </td><td>20.162797</td></tr>
	<tr><th scope=row>6</th><td> 6       </td><td>17.17425 </td><td>14.36932 </td><td>18.55487 </td><td>13.498498</td></tr>
	<tr><th scope=row>7</th><td> 7       </td><td>20.15380 </td><td>18.00987 </td><td>15.99028 </td><td>14.325000</td></tr>
	<tr><th scope=row>8</th><td> 8       </td><td>20.68866 </td><td>12.83505 </td><td>25.24119 </td><td>24.538494</td></tr>
	<tr><th scope=row>9</th><td> 9       </td><td>18.84664 </td><td>24.01079 </td><td>12.69775 </td><td> 8.095156</td></tr>
	<tr><th scope=row>10</th><td>10       </td><td>16.29913 </td><td>21.51270 </td><td>15.14676 </td><td>23.722103</td></tr>
	<tr><th scope=row>11</th><td> 1       </td><td>16.19873 </td><td>14.41495 </td><td>27.73225 </td><td>15.564688</td></tr>
	<tr><th scope=row>21</th><td> 2       </td><td>27.76677 </td><td>18.54772 </td><td>21.06688 </td><td>15.697810</td></tr>
	<tr><th scope=row>31</th><td> 3       </td><td>11.68592 </td><td>14.91207 </td><td>22.82086 </td><td>15.666790</td></tr>
	<tr><th scope=row>41</th><td> 4       </td><td>16.39982 </td><td>28.30284 </td><td>10.97550 </td><td>19.083633</td></tr>
	<tr><th scope=row>51</th><td> 5       </td><td>22.16232 </td><td>16.82574 </td><td>14.28676 </td><td>20.162797</td></tr>
	<tr><th scope=row>61</th><td> 6       </td><td>17.17425 </td><td>14.36932 </td><td>18.55487 </td><td>13.498498</td></tr>
	<tr><th scope=row>71</th><td> 7       </td><td>20.15380 </td><td>18.00987 </td><td>15.99028 </td><td>14.325000</td></tr>
	<tr><th scope=row>81</th><td> 8       </td><td>20.68866 </td><td>12.83505 </td><td>25.24119 </td><td>24.538494</td></tr>
	<tr><th scope=row>91</th><td> 9       </td><td>18.84664 </td><td>24.01079 </td><td>12.69775 </td><td> 8.095156</td></tr>
	<tr><th scope=row>101</th><td>10       </td><td>16.29913 </td><td>21.51270 </td><td>15.14676 </td><td>23.722103</td></tr>
</tbody>
</table>

</div>

</div>
</div>
<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">

<div markdown="0" class="output output_html">
<table>
<thead><tr><th scope=col>key</th><th scope=col>a</th><th scope=col>b</th><th scope=col>c</th><th scope=col>d</th></tr></thead>
<tbody>
	<tr><td> 1       </td><td>16.19873 </td><td>14.41495 </td><td>27.73225 </td><td>15.564688</td></tr>
	<tr><td> 2       </td><td>27.76677 </td><td>18.54772 </td><td>21.06688 </td><td>15.697810</td></tr>
	<tr><td> 3       </td><td>11.68592 </td><td>14.91207 </td><td>22.82086 </td><td>15.666790</td></tr>
	<tr><td> 4       </td><td>16.39982 </td><td>28.30284 </td><td>10.97550 </td><td>19.083633</td></tr>
	<tr><td> 5       </td><td>22.16232 </td><td>16.82574 </td><td>14.28676 </td><td>20.162797</td></tr>
	<tr><td> 6       </td><td>17.17425 </td><td>14.36932 </td><td>18.55487 </td><td>13.498498</td></tr>
	<tr><td> 7       </td><td>20.15380 </td><td>18.00987 </td><td>15.99028 </td><td>14.325000</td></tr>
	<tr><td> 8       </td><td>20.68866 </td><td>12.83505 </td><td>25.24119 </td><td>24.538494</td></tr>
	<tr><td> 9       </td><td>18.84664 </td><td>24.01079 </td><td>12.69775 </td><td> 8.095156</td></tr>
	<tr><td>10       </td><td>16.29913 </td><td>21.51270 </td><td>15.14676 </td><td>23.722103</td></tr>
	<tr><td> 1       </td><td>16.19873 </td><td>14.41495 </td><td>27.73225 </td><td>15.564688</td></tr>
	<tr><td> 2       </td><td>27.76677 </td><td>18.54772 </td><td>21.06688 </td><td>15.697810</td></tr>
	<tr><td> 3       </td><td>11.68592 </td><td>14.91207 </td><td>22.82086 </td><td>15.666790</td></tr>
	<tr><td> 4       </td><td>16.39982 </td><td>28.30284 </td><td>10.97550 </td><td>19.083633</td></tr>
	<tr><td> 5       </td><td>22.16232 </td><td>16.82574 </td><td>14.28676 </td><td>20.162797</td></tr>
	<tr><td> 6       </td><td>17.17425 </td><td>14.36932 </td><td>18.55487 </td><td>13.498498</td></tr>
	<tr><td> 7       </td><td>20.15380 </td><td>18.00987 </td><td>15.99028 </td><td>14.325000</td></tr>
	<tr><td> 8       </td><td>20.68866 </td><td>12.83505 </td><td>25.24119 </td><td>24.538494</td></tr>
	<tr><td> 9       </td><td>18.84664 </td><td>24.01079 </td><td>12.69775 </td><td> 8.095156</td></tr>
	<tr><td>10       </td><td>16.29913 </td><td>21.51270 </td><td>15.14676 </td><td>23.722103</td></tr>
</tbody>
</table>

</div>

</div>
</div>
</div>



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```R
df7

```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">

<div markdown="0" class="output output_html">
<table>
<thead><tr><th scope=col>key</th><th scope=col>a</th><th scope=col>b</th><th scope=col>c</th><th scope=col>d</th></tr></thead>
<tbody>
	<tr><td> 1       </td><td>16.19873 </td><td>14.41495 </td><td>27.73225 </td><td>15.564688</td></tr>
	<tr><td> 2       </td><td>27.76677 </td><td>18.54772 </td><td>21.06688 </td><td>15.697810</td></tr>
	<tr><td> 3       </td><td>11.68592 </td><td>14.91207 </td><td>22.82086 </td><td>15.666790</td></tr>
	<tr><td> 4       </td><td>16.39982 </td><td>28.30284 </td><td>10.97550 </td><td>19.083633</td></tr>
	<tr><td> 5       </td><td>22.16232 </td><td>16.82574 </td><td>14.28676 </td><td>20.162797</td></tr>
	<tr><td> 6       </td><td>17.17425 </td><td>14.36932 </td><td>18.55487 </td><td>13.498498</td></tr>
	<tr><td> 7       </td><td>20.15380 </td><td>18.00987 </td><td>15.99028 </td><td>14.325000</td></tr>
	<tr><td> 8       </td><td>20.68866 </td><td>12.83505 </td><td>25.24119 </td><td>24.538494</td></tr>
	<tr><td> 9       </td><td>18.84664 </td><td>24.01079 </td><td>12.69775 </td><td> 8.095156</td></tr>
	<tr><td>10       </td><td>16.29913 </td><td>21.51270 </td><td>15.14676 </td><td>23.722103</td></tr>
	<tr><td> 1       </td><td>16.19873 </td><td>14.41495 </td><td>27.73225 </td><td>15.564688</td></tr>
	<tr><td> 2       </td><td>27.76677 </td><td>18.54772 </td><td>21.06688 </td><td>15.697810</td></tr>
	<tr><td> 3       </td><td>11.68592 </td><td>14.91207 </td><td>22.82086 </td><td>15.666790</td></tr>
	<tr><td> 4       </td><td>16.39982 </td><td>28.30284 </td><td>10.97550 </td><td>19.083633</td></tr>
	<tr><td> 5       </td><td>22.16232 </td><td>16.82574 </td><td>14.28676 </td><td>20.162797</td></tr>
	<tr><td> 6       </td><td>17.17425 </td><td>14.36932 </td><td>18.55487 </td><td>13.498498</td></tr>
	<tr><td> 7       </td><td>20.15380 </td><td>18.00987 </td><td>15.99028 </td><td>14.325000</td></tr>
	<tr><td> 8       </td><td>20.68866 </td><td>12.83505 </td><td>25.24119 </td><td>24.538494</td></tr>
	<tr><td> 9       </td><td>18.84664 </td><td>24.01079 </td><td>12.69775 </td><td> 8.095156</td></tr>
	<tr><td>10       </td><td>16.29913 </td><td>21.51270 </td><td>15.14676 </td><td>23.722103</td></tr>
</tbody>
</table>

</div>

</div>
</div>
</div>



## `aggregate` and `by`
- Aggregation is a very important function.
- Can have variables/analyses that happen at different levels.
- `by(x, by, FUN)` provides similar functionality.



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```R
iris=read.csv(file="../../input/iris.csv", header=TRUE,sep=",")
head(iris)

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
</tbody>
</table>

</div>

</div>
</div>
</div>



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```R
iris<-read.csv(file="../../input/iris.csv", header=TRUE,sep=",")

#Aggregate by Species  aggregate(x, by, FUN, ...)
iris.agg<-aggregate(iris[,1:4], by=list("species" = iris$species), mean)
print(iris.agg)

#Notice this gives us the same output but structured differently. 
by(iris[, 1:4], iris$species, colMeans)

```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">
{:.output_stream}
```
     species sepal_length sepal_width petal_length petal_width
1     setosa        5.006       3.418        1.464       0.244
2 versicolor        5.936       2.770        4.260       1.326
3  virginica        6.588       2.974        5.552       2.026
```
</div>
</div>
<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">
{:.output_data_text}
```
iris$species: setosa
sepal_length  sepal_width petal_length  petal_width 
       5.006        3.418        1.464        0.244 
------------------------------------------------------------ 
iris$species: versicolor
sepal_length  sepal_width petal_length  petal_width 
       5.936        2.770        4.260        1.326 
------------------------------------------------------------ 
iris$species: virginica
sepal_length  sepal_width petal_length  petal_width 
       6.588        2.974        5.552        2.026 
```

</div>
</div>
</div>



## `apply`(plus `lapply`/`sapply`/`tapply`/`rapply`)
- `apply` - Applying a function to **an array or matrix**, return a vector or array or list of values. `apply(X, MARGIN, FUN, ...)`
- [`lapply`](https://stat.ethz.ch/R-manual/R-devel/library/base/html/lapply.html) - Apply a function to **each element of a list or vector**, return a **list**. 
- [`sapply`](https://stat.ethz.ch/R-manual/R-devel/library/base/html/lapply.html) - A user-friendly version if `lapply`. Apply a function to **each element of a list or vector**, return a **vector**.
- `tapply` - Apply a function to **subsets of a vector** (and the subsets are defined by some other vector, usually a factor), return a **vector**. 
- `rapply` - Apply a function to **each element of a nested list structure, recursively,** return a list.
- Some functions aren't vectorized, or you may want to use a function on every row or column of a matrix/data frame, every element of a list, etc.
- For more info see this [tutorial](https://nsaunders.wordpress.com/2010/08/20/a-brief-introduction-to-apply-in-r/)




## `apply`
- `apply` - Applying a function to **an array or matrix**, return a vector or array or list of values. `apply(X, MARGIN, FUN, ...)`
- If you are using a data frame the data types must all be the same. 
- `apply(X, MARGIN, FUN, ...) where X is an array or matrix. 
- `MARGIN` is a vector giving the where function should be applied. E.g., for a matrix 1 indicates rows, 2 indicates columns, c(1, 2) indicates rows and columns.
- `FUN` is any function.  



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```R
iris<-read.csv(file="../../input/iris.csv", header=TRUE,sep=",")
iris$sum<-apply(iris[1:4], 1, sum) #This provides a sum across  for each row. 
iris$mean<-apply(iris[1:4], 1, mean)#This provides a mean across collumns for each row. 
head(iris)
apply(iris[1:4], 2, mean)

```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">

<div markdown="0" class="output output_html">
<table>
<thead><tr><th scope=col>sepal_length</th><th scope=col>sepal_width</th><th scope=col>petal_length</th><th scope=col>petal_width</th><th scope=col>species</th><th scope=col>sum</th><th scope=col>mean</th></tr></thead>
<tbody>
	<tr><td>5.1   </td><td>3.5   </td><td>1.4   </td><td>0.2   </td><td>setosa</td><td>10.2  </td><td>2.550 </td></tr>
	<tr><td>4.9   </td><td>3.0   </td><td>1.4   </td><td>0.2   </td><td>setosa</td><td> 9.5  </td><td>2.375 </td></tr>
	<tr><td>4.7   </td><td>3.2   </td><td>1.3   </td><td>0.2   </td><td>setosa</td><td> 9.4  </td><td>2.350 </td></tr>
	<tr><td>4.6   </td><td>3.1   </td><td>1.5   </td><td>0.2   </td><td>setosa</td><td> 9.4  </td><td>2.350 </td></tr>
	<tr><td>5.0   </td><td>3.6   </td><td>1.4   </td><td>0.2   </td><td>setosa</td><td>10.2  </td><td>2.550 </td></tr>
	<tr><td>5.4   </td><td>3.9   </td><td>1.7   </td><td>0.4   </td><td>setosa</td><td>11.4  </td><td>2.850 </td></tr>
</tbody>
</table>

</div>

</div>
</div>
<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">

<div markdown="0" class="output output_html">
<dl class=dl-horizontal>
	<dt>sepal_length</dt>
		<dd>5.84333333333333</dd>
	<dt>sepal_width</dt>
		<dd>3.054</dd>
	<dt>petal_length</dt>
		<dd>3.75866666666667</dd>
	<dt>petal_width</dt>
		<dd>1.19866666666667</dd>
</dl>

</div>

</div>
</div>
</div>



## `lapply` & `sapply`
- [`lapply`](https://stat.ethz.ch/R-manual/R-devel/library/base/html/lapply.html) - Apply a function to **each element of a list or vector**, return a **list**.
- `lapply(X, FUN, ...)`
- [`sapply`](https://stat.ethz.ch/R-manual/R-devel/library/base/html/lapply.html) - A user-friendly version if `lapply`. Apply a function to **each element of a list or vector**, return a **vector**.
- `sapply(X, FUN, ...)`



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```R
# create a list with 2 elements
sample <- list("count" = 1:5, "numbers" =5:10)

# sum each and return as a list. 
sample.sum<-lapply(sample, sum)

class(sample.sum)
print(c(sample.sum, sample.sum["numbers"],sample.sum["count"]))


```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">

<div markdown="0" class="output output_html">
'list'
</div>

</div>
</div>
<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">
{:.output_stream}
```
$count
[1] 15

$numbers
[1] 45

$numbers
[1] 45

$count
[1] 15

```
</div>
</div>
</div>



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```R
# create a list with 2 elements
sample <- list("count" = 1:5, "numbers" =5:10)

# sum each and return as a list. 
sample.sum<-sapply(sample, sum)

class(sample.sum)
print(c(sample.sum, sample.sum["numbers"],sample.sum["count"],sample.sum[["count"]]))

#Note the differenece between #sample.sum[["count"]] and sample.sum["count"]

```
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
  count numbers numbers   count         
     15      45      45      15      15 
```
</div>
</div>
</div>



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```R
# We can also utilize simple 
square<-function(x) x^2
square(1:5)

# We can use our own function here.     
sapply(1:10, square)

#We can also specify the function directly in sapply.
sapply(1:10, function(x) x^2)


```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">

<div markdown="0" class="output output_html">
<ol class=list-inline>
	<li>1</li>
	<li>4</li>
	<li>9</li>
	<li>16</li>
	<li>25</li>
</ol>

</div>

</div>
</div>
<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">

<div markdown="0" class="output output_html">
<ol class=list-inline>
	<li>1</li>
	<li>4</li>
	<li>9</li>
	<li>16</li>
	<li>25</li>
	<li>36</li>
	<li>49</li>
	<li>64</li>
	<li>81</li>
	<li>100</li>
</ol>

</div>

</div>
</div>
<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">

<div markdown="0" class="output output_html">
<ol class=list-inline>
	<li>1</li>
	<li>4</li>
	<li>9</li>
	<li>16</li>
	<li>25</li>
	<li>36</li>
	<li>49</li>
	<li>64</li>
	<li>81</li>
	<li>100</li>
</ol>

</div>

</div>
</div>
</div>



## `tapply`
- `tapply` - Apply a function to subsets of a vector (and the subsets are defined by some other vector, usually a factor), return a vector.
- Can do something similar to aggregate. 



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```R
#Tapply example
#tapply(X, INDEX, FUN, …) 
#X = a vector, INDEX = list of one or more factor, FUN = Function or operation that needs to be applied. 
iris<-read.csv(file="../../input/iris.csv", header=TRUE,sep=",")
iris.sepal_length.agg<-tapply(iris$sepal_length, iris$species, mean)
print(iris.sepal_length.agg)



```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">
{:.output_stream}
```
    setosa versicolor  virginica 
     5.006      5.936      6.588 
```
</div>
</div>
</div>



## CREDITS


Copyright [AnalyticsDojo](http://rpi.analyticsdojo.com) 2016.
This work is licensed under the [Creative Commons Attribution 4.0 International](https://creativecommons.org/licenses/by/4.0/) license agreement.
This work is adopted from the Berkley R Bootcamp.  

