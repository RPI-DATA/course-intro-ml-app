---
interact_link: content/notebooks/10-intro-r/09-titanic-intro.ipynb
kernel_name: ir
has_widgets: false
title: 'Titanic'
prev_page:
  url: /notebooks/10-intro-r/08-intro-r-tidyverse.html
  title: 'Tidyvere'
next_page:
  url: /notebooks/12-intro-modeling-2/01-matrix-regression-gradient-decent-python.html
  title: 'Regression - Matrix'
comment: "***PROGRAMMATICALLY GENERATED, DO NOT EDIT. SEE ORIGINAL FILES IN /content***"
---


[![AnalyticsDojo](https://github.com/rpi-techfundamentals/spring2019-materials/blob/master/fig/final-logo.png?raw=1)](http://www.analyticsdojo.com)
<center><h1>Introduction to R - Titanic Baseline </h1></center>
<center><h3><a href = 'http://rpi.analyticsdojo.com'>rpi.analyticsdojo.com</a></h3></center>



## Running Code using Kaggle Notebooks
- Kaggle utilizes Docker to create a fully functional environment for hosting competitions in data science.
- You could download/run kaggle/python docker image from [GitHub](https://github.com/kaggle/docker-python) and run it as an alternative to the standard Jupyter Stack for Data Science we have been using.
- Kaggle has created an incredible resource for learning analytics.  You can view a number of *toy* examples that can be used to understand data science and also compete in real problems faced by top companies. 



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```R
train <- read.csv('../../input/train.csv', stringsAsFactors = F)
test  <- read.csv('../../input/test.csv', stringsAsFactors = F)

```
</div>

</div>



## `train` and `test` set on Kaggle
- The `train` file contains a wide variety of information that might be useful in understanding whether they survived or not. It also includes a record as to whether they survived or not.
- The `test` file contains all of the columns of the first file except whether they survived. Our goal is to predict whether the individuals survived.



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```R
head(train)

```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">

<div markdown="0" class="output output_html">
<table>
<thead><tr><th scope=col>PassengerId</th><th scope=col>Survived</th><th scope=col>Pclass</th><th scope=col>Name</th><th scope=col>Sex</th><th scope=col>Age</th><th scope=col>SibSp</th><th scope=col>Parch</th><th scope=col>Ticket</th><th scope=col>Fare</th><th scope=col>Cabin</th><th scope=col>Embarked</th></tr></thead>
<tbody>
	<tr><td>1                                                  </td><td>0                                                  </td><td>3                                                  </td><td>Braund, Mr. Owen Harris                            </td><td>male                                               </td><td>22                                                 </td><td>1                                                  </td><td>0                                                  </td><td>A/5 21171                                          </td><td> 7.2500                                            </td><td>                                                   </td><td>S                                                  </td></tr>
	<tr><td>2                                                  </td><td>1                                                  </td><td>1                                                  </td><td>Cumings, Mrs. John Bradley (Florence Briggs Thayer)</td><td>female                                             </td><td>38                                                 </td><td>1                                                  </td><td>0                                                  </td><td>PC 17599                                           </td><td>71.2833                                            </td><td>C85                                                </td><td>C                                                  </td></tr>
	<tr><td>3                                                  </td><td>1                                                  </td><td>3                                                  </td><td>Heikkinen, Miss. Laina                             </td><td>female                                             </td><td>26                                                 </td><td>0                                                  </td><td>0                                                  </td><td>STON/O2. 3101282                                   </td><td> 7.9250                                            </td><td>                                                   </td><td>S                                                  </td></tr>
	<tr><td>4                                                  </td><td>1                                                  </td><td>1                                                  </td><td>Futrelle, Mrs. Jacques Heath (Lily May Peel)       </td><td>female                                             </td><td>35                                                 </td><td>1                                                  </td><td>0                                                  </td><td>113803                                             </td><td>53.1000                                            </td><td>C123                                               </td><td>S                                                  </td></tr>
	<tr><td>5                                                  </td><td>0                                                  </td><td>3                                                  </td><td>Allen, Mr. William Henry                           </td><td>male                                               </td><td>35                                                 </td><td>0                                                  </td><td>0                                                  </td><td>373450                                             </td><td> 8.0500                                            </td><td>                                                   </td><td>S                                                  </td></tr>
	<tr><td>6                                                  </td><td>0                                                  </td><td>3                                                  </td><td>Moran, Mr. James                                   </td><td>male                                               </td><td>NA                                                 </td><td>0                                                  </td><td>0                                                  </td><td>330877                                             </td><td> 8.4583                                            </td><td>                                                   </td><td>Q                                                  </td></tr>
</tbody>
</table>

</div>

</div>
</div>
</div>



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```R
head(test)

```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">

<div markdown="0" class="output output_html">
<table>
<thead><tr><th scope=col>PassengerId</th><th scope=col>Pclass</th><th scope=col>Name</th><th scope=col>Sex</th><th scope=col>Age</th><th scope=col>SibSp</th><th scope=col>Parch</th><th scope=col>Ticket</th><th scope=col>Fare</th><th scope=col>Cabin</th><th scope=col>Embarked</th></tr></thead>
<tbody>
	<tr><td>892                                         </td><td>3                                           </td><td>Kelly, Mr. James                            </td><td>male                                        </td><td>34.5                                        </td><td>0                                           </td><td>0                                           </td><td>330911                                      </td><td> 7.8292                                     </td><td>                                            </td><td>Q                                           </td></tr>
	<tr><td>893                                         </td><td>3                                           </td><td>Wilkes, Mrs. James (Ellen Needs)            </td><td>female                                      </td><td>47.0                                        </td><td>1                                           </td><td>0                                           </td><td>363272                                      </td><td> 7.0000                                     </td><td>                                            </td><td>S                                           </td></tr>
	<tr><td>894                                         </td><td>2                                           </td><td>Myles, Mr. Thomas Francis                   </td><td>male                                        </td><td>62.0                                        </td><td>0                                           </td><td>0                                           </td><td>240276                                      </td><td> 9.6875                                     </td><td>                                            </td><td>Q                                           </td></tr>
	<tr><td>895                                         </td><td>3                                           </td><td>Wirz, Mr. Albert                            </td><td>male                                        </td><td>27.0                                        </td><td>0                                           </td><td>0                                           </td><td>315154                                      </td><td> 8.6625                                     </td><td>                                            </td><td>S                                           </td></tr>
	<tr><td>896                                         </td><td>3                                           </td><td>Hirvonen, Mrs. Alexander (Helga E Lindqvist)</td><td>female                                      </td><td>22.0                                        </td><td>1                                           </td><td>1                                           </td><td>3101298                                     </td><td>12.2875                                     </td><td>                                            </td><td>S                                           </td></tr>
	<tr><td>897                                         </td><td>3                                           </td><td>Svensson, Mr. Johan Cervin                  </td><td>male                                        </td><td>14.0                                        </td><td>0                                           </td><td>0                                           </td><td>7538                                        </td><td> 9.2250                                     </td><td>                                            </td><td>S                                           </td></tr>
</tbody>
</table>

</div>

</div>
</div>
</div>



## Baseline Model: No Survivors
- The Titanic problem is one of classification, and often the simplest baseline of all 0/1 is an appropriate baseline.
- Even if you aren't familiar with the history of the tragedy, by checking out the [Wikipedia Page](https://en.wikipedia.org/wiki/RMS_Titanic) we can quickly see that the majority of people (68%) died.
- As a result, our baseline model will be for no survivors.



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```R
test["Survived"] <- 0

```
</div>

</div>



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```R
submission <- test[,c("PassengerId", "Survived")]

```
</div>

</div>



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```R
head(submission)

```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">

<div markdown="0" class="output output_html">
<table>
<thead><tr><th scope=col>PassengerId</th><th scope=col>Survived</th></tr></thead>
<tbody>
	<tr><td>892</td><td>0  </td></tr>
	<tr><td>893</td><td>0  </td></tr>
	<tr><td>894</td><td>0  </td></tr>
	<tr><td>895</td><td>0  </td></tr>
	<tr><td>896</td><td>0  </td></tr>
	<tr><td>897</td><td>0  </td></tr>
</tbody>
</table>

</div>

</div>
</div>
</div>



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```R
# Write the solution to file
write.csv(submission, file = 'nosurvivors.csv', row.names = F)

```
</div>

</div>



## The First Rule of Shipwrecks
- You may have seen it in a movie or read it in a novel, but [women and children first](https://en.wikipedia.org/wiki/Women_and_children_first) has at it's roots something that could provide our first model.
- Now let's recode the `Survived` column based on whether was a man or a woman.  
- We are using conditionals to *select* rows of interest (for example, where test['Sex'] == 'male') and recoding appropriate columns.



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```R
#Here we can code it as Survived, but if we do so we will overwrite our other prediction. 
#Instead, let's code it as PredGender

test[test$Sex == "male", "PredGender"] <- 0
test[test$Sex == "female", "PredGender"] <- 1

```
</div>

</div>



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```R
submission = test[,c("PassengerId", "PredGender")]
#This will Rename the survived column
names(submission)[2] <- "Survived"
head(submission)

```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">

<div markdown="0" class="output output_html">
<table>
<thead><tr><th scope=col>PassengerId</th><th scope=col>Survived</th></tr></thead>
<tbody>
	<tr><td>892</td><td>0  </td></tr>
	<tr><td>893</td><td>1  </td></tr>
	<tr><td>894</td><td>0  </td></tr>
	<tr><td>895</td><td>0  </td></tr>
	<tr><td>896</td><td>1  </td></tr>
	<tr><td>897</td><td>0  </td></tr>
</tbody>
</table>

</div>

</div>
</div>
</div>



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```R
names(submission)[2]<-"new"
submission

```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">

<div markdown="0" class="output output_html">
<table>
<thead><tr><th scope=col>PassengerId</th><th scope=col>new</th></tr></thead>
<tbody>
	<tr><td>892</td><td>0  </td></tr>
	<tr><td>893</td><td>1  </td></tr>
	<tr><td>894</td><td>0  </td></tr>
	<tr><td>895</td><td>0  </td></tr>
	<tr><td>896</td><td>1  </td></tr>
	<tr><td>897</td><td>0  </td></tr>
	<tr><td>898</td><td>1  </td></tr>
	<tr><td>899</td><td>0  </td></tr>
	<tr><td>900</td><td>1  </td></tr>
	<tr><td>901</td><td>0  </td></tr>
	<tr><td>902</td><td>0  </td></tr>
	<tr><td>903</td><td>0  </td></tr>
	<tr><td>904</td><td>1  </td></tr>
	<tr><td>905</td><td>0  </td></tr>
	<tr><td>906</td><td>1  </td></tr>
	<tr><td>907</td><td>1  </td></tr>
	<tr><td>908</td><td>0  </td></tr>
	<tr><td>909</td><td>0  </td></tr>
	<tr><td>910</td><td>1  </td></tr>
	<tr><td>911</td><td>1  </td></tr>
	<tr><td>912</td><td>0  </td></tr>
	<tr><td>913</td><td>0  </td></tr>
	<tr><td>914</td><td>1  </td></tr>
	<tr><td>915</td><td>0  </td></tr>
	<tr><td>916</td><td>1  </td></tr>
	<tr><td>917</td><td>0  </td></tr>
	<tr><td>918</td><td>1  </td></tr>
	<tr><td>919</td><td>0  </td></tr>
	<tr><td>920</td><td>0  </td></tr>
	<tr><td>921</td><td>0  </td></tr>
	<tr><td>⋮</td><td>⋮</td></tr>
	<tr><td>1280</td><td>0   </td></tr>
	<tr><td>1281</td><td>0   </td></tr>
	<tr><td>1282</td><td>0   </td></tr>
	<tr><td>1283</td><td>1   </td></tr>
	<tr><td>1284</td><td>0   </td></tr>
	<tr><td>1285</td><td>0   </td></tr>
	<tr><td>1286</td><td>0   </td></tr>
	<tr><td>1287</td><td>1   </td></tr>
	<tr><td>1288</td><td>0   </td></tr>
	<tr><td>1289</td><td>1   </td></tr>
	<tr><td>1290</td><td>0   </td></tr>
	<tr><td>1291</td><td>0   </td></tr>
	<tr><td>1292</td><td>1   </td></tr>
	<tr><td>1293</td><td>0   </td></tr>
	<tr><td>1294</td><td>1   </td></tr>
	<tr><td>1295</td><td>0   </td></tr>
	<tr><td>1296</td><td>0   </td></tr>
	<tr><td>1297</td><td>0   </td></tr>
	<tr><td>1298</td><td>0   </td></tr>
	<tr><td>1299</td><td>0   </td></tr>
	<tr><td>1300</td><td>1   </td></tr>
	<tr><td>1301</td><td>1   </td></tr>
	<tr><td>1302</td><td>1   </td></tr>
	<tr><td>1303</td><td>1   </td></tr>
	<tr><td>1304</td><td>1   </td></tr>
	<tr><td>1305</td><td>0   </td></tr>
	<tr><td>1306</td><td>1   </td></tr>
	<tr><td>1307</td><td>0   </td></tr>
	<tr><td>1308</td><td>0   </td></tr>
	<tr><td>1309</td><td>0   </td></tr>
</tbody>
</table>

</div>

</div>
</div>
</div>



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```R
write.csv(submission, file = 'womensurvive.csv', row.names = F)

```
</div>

</div>

