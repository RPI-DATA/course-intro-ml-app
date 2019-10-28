---
interact_link: content/notebooks/16-intro-nlp/04-what-cooking-python.ipynb
kernel_name: python3
has_widgets: false
title: 'What's Cooking Python'
prev_page:
  url: /notebooks/16-intro-nlp/03-scikit-learn-text.html
  title: 'Scikit Learn Text'
next_page:
  url: /notebooks/16-intro-nlp/05-bag-popcorn-bag-words.html
  title: 'Bag of Popcorn Bag of Words'
comment: "***PROGRAMMATICALLY GENERATED, DO NOT EDIT. SEE ORIGINAL FILES IN /content***"
---


[![AnalyticsDojo](https://github.com/rpi-techfundamentals/spring2019-materials/blob/master/fig/final-logo.png?raw=1)](http://rpi.analyticsdojo.com)
<center><h1> What's Cooking  in Python</h1></center>
<center><h3><a href = 'http://rpi.analyticsdojo.com'>rpi.analyticsdojo.com</a></h3></center>




This was adopted from. 
https://www.kaggle.com/manuelatadvice/whats-cooking/noname/code



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```#This imports a bunch of packages.  
import pandas as pd
import matplotlib.pyplot as plt
import nltk
from nltk.stem import WordNetLemmatizer
from collections import Counter
import json
from nltk.stem import WordNetLemmatizer
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
#from sklearn import grid_search



```
</div>

</div>



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```
#If you import the codes locally, this seems to cause some issues.  
import json
from urllib.request import urlopen

urltrain= 'https://raw.githubusercontent.com/RPI-Analytics/MGMT6963-2015/master/data/whatscooking/whatscookingtrain.json'
urltest = 'https://raw.githubusercontent.com/RPI-Analytics/MGMT6963-2015/master/data/whatscooking/whatscookingtest.json'


train = pd.read_json(urlopen(urltrain))
test = pd.read_json(urlopen(urltest))

```
</div>

</div>



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```#First we want to see the most popular cuisine for the naive model. 
train.groupby('cuisine').size()

```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">


{:.output_data_text}
```
cuisine
brazilian        467
british          804
cajun_creole    1546
chinese         2673
filipino         755
french          2646
greek           1175
indian          3003
irish            667
italian         7838
jamaican         526
japanese        1423
korean           830
mexican         6438
moroccan         821
russian          489
southern_us     4320
spanish          989
thai            1539
vietnamese       825
dtype: int64
```


</div>
</div>
</div>



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```#Here we write the most popular selection.  This is the baseline by which we will judge other models. 
test['cuisine']='italian'

```
</div>

</div>



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```#THis is a much more simple version that selects out the columns ID and cuisinte
submission=test[['id' ,  'cuisine' ]]
#This is a more complex method I showed that gives same.
#submission=pd.DataFrame(test.ix[:,['id' ,  'cuisine' ]])

```
</div>

</div>



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```#This outputs the file.
submission.to_csv("1_cookingSubmission.csv",index=False)


```
</div>

</div>



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```#So it seems there is some data we need to use the NLTK leemmatizer.  
stemmer = WordNetLemmatizer()
nltk.download('wordnet')

 

```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">
{:.output_stream}
```
[nltk_data] Downloading package wordnet to /root/nltk_data...
[nltk_data]   Unzipping corpora/wordnet.zip.
```
</div>
</div>
<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">


{:.output_data_text}
```
True
```


</div>
</div>
</div>



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```train

```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">



<div markdown="0" class="output output_html">
<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>cuisine</th>
      <th>id</th>
      <th>ingredients</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>greek</td>
      <td>10259</td>
      <td>[romaine lettuce, black olives, grape tomatoes...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>southern_us</td>
      <td>25693</td>
      <td>[plain flour, ground pepper, salt, tomatoes, g...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>filipino</td>
      <td>20130</td>
      <td>[eggs, pepper, salt, mayonaise, cooking oil, g...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>indian</td>
      <td>22213</td>
      <td>[water, vegetable oil, wheat, salt]</td>
    </tr>
    <tr>
      <th>4</th>
      <td>indian</td>
      <td>13162</td>
      <td>[black pepper, shallots, cornflour, cayenne pe...</td>
    </tr>
    <tr>
      <th>5</th>
      <td>jamaican</td>
      <td>6602</td>
      <td>[plain flour, sugar, butter, eggs, fresh ginge...</td>
    </tr>
    <tr>
      <th>6</th>
      <td>spanish</td>
      <td>42779</td>
      <td>[olive oil, salt, medium shrimp, pepper, garli...</td>
    </tr>
    <tr>
      <th>7</th>
      <td>italian</td>
      <td>3735</td>
      <td>[sugar, pistachio nuts, white almond bark, flo...</td>
    </tr>
    <tr>
      <th>8</th>
      <td>mexican</td>
      <td>16903</td>
      <td>[olive oil, purple onion, fresh pineapple, por...</td>
    </tr>
    <tr>
      <th>9</th>
      <td>italian</td>
      <td>12734</td>
      <td>[chopped tomatoes, fresh basil, garlic, extra-...</td>
    </tr>
    <tr>
      <th>10</th>
      <td>italian</td>
      <td>5875</td>
      <td>[pimentos, sweet pepper, dried oregano, olive ...</td>
    </tr>
    <tr>
      <th>11</th>
      <td>chinese</td>
      <td>45887</td>
      <td>[low sodium soy sauce, fresh ginger, dry musta...</td>
    </tr>
    <tr>
      <th>12</th>
      <td>italian</td>
      <td>2698</td>
      <td>[Italian parsley leaves, walnuts, hot red pepp...</td>
    </tr>
    <tr>
      <th>13</th>
      <td>mexican</td>
      <td>41995</td>
      <td>[ground cinnamon, fresh cilantro, chili powder...</td>
    </tr>
    <tr>
      <th>14</th>
      <td>italian</td>
      <td>31908</td>
      <td>[fresh parmesan cheese, butter, all-purpose fl...</td>
    </tr>
    <tr>
      <th>15</th>
      <td>indian</td>
      <td>24717</td>
      <td>[tumeric, vegetable stock, tomatoes, garam mas...</td>
    </tr>
    <tr>
      <th>16</th>
      <td>british</td>
      <td>34466</td>
      <td>[greek yogurt, lemon curd, confectioners sugar...</td>
    </tr>
    <tr>
      <th>17</th>
      <td>italian</td>
      <td>1420</td>
      <td>[italian seasoning, broiler-fryer chicken, may...</td>
    </tr>
    <tr>
      <th>18</th>
      <td>thai</td>
      <td>2941</td>
      <td>[sugar, hot chili, asian fish sauce, lime juice]</td>
    </tr>
    <tr>
      <th>19</th>
      <td>vietnamese</td>
      <td>8152</td>
      <td>[soy sauce, vegetable oil, red bell pepper, ch...</td>
    </tr>
    <tr>
      <th>20</th>
      <td>thai</td>
      <td>13121</td>
      <td>[pork loin, roasted peanuts, chopped cilantro ...</td>
    </tr>
    <tr>
      <th>21</th>
      <td>mexican</td>
      <td>40523</td>
      <td>[roma tomatoes, kosher salt, purple onion, jal...</td>
    </tr>
    <tr>
      <th>22</th>
      <td>southern_us</td>
      <td>40989</td>
      <td>[low-fat mayonnaise, pepper, salt, baking pota...</td>
    </tr>
    <tr>
      <th>23</th>
      <td>chinese</td>
      <td>29630</td>
      <td>[sesame seeds, red pepper, yellow peppers, wat...</td>
    </tr>
    <tr>
      <th>24</th>
      <td>italian</td>
      <td>49136</td>
      <td>[marinara sauce, flat leaf parsley, olive oil,...</td>
    </tr>
    <tr>
      <th>25</th>
      <td>chinese</td>
      <td>26705</td>
      <td>[sugar, lo mein noodles, salt, chicken broth, ...</td>
    </tr>
    <tr>
      <th>26</th>
      <td>cajun_creole</td>
      <td>27976</td>
      <td>[herbs, lemon juice, fresh tomatoes, paprika, ...</td>
    </tr>
    <tr>
      <th>27</th>
      <td>italian</td>
      <td>22087</td>
      <td>[ground black pepper, butter, sliced mushrooms...</td>
    </tr>
    <tr>
      <th>28</th>
      <td>chinese</td>
      <td>9197</td>
      <td>[green bell pepper, egg roll wrappers, sweet a...</td>
    </tr>
    <tr>
      <th>29</th>
      <td>mexican</td>
      <td>1299</td>
      <td>[flour tortillas, cheese, breakfast sausages, ...</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>39744</th>
      <td>greek</td>
      <td>5680</td>
      <td>[extra-virgin olive oil, oregano, potatoes, ga...</td>
    </tr>
    <tr>
      <th>39745</th>
      <td>spanish</td>
      <td>5511</td>
      <td>[quinoa, extra-virgin olive oil, fresh thyme l...</td>
    </tr>
    <tr>
      <th>39746</th>
      <td>indian</td>
      <td>32051</td>
      <td>[clove, bay leaves, ginger, chopped cilantro, ...</td>
    </tr>
    <tr>
      <th>39747</th>
      <td>moroccan</td>
      <td>5119</td>
      <td>[water, sugar, grated lemon zest, butter, pitt...</td>
    </tr>
    <tr>
      <th>39748</th>
      <td>italian</td>
      <td>9526</td>
      <td>[sea salt, pizza doughs, all-purpose flour, co...</td>
    </tr>
    <tr>
      <th>39749</th>
      <td>mexican</td>
      <td>45599</td>
      <td>[kosher salt, minced onion, tortilla chips, su...</td>
    </tr>
    <tr>
      <th>39750</th>
      <td>mexican</td>
      <td>49670</td>
      <td>[ground black pepper, chicken breasts, salsa, ...</td>
    </tr>
    <tr>
      <th>39751</th>
      <td>moroccan</td>
      <td>30735</td>
      <td>[olive oil, cayenne pepper, chopped cilantro f...</td>
    </tr>
    <tr>
      <th>39752</th>
      <td>southern_us</td>
      <td>5911</td>
      <td>[self rising flour, milk, white sugar, butter,...</td>
    </tr>
    <tr>
      <th>39753</th>
      <td>italian</td>
      <td>33294</td>
      <td>[rosemary sprigs, lemon zest, garlic cloves, g...</td>
    </tr>
    <tr>
      <th>39754</th>
      <td>vietnamese</td>
      <td>27082</td>
      <td>[jasmine rice, bay leaves, sticky rice, rotiss...</td>
    </tr>
    <tr>
      <th>39755</th>
      <td>indian</td>
      <td>36337</td>
      <td>[mint leaves, cilantro leaves, ghee, tomatoes,...</td>
    </tr>
    <tr>
      <th>39756</th>
      <td>mexican</td>
      <td>15508</td>
      <td>[vegetable oil, cinnamon sticks, water, all-pu...</td>
    </tr>
    <tr>
      <th>39757</th>
      <td>greek</td>
      <td>34331</td>
      <td>[red bell pepper, garlic cloves, extra-virgin ...</td>
    </tr>
    <tr>
      <th>39758</th>
      <td>greek</td>
      <td>47387</td>
      <td>[milk, salt, ground cayenne pepper, ground lam...</td>
    </tr>
    <tr>
      <th>39759</th>
      <td>korean</td>
      <td>12153</td>
      <td>[red chili peppers, sea salt, onions, water, c...</td>
    </tr>
    <tr>
      <th>39760</th>
      <td>southern_us</td>
      <td>41840</td>
      <td>[butter, large eggs, cornmeal, baking powder, ...</td>
    </tr>
    <tr>
      <th>39761</th>
      <td>chinese</td>
      <td>6487</td>
      <td>[honey, chicken breast halves, cilantro leaves...</td>
    </tr>
    <tr>
      <th>39762</th>
      <td>indian</td>
      <td>26646</td>
      <td>[curry powder, salt, chicken, water, vegetable...</td>
    </tr>
    <tr>
      <th>39763</th>
      <td>italian</td>
      <td>44798</td>
      <td>[fettuccine pasta, low-fat cream cheese, garli...</td>
    </tr>
    <tr>
      <th>39764</th>
      <td>mexican</td>
      <td>8089</td>
      <td>[chili powder, worcestershire sauce, celery, r...</td>
    </tr>
    <tr>
      <th>39765</th>
      <td>indian</td>
      <td>6153</td>
      <td>[coconut, unsweetened coconut milk, mint leave...</td>
    </tr>
    <tr>
      <th>39766</th>
      <td>irish</td>
      <td>25557</td>
      <td>[rutabaga, ham, thick-cut bacon, potatoes, fre...</td>
    </tr>
    <tr>
      <th>39767</th>
      <td>italian</td>
      <td>24348</td>
      <td>[low-fat sour cream, grated parmesan cheese, s...</td>
    </tr>
    <tr>
      <th>39768</th>
      <td>mexican</td>
      <td>7377</td>
      <td>[shredded cheddar cheese, crushed cheese crack...</td>
    </tr>
    <tr>
      <th>39769</th>
      <td>irish</td>
      <td>29109</td>
      <td>[light brown sugar, granulated sugar, butter, ...</td>
    </tr>
    <tr>
      <th>39770</th>
      <td>italian</td>
      <td>11462</td>
      <td>[KRAFT Zesty Italian Dressing, purple onion, b...</td>
    </tr>
    <tr>
      <th>39771</th>
      <td>irish</td>
      <td>2238</td>
      <td>[eggs, citrus fruit, raisins, sourdough starte...</td>
    </tr>
    <tr>
      <th>39772</th>
      <td>chinese</td>
      <td>41882</td>
      <td>[boneless chicken skinless thigh, minced garli...</td>
    </tr>
    <tr>
      <th>39773</th>
      <td>mexican</td>
      <td>2362</td>
      <td>[green chile, jalapeno chilies, onions, ground...</td>
    </tr>
  </tbody>
</table>
<p>39774 rows × 3 columns</p>
</div>
</div>


</div>
</div>
</div>



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```#We see this in a Python Solution. 
train['ingredients_clean_string1'] = [','.join(z).strip() for z in train['ingredients']] 

#We also know that we can do something similar though a Lambda function. 
strip = lambda x: ' , '.join(x).strip() 
#Finally, we call the function for name
train['ingredients_clean_string2'] = train['ingredients'].map(strip)

#Now that we used the lambda function, we can reuse this for the test dataset. 
test['ingredients_clean_string1'] = test['ingredients'].map(strip)
 


```
</div>

</div>



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```#We see this in one of the solutions.  We can reconstruct it in a way that makes it abit easier to follow, but I found when doing that it took forever.  

#To interpret this, read from right to left. 
train['ingredients_string1'] = [' '.join([WordNetLemmatizer().lemmatize(re.sub('[^A-Za-z]', ' ', line)) for line in lists]).strip() for lists in train['ingredients']]       
test['ingredients_string1'] = [' '.join([WordNetLemmatizer().lemmatize(re.sub('[^A-Za-z]', ' ', line)) for line in lists]).strip() for lists in test['ingredients']]       




```
</div>

</div>



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```train['ingredients_string1']

```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">


{:.output_data_text}
```
0        romaine lettuce black olives grape tomatoes ga...
1        plain flour ground pepper salt tomato ground b...
2        egg pepper salt mayonaise cooking oil green ch...
3                           water vegetable oil wheat salt
4        black pepper shallot cornflour cayenne pepper ...
5        plain flour sugar butter egg fresh ginger root...
6        olive oil salt medium shrimp pepper garlic cho...
7        sugar pistachio nuts white almond bark flour v...
8        olive oil purple onion fresh pineapple pork po...
9        chopped tomatoes fresh basil garlic extra virg...
10       pimento sweet pepper dried oregano olive oil g...
11       low sodium soy sauce fresh ginger dry mustard ...
12       Italian parsley leaves walnut hot red pepper f...
13       ground cinnamon fresh cilantro chili powder gr...
14       fresh parmesan cheese butter all purpose flour...
15       tumeric vegetable stock tomato garam masala na...
16       greek yogurt lemon curd confectioners sugar ra...
17       italian seasoning broiler fryer chicken mayona...
18             sugar hot chili asian fish sauce lime juice
19       soy sauce vegetable oil red bell pepper chicke...
20       pork loin roasted peanuts chopped cilantro fre...
21       roma tomatoes kosher salt purple onion jalapen...
22       low fat mayonnaise pepper salt baking potatoes...
23       sesame seeds red pepper yellow peppers water e...
24       marinara sauce flat leaf parsley olive oil lin...
25       sugar lo mein noodles salt chicken broth light...
26       herb lemon juice fresh tomatoes paprika mango ...
27       ground black pepper butter sliced mushrooms sh...
28       green bell pepper egg roll wrappers sweet and ...
29       flour tortillas cheese breakfast sausages larg...
                               ...                        
39744    extra virgin olive oil oregano potato garlic c...
39745    quinoa extra virgin olive oil fresh thyme leav...
39746    clove bay leaves ginger chopped cilantro groun...
39747    water sugar grated lemon zest butter pitted da...
39748    sea salt pizza doughs all purpose flour cornme...
39749    kosher salt minced onion tortilla chips sugar ...
39750    ground black pepper chicken breasts salsa ched...
39751    olive oil cayenne pepper chopped cilantro fres...
39752    self rising flour milk white sugar butter peac...
39753    rosemary sprigs lemon zest garlic cloves groun...
39754    jasmine rice bay leaves sticky rice rotisserie...
39755    mint leaves cilantro leaves ghee tomato cinnam...
39756    vegetable oil cinnamon sticks water all purpos...
39757    red bell pepper garlic cloves extra virgin oli...
39758    milk salt ground cayenne pepper ground lamb gr...
39759    red chili peppers sea salt onion water chilli ...
39760    butter large eggs cornmeal baking powder boili...
39761    honey chicken breast halves cilantro leaves ca...
39762    curry powder salt chicken water vegetable oil ...
39763    fettuccine pasta low fat cream cheese garlic n...
39764    chili powder worcestershire sauce celery red k...
39765    coconut unsweetened coconut milk mint leaves p...
39766    rutabaga ham thick cut bacon potato fresh pars...
39767    low fat sour cream grated parmesan cheese salt...
39768    shredded cheddar cheese crushed cheese cracker...
39769    light brown sugar granulated sugar butter warm...
39770    KRAFT Zesty Italian Dressing purple onion broc...
39771    egg citrus fruit raisin sourdough starter flou...
39772    boneless chicken skinless thigh minced garlic ...
39773    green chile jalapeno chilies onion ground blac...
Name: ingredients_string1, Length: 39774, dtype: object
```


</div>
</div>
</div>



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```ingredients = train['ingredients'].apply(lambda x:','.join(x))
ingredients

```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">


{:.output_data_text}
```
0        romaine lettuce,black olives,grape tomatoes,ga...
1        plain flour,ground pepper,salt,tomatoes,ground...
2        eggs,pepper,salt,mayonaise,cooking oil,green c...
3                           water,vegetable oil,wheat,salt
4        black pepper,shallots,cornflour,cayenne pepper...
5        plain flour,sugar,butter,eggs,fresh ginger roo...
6        olive oil,salt,medium shrimp,pepper,garlic,cho...
7        sugar,pistachio nuts,white almond bark,flour,v...
8        olive oil,purple onion,fresh pineapple,pork,po...
9        chopped tomatoes,fresh basil,garlic,extra-virg...
10       pimentos,sweet pepper,dried oregano,olive oil,...
11       low sodium soy sauce,fresh ginger,dry mustard,...
12       Italian parsley leaves,walnuts,hot red pepper ...
13       ground cinnamon,fresh cilantro,chili powder,gr...
14       fresh parmesan cheese,butter,all-purpose flour...
15       tumeric,vegetable stock,tomatoes,garam masala,...
16       greek yogurt,lemon curd,confectioners sugar,ra...
17       italian seasoning,broiler-fryer chicken,mayona...
18             sugar,hot chili,asian fish sauce,lime juice
19       soy sauce,vegetable oil,red bell pepper,chicke...
20       pork loin,roasted peanuts,chopped cilantro fre...
21       roma tomatoes,kosher salt,purple onion,jalapen...
22       low-fat mayonnaise,pepper,salt,baking potatoes...
23       sesame seeds,red pepper,yellow peppers,water,e...
24       marinara sauce,flat leaf parsley,olive oil,lin...
25       sugar,lo mein noodles,salt,chicken broth,light...
26       herbs,lemon juice,fresh tomatoes,paprika,mango...
27       ground black pepper,butter,sliced mushrooms,sh...
28       green bell pepper,egg roll wrappers,sweet and ...
29       flour tortillas,cheese,breakfast sausages,larg...
                               ...                        
39744    extra-virgin olive oil,oregano,potatoes,garlic...
39745    quinoa,extra-virgin olive oil,fresh thyme leav...
39746    clove,bay leaves,ginger,chopped cilantro,groun...
39747    water,sugar,grated lemon zest,butter,pitted da...
39748    sea salt,pizza doughs,all-purpose flour,cornme...
39749    kosher salt,minced onion,tortilla chips,sugar,...
39750    ground black pepper,chicken breasts,salsa,ched...
39751    olive oil,cayenne pepper,chopped cilantro fres...
39752    self rising flour,milk,white sugar,butter,peac...
39753    rosemary sprigs,lemon zest,garlic cloves,groun...
39754    jasmine rice,bay leaves,sticky rice,rotisserie...
39755    mint leaves,cilantro leaves,ghee,tomatoes,cinn...
39756    vegetable oil,cinnamon sticks,water,all-purpos...
39757    red bell pepper,garlic cloves,extra-virgin oli...
39758    milk,salt,ground cayenne pepper,ground lamb,gr...
39759    red chili peppers,sea salt,onions,water,chilli...
39760    butter,large eggs,cornmeal,baking powder,boili...
39761    honey,chicken breast halves,cilantro leaves,ca...
39762    curry powder,salt,chicken,water,vegetable oil,...
39763    fettuccine pasta,low-fat cream cheese,garlic,n...
39764    chili powder,worcestershire sauce,celery,red k...
39765    coconut,unsweetened coconut milk,mint leaves,p...
39766    rutabaga,ham,thick-cut bacon,potatoes,fresh pa...
39767    low-fat sour cream,grated parmesan cheese,salt...
39768    shredded cheddar cheese,crushed cheese cracker...
39769    light brown sugar,granulated sugar,butter,warm...
39770    KRAFT Zesty Italian Dressing,purple onion,broc...
39771    eggs,citrus fruit,raisins,sourdough starter,fl...
39772    boneless chicken skinless thigh,minced garlic,...
39773    green chile,jalapeno chilies,onions,ground bla...
Name: ingredients, Length: 39774, dtype: object
```


</div>
</div>
</div>



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```#Now we will create a corpus.
corpustr = train['ingredients_string1']
corpusts = test['ingredients_string1']
corpustr

```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">


{:.output_data_text}
```
0        romaine lettuce black olives grape tomatoes ga...
1        plain flour ground pepper salt tomato ground b...
2        egg pepper salt mayonaise cooking oil green ch...
3                           water vegetable oil wheat salt
4        black pepper shallot cornflour cayenne pepper ...
5        plain flour sugar butter egg fresh ginger root...
6        olive oil salt medium shrimp pepper garlic cho...
7        sugar pistachio nuts white almond bark flour v...
8        olive oil purple onion fresh pineapple pork po...
9        chopped tomatoes fresh basil garlic extra virg...
10       pimento sweet pepper dried oregano olive oil g...
11       low sodium soy sauce fresh ginger dry mustard ...
12       Italian parsley leaves walnut hot red pepper f...
13       ground cinnamon fresh cilantro chili powder gr...
14       fresh parmesan cheese butter all purpose flour...
15       tumeric vegetable stock tomato garam masala na...
16       greek yogurt lemon curd confectioners sugar ra...
17       italian seasoning broiler fryer chicken mayona...
18             sugar hot chili asian fish sauce lime juice
19       soy sauce vegetable oil red bell pepper chicke...
20       pork loin roasted peanuts chopped cilantro fre...
21       roma tomatoes kosher salt purple onion jalapen...
22       low fat mayonnaise pepper salt baking potatoes...
23       sesame seeds red pepper yellow peppers water e...
24       marinara sauce flat leaf parsley olive oil lin...
25       sugar lo mein noodles salt chicken broth light...
26       herb lemon juice fresh tomatoes paprika mango ...
27       ground black pepper butter sliced mushrooms sh...
28       green bell pepper egg roll wrappers sweet and ...
29       flour tortillas cheese breakfast sausages larg...
                               ...                        
39744    extra virgin olive oil oregano potato garlic c...
39745    quinoa extra virgin olive oil fresh thyme leav...
39746    clove bay leaves ginger chopped cilantro groun...
39747    water sugar grated lemon zest butter pitted da...
39748    sea salt pizza doughs all purpose flour cornme...
39749    kosher salt minced onion tortilla chips sugar ...
39750    ground black pepper chicken breasts salsa ched...
39751    olive oil cayenne pepper chopped cilantro fres...
39752    self rising flour milk white sugar butter peac...
39753    rosemary sprigs lemon zest garlic cloves groun...
39754    jasmine rice bay leaves sticky rice rotisserie...
39755    mint leaves cilantro leaves ghee tomato cinnam...
39756    vegetable oil cinnamon sticks water all purpos...
39757    red bell pepper garlic cloves extra virgin oli...
39758    milk salt ground cayenne pepper ground lamb gr...
39759    red chili peppers sea salt onion water chilli ...
39760    butter large eggs cornmeal baking powder boili...
39761    honey chicken breast halves cilantro leaves ca...
39762    curry powder salt chicken water vegetable oil ...
39763    fettuccine pasta low fat cream cheese garlic n...
39764    chili powder worcestershire sauce celery red k...
39765    coconut unsweetened coconut milk mint leaves p...
39766    rutabaga ham thick cut bacon potato fresh pars...
39767    low fat sour cream grated parmesan cheese salt...
39768    shredded cheddar cheese crushed cheese cracker...
39769    light brown sugar granulated sugar butter warm...
39770    KRAFT Zesty Italian Dressing purple onion broc...
39771    egg citrus fruit raisin sourdough starter flou...
39772    boneless chicken skinless thigh minced garlic ...
39773    green chile jalapeno chilies onion ground blac...
Name: ingredients_string1, Length: 39774, dtype: object
```


</div>
</div>
</div>



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```#http://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html
#You could develop an understanding based on each.  
vectorizertr = TfidfVectorizer(stop_words='english',
                             ngram_range = ( 1 , 1 ),analyzer="word", 
                             max_df = .57 , binary=False , token_pattern=r'\w+' , sublinear_tf=False)
vectorizerts = TfidfVectorizer(stop_words='english')

```
</div>

</div>



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```#Note that this doesn't work with the #todense option.  
tfidftr=vectorizertr.fit_transform(corpustr)
predictors_tr = tfidftr

```
</div>

</div>



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```#Note that this doesn't work with the #todense option.  This creates a matrix of predictors from the corpus. 
tfidfts=vectorizertr.transform(corpusts)
predictors_ts= tfidfts

```
</div>

</div>



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```#This is target variable.  
targets_tr = train['cuisine']


```
</div>

</div>



## Logistic Regression and Regularization.

- Regularlization can help us with the large matrix by adding a penalty for each parameter. 
- Finding out how much regularization via grid search (search across hyperparameters.)

http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html

```C : float, default: 1.0
Inverse of regularization strength; must be a positive float. Like in support vector machines, smaller values specify stronger regularization.```



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```#Logistic Regression. 
parameters = {'C':[1, 10]}
#clf = LinearSVC()
clf = LogisticRegression()



```
</div>

</div>



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```predictors_tr

```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">


{:.output_data_text}
```
<39774x2963 sparse matrix of type '<class 'numpy.float64'>'
	with 727921 stored elements in Compressed Sparse Row format>
```


</div>
</div>
</div>



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```from sklearn.model_selection import GridSearchCV
#This uses that associated paramters to search a grid space. 
classifier = GridSearchCV(clf, parameters)
classifier=classifier.fit(predictors_tr,targets_tr)



```
</div>

</div>



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```#This predicts the outcome for the test set. 
predictions=classifier.predict(predictors_ts)

```
</div>

</div>



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```#This adds it to the resulting dataframe. 
test['cuisine'] = predictions

```
</div>

</div>



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```#This creates the submision dataframe
submission2=test[['id' ,  'cuisine' ]]

```
</div>

</div>



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```#This outputs the file.
submission2.to_csv("2_logisticSubmission.csv",index=False)

```
</div>

</div>



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```from sklearn.ensemble import RandomForestClassifier 



```
</div>

</div>



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```# Create the random forest object which will include all the parameters
# for the fit
forest = RandomForestClassifier(n_estimators = 10)



```
</div>

</div>



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```# Fit the training data to the Survived labels and create the decision trees
forest = forest.fit(predictors_tr,targets_tr)



```
</div>

</div>



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```# Take the same decision trees and run it on the test data
predictions = forest.predict(predictors_ts)

```
</div>

</div>



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```#This adds it to the resulting dataframe. 
test['cuisine'] = predictions

```
</div>

</div>



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```#This creates the submision dataframe
submission2=test[['id' ,  'cuisine' ]]

```
</div>

</div>



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```ingredients = train['ingredients'].apply(lambda x:','.join(x))
ingredients
train

```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">



<div markdown="0" class="output output_html">
<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>cuisine</th>
      <th>id</th>
      <th>ingredients</th>
      <th>ingredients_clean_string1</th>
      <th>ingredients_clean_string2</th>
      <th>ingredients_string1</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>greek</td>
      <td>10259</td>
      <td>[romaine lettuce, black olives, grape tomatoes...</td>
      <td>romaine lettuce,black olives,grape tomatoes,ga...</td>
      <td>romaine lettuce , black olives , grape tomatoe...</td>
      <td>romaine lettuce black olives grape tomatoes ga...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>southern_us</td>
      <td>25693</td>
      <td>[plain flour, ground pepper, salt, tomatoes, g...</td>
      <td>plain flour,ground pepper,salt,tomatoes,ground...</td>
      <td>plain flour , ground pepper , salt , tomatoes ...</td>
      <td>plain flour ground pepper salt tomato ground b...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>filipino</td>
      <td>20130</td>
      <td>[eggs, pepper, salt, mayonaise, cooking oil, g...</td>
      <td>eggs,pepper,salt,mayonaise,cooking oil,green c...</td>
      <td>eggs , pepper , salt , mayonaise , cooking oil...</td>
      <td>egg pepper salt mayonaise cooking oil green ch...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>indian</td>
      <td>22213</td>
      <td>[water, vegetable oil, wheat, salt]</td>
      <td>water,vegetable oil,wheat,salt</td>
      <td>water , vegetable oil , wheat , salt</td>
      <td>water vegetable oil wheat salt</td>
    </tr>
    <tr>
      <th>4</th>
      <td>indian</td>
      <td>13162</td>
      <td>[black pepper, shallots, cornflour, cayenne pe...</td>
      <td>black pepper,shallots,cornflour,cayenne pepper...</td>
      <td>black pepper , shallots , cornflour , cayenne ...</td>
      <td>black pepper shallot cornflour cayenne pepper ...</td>
    </tr>
    <tr>
      <th>5</th>
      <td>jamaican</td>
      <td>6602</td>
      <td>[plain flour, sugar, butter, eggs, fresh ginge...</td>
      <td>plain flour,sugar,butter,eggs,fresh ginger roo...</td>
      <td>plain flour , sugar , butter , eggs , fresh gi...</td>
      <td>plain flour sugar butter egg fresh ginger root...</td>
    </tr>
    <tr>
      <th>6</th>
      <td>spanish</td>
      <td>42779</td>
      <td>[olive oil, salt, medium shrimp, pepper, garli...</td>
      <td>olive oil,salt,medium shrimp,pepper,garlic,cho...</td>
      <td>olive oil , salt , medium shrimp , pepper , ga...</td>
      <td>olive oil salt medium shrimp pepper garlic cho...</td>
    </tr>
    <tr>
      <th>7</th>
      <td>italian</td>
      <td>3735</td>
      <td>[sugar, pistachio nuts, white almond bark, flo...</td>
      <td>sugar,pistachio nuts,white almond bark,flour,v...</td>
      <td>sugar , pistachio nuts , white almond bark , f...</td>
      <td>sugar pistachio nuts white almond bark flour v...</td>
    </tr>
    <tr>
      <th>8</th>
      <td>mexican</td>
      <td>16903</td>
      <td>[olive oil, purple onion, fresh pineapple, por...</td>
      <td>olive oil,purple onion,fresh pineapple,pork,po...</td>
      <td>olive oil , purple onion , fresh pineapple , p...</td>
      <td>olive oil purple onion fresh pineapple pork po...</td>
    </tr>
    <tr>
      <th>9</th>
      <td>italian</td>
      <td>12734</td>
      <td>[chopped tomatoes, fresh basil, garlic, extra-...</td>
      <td>chopped tomatoes,fresh basil,garlic,extra-virg...</td>
      <td>chopped tomatoes , fresh basil , garlic , extr...</td>
      <td>chopped tomatoes fresh basil garlic extra virg...</td>
    </tr>
    <tr>
      <th>10</th>
      <td>italian</td>
      <td>5875</td>
      <td>[pimentos, sweet pepper, dried oregano, olive ...</td>
      <td>pimentos,sweet pepper,dried oregano,olive oil,...</td>
      <td>pimentos , sweet pepper , dried oregano , oliv...</td>
      <td>pimento sweet pepper dried oregano olive oil g...</td>
    </tr>
    <tr>
      <th>11</th>
      <td>chinese</td>
      <td>45887</td>
      <td>[low sodium soy sauce, fresh ginger, dry musta...</td>
      <td>low sodium soy sauce,fresh ginger,dry mustard,...</td>
      <td>low sodium soy sauce , fresh ginger , dry must...</td>
      <td>low sodium soy sauce fresh ginger dry mustard ...</td>
    </tr>
    <tr>
      <th>12</th>
      <td>italian</td>
      <td>2698</td>
      <td>[Italian parsley leaves, walnuts, hot red pepp...</td>
      <td>Italian parsley leaves,walnuts,hot red pepper ...</td>
      <td>Italian parsley leaves , walnuts , hot red pep...</td>
      <td>Italian parsley leaves walnut hot red pepper f...</td>
    </tr>
    <tr>
      <th>13</th>
      <td>mexican</td>
      <td>41995</td>
      <td>[ground cinnamon, fresh cilantro, chili powder...</td>
      <td>ground cinnamon,fresh cilantro,chili powder,gr...</td>
      <td>ground cinnamon , fresh cilantro , chili powde...</td>
      <td>ground cinnamon fresh cilantro chili powder gr...</td>
    </tr>
    <tr>
      <th>14</th>
      <td>italian</td>
      <td>31908</td>
      <td>[fresh parmesan cheese, butter, all-purpose fl...</td>
      <td>fresh parmesan cheese,butter,all-purpose flour...</td>
      <td>fresh parmesan cheese , butter , all-purpose f...</td>
      <td>fresh parmesan cheese butter all purpose flour...</td>
    </tr>
    <tr>
      <th>15</th>
      <td>indian</td>
      <td>24717</td>
      <td>[tumeric, vegetable stock, tomatoes, garam mas...</td>
      <td>tumeric,vegetable stock,tomatoes,garam masala,...</td>
      <td>tumeric , vegetable stock , tomatoes , garam m...</td>
      <td>tumeric vegetable stock tomato garam masala na...</td>
    </tr>
    <tr>
      <th>16</th>
      <td>british</td>
      <td>34466</td>
      <td>[greek yogurt, lemon curd, confectioners sugar...</td>
      <td>greek yogurt,lemon curd,confectioners sugar,ra...</td>
      <td>greek yogurt , lemon curd , confectioners suga...</td>
      <td>greek yogurt lemon curd confectioners sugar ra...</td>
    </tr>
    <tr>
      <th>17</th>
      <td>italian</td>
      <td>1420</td>
      <td>[italian seasoning, broiler-fryer chicken, may...</td>
      <td>italian seasoning,broiler-fryer chicken,mayona...</td>
      <td>italian seasoning , broiler-fryer chicken , ma...</td>
      <td>italian seasoning broiler fryer chicken mayona...</td>
    </tr>
    <tr>
      <th>18</th>
      <td>thai</td>
      <td>2941</td>
      <td>[sugar, hot chili, asian fish sauce, lime juice]</td>
      <td>sugar,hot chili,asian fish sauce,lime juice</td>
      <td>sugar , hot chili , asian fish sauce , lime juice</td>
      <td>sugar hot chili asian fish sauce lime juice</td>
    </tr>
    <tr>
      <th>19</th>
      <td>vietnamese</td>
      <td>8152</td>
      <td>[soy sauce, vegetable oil, red bell pepper, ch...</td>
      <td>soy sauce,vegetable oil,red bell pepper,chicke...</td>
      <td>soy sauce , vegetable oil , red bell pepper , ...</td>
      <td>soy sauce vegetable oil red bell pepper chicke...</td>
    </tr>
    <tr>
      <th>20</th>
      <td>thai</td>
      <td>13121</td>
      <td>[pork loin, roasted peanuts, chopped cilantro ...</td>
      <td>pork loin,roasted peanuts,chopped cilantro fre...</td>
      <td>pork loin , roasted peanuts , chopped cilantro...</td>
      <td>pork loin roasted peanuts chopped cilantro fre...</td>
    </tr>
    <tr>
      <th>21</th>
      <td>mexican</td>
      <td>40523</td>
      <td>[roma tomatoes, kosher salt, purple onion, jal...</td>
      <td>roma tomatoes,kosher salt,purple onion,jalapen...</td>
      <td>roma tomatoes , kosher salt , purple onion , j...</td>
      <td>roma tomatoes kosher salt purple onion jalapen...</td>
    </tr>
    <tr>
      <th>22</th>
      <td>southern_us</td>
      <td>40989</td>
      <td>[low-fat mayonnaise, pepper, salt, baking pota...</td>
      <td>low-fat mayonnaise,pepper,salt,baking potatoes...</td>
      <td>low-fat mayonnaise , pepper , salt , baking po...</td>
      <td>low fat mayonnaise pepper salt baking potatoes...</td>
    </tr>
    <tr>
      <th>23</th>
      <td>chinese</td>
      <td>29630</td>
      <td>[sesame seeds, red pepper, yellow peppers, wat...</td>
      <td>sesame seeds,red pepper,yellow peppers,water,e...</td>
      <td>sesame seeds , red pepper , yellow peppers , w...</td>
      <td>sesame seeds red pepper yellow peppers water e...</td>
    </tr>
    <tr>
      <th>24</th>
      <td>italian</td>
      <td>49136</td>
      <td>[marinara sauce, flat leaf parsley, olive oil,...</td>
      <td>marinara sauce,flat leaf parsley,olive oil,lin...</td>
      <td>marinara sauce , flat leaf parsley , olive oil...</td>
      <td>marinara sauce flat leaf parsley olive oil lin...</td>
    </tr>
    <tr>
      <th>25</th>
      <td>chinese</td>
      <td>26705</td>
      <td>[sugar, lo mein noodles, salt, chicken broth, ...</td>
      <td>sugar,lo mein noodles,salt,chicken broth,light...</td>
      <td>sugar , lo mein noodles , salt , chicken broth...</td>
      <td>sugar lo mein noodles salt chicken broth light...</td>
    </tr>
    <tr>
      <th>26</th>
      <td>cajun_creole</td>
      <td>27976</td>
      <td>[herbs, lemon juice, fresh tomatoes, paprika, ...</td>
      <td>herbs,lemon juice,fresh tomatoes,paprika,mango...</td>
      <td>herbs , lemon juice , fresh tomatoes , paprika...</td>
      <td>herb lemon juice fresh tomatoes paprika mango ...</td>
    </tr>
    <tr>
      <th>27</th>
      <td>italian</td>
      <td>22087</td>
      <td>[ground black pepper, butter, sliced mushrooms...</td>
      <td>ground black pepper,butter,sliced mushrooms,sh...</td>
      <td>ground black pepper , butter , sliced mushroom...</td>
      <td>ground black pepper butter sliced mushrooms sh...</td>
    </tr>
    <tr>
      <th>28</th>
      <td>chinese</td>
      <td>9197</td>
      <td>[green bell pepper, egg roll wrappers, sweet a...</td>
      <td>green bell pepper,egg roll wrappers,sweet and ...</td>
      <td>green bell pepper , egg roll wrappers , sweet ...</td>
      <td>green bell pepper egg roll wrappers sweet and ...</td>
    </tr>
    <tr>
      <th>29</th>
      <td>mexican</td>
      <td>1299</td>
      <td>[flour tortillas, cheese, breakfast sausages, ...</td>
      <td>flour tortillas,cheese,breakfast sausages,larg...</td>
      <td>flour tortillas , cheese , breakfast sausages ...</td>
      <td>flour tortillas cheese breakfast sausages larg...</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>39744</th>
      <td>greek</td>
      <td>5680</td>
      <td>[extra-virgin olive oil, oregano, potatoes, ga...</td>
      <td>extra-virgin olive oil,oregano,potatoes,garlic...</td>
      <td>extra-virgin olive oil , oregano , potatoes , ...</td>
      <td>extra virgin olive oil oregano potato garlic c...</td>
    </tr>
    <tr>
      <th>39745</th>
      <td>spanish</td>
      <td>5511</td>
      <td>[quinoa, extra-virgin olive oil, fresh thyme l...</td>
      <td>quinoa,extra-virgin olive oil,fresh thyme leav...</td>
      <td>quinoa , extra-virgin olive oil , fresh thyme ...</td>
      <td>quinoa extra virgin olive oil fresh thyme leav...</td>
    </tr>
    <tr>
      <th>39746</th>
      <td>indian</td>
      <td>32051</td>
      <td>[clove, bay leaves, ginger, chopped cilantro, ...</td>
      <td>clove,bay leaves,ginger,chopped cilantro,groun...</td>
      <td>clove , bay leaves , ginger , chopped cilantro...</td>
      <td>clove bay leaves ginger chopped cilantro groun...</td>
    </tr>
    <tr>
      <th>39747</th>
      <td>moroccan</td>
      <td>5119</td>
      <td>[water, sugar, grated lemon zest, butter, pitt...</td>
      <td>water,sugar,grated lemon zest,butter,pitted da...</td>
      <td>water , sugar , grated lemon zest , butter , p...</td>
      <td>water sugar grated lemon zest butter pitted da...</td>
    </tr>
    <tr>
      <th>39748</th>
      <td>italian</td>
      <td>9526</td>
      <td>[sea salt, pizza doughs, all-purpose flour, co...</td>
      <td>sea salt,pizza doughs,all-purpose flour,cornme...</td>
      <td>sea salt , pizza doughs , all-purpose flour , ...</td>
      <td>sea salt pizza doughs all purpose flour cornme...</td>
    </tr>
    <tr>
      <th>39749</th>
      <td>mexican</td>
      <td>45599</td>
      <td>[kosher salt, minced onion, tortilla chips, su...</td>
      <td>kosher salt,minced onion,tortilla chips,sugar,...</td>
      <td>kosher salt , minced onion , tortilla chips , ...</td>
      <td>kosher salt minced onion tortilla chips sugar ...</td>
    </tr>
    <tr>
      <th>39750</th>
      <td>mexican</td>
      <td>49670</td>
      <td>[ground black pepper, chicken breasts, salsa, ...</td>
      <td>ground black pepper,chicken breasts,salsa,ched...</td>
      <td>ground black pepper , chicken breasts , salsa ...</td>
      <td>ground black pepper chicken breasts salsa ched...</td>
    </tr>
    <tr>
      <th>39751</th>
      <td>moroccan</td>
      <td>30735</td>
      <td>[olive oil, cayenne pepper, chopped cilantro f...</td>
      <td>olive oil,cayenne pepper,chopped cilantro fres...</td>
      <td>olive oil , cayenne pepper , chopped cilantro ...</td>
      <td>olive oil cayenne pepper chopped cilantro fres...</td>
    </tr>
    <tr>
      <th>39752</th>
      <td>southern_us</td>
      <td>5911</td>
      <td>[self rising flour, milk, white sugar, butter,...</td>
      <td>self rising flour,milk,white sugar,butter,peac...</td>
      <td>self rising flour , milk , white sugar , butte...</td>
      <td>self rising flour milk white sugar butter peac...</td>
    </tr>
    <tr>
      <th>39753</th>
      <td>italian</td>
      <td>33294</td>
      <td>[rosemary sprigs, lemon zest, garlic cloves, g...</td>
      <td>rosemary sprigs,lemon zest,garlic cloves,groun...</td>
      <td>rosemary sprigs , lemon zest , garlic cloves ,...</td>
      <td>rosemary sprigs lemon zest garlic cloves groun...</td>
    </tr>
    <tr>
      <th>39754</th>
      <td>vietnamese</td>
      <td>27082</td>
      <td>[jasmine rice, bay leaves, sticky rice, rotiss...</td>
      <td>jasmine rice,bay leaves,sticky rice,rotisserie...</td>
      <td>jasmine rice , bay leaves , sticky rice , roti...</td>
      <td>jasmine rice bay leaves sticky rice rotisserie...</td>
    </tr>
    <tr>
      <th>39755</th>
      <td>indian</td>
      <td>36337</td>
      <td>[mint leaves, cilantro leaves, ghee, tomatoes,...</td>
      <td>mint leaves,cilantro leaves,ghee,tomatoes,cinn...</td>
      <td>mint leaves , cilantro leaves , ghee , tomatoe...</td>
      <td>mint leaves cilantro leaves ghee tomato cinnam...</td>
    </tr>
    <tr>
      <th>39756</th>
      <td>mexican</td>
      <td>15508</td>
      <td>[vegetable oil, cinnamon sticks, water, all-pu...</td>
      <td>vegetable oil,cinnamon sticks,water,all-purpos...</td>
      <td>vegetable oil , cinnamon sticks , water , all-...</td>
      <td>vegetable oil cinnamon sticks water all purpos...</td>
    </tr>
    <tr>
      <th>39757</th>
      <td>greek</td>
      <td>34331</td>
      <td>[red bell pepper, garlic cloves, extra-virgin ...</td>
      <td>red bell pepper,garlic cloves,extra-virgin oli...</td>
      <td>red bell pepper , garlic cloves , extra-virgin...</td>
      <td>red bell pepper garlic cloves extra virgin oli...</td>
    </tr>
    <tr>
      <th>39758</th>
      <td>greek</td>
      <td>47387</td>
      <td>[milk, salt, ground cayenne pepper, ground lam...</td>
      <td>milk,salt,ground cayenne pepper,ground lamb,gr...</td>
      <td>milk , salt , ground cayenne pepper , ground l...</td>
      <td>milk salt ground cayenne pepper ground lamb gr...</td>
    </tr>
    <tr>
      <th>39759</th>
      <td>korean</td>
      <td>12153</td>
      <td>[red chili peppers, sea salt, onions, water, c...</td>
      <td>red chili peppers,sea salt,onions,water,chilli...</td>
      <td>red chili peppers , sea salt , onions , water ...</td>
      <td>red chili peppers sea salt onion water chilli ...</td>
    </tr>
    <tr>
      <th>39760</th>
      <td>southern_us</td>
      <td>41840</td>
      <td>[butter, large eggs, cornmeal, baking powder, ...</td>
      <td>butter,large eggs,cornmeal,baking powder,boili...</td>
      <td>butter , large eggs , cornmeal , baking powder...</td>
      <td>butter large eggs cornmeal baking powder boili...</td>
    </tr>
    <tr>
      <th>39761</th>
      <td>chinese</td>
      <td>6487</td>
      <td>[honey, chicken breast halves, cilantro leaves...</td>
      <td>honey,chicken breast halves,cilantro leaves,ca...</td>
      <td>honey , chicken breast halves , cilantro leave...</td>
      <td>honey chicken breast halves cilantro leaves ca...</td>
    </tr>
    <tr>
      <th>39762</th>
      <td>indian</td>
      <td>26646</td>
      <td>[curry powder, salt, chicken, water, vegetable...</td>
      <td>curry powder,salt,chicken,water,vegetable oil,...</td>
      <td>curry powder , salt , chicken , water , vegeta...</td>
      <td>curry powder salt chicken water vegetable oil ...</td>
    </tr>
    <tr>
      <th>39763</th>
      <td>italian</td>
      <td>44798</td>
      <td>[fettuccine pasta, low-fat cream cheese, garli...</td>
      <td>fettuccine pasta,low-fat cream cheese,garlic,n...</td>
      <td>fettuccine pasta , low-fat cream cheese , garl...</td>
      <td>fettuccine pasta low fat cream cheese garlic n...</td>
    </tr>
    <tr>
      <th>39764</th>
      <td>mexican</td>
      <td>8089</td>
      <td>[chili powder, worcestershire sauce, celery, r...</td>
      <td>chili powder,worcestershire sauce,celery,red k...</td>
      <td>chili powder , worcestershire sauce , celery ,...</td>
      <td>chili powder worcestershire sauce celery red k...</td>
    </tr>
    <tr>
      <th>39765</th>
      <td>indian</td>
      <td>6153</td>
      <td>[coconut, unsweetened coconut milk, mint leave...</td>
      <td>coconut,unsweetened coconut milk,mint leaves,p...</td>
      <td>coconut , unsweetened coconut milk , mint leav...</td>
      <td>coconut unsweetened coconut milk mint leaves p...</td>
    </tr>
    <tr>
      <th>39766</th>
      <td>irish</td>
      <td>25557</td>
      <td>[rutabaga, ham, thick-cut bacon, potatoes, fre...</td>
      <td>rutabaga,ham,thick-cut bacon,potatoes,fresh pa...</td>
      <td>rutabaga , ham , thick-cut bacon , potatoes , ...</td>
      <td>rutabaga ham thick cut bacon potato fresh pars...</td>
    </tr>
    <tr>
      <th>39767</th>
      <td>italian</td>
      <td>24348</td>
      <td>[low-fat sour cream, grated parmesan cheese, s...</td>
      <td>low-fat sour cream,grated parmesan cheese,salt...</td>
      <td>low-fat sour cream , grated parmesan cheese , ...</td>
      <td>low fat sour cream grated parmesan cheese salt...</td>
    </tr>
    <tr>
      <th>39768</th>
      <td>mexican</td>
      <td>7377</td>
      <td>[shredded cheddar cheese, crushed cheese crack...</td>
      <td>shredded cheddar cheese,crushed cheese cracker...</td>
      <td>shredded cheddar cheese , crushed cheese crack...</td>
      <td>shredded cheddar cheese crushed cheese cracker...</td>
    </tr>
    <tr>
      <th>39769</th>
      <td>irish</td>
      <td>29109</td>
      <td>[light brown sugar, granulated sugar, butter, ...</td>
      <td>light brown sugar,granulated sugar,butter,warm...</td>
      <td>light brown sugar , granulated sugar , butter ...</td>
      <td>light brown sugar granulated sugar butter warm...</td>
    </tr>
    <tr>
      <th>39770</th>
      <td>italian</td>
      <td>11462</td>
      <td>[KRAFT Zesty Italian Dressing, purple onion, b...</td>
      <td>KRAFT Zesty Italian Dressing,purple onion,broc...</td>
      <td>KRAFT Zesty Italian Dressing , purple onion , ...</td>
      <td>KRAFT Zesty Italian Dressing purple onion broc...</td>
    </tr>
    <tr>
      <th>39771</th>
      <td>irish</td>
      <td>2238</td>
      <td>[eggs, citrus fruit, raisins, sourdough starte...</td>
      <td>eggs,citrus fruit,raisins,sourdough starter,fl...</td>
      <td>eggs , citrus fruit , raisins , sourdough star...</td>
      <td>egg citrus fruit raisin sourdough starter flou...</td>
    </tr>
    <tr>
      <th>39772</th>
      <td>chinese</td>
      <td>41882</td>
      <td>[boneless chicken skinless thigh, minced garli...</td>
      <td>boneless chicken skinless thigh,minced garlic,...</td>
      <td>boneless chicken skinless thigh , minced garli...</td>
      <td>boneless chicken skinless thigh minced garlic ...</td>
    </tr>
    <tr>
      <th>39773</th>
      <td>mexican</td>
      <td>2362</td>
      <td>[green chile, jalapeno chilies, onions, ground...</td>
      <td>green chile,jalapeno chilies,onions,ground bla...</td>
      <td>green chile , jalapeno chilies , onions , grou...</td>
      <td>green chile jalapeno chilies onion ground blac...</td>
    </tr>
  </tbody>
</table>
<p>39774 rows × 6 columns</p>
</div>
</div>


</div>
</div>
</div>



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```#What we really need is a ingredient freqency, inverse cuisine frequency.  
#Let's first create a function to create a count
def tf(count, N):
        return count / float(N)

#Now let's create a function for IDF....
def idf(count, N ):

    # tf-idf calc involves multiplying against a tf value less than 0, so it's
    # necessary to return a value greater than 1 for consistent scoring. 
    # (Multiplying two values less than 1 returns a value less than each of 
    # them.)

    try:
        return 1.0 + log(float(N) / count)
    except ZeroDivisionError:
        return 1.0
    
    

```
</div>

</div>



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```from collections import Counter

#This creates a Corpus for each incredient
corpus=','.join(x.strip() for x in train['ingredients_clean_string1']).split(',')
total=Counter(corpus)
#This creates a corpus for each word
corpus2=','.join(x.strip() for x in train['ingredients_string1']).split(' ')
#This just does some extra cleaning....because of some issues with file. 
corpus3=','.join(x.strip() for x in corpus2).split(',')
total2=Counter(corpus3)


```
</div>

</div>



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```print(len(train))
countcuisines=Counter(train['cuisine'])
countcuisines

```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">
{:.output_stream}
```
39774
```
</div>
</div>
<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">


{:.output_data_text}
```
Counter({'brazilian': 467,
         'british': 804,
         'cajun_creole': 1546,
         'chinese': 2673,
         'filipino': 755,
         'french': 2646,
         'greek': 1175,
         'indian': 3003,
         'irish': 667,
         'italian': 7838,
         'jamaican': 526,
         'japanese': 1423,
         'korean': 830,
         'mexican': 6438,
         'moroccan': 821,
         'russian': 489,
         'southern_us': 4320,
         'spanish': 989,
         'thai': 1539,
         'vietnamese': 825})
```


</div>
</div>
</div>



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```# Now we want to create a function that can take in a dictionary [a lookup that has terms and values] and 
# score different recipes by the lookup.

def score(dictionary, stringx, delimiter):
    sumt=sum((dictionary.get(x,0) for x in stringx.split(delimiter)))
    return  sumt/len(stringx.split(delimiter))

#Alt.   
#    sumt=0
#    for x in stringx.split(delimiter):
#        sumt+=dictionary[x]
    
testdic={'salt': .5, 'vanilla': .2, 'butter': .3,'sugar': .1}
teststring='salt,vanilla,sugar,stink'
teststring2='salt vanilla sugar butter stink'
print(score(testdic,teststring,','))
print(score(testdic,teststring2,' '))
#print(score(testdic,teststring,' '))
print("Test1 Value: ", (.5+.2+.1)/4)
print("Test2 Value: ", (.5+.2+.1+.3)/5)


#Now we are using that separate function in another function.  
#title_fn = lambda x: 1 if has_title(x) else 0
#Finally, we call the function for name
#train['Title'] = train['Name'].map(title_fn)

```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">
{:.output_stream}
```
0.19999999999999998
0.21999999999999997
Test1 Value:  0.19999999999999998
Test2 Value:  0.21999999999999997
```
</div>
</div>
</div>



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```from math import log
trainc = pd.DataFrame()
trainc2 = pd.DataFrame()
testc = pd.DataFrame()
testc2 = pd.DataFrame()

#This will loop through each unique cuisine
for cuisine in train['cuisine'].unique():
#This selects only rows that are greek.

#Number of rows in the cuisine
    cuisinerows=train[train['cuisine'] == cuisine]
    notcuisinerows=train[train['cuisine'] != cuisine]
    
    #print (cuisinerows)
    
    #','.join(x.strip() for x in line.split(':'))
    
    #This looks at specific ingredients in the cuisine corpus
    cuisinecorpus=','.join(x.strip() for x in cuisinerows['ingredients_clean_string1']).split(',')
    #This looks at specific ingredients not in cuisine
    notcuisinecorpus=','.join(x.strip() for x in notcuisinerows['ingredients_clean_string1']).split(',')
   
    #This treats all words individually in cuisine
    cuisinecorpus2=','.join(x.strip() for x in cuisinerows['ingredients_string1']).split(' ')
    notcuisinecorpus2=','.join(x.strip() for x in notcuisinerows['ingredients_string1']).split(' ')
    #this extra line is just some additional cuisine
    cuisinecorpus2=','.join(x.strip() for x in cuisinecorpus2).split(',')
    notcuisinecorpus2=','.join(x.strip() for x in cuisinecorpus2).split(',')
    
    #This creates the document term matrix for each.
    tfcuisine=Counter(cuisinecorpus)
    tfnotcuisine=Counter(notcuisinecorpus)
    
    tfcuisine2=Counter(cuisinecorpus2)
    tfnotcuisine2=Counter(notcuisinecorpus2)
     
    #We didn't talk about it much in class, but this is creating a dict that indicates TFIDF for each ingredient
  
    cfincf={k: (tf(tfcuisine[k],len(cuisinecorpus))*idf(tfnotcuisine[k],len(notcuisinecorpus)))  for k in  tfcuisine.keys()}    
    
    #We didn't talk about it much in class, but this is creating a dict that indicates TFIDF for each word
    cfincf2={k: (tf(tfcuisine2[k],len(cuisinecorpus2))*idf(tfnotcuisine2[k],len(notcuisinecorpus2)))  for k in  tfcuisine2.keys()}    
    
    #Now we will use our strings to score each outcome. 
    score_fn = lambda x: score(cfincf, x, ',') 
    score_fn2 = lambda x: score(cfincf2, x, ' ')
    
#Finally, we call the function for name
    trainc[cuisine] = train['ingredients_clean_string1'].map(score_fn)
    trainc2[cuisine] = train['ingredients_clean_string1'].map(score_fn2)
    testc[cuisine] = test['ingredients_clean_string1'].map(score_fn)
    testc2[cuisine] = test['ingredients_clean_string1'].map(score_fn2)
    
    

```
</div>

</div>



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```trainc 

```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">



<div markdown="0" class="output output_html">
<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>greek</th>
      <th>southern_us</th>
      <th>filipino</th>
      <th>indian</th>
      <th>jamaican</th>
      <th>spanish</th>
      <th>italian</th>
      <th>mexican</th>
      <th>chinese</th>
      <th>british</th>
      <th>thai</th>
      <th>vietnamese</th>
      <th>cajun_creole</th>
      <th>brazilian</th>
      <th>french</th>
      <th>japanese</th>
      <th>irish</th>
      <th>korean</th>
      <th>moroccan</th>
      <th>russian</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.063402</td>
      <td>0.014313</td>
      <td>0.042993</td>
      <td>0.018555</td>
      <td>0.027734</td>
      <td>0.021350</td>
      <td>0.025247</td>
      <td>0.028564</td>
      <td>0.018063</td>
      <td>0.010681</td>
      <td>0.019183</td>
      <td>0.021842</td>
      <td>0.020400</td>
      <td>0.019667</td>
      <td>0.013159</td>
      <td>0.010634</td>
      <td>0.014112</td>
      <td>0.026451</td>
      <td>0.022275</td>
      <td>0.011072</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.043487</td>
      <td>0.058533</td>
      <td>0.049211</td>
      <td>0.041847</td>
      <td>0.054144</td>
      <td>0.043117</td>
      <td>0.040338</td>
      <td>0.036738</td>
      <td>0.029228</td>
      <td>0.062378</td>
      <td>0.022971</td>
      <td>0.025563</td>
      <td>0.039871</td>
      <td>0.047792</td>
      <td>0.040042</td>
      <td>0.033356</td>
      <td>0.055036</td>
      <td>0.028616</td>
      <td>0.031038</td>
      <td>0.058220</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.034594</td>
      <td>0.054670</td>
      <td>0.073175</td>
      <td>0.037089</td>
      <td>0.048051</td>
      <td>0.030905</td>
      <td>0.035124</td>
      <td>0.033269</td>
      <td>0.044232</td>
      <td>0.051751</td>
      <td>0.024760</td>
      <td>0.028254</td>
      <td>0.038588</td>
      <td>0.037735</td>
      <td>0.035919</td>
      <td>0.045915</td>
      <td>0.054855</td>
      <td>0.047717</td>
      <td>0.025595</td>
      <td>0.052095</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.068897</td>
      <td>0.093675</td>
      <td>0.127790</td>
      <td>0.102428</td>
      <td>0.104767</td>
      <td>0.071939</td>
      <td>0.067849</td>
      <td>0.072040</td>
      <td>0.086195</td>
      <td>0.082342</td>
      <td>0.066098</td>
      <td>0.073295</td>
      <td>0.075210</td>
      <td>0.083617</td>
      <td>0.075126</td>
      <td>0.091498</td>
      <td>0.092381</td>
      <td>0.078978</td>
      <td>0.067758</td>
      <td>0.107309</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.031048</td>
      <td>0.040054</td>
      <td>0.047440</td>
      <td>0.064445</td>
      <td>0.039515</td>
      <td>0.031350</td>
      <td>0.027690</td>
      <td>0.036956</td>
      <td>0.022852</td>
      <td>0.042136</td>
      <td>0.020566</td>
      <td>0.023596</td>
      <td>0.038502</td>
      <td>0.036080</td>
      <td>0.035890</td>
      <td>0.029748</td>
      <td>0.042737</td>
      <td>0.023796</td>
      <td>0.041862</td>
      <td>0.042769</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0.035193</td>
      <td>0.083304</td>
      <td>0.044253</td>
      <td>0.035216</td>
      <td>0.051046</td>
      <td>0.033618</td>
      <td>0.035672</td>
      <td>0.025472</td>
      <td>0.032754</td>
      <td>0.091074</td>
      <td>0.020535</td>
      <td>0.028287</td>
      <td>0.032436</td>
      <td>0.042970</td>
      <td>0.054128</td>
      <td>0.037862</td>
      <td>0.081313</td>
      <td>0.030673</td>
      <td>0.046190</td>
      <td>0.074101</td>
    </tr>
    <tr>
      <th>6</th>
      <td>0.052469</td>
      <td>0.034019</td>
      <td>0.056384</td>
      <td>0.037385</td>
      <td>0.044352</td>
      <td>0.055274</td>
      <td>0.050894</td>
      <td>0.048175</td>
      <td>0.025828</td>
      <td>0.029773</td>
      <td>0.027076</td>
      <td>0.030762</td>
      <td>0.039665</td>
      <td>0.050097</td>
      <td>0.037727</td>
      <td>0.021628</td>
      <td>0.036278</td>
      <td>0.032287</td>
      <td>0.044278</td>
      <td>0.034476</td>
    </tr>
    <tr>
      <th>7</th>
      <td>0.032569</td>
      <td>0.051098</td>
      <td>0.026654</td>
      <td>0.012860</td>
      <td>0.026457</td>
      <td>0.034987</td>
      <td>0.036313</td>
      <td>0.018347</td>
      <td>0.024357</td>
      <td>0.054291</td>
      <td>0.015490</td>
      <td>0.022329</td>
      <td>0.020148</td>
      <td>0.033981</td>
      <td>0.038185</td>
      <td>0.028440</td>
      <td>0.044628</td>
      <td>0.025368</td>
      <td>0.028201</td>
      <td>0.052309</td>
    </tr>
    <tr>
      <th>8</th>
      <td>0.048198</td>
      <td>0.030028</td>
      <td>0.034463</td>
      <td>0.030507</td>
      <td>0.035929</td>
      <td>0.042568</td>
      <td>0.041386</td>
      <td>0.066220</td>
      <td>0.016749</td>
      <td>0.027119</td>
      <td>0.026904</td>
      <td>0.028878</td>
      <td>0.027301</td>
      <td>0.044445</td>
      <td>0.030485</td>
      <td>0.015853</td>
      <td>0.029126</td>
      <td>0.017866</td>
      <td>0.042531</td>
      <td>0.028645</td>
    </tr>
    <tr>
      <th>9</th>
      <td>0.052611</td>
      <td>0.018345</td>
      <td>0.040204</td>
      <td>0.026628</td>
      <td>0.025688</td>
      <td>0.059186</td>
      <td>0.066899</td>
      <td>0.030249</td>
      <td>0.026617</td>
      <td>0.014946</td>
      <td>0.028922</td>
      <td>0.031394</td>
      <td>0.028958</td>
      <td>0.027223</td>
      <td>0.031454</td>
      <td>0.018373</td>
      <td>0.016014</td>
      <td>0.035546</td>
      <td>0.042751</td>
      <td>0.012981</td>
    </tr>
    <tr>
      <th>10</th>
      <td>0.044711</td>
      <td>0.015678</td>
      <td>0.033478</td>
      <td>0.015784</td>
      <td>0.023039</td>
      <td>0.030164</td>
      <td>0.037720</td>
      <td>0.027123</td>
      <td>0.017532</td>
      <td>0.014188</td>
      <td>0.015861</td>
      <td>0.018076</td>
      <td>0.027358</td>
      <td>0.027688</td>
      <td>0.018872</td>
      <td>0.011806</td>
      <td>0.014721</td>
      <td>0.024026</td>
      <td>0.025974</td>
      <td>0.014179</td>
    </tr>
    <tr>
      <th>11</th>
      <td>0.014935</td>
      <td>0.022308</td>
      <td>0.046625</td>
      <td>0.025299</td>
      <td>0.032608</td>
      <td>0.017119</td>
      <td>0.017591</td>
      <td>0.016610</td>
      <td>0.085211</td>
      <td>0.019256</td>
      <td>0.037316</td>
      <td>0.046452</td>
      <td>0.019584</td>
      <td>0.023416</td>
      <td>0.021237</td>
      <td>0.049261</td>
      <td>0.018267</td>
      <td>0.071286</td>
      <td>0.016275</td>
      <td>0.022955</td>
    </tr>
    <tr>
      <th>12</th>
      <td>0.040690</td>
      <td>0.009220</td>
      <td>0.007786</td>
      <td>0.010033</td>
      <td>0.009602</td>
      <td>0.041333</td>
      <td>0.035441</td>
      <td>0.012257</td>
      <td>0.008448</td>
      <td>0.006883</td>
      <td>0.009609</td>
      <td>0.008843</td>
      <td>0.013354</td>
      <td>0.015438</td>
      <td>0.026803</td>
      <td>0.006866</td>
      <td>0.007666</td>
      <td>0.011848</td>
      <td>0.032162</td>
      <td>0.011539</td>
    </tr>
    <tr>
      <th>13</th>
      <td>0.051777</td>
      <td>0.034423</td>
      <td>0.050433</td>
      <td>0.056168</td>
      <td>0.053739</td>
      <td>0.051638</td>
      <td>0.049714</td>
      <td>0.066204</td>
      <td>0.025903</td>
      <td>0.031596</td>
      <td>0.031020</td>
      <td>0.033736</td>
      <td>0.040892</td>
      <td>0.049589</td>
      <td>0.035481</td>
      <td>0.024572</td>
      <td>0.035938</td>
      <td>0.036255</td>
      <td>0.073250</td>
      <td>0.036863</td>
    </tr>
    <tr>
      <th>14</th>
      <td>0.036834</td>
      <td>0.057491</td>
      <td>0.032764</td>
      <td>0.026794</td>
      <td>0.033535</td>
      <td>0.031436</td>
      <td>0.045151</td>
      <td>0.024746</td>
      <td>0.015766</td>
      <td>0.050525</td>
      <td>0.011089</td>
      <td>0.015502</td>
      <td>0.036737</td>
      <td>0.028376</td>
      <td>0.053662</td>
      <td>0.018475</td>
      <td>0.065886</td>
      <td>0.016770</td>
      <td>0.027929</td>
      <td>0.047768</td>
    </tr>
    <tr>
      <th>15</th>
      <td>0.017551</td>
      <td>0.011240</td>
      <td>0.027412</td>
      <td>0.066878</td>
      <td>0.022906</td>
      <td>0.023089</td>
      <td>0.014533</td>
      <td>0.019909</td>
      <td>0.008662</td>
      <td>0.012057</td>
      <td>0.014859</td>
      <td>0.012494</td>
      <td>0.018123</td>
      <td>0.027198</td>
      <td>0.010493</td>
      <td>0.014744</td>
      <td>0.012451</td>
      <td>0.017692</td>
      <td>0.032245</td>
      <td>0.018569</td>
    </tr>
    <tr>
      <th>16</th>
      <td>0.017166</td>
      <td>0.003829</td>
      <td>0.001057</td>
      <td>0.005451</td>
      <td>0.000641</td>
      <td>0.001197</td>
      <td>0.003415</td>
      <td>0.001380</td>
      <td>0.000623</td>
      <td>0.010019</td>
      <td>0.000000</td>
      <td>0.000190</td>
      <td>0.002788</td>
      <td>0.003153</td>
      <td>0.009591</td>
      <td>0.000747</td>
      <td>0.007128</td>
      <td>0.000227</td>
      <td>0.004519</td>
      <td>0.007548</td>
    </tr>
    <tr>
      <th>17</th>
      <td>0.003336</td>
      <td>0.009200</td>
      <td>0.003562</td>
      <td>0.000616</td>
      <td>0.002555</td>
      <td>0.002845</td>
      <td>0.012942</td>
      <td>0.004726</td>
      <td>0.000833</td>
      <td>0.001891</td>
      <td>0.000941</td>
      <td>0.005777</td>
      <td>0.008413</td>
      <td>0.004107</td>
      <td>0.003836</td>
      <td>0.006900</td>
      <td>0.002995</td>
      <td>0.001753</td>
      <td>0.001204</td>
      <td>0.011760</td>
    </tr>
    <tr>
      <th>18</th>
      <td>0.008620</td>
      <td>0.034108</td>
      <td>0.025559</td>
      <td>0.014156</td>
      <td>0.029261</td>
      <td>0.017067</td>
      <td>0.012636</td>
      <td>0.020661</td>
      <td>0.035407</td>
      <td>0.029928</td>
      <td>0.054964</td>
      <td>0.073268</td>
      <td>0.008233</td>
      <td>0.029996</td>
      <td>0.032294</td>
      <td>0.039515</td>
      <td>0.025144</td>
      <td>0.039231</td>
      <td>0.009663</td>
      <td>0.040355</td>
    </tr>
    <tr>
      <th>19</th>
      <td>0.021389</td>
      <td>0.030344</td>
      <td>0.049455</td>
      <td>0.028647</td>
      <td>0.035496</td>
      <td>0.028708</td>
      <td>0.021626</td>
      <td>0.032226</td>
      <td>0.065140</td>
      <td>0.022815</td>
      <td>0.043816</td>
      <td>0.034984</td>
      <td>0.033886</td>
      <td>0.028299</td>
      <td>0.021785</td>
      <td>0.045085</td>
      <td>0.025661</td>
      <td>0.041838</td>
      <td>0.019556</td>
      <td>0.027534</td>
    </tr>
    <tr>
      <th>20</th>
      <td>0.010162</td>
      <td>0.008295</td>
      <td>0.022779</td>
      <td>0.015937</td>
      <td>0.012383</td>
      <td>0.012061</td>
      <td>0.007016</td>
      <td>0.014653</td>
      <td>0.025174</td>
      <td>0.006212</td>
      <td>0.037328</td>
      <td>0.042241</td>
      <td>0.011736</td>
      <td>0.013620</td>
      <td>0.007801</td>
      <td>0.016115</td>
      <td>0.006916</td>
      <td>0.016767</td>
      <td>0.016196</td>
      <td>0.010217</td>
    </tr>
    <tr>
      <th>21</th>
      <td>0.025744</td>
      <td>0.014375</td>
      <td>0.011183</td>
      <td>0.022722</td>
      <td>0.023651</td>
      <td>0.024042</td>
      <td>0.014468</td>
      <td>0.060634</td>
      <td>0.009990</td>
      <td>0.009518</td>
      <td>0.034560</td>
      <td>0.034047</td>
      <td>0.010966</td>
      <td>0.038359</td>
      <td>0.008969</td>
      <td>0.008927</td>
      <td>0.006469</td>
      <td>0.011122</td>
      <td>0.020277</td>
      <td>0.006964</td>
    </tr>
    <tr>
      <th>22</th>
      <td>0.056933</td>
      <td>0.066873</td>
      <td>0.075350</td>
      <td>0.043281</td>
      <td>0.060178</td>
      <td>0.054031</td>
      <td>0.051068</td>
      <td>0.042265</td>
      <td>0.032877</td>
      <td>0.073093</td>
      <td>0.023354</td>
      <td>0.027387</td>
      <td>0.040654</td>
      <td>0.053363</td>
      <td>0.048982</td>
      <td>0.035385</td>
      <td>0.075043</td>
      <td>0.039472</td>
      <td>0.038570</td>
      <td>0.077110</td>
    </tr>
    <tr>
      <th>23</th>
      <td>0.006597</td>
      <td>0.007856</td>
      <td>0.038092</td>
      <td>0.016841</td>
      <td>0.019495</td>
      <td>0.008334</td>
      <td>0.007234</td>
      <td>0.007905</td>
      <td>0.065189</td>
      <td>0.006482</td>
      <td>0.036011</td>
      <td>0.028269</td>
      <td>0.008258</td>
      <td>0.011601</td>
      <td>0.007768</td>
      <td>0.051924</td>
      <td>0.007876</td>
      <td>0.083163</td>
      <td>0.010744</td>
      <td>0.010118</td>
    </tr>
    <tr>
      <th>24</th>
      <td>0.043876</td>
      <td>0.009936</td>
      <td>0.028736</td>
      <td>0.016953</td>
      <td>0.019271</td>
      <td>0.038957</td>
      <td>0.054159</td>
      <td>0.022452</td>
      <td>0.018126</td>
      <td>0.008484</td>
      <td>0.018232</td>
      <td>0.016218</td>
      <td>0.025099</td>
      <td>0.029277</td>
      <td>0.024636</td>
      <td>0.009942</td>
      <td>0.011226</td>
      <td>0.022776</td>
      <td>0.037319</td>
      <td>0.013639</td>
    </tr>
    <tr>
      <th>25</th>
      <td>0.023218</td>
      <td>0.032824</td>
      <td>0.041970</td>
      <td>0.020620</td>
      <td>0.027782</td>
      <td>0.023622</td>
      <td>0.022130</td>
      <td>0.020285</td>
      <td>0.056572</td>
      <td>0.027826</td>
      <td>0.031386</td>
      <td>0.038662</td>
      <td>0.020888</td>
      <td>0.024252</td>
      <td>0.026030</td>
      <td>0.032600</td>
      <td>0.035145</td>
      <td>0.048479</td>
      <td>0.018331</td>
      <td>0.036235</td>
    </tr>
    <tr>
      <th>26</th>
      <td>0.020897</td>
      <td>0.013815</td>
      <td>0.037985</td>
      <td>0.040500</td>
      <td>0.024508</td>
      <td>0.022653</td>
      <td>0.012125</td>
      <td>0.020043</td>
      <td>0.016462</td>
      <td>0.014394</td>
      <td>0.016247</td>
      <td>0.018186</td>
      <td>0.025213</td>
      <td>0.024360</td>
      <td>0.011600</td>
      <td>0.016914</td>
      <td>0.014853</td>
      <td>0.019048</td>
      <td>0.030097</td>
      <td>0.023192</td>
    </tr>
    <tr>
      <th>27</th>
      <td>0.038889</td>
      <td>0.065191</td>
      <td>0.039325</td>
      <td>0.030035</td>
      <td>0.035895</td>
      <td>0.033544</td>
      <td>0.063516</td>
      <td>0.029824</td>
      <td>0.021639</td>
      <td>0.063166</td>
      <td>0.014085</td>
      <td>0.017656</td>
      <td>0.047188</td>
      <td>0.033591</td>
      <td>0.055188</td>
      <td>0.020889</td>
      <td>0.069487</td>
      <td>0.018144</td>
      <td>0.028729</td>
      <td>0.053462</td>
    </tr>
    <tr>
      <th>28</th>
      <td>0.016691</td>
      <td>0.016813</td>
      <td>0.076328</td>
      <td>0.036513</td>
      <td>0.039193</td>
      <td>0.021910</td>
      <td>0.018646</td>
      <td>0.023948</td>
      <td>0.069801</td>
      <td>0.018834</td>
      <td>0.036073</td>
      <td>0.041090</td>
      <td>0.033610</td>
      <td>0.028982</td>
      <td>0.015818</td>
      <td>0.049317</td>
      <td>0.024027</td>
      <td>0.057404</td>
      <td>0.023862</td>
      <td>0.029668</td>
    </tr>
    <tr>
      <th>29</th>
      <td>0.011965</td>
      <td>0.030276</td>
      <td>0.004873</td>
      <td>0.001403</td>
      <td>0.004113</td>
      <td>0.017928</td>
      <td>0.015687</td>
      <td>0.039101</td>
      <td>0.007292</td>
      <td>0.023230</td>
      <td>0.005690</td>
      <td>0.002353</td>
      <td>0.007210</td>
      <td>0.008253</td>
      <td>0.026653</td>
      <td>0.008082</td>
      <td>0.020876</td>
      <td>0.005786</td>
      <td>0.003093</td>
      <td>0.026250</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>39744</th>
      <td>0.086324</td>
      <td>0.049277</td>
      <td>0.062285</td>
      <td>0.047769</td>
      <td>0.055648</td>
      <td>0.079797</td>
      <td>0.067229</td>
      <td>0.043494</td>
      <td>0.027615</td>
      <td>0.049303</td>
      <td>0.025243</td>
      <td>0.028567</td>
      <td>0.043743</td>
      <td>0.050433</td>
      <td>0.057366</td>
      <td>0.028352</td>
      <td>0.065633</td>
      <td>0.036608</td>
      <td>0.060707</td>
      <td>0.059449</td>
    </tr>
    <tr>
      <th>39745</th>
      <td>0.032008</td>
      <td>0.004693</td>
      <td>0.001629</td>
      <td>0.002926</td>
      <td>0.006625</td>
      <td>0.039328</td>
      <td>0.030379</td>
      <td>0.006699</td>
      <td>0.002181</td>
      <td>0.006760</td>
      <td>0.001918</td>
      <td>0.001396</td>
      <td>0.006652</td>
      <td>0.005422</td>
      <td>0.019828</td>
      <td>0.004137</td>
      <td>0.005383</td>
      <td>0.001385</td>
      <td>0.020475</td>
      <td>0.003174</td>
    </tr>
    <tr>
      <th>39746</th>
      <td>0.012079</td>
      <td>0.008532</td>
      <td>0.025263</td>
      <td>0.044370</td>
      <td>0.016875</td>
      <td>0.014966</td>
      <td>0.011437</td>
      <td>0.017948</td>
      <td>0.018411</td>
      <td>0.011184</td>
      <td>0.016898</td>
      <td>0.020714</td>
      <td>0.014922</td>
      <td>0.016471</td>
      <td>0.009941</td>
      <td>0.015168</td>
      <td>0.011302</td>
      <td>0.019800</td>
      <td>0.026021</td>
      <td>0.009496</td>
    </tr>
    <tr>
      <th>39747</th>
      <td>0.027555</td>
      <td>0.065636</td>
      <td>0.057759</td>
      <td>0.033013</td>
      <td>0.039629</td>
      <td>0.033036</td>
      <td>0.034736</td>
      <td>0.020165</td>
      <td>0.044036</td>
      <td>0.061891</td>
      <td>0.029932</td>
      <td>0.047905</td>
      <td>0.033359</td>
      <td>0.040772</td>
      <td>0.056520</td>
      <td>0.053715</td>
      <td>0.066350</td>
      <td>0.047595</td>
      <td>0.032493</td>
      <td>0.067114</td>
    </tr>
    <tr>
      <th>39748</th>
      <td>0.029030</td>
      <td>0.035618</td>
      <td>0.007109</td>
      <td>0.010674</td>
      <td>0.013708</td>
      <td>0.034282</td>
      <td>0.041982</td>
      <td>0.014640</td>
      <td>0.008440</td>
      <td>0.031966</td>
      <td>0.005656</td>
      <td>0.008474</td>
      <td>0.018536</td>
      <td>0.011332</td>
      <td>0.031963</td>
      <td>0.011412</td>
      <td>0.031794</td>
      <td>0.011257</td>
      <td>0.021181</td>
      <td>0.023184</td>
    </tr>
    <tr>
      <th>39749</th>
      <td>0.021890</td>
      <td>0.016676</td>
      <td>0.012776</td>
      <td>0.016873</td>
      <td>0.015811</td>
      <td>0.027686</td>
      <td>0.023400</td>
      <td>0.035223</td>
      <td>0.015357</td>
      <td>0.014923</td>
      <td>0.022195</td>
      <td>0.029460</td>
      <td>0.011462</td>
      <td>0.025182</td>
      <td>0.019494</td>
      <td>0.018990</td>
      <td>0.012843</td>
      <td>0.015840</td>
      <td>0.022629</td>
      <td>0.017962</td>
    </tr>
    <tr>
      <th>39750</th>
      <td>0.016826</td>
      <td>0.028852</td>
      <td>0.011253</td>
      <td>0.014596</td>
      <td>0.012092</td>
      <td>0.019902</td>
      <td>0.020986</td>
      <td>0.044687</td>
      <td>0.008762</td>
      <td>0.032757</td>
      <td>0.007474</td>
      <td>0.009603</td>
      <td>0.017093</td>
      <td>0.013899</td>
      <td>0.030469</td>
      <td>0.009724</td>
      <td>0.023713</td>
      <td>0.010182</td>
      <td>0.013220</td>
      <td>0.034891</td>
    </tr>
    <tr>
      <th>39751</th>
      <td>0.036433</td>
      <td>0.014271</td>
      <td>0.005749</td>
      <td>0.029265</td>
      <td>0.014308</td>
      <td>0.035207</td>
      <td>0.027638</td>
      <td>0.034771</td>
      <td>0.005720</td>
      <td>0.008445</td>
      <td>0.013089</td>
      <td>0.007780</td>
      <td>0.028403</td>
      <td>0.021090</td>
      <td>0.021351</td>
      <td>0.008030</td>
      <td>0.008713</td>
      <td>0.003991</td>
      <td>0.068518</td>
      <td>0.009849</td>
    </tr>
    <tr>
      <th>39752</th>
      <td>0.016740</td>
      <td>0.066711</td>
      <td>0.020790</td>
      <td>0.014917</td>
      <td>0.017592</td>
      <td>0.013041</td>
      <td>0.023180</td>
      <td>0.011419</td>
      <td>0.007444</td>
      <td>0.079182</td>
      <td>0.004667</td>
      <td>0.006060</td>
      <td>0.025833</td>
      <td>0.027954</td>
      <td>0.033967</td>
      <td>0.016384</td>
      <td>0.070299</td>
      <td>0.006569</td>
      <td>0.007669</td>
      <td>0.044641</td>
    </tr>
    <tr>
      <th>39753</th>
      <td>0.035281</td>
      <td>0.012250</td>
      <td>0.016242</td>
      <td>0.010874</td>
      <td>0.016964</td>
      <td>0.037673</td>
      <td>0.036695</td>
      <td>0.013723</td>
      <td>0.012462</td>
      <td>0.009874</td>
      <td>0.013667</td>
      <td>0.016015</td>
      <td>0.018795</td>
      <td>0.016984</td>
      <td>0.026600</td>
      <td>0.008367</td>
      <td>0.012948</td>
      <td>0.018793</td>
      <td>0.027134</td>
      <td>0.012644</td>
    </tr>
    <tr>
      <th>39754</th>
      <td>0.029001</td>
      <td>0.032835</td>
      <td>0.051980</td>
      <td>0.034839</td>
      <td>0.040539</td>
      <td>0.033824</td>
      <td>0.027623</td>
      <td>0.033533</td>
      <td>0.038415</td>
      <td>0.029023</td>
      <td>0.041005</td>
      <td>0.042365</td>
      <td>0.033202</td>
      <td>0.041555</td>
      <td>0.030888</td>
      <td>0.033593</td>
      <td>0.036461</td>
      <td>0.045893</td>
      <td>0.030584</td>
      <td>0.039121</td>
    </tr>
    <tr>
      <th>39755</th>
      <td>0.028987</td>
      <td>0.026023</td>
      <td>0.057608</td>
      <td>0.077765</td>
      <td>0.042992</td>
      <td>0.032929</td>
      <td>0.023923</td>
      <td>0.031137</td>
      <td>0.019130</td>
      <td>0.026803</td>
      <td>0.030397</td>
      <td>0.027382</td>
      <td>0.031284</td>
      <td>0.045782</td>
      <td>0.024803</td>
      <td>0.025424</td>
      <td>0.030481</td>
      <td>0.019763</td>
      <td>0.035134</td>
      <td>0.036266</td>
    </tr>
    <tr>
      <th>39756</th>
      <td>0.040467</td>
      <td>0.075034</td>
      <td>0.062623</td>
      <td>0.055329</td>
      <td>0.057027</td>
      <td>0.040249</td>
      <td>0.040588</td>
      <td>0.038078</td>
      <td>0.044478</td>
      <td>0.069482</td>
      <td>0.031287</td>
      <td>0.038340</td>
      <td>0.044655</td>
      <td>0.044244</td>
      <td>0.051629</td>
      <td>0.047457</td>
      <td>0.077445</td>
      <td>0.038053</td>
      <td>0.039844</td>
      <td>0.070165</td>
    </tr>
    <tr>
      <th>39757</th>
      <td>0.112190</td>
      <td>0.017192</td>
      <td>0.025209</td>
      <td>0.020450</td>
      <td>0.029405</td>
      <td>0.097426</td>
      <td>0.065367</td>
      <td>0.031348</td>
      <td>0.025605</td>
      <td>0.008721</td>
      <td>0.036381</td>
      <td>0.023793</td>
      <td>0.035609</td>
      <td>0.039246</td>
      <td>0.039590</td>
      <td>0.014486</td>
      <td>0.010978</td>
      <td>0.027298</td>
      <td>0.053976</td>
      <td>0.014949</td>
    </tr>
    <tr>
      <th>39758</th>
      <td>0.047906</td>
      <td>0.033670</td>
      <td>0.025546</td>
      <td>0.039313</td>
      <td>0.031314</td>
      <td>0.029433</td>
      <td>0.026207</td>
      <td>0.032714</td>
      <td>0.014691</td>
      <td>0.034965</td>
      <td>0.017078</td>
      <td>0.023541</td>
      <td>0.022837</td>
      <td>0.026816</td>
      <td>0.025586</td>
      <td>0.016453</td>
      <td>0.031659</td>
      <td>0.016193</td>
      <td>0.057030</td>
      <td>0.029049</td>
    </tr>
    <tr>
      <th>39759</th>
      <td>0.038217</td>
      <td>0.020502</td>
      <td>0.070465</td>
      <td>0.043308</td>
      <td>0.041877</td>
      <td>0.033393</td>
      <td>0.026785</td>
      <td>0.030035</td>
      <td>0.038172</td>
      <td>0.024226</td>
      <td>0.033689</td>
      <td>0.042708</td>
      <td>0.031869</td>
      <td>0.038698</td>
      <td>0.023329</td>
      <td>0.035764</td>
      <td>0.025300</td>
      <td>0.052615</td>
      <td>0.032063</td>
      <td>0.033747</td>
    </tr>
    <tr>
      <th>39760</th>
      <td>0.046432</td>
      <td>0.111874</td>
      <td>0.048214</td>
      <td>0.041790</td>
      <td>0.052640</td>
      <td>0.045853</td>
      <td>0.051156</td>
      <td>0.035714</td>
      <td>0.026036</td>
      <td>0.107232</td>
      <td>0.017106</td>
      <td>0.018719</td>
      <td>0.046687</td>
      <td>0.051179</td>
      <td>0.067226</td>
      <td>0.033531</td>
      <td>0.109142</td>
      <td>0.021027</td>
      <td>0.032921</td>
      <td>0.085083</td>
    </tr>
    <tr>
      <th>39761</th>
      <td>0.017444</td>
      <td>0.013278</td>
      <td>0.032610</td>
      <td>0.019138</td>
      <td>0.021444</td>
      <td>0.023126</td>
      <td>0.017249</td>
      <td>0.017894</td>
      <td>0.056606</td>
      <td>0.009723</td>
      <td>0.046915</td>
      <td>0.053410</td>
      <td>0.017671</td>
      <td>0.018570</td>
      <td>0.018921</td>
      <td>0.046092</td>
      <td>0.013808</td>
      <td>0.060507</td>
      <td>0.025530</td>
      <td>0.014857</td>
    </tr>
    <tr>
      <th>39762</th>
      <td>0.045148</td>
      <td>0.049539</td>
      <td>0.065605</td>
      <td>0.057935</td>
      <td>0.058796</td>
      <td>0.038099</td>
      <td>0.035812</td>
      <td>0.035026</td>
      <td>0.038247</td>
      <td>0.049864</td>
      <td>0.032031</td>
      <td>0.032979</td>
      <td>0.035857</td>
      <td>0.043259</td>
      <td>0.036617</td>
      <td>0.041232</td>
      <td>0.046299</td>
      <td>0.038355</td>
      <td>0.036318</td>
      <td>0.057021</td>
    </tr>
    <tr>
      <th>39763</th>
      <td>0.016600</td>
      <td>0.009887</td>
      <td>0.038419</td>
      <td>0.015095</td>
      <td>0.017511</td>
      <td>0.011007</td>
      <td>0.038909</td>
      <td>0.015091</td>
      <td>0.046293</td>
      <td>0.007765</td>
      <td>0.019067</td>
      <td>0.020984</td>
      <td>0.017207</td>
      <td>0.017470</td>
      <td>0.011965</td>
      <td>0.015023</td>
      <td>0.008831</td>
      <td>0.026754</td>
      <td>0.010003</td>
      <td>0.004367</td>
    </tr>
    <tr>
      <th>39764</th>
      <td>0.032426</td>
      <td>0.029096</td>
      <td>0.032518</td>
      <td>0.032771</td>
      <td>0.033267</td>
      <td>0.033627</td>
      <td>0.029887</td>
      <td>0.040822</td>
      <td>0.015456</td>
      <td>0.027908</td>
      <td>0.014678</td>
      <td>0.012716</td>
      <td>0.038601</td>
      <td>0.025060</td>
      <td>0.023795</td>
      <td>0.015247</td>
      <td>0.028729</td>
      <td>0.014586</td>
      <td>0.034114</td>
      <td>0.028912</td>
    </tr>
    <tr>
      <th>39765</th>
      <td>0.012234</td>
      <td>0.002012</td>
      <td>0.004086</td>
      <td>0.027333</td>
      <td>0.007519</td>
      <td>0.000999</td>
      <td>0.000709</td>
      <td>0.000927</td>
      <td>0.000531</td>
      <td>0.000537</td>
      <td>0.020029</td>
      <td>0.012559</td>
      <td>0.000112</td>
      <td>0.015651</td>
      <td>0.001109</td>
      <td>0.000929</td>
      <td>0.001650</td>
      <td>0.000234</td>
      <td>0.004739</td>
      <td>0.002451</td>
    </tr>
    <tr>
      <th>39766</th>
      <td>0.046742</td>
      <td>0.042740</td>
      <td>0.074044</td>
      <td>0.048202</td>
      <td>0.053106</td>
      <td>0.052404</td>
      <td>0.043170</td>
      <td>0.035914</td>
      <td>0.025135</td>
      <td>0.051718</td>
      <td>0.021005</td>
      <td>0.031504</td>
      <td>0.045124</td>
      <td>0.049944</td>
      <td>0.043731</td>
      <td>0.030968</td>
      <td>0.078021</td>
      <td>0.041203</td>
      <td>0.049883</td>
      <td>0.068234</td>
    </tr>
    <tr>
      <th>39767</th>
      <td>0.069343</td>
      <td>0.047159</td>
      <td>0.063694</td>
      <td>0.045233</td>
      <td>0.049950</td>
      <td>0.053623</td>
      <td>0.078013</td>
      <td>0.045394</td>
      <td>0.026207</td>
      <td>0.044816</td>
      <td>0.022559</td>
      <td>0.027021</td>
      <td>0.055449</td>
      <td>0.055049</td>
      <td>0.047962</td>
      <td>0.024653</td>
      <td>0.054266</td>
      <td>0.035864</td>
      <td>0.052654</td>
      <td>0.048519</td>
    </tr>
    <tr>
      <th>39768</th>
      <td>0.030538</td>
      <td>0.036546</td>
      <td>0.038504</td>
      <td>0.024420</td>
      <td>0.033325</td>
      <td>0.027172</td>
      <td>0.027191</td>
      <td>0.036371</td>
      <td>0.015412</td>
      <td>0.029062</td>
      <td>0.011873</td>
      <td>0.015691</td>
      <td>0.029110</td>
      <td>0.027381</td>
      <td>0.024385</td>
      <td>0.015818</td>
      <td>0.034290</td>
      <td>0.018076</td>
      <td>0.021244</td>
      <td>0.031317</td>
    </tr>
    <tr>
      <th>39769</th>
      <td>0.032150</td>
      <td>0.066033</td>
      <td>0.029184</td>
      <td>0.026894</td>
      <td>0.030544</td>
      <td>0.029705</td>
      <td>0.038385</td>
      <td>0.022863</td>
      <td>0.018367</td>
      <td>0.060388</td>
      <td>0.014152</td>
      <td>0.015796</td>
      <td>0.033822</td>
      <td>0.028824</td>
      <td>0.053320</td>
      <td>0.022414</td>
      <td>0.075838</td>
      <td>0.017011</td>
      <td>0.024165</td>
      <td>0.056238</td>
    </tr>
    <tr>
      <th>39770</th>
      <td>0.016458</td>
      <td>0.002561</td>
      <td>0.002811</td>
      <td>0.005192</td>
      <td>0.005450</td>
      <td>0.006930</td>
      <td>0.007344</td>
      <td>0.010403</td>
      <td>0.006324</td>
      <td>0.001576</td>
      <td>0.008120</td>
      <td>0.004943</td>
      <td>0.004182</td>
      <td>0.003158</td>
      <td>0.002832</td>
      <td>0.001856</td>
      <td>0.000442</td>
      <td>0.005362</td>
      <td>0.007318</td>
      <td>0.002201</td>
    </tr>
    <tr>
      <th>39771</th>
      <td>0.033873</td>
      <td>0.070026</td>
      <td>0.043007</td>
      <td>0.032490</td>
      <td>0.051022</td>
      <td>0.034044</td>
      <td>0.034440</td>
      <td>0.025197</td>
      <td>0.028556</td>
      <td>0.082214</td>
      <td>0.018631</td>
      <td>0.026223</td>
      <td>0.032494</td>
      <td>0.041096</td>
      <td>0.049088</td>
      <td>0.034924</td>
      <td>0.075519</td>
      <td>0.029254</td>
      <td>0.038960</td>
      <td>0.073456</td>
    </tr>
    <tr>
      <th>39772</th>
      <td>0.008418</td>
      <td>0.024918</td>
      <td>0.022017</td>
      <td>0.015611</td>
      <td>0.020428</td>
      <td>0.011465</td>
      <td>0.009731</td>
      <td>0.008582</td>
      <td>0.047711</td>
      <td>0.023071</td>
      <td>0.021034</td>
      <td>0.026121</td>
      <td>0.010759</td>
      <td>0.013128</td>
      <td>0.013049</td>
      <td>0.029379</td>
      <td>0.023059</td>
      <td>0.032138</td>
      <td>0.008820</td>
      <td>0.020910</td>
    </tr>
    <tr>
      <th>39773</th>
      <td>0.056693</td>
      <td>0.039890</td>
      <td>0.070805</td>
      <td>0.048906</td>
      <td>0.051819</td>
      <td>0.048874</td>
      <td>0.048325</td>
      <td>0.062757</td>
      <td>0.031298</td>
      <td>0.036374</td>
      <td>0.030498</td>
      <td>0.036206</td>
      <td>0.063782</td>
      <td>0.049614</td>
      <td>0.035829</td>
      <td>0.026607</td>
      <td>0.043507</td>
      <td>0.040689</td>
      <td>0.045417</td>
      <td>0.043859</td>
    </tr>
  </tbody>
</table>
<p>39774 rows × 20 columns</p>
</div>
</div>


</div>
</div>
</div>



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```
#This creates a prediction from the column that has the maximum value and outputs it to a file. 
test['cuisine']=testc.idxmax(axis=1)
submission=test[['id' ,  'cuisine' ]]
submission.to_csv("4_tfidf1.csv",index=False)
test['cuisine']=testc2.idxmax(axis=1)
submission=test[['id' ,  'cuisine' ]]
submission.to_csv("5_tfidf2.csv",index=False)


```
</div>

</div>



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```trainc['prediction']=trainc.idxmax(axis=1)
trainc2['prediction']=trainc2.idxmax(axis=1)
trainc['cuisine']=train['cuisine']
trainc2['cuisine']=train['cuisine']
trainc.to_csv("coded1.csv",index=False)
trainc2.to_csv("coded2.csv",index=False)

```
</div>

</div>

