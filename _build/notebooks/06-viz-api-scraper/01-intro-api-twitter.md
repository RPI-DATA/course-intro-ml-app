---
interact_link: content/notebooks/06-viz-api-scraper/01-intro-api-twitter.ipynb
kernel_name: python3
has_widgets: false
title: 'Twitter'
prev_page:
  url: /notebooks/04-python/05-intro-kaggle-baseline.html
  title: 'Kaggle Baseline'
next_page:
  url: /notebooks/06-viz-api-scraper/02-intro-python-webmining.html
  title: 'Web Mining'
comment: "***PROGRAMMATICALLY GENERATED, DO NOT EDIT. SEE ORIGINAL FILES IN /content***"
---



[![AnalyticsDojo](https://github.com/rpi-techfundamentals/spring2019-materials/blob/master/fig/final-logo.png?raw=1)](http://rpi.analyticsdojo.com)
<center><h1>Introduction to API's with Python</h1></center>
<center><h3><a href = 'http://rpi.analyticsdojo.com'>rpi.analyticsdojo.com</a></h3></center>





This is adopted from [Mining the Social Web, 2nd Edition](http://bit.ly/16kGNyb)
Copyright (c) 2013, Matthew A. Russell
All rights reserved.

This work is licensed under the [Simplified BSD License](https://github.com/ptwobrussell/Mining-the-Social-Web-2nd-Edition/blob/master/LICENSE.txt).



### Before you Begin #1
If you are working locally or on colab, this exercise requires the twitter package and the ruamel.yaml package.   Yaml files are structured files useful for storing configuration. 




<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
``` !pip install twitter ruamel.yaml

```
</div>

</div>



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```#see if it worked by importing the twitter package & some other things we will use.  
from  twitter import *
import datetime, traceback 
import json
import time
import sys


```
</div>

</div>



### Before you Begin #2
Download the sample configuration.  




<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```!!wget https://raw.githubusercontent.com/rpi-techfundamentals/spring2019-materials/master/04-viz-api-scraper/screen_names.csv && wget https://raw.githubusercontent.com/rpi-techfundamentals/spring2019-materials/master/04-viz-api-scraper/twitlab.py 


```
</div>

</div>



# Download Authorization file (look at Slack for file)



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```!wget https://www.dropbox.com/s/ojjs2aj14gjuqjb/config.yaml 

```
</div>

</div>



## Step1.  Loading Authorization Data
- Here we are going to store the authorization data in a .YAML file rather than directly in the notebook.  
- We have also added `config.yaml` to the `.gitignore` file so we won't accidentally commit our sensitive data to the repository.
- You should generally keep sensitive data out of all git repositories (public or private) but definitely Public. 
- If you ever accidentally commit data to a public repository you must consider it compromised.
- A .yaml file is a common way to store configuration data, but it is not really secure. 



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```#This will import some required libraries.
import sys 
import ruamel.yaml #A .yaml file 
#This is your configuration file. 
twitter_yaml='config.yaml'
with open(twitter_yaml, 'r') as yaml_t:
    cf_t=ruamel.yaml.round_trip_load(yaml_t, preserve_quotes=True)

#You can check your config was loaded by printing, but you should not commit this.
cf_t


```
</div>

</div>



# `cat` command to look at files
We can use the `cat` command to look at the structure of our files. 



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```!cat config.yaml

```
</div>

</div>



## Create Some Relevant Functions
- We first will create a Twitter object we can used to authorize data.
- Then we will get profiles.
- Finally we will get some tweets.  

**Don't worry about not understanding all the code.  Here we are pushing you us more complex functions.**



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```#@title
def create_twitter_auth(cf_t):
        """Function to create a twitter object
           Args: cf_t is configuration dictionary. 
           Returns: Twitter object.
            """
        # When using twitter stream you must authorize.
        # these tokens are necessary for user authentication
        # create twitter API object

        auth = OAuth(cf_t['access_token'], cf_t['access_token_secret'], cf_t['consumer_key'], cf_t['consumer_secret'])

        try:
            # create twitter API object
            twitter = Twitter(auth = auth)
        except TwitterHTTPError:
            traceback.print_exc()
            time.sleep(cf_t['sleep_interval'])
        return twitter

```
</div>

</div>



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```def get_profiles(twitter, names, cf_t):
    """Function write profiles to a file with the form *data-user-profiles.json*
       Args: names is a list of names
             cf_t is a list of twitter config
       Returns: Nothing
        """
    # file name for daily tracking
    dt = datetime.datetime.now()
    fn = cf_t['data']+'/profiles/'+dt.strftime('%Y-%m-%d-user-profiles.json')
    with open(fn, 'w') as f:
        for name in names:
            print("Searching twitter for User profile: ", name)
            try:
                # create a subquery, looking up information about these users
                # twitter API docs: https://dev.twitter.com/docs/api/1/get/users/lookup
                profiles = twitter.users.lookup(screen_name = name)
                sub_start_time = time.time()
                for profile in profiles:
                    print("User found. Total tweets:", profile['statuses_count'])
                    # now save user info
                    f.write(json.dumps(profile))
                    f.write("\n")
                sub_elapsed_time = time.time() - sub_start_time;
                if sub_elapsed_time < cf_t['sleep_interval']:
                    time.sleep(cf_t['sleep_interval'] + 1 - sub_elapsed_time)
            except TwitterHTTPError:
                traceback.print_exc()
                time.sleep(cf_t['sleep_interval'])
                continue
    f.close()
    return fn

```
</div>

</div>



## Load Twitter Handle From CSV
- This is a .csv that has individuals we want to collect data on. 
- Go ahead and follow [AnalyticsDojo](https://twitter.com/AnalyticsDojo).  



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```import pandas as pd
df=pd.read_csv(cf_t['file'])
df

```
</div>

</div>



## Create Twitter Object



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```import twitlab

```
</div>

</div>



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```#Create Twitter Object
twitter= twitlab.create_twitter_auth(cf_t)

```
</div>

</div>



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```df['screen_name']

```
</div>

</div>



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```!mkdir data && mkdir data/profiles

```
</div>

</div>



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```#This will get general profile data
profiles_fn=twitlab.get_profiles(twitter, df['screen_name'], cf_t)

```
</div>

</div>



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```!ls data/profiles

```
</div>

</div>



# Let's look at the files created.



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```!cat data/profiles/*-user-profiles.json

```
</div>

</div>



The outcoming of running the above API is to generate a twitter object. 



## Step 2. Getting Help



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```# We can get some help on how to use the twitter api with the following. 
help(twitter)

```
</div>

</div>




Go ahead and take a look at the [twitter docs](https://dev.twitter.com/rest/public).





<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```# The Yahoo! Where On Earth ID for the entire world is 1.
# See https://dev.twitter.com/docs/api/1.1/get/trends/place and
# http://developer.yahoo.com/geo/geoplanet/

WORLD_WOE_ID = 1
US_WOE_ID = 23424977

# Prefix ID with the underscore for query string parameterization.
# Without the underscore, the twitter package appends the ID value
# to the URL itself as a special case keyword argument.

world_trends = twitter.trends.place(_id=WORLD_WOE_ID)
us_trends = twitter.trends.place(_id=US_WOE_ID)

print (world_trends)
print (us_trends)

```
</div>

</div>



## Step 3. Displaying API responses as pretty-printed JSON



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```import json

print (json.dumps(world_trends, indent=1))
print (json.dumps(us_trends, indent=1))

```
</div>

</div>



Take a look at the [api docs](https://dev.twitter.com/rest/reference/get/trends/place) for the /trends/place call made above. 



## Step 4. Collecting search results for a targeted hashtag.



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```# Import unquote to prevent url encoding errors in next_results
#from urllib3 import unquote

#This can be any trending topic, but let's focus on a hashtag that is relevant to the class. 
q = '#analytics' 

count = 100

# See https://dev.twitter.com/rest/reference/get/search/tweets
search_results = twitter.search.tweets(q=q, count=count)

#This selects out 
statuses = search_results['statuses']


# Iterate through 5 more batches of results by following the cursor
for _ in range(5):
    print ("Length of statuses", len(statuses))
    try:
        next_results = search_results['search_metadata']['next_results']
        print ("next_results", next_results)
    except: # No more results when next_results doesn't exist
        break
        
    # Create a dictionary from next_results, which has the following form:
    # ?max_id=313519052523986943&q=NCAA&include_entities=1
    kwargs = dict([ kv.split('=') for kv in next_results[1:].split("&") ])
    print (kwargs)
    search_results = twitter.search.tweets(**kwargs)
    statuses += search_results['statuses']

# Show one sample search result by slicing the list...
print (json.dumps(statuses[0], indent=1))

```
</div>

</div>



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```#Print several
print (json.dumps(statuses[0:5], indent=1))

```
</div>

</div>



## Step 5. Extracting text, screen names, and hashtags from tweets



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```#We can access an individual tweet like so:
statuses[1]['text']





```
</div>

</div>



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```statuses[1]['entities']

```
</div>

</div>



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```#notice the nested relationships.  We have to take notice of this to further access the data.
statuses[1]['entities']['hashtags']

```
</div>

</div>



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```status_texts = [ status['text'] 
                 for status in statuses ]

screen_names = [ user_mention['screen_name'] 
                 for status in statuses
                     for user_mention in status['entities']['user_mentions'] ]

hashtags = [ hashtag['text'] 
             for status in statuses
                 for hashtag in status['entities']['hashtags'] ]

urls = [ url['url'] 
             for status in statuses
                 for url in status['entities']['urls'] ]



# Compute a collection of all words from all tweets
words = [ w 
          for t in status_texts 
              for w in t.split() ]

# Explore the first 5 items for each...

print (json.dumps(status_texts[0:5], indent=1))
print (json.dumps(screen_names[0:5], indent=1)) 
print (json.dumps(hashtags[0:5], indent=1))
print (json.dumps(words[0:5], indent=1))

```
</div>

</div>



## Step 6. Creating a basic frequency distribution from the words in tweets



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```from collections import Counter

for item in [words, screen_names, hashtags]:
    c = Counter(item)
    print (c.most_common()[:10]) # top 10, "\n")
    

```
</div>

</div>

