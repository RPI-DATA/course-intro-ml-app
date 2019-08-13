#!/usr/bin/env python
# coding: utf-8

# # YAML Files
# ---
# These files are used to configure and organize the website's contents.

# In[ ]:


# Only need to run once
#!pip install ruamel.yaml


# In[1]:


# Always run this before any of the following cells
import pandas as pd
import numpy as np
import csv
import logging
import subprocess
import ruamel.yaml


# In[2]:


def load_yaml_file(file):
    """
    Loads a yaml file from file system.
    @param file Path to file to be loaded.
    """
    try:
        with open(file, 'r') as yaml:
            kwargs = ruamel.yaml.round_trip_load(yaml, preserve_quotes=True)
        return kwargs
    except subprocess.CalledProcessError as e:
        print("error")
    return(e.output.decode("utf-8"))

def update_yaml_file(file, kwargs):
    """
    Updates a yaml file.
    @param kwargs dictionary.
    """
    print("Updating the file: " + file)
    try:
        ruamel.yaml.round_trip_dump(kwargs, open(file, 'w'))
    except subprocess.CalledProcessError as e:
        print("error: " + e)
        
def write_md_file(filename, df):
    print("Updating the file: " + filename)
    df.to_csv(filename,  index=None, sep=' ',quoting = csv.QUOTE_NONE, escapechar = ' ')
    


# In[3]:


# Configuration
config = load_yaml_file('../_config.yml') # Load the file.
config_xl= pd.read_excel('../book.xlsx', sheet_name = '_config_yml', header=None, index_col=None)
for x in range(len(config_xl)):           # Update the Yaml with the config from excel
    config[config_xl.iloc[x,0]]=config_xl.iloc[x,1]
update_yaml_file('../_config.yml', config)


# In[4]:


# Table of contents (current)
# 1. read the Excel sheet and create a yaml file from it.
import re
import os
toc_yml= pd.read_excel('../book.xlsx', sheet_name = 'toc_yml', header=0)
toc_yml.to_csv('../_data/toc2.yml',index=None,quoting=csv.QUOTE_NONE,escapechar=' ')

# 2. replace double spaces with single spaces.
with open('../_data/toc.yml', 'w') as out:
    with open('../_data/toc2.yml', 'r') as f:
        for line in f:
            line = re.sub(r"  ", " ", line)
            out.write(line)
            
# 3. delete toc2.yml
os.remove('../_data/toc2.yml')


# In[5]:


# Table of contents (old approach - only works for an unchanging number of fields)
# toc = load_yaml_file('../_data/toc.yml')
# toc_xl= pd.read_excel('../book.xlsx', sheet_name = 'toc_yml',  index_col=None)
# for x in range(len(toc_xl)):
#     toc[toc_xl.loc[x,'index']]['title']=toc_xl.loc[x,'title']
#     toc[toc_xl.loc[x,'index']]['url']=toc_xl.loc[x,'url']
# update_yaml_file('../_data/toc.yml', toc)


# In[6]:


# Table of contents (experimental - currently doesn't work; see issue #3 in the repo)
# from collections import OrderedDict
# toc = load_yaml_file('../_data/toc.yml')                                 # load original yaml file
# toc_xl= pd.read_excel('../book.xlsx',sheet_name ='toc_yml3',index_col=0) # load excel data
# toc_ses= toc_xl.to_dict(into=OrderedDict,orient='records')               # convert excel df to list of OrderedDict
# toc[3]['sections']= toc_ses
# update_yaml_file('../_data/toc2.yml', toc)


# # Markdown files
# ---
# These files comprise the site's content, aside from the notebooks already created.

# In[7]:


# Always run this before any of the following cells
import pandas as pd
import numpy as np
import csv


# In[8]:


# Home
index_file = '../content/index.md'
index_md= pd.read_excel('../book.xlsx', sheet_name = 'index_md', header=0)
write_md_file(index_file, index_md)


# In[9]:


# Schedule
schedule_file='../content/sessions/index.md'
schedule_md= pd.read_excel('../book.xlsx', sheet_name = 'schedule_md', header=0)
write_md_file(schedule_file, schedule_md)


# In[10]:


# Sessions
session_md= pd.read_excel('../book.xlsx',sheet_name='session_md',header=0,index_col=0,usecols="A:B")
session_md=session_md.dropna()
for index, row in session_md.iterrows():
    session_file='../content/sessions/'+str(index)+'.md'
    print("Updating the file: " + session_file)
    row.to_csv(session_file,index=False,header=False,sep=' ',quoting = csv.QUOTE_NONE,
               escapechar = ' ')


# In[11]:


#Assignments
assignments_file='../content/assignments/index.md'
assignments_md= pd.read_excel('../book.xlsx', sheet_name = 'assignments_md', header=0)
write_md_file(assignments_file, assignments_md)


# In[12]:


# Grading
grading_file='../content/grading.md'
grading_md= pd.read_excel('../book.xlsx', sheet_name = 'grading_md', header=0)
write_md_file(grading_file, grading_md)


# In[15]:


# Notebooks
notebooks_file='../content/notebooks/index.md'
notebooks_md= pd.read_excel('../book.xlsx', sheet_name = 'notebooks_md', header=0)
write_md_file(notebooks_file, notebooks_md)


# In[16]:


# Readings
readings_file='../content/sessions/readings.md'
readings_md= pd.read_excel('../book.xlsx', sheet_name = 'readings_md', header=0)
write_md_file(readings_file, readings_md)


# In[ ]:




