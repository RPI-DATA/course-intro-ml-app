#!/usr/bin/env python
# coding: utf-8

# # YAML Files
# ---
# These files are used to configure and organize the website's contents.

# In[ ]:


# Only need to run once
get_ipython().system('pip install ruamel.yaml')


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


# In[9]:


# Configuration
config = load_yaml_file('../_config.yml') # Load the file.
config_xl= pd.read_excel('../book.xlsx', sheet_name = '_config_yml', header=None, index_col=None)
for x in range(len(config_xl)):           # Update the Yaml with the config from excel
    config[config_xl.iloc[x,0]]=config_xl.iloc[x,1]
update_yaml_file('../_config.yml', config)


# In[ ]:


# Table of contents (old approach - only works for an unchanging number of fields)
toc = load_yaml_file('../_data/toc.yml')
toc_xl= pd.read_excel('../book.xlsx', sheet_name = 'toc_yml',  index_col=None)
for x in range(len(toc_xl)):
    toc[toc_xl.loc[x,'index']]['title']=toc_xl.loc[x,'title']
    toc[toc_xl.loc[x,'index']]['url']=toc_xl.loc[x,'url']
update_yaml_file('../_data/toc.yml', toc)


# *Note: The toc.yml file is too complex to handle as a dataframe, since the number of fields varies as users add in more sessions, assignments, etc., so my workaround is just recreating the file in an Excel tab, then converting that using to_csv. The secondary operation replaces all of the double spaces that messes with the yaml's mapping and turns them into single spaces.*

# In[8]:


# 1. read the Excel sheet and create a yaml file from it.
import re
toc_yml= pd.read_excel('../book.xlsx', sheet_name = 'toc_yml2', header=0)
toc_yml.to_csv('../_data/toc2.yml',index=None,quoting=csv.QUOTE_NONE,escapechar=' ')

#2. replace double spaces with single spaces.
with open('../_data/toc.yml', 'w') as out:
    with open('../_data/toc2.yml', 'r') as f:
        for line in f:
            line = re.sub(r"  ", " ", line)
            out.write(line)


# *The following is just me trying another approach.*

# In[16]:


toc = load_yaml_file('../_data/toc.yml')
toc_xl= pd.read_excel('../book.xlsx', sheet_name = 'toc_yml3',  index_col=0)
toc_xl


# In[28]:


toc_gr= toc_xl.groupby(level=0,by='index')
print(toc_gr)


# In[22]:


from collections import OrderedDict
toc_dict= toc_xl.to_dict(into=OrderedDict, orient='index')
toc_dict


# # Markdown files
# ---
# These files comprise the site's content, aside from the notebooks already created.

# In[24]:


# Always run this before any of the following cells
import pandas as pd
import numpy as np
import csv


# In[68]:


# Home
index_md= pd.read_excel('../book.xlsx', sheet_name = 'index_md', header=0)
index_md.to_csv('../content/index.md',  index=None, sep=' ',quoting = csv.QUOTE_NONE, escapechar = ' ')


# In[69]:


# Schedule
schedule_md= pd.read_excel('../book.xlsx', sheet_name = 'schedule_md', header=0)
schedule_md.to_csv('../content/sessions/index.md',  index=None, sep=' ',quoting = csv.QUOTE_NONE, escapechar = ' ')


# In[58]:


# Sessions (individual pages)
session_md= pd.read_excel('../book.xlsx', sheet_name = 'session_md', header=0, index_col=0, usecols="A:B")
session_md=session_md.dropna()
for index, row in session_md.iterrows():
    row.to_csv('../content/sessions/'+str(index)+'.md',index=False,header=False,sep=' ',quoting = csv.QUOTE_NONE,
               escapechar = ' ')


# In[70]:


# Assignments
assignments_md= pd.read_excel('../book.xlsx', sheet_name = 'assignments_md', header=0)
assignments_md.to_csv('../content/assignments/index.md',  index=None, sep=' ',quoting = csv.QUOTE_NONE, escapechar = ' ')


# In[60]:


# Assignments (individual pages)
assign_md= pd.read_excel('../book.xlsx', sheet_name = 'assign_md', header=0, index_col=0, usecols="A:B")
assign_md=assign_md.dropna()
for index, row in assign_md.iterrows(): 
    row.to_csv('../content/assignments/'+str(index)+'.md',index=False,header=False,sep=' ',quoting=csv.QUOTE_NONE,
               escapechar=' ')


# In[ ]:


# Grading
grading_md= pd.read_excel('../book.xlsx', sheet_name = 'grading_md', header=0)
grading_md.to_csv('../content/grading.md',  index=None, sep=' ',quoting = csv.QUOTE_NONE, escapechar = ' ')


# In[72]:


# Notebooks
notebooks_md= pd.read_excel('../book.xlsx', sheet_name = 'notebooks_md', header=0)
notebooks_md.to_csv('../content/notebooks/index.md', index=None, sep=' ',quoting = csv.QUOTE_NONE, escapechar = ' ')


# In[63]:


# Readings
readings_md= pd.read_excel('../book.xlsx', sheet_name = 'readings_md', header=0)
readings_md.to_csv('../content/sessions/readings.md',  index=None, sep=' ',quoting = csv.QUOTE_NONE, escapechar = ' ')

