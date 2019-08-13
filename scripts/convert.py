#!/usr/bin/env python
# coding: utf-8

# YAML Files
# ---
# These files are used to configure and organize the website's contents.

# Only need to run once
get_ipython().system('pip install ruamel.yaml')

# Always run this before any of the following cells
import pandas as pd
import numpy as np
import csv
import logging
import subprocess
import ruamel.yaml

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

# Configuration
config = load_yaml_file('../_config.yml') # Load the file.
config_xl= pd.read_excel('../book.xlsx', sheet_name = '_config_yml', header=None, index_col=None)
for x in range(len(config_xl)):           # Update the Yaml with the config from excel
    config[config_xl.iloc[x,0]]=config_xl.iloc[x,1]
update_yaml_file('../_config.yml', config)

# Table of contents
# 1. read the Excel sheet and create a yaml file from it.
import re
import os
toc_yml= pd.read_excel('../book.xlsx', sheet_name = 'toc_yml', header=0)
toc_yml.to_csv('../_data/toc2.yml',index=None,quoting=csv.QUOTE_NONE,escapechar=' ')
#2. replace double spaces with single spaces.
with open('../_data/toc.yml', 'w') as out:
    with open('../_data/toc2.yml', 'r') as f:
        for line in f:
            line = re.sub(r"  ", " ", line)
            out.write(line)
# 3. delete toc2.yml
os.remove('../_data/toc2.yml')

# # Markdown files
# ---
# These files comprise the site's content, aside from the notebooks already created.

# Always run this before any of the following cells
import pandas as pd
import numpy as np
import csv

# Home
index_md= pd.read_excel('../book.xlsx', sheet_name = 'index_md', header=0)
index_md.to_csv('../content/index.md',  index=None, sep=' ',quoting = csv.QUOTE_NONE, escapechar = ' ')

# Schedule
schedule_md= pd.read_excel('../book.xlsx', sheet_name = 'schedule_md', header=0)
schedule_md.to_csv('../content/sessions/index.md',  index=None, sep=' ',quoting = csv.QUOTE_NONE, escapechar = ' ')

# Sessions (individual pages)
session_md= pd.read_excel('../book.xlsx', sheet_name = 'session_md', header=0, index_col=0, usecols="A:B")
session_md=session_md.dropna()
for index, row in session_md.iterrows():
    row.to_csv('../content/sessions/'+str(index)+'.md',index=False,header=False,sep=' ',quoting = csv.QUOTE_NONE,
               escapechar = ' ')

# Assignments
assignments_md= pd.read_excel('../book.xlsx', sheet_name = 'assignments_md', header=0)
assignments_md.to_csv('../content/assignments/index.md',  index=None, sep=' ',quoting = csv.QUOTE_NONE, escapechar = ' ')

# Grading
grading_md= pd.read_excel('../book.xlsx', sheet_name = 'grading_md', header=0)
grading_md.to_csv('../content/grading.md',  index=None, sep=' ',quoting = csv.QUOTE_NONE, escapechar = ' ')

# Notebooks
notebooks_md= pd.read_excel('../book.xlsx', sheet_name = 'notebooks_md', header=0)
notebooks_md.to_csv('../content/notebooks/index.md', index=None, sep=' ',quoting = csv.QUOTE_NONE, escapechar = ' ')

# Readings
readings_md= pd.read_excel('../book.xlsx', sheet_name = 'readings_md', header=0)
readings_md.to_csv('../content/sessions/readings.md',  index=None, sep=' ',quoting = csv.QUOTE_NONE, escapechar = ' ')