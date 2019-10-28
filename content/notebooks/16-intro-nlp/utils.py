import ruamel.yaml
import subprocess
import urllib.request
import json
import pandas as pd
def bash_command(command):
    try:
        print("executing the Bash command:\n", command)
        result=subprocess.check_output(command, stderr=subprocess.STDOUT, shell=True)
        result=result.decode("utf-8")
        return result
    except subprocess.CalledProcessError as e:
        return(e.output.decode("utf-8"))

def initialize(file):
    """This functions creates a variety of commands to load.
       It returns the cf.
    """
    #load the config file
    with open(file, 'r') as yaml:
         cf=ruamel.yaml.round_trip_load(yaml, preserve_quotes=True)
    return cf

def azure_request(command, endpoint, key, postdata):
    #Set URI
    uri=endpoint+"/"+command
    #Set header
    headers = {}
    headers['Ocp-Apim-Subscription-Key'] = key
    headers['Content-Type'] = 'application/json'
    headers['Accept'] = 'application/json'
    #Make request
    request = urllib.request.Request(uri, postdata, headers)
    response = urllib.request.urlopen(request)
    return json.loads(response.read().decode('utf-8'))

def merge_results(df, results):
    df_results=pd.DataFrame(results['documents'])
    df_merged = pd.concat([df, df_results], axis=1, join_axes=[df.index])
    return df_merged.drop(['id'], axis=1)

def df_to_postdata(df):
    df=df.loc[:,["Review_ID","language","Review_Content"]]
    df=df.rename(index=str, columns={"Review_ID": "id", "Review_Content": "text"})
    df['language']='en'
    return json.dumps({'documents': json.loads(df.to_json(orient='records')) }).encode('utf-8')
from os import listdir

def find_filenames( path_to_dir, suffix=".csv" ):
    filenames = listdir(path_to_dir)
    return [ filename for filename in filenames if filename.endswith( suffix ) ]
