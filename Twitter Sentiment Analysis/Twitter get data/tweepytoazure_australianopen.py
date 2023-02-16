#https://stackoverflow.com/questions/50046158/pyodbc-login-timeout-error

import tweepy
import pandas as pd
import configparser
import re
from datetime import date
from datetime import datetime
import sys
import json
import pyodbc
import yaml
#from azureml.core import Datastore, Dataset, Workspace 
#from azureml.core.authentication import InteractiveLoginAuthentication

from select import select
import textwrap
import pyodbc
import yaml
import os




# Reading new .ini config file with peter's twitter details
config = configparser.ConfigParser()
config.read('/home/ubuntu/Documents/PythonProjects/TweepyAzure/config.ini')

#config_json = json.load(open('config.json'))

# From peter's config file, read the API keys & Access tokens..
api_key = config['twitter']['api_key']
api_key_secret = config['twitter']['api_key_secret']

access_token = config['twitter']['access_token']
access_token_secret = config['twitter']['access_token_secret']

search_term = 'australianopen'
tweet_amount = 100

tweet_list = []

today = date.today()
now = datetime.now()
now = f"{now}".replace(" ", "-").replace(":","").replace(".","")

# Access data
auth = tweepy.OAuthHandler(api_key,api_key_secret)
auth.set_access_token(access_token, access_token_secret)

api = tweepy.API(auth)

# OPTION2: Search for tweets on a search term

# https://stackoverflow.com/questions/24002536/get-tweepy-search-results-as-json
searched_tweets = [status._json for status in tweepy.Cursor(api.search_tweets,  q=search_term).items(tweet_amount)]
json_strings = [json.dumps(json_obj) for json_obj in searched_tweets]  

#print(json_strings)
# Create list of tweets
for json_obj in searched_tweets:
    tweet_dict ={
        "tweetJSON": json.dumps(json_obj)
    }
    tweet_list.append(tweet_dict)

# Save as df and save to csv file
tweet_df = pd.DataFrame(tweet_list)
# date_column = tweet_df[['todayDate']]
# tweet_column = tweet_df[['tweetJSON']]

export_file_name = f"PeterTwitterTestExport-{now}.csv"
tweet_df.to_csv(export_file_name, header = True, index = False)



# EXECUTING THE AZURE CONNECTION



config_path = r'/home/ubuntu/Documents/PythonProjects/TweepyAzure/config_australianopen.yaml'
with open(config_path, "r") as ymlfile:
    cfg = yaml.safe_load(ymlfile)

server_name= "twittertestdb"
server = 'tcp:{server_name}.database.windows.net,1433'.format(server_name =server_name)
database = cfg["azure"]["database"]
username = cfg["details"]["username"]
password = cfg["details"]["password"]
driver = cfg["details"]["driver"]

target_table = cfg["table_details"]["target_table"]
target_column = cfg["table_details"]["target_column"]

# take the connection string from the "ConnectionString" tab in your SQLDB
# I'm using a Database Owner user rather than my own details because own details require MFA
connection_string = 'DRIVER='+driver+';SERVER='+server+';DATABASE='+database+';UID='+username+';PWD='+password
cnxn = pyodbc.connect(connection_string)

# cnxn: pyodbc.Connection = pyodbc.connect(connection_string)
cursor = cnxn.cursor()

insert_sql = f"INSERT INTO {target_table} ({target_column}) VALUES (?)"

# insert_sql = f"INSERT INTO {target_table} INSERT_DATETIME, JSON_PAYLOAD VALUES (?)"

params = list(tuple(row) for row in tweet_df.values)
cursor.executemany(insert_sql, params)

cursor.commit()
cursor.close()
