# -*- coding: utf-8 -*-
"""
Created on Thu Jan 19 13:23:58 2023

@author: PascalleVeltman
"""
import pandas as pd
from azureml.core import Workspace, Dataset, Datastore
from textblob import TextBlob
from wordcloud import wordcloud
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import sys
plt.style.use('fivethirtyeight')
import pyodbc
import json
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
import nltk
from rake_nltk import Rake
import os
import yaml
from better_profanity import profanity
# from .functions.py import sentiment_vader
sys.path.append('D:/projects/base/app/modules')       
from nltk.sentiment.vader import SentimentIntensityAnalyzer
sid = SentimentIntensityAnalyzer()
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut
from geopy.exc import GeocoderServiceError
import geopy
import math
import locationtagger
import nltk
import spacy
from geopy.geocoders import Nominatim
import folium
import time

# =============================================================================
# LOAD DATA
# =============================================================================

# read config file
path = os.getcwd()
if '\ONE51 CONSULTING PTY LTD\Projects - Twitter Mock Project\@code' not in path:
    sys.path.append('\ONE51 CONSULTING PTY LTD\Projects - Twitter Mock Project')
    path = path + '\ONE51 CONSULTING PTY LTD\Projects - Twitter Mock Project\@code'
import os
os.chdir(path)
config_path = path + '\\config.yaml'
with open(config_path, "r") as ymlfile:
    cfg = yaml.safe_load(ymlfile)

# Get server and login details
server_name= "twittertestdb"
server = 'tcp:{server_name}.database.windows.net,1433'.format(server_name =server_name)
database = cfg["azure"]["database"]
username = cfg["sql_login2"]["name"]
password = cfg["sql_login2"]["password"]
driver = cfg["details"]["driver"]

# Create connection string for SQL
cnxn = pyodbc.connect('DRIVER={ODBC Driver 17 for SQL Server};SERVER='+server+';DATABASE='+database+';ENCRYPT=yes;UID='+username+';PWD='+ password)
cursor = cnxn.cursor()

# Select new Harry Tweets
harry_new_select = """SELECT * 
    FROM [dbo].[PRINCEHARRY]
    WHERE CAST(
        concat(
            SUBSTRING(JSON_VALUE(CAST(JSON_PAYLOAD AS NVARCHAR(MAX)) ,'$.created_at'),9,2),
            '-',
            SUBSTRING(JSON_VALUE(CAST(JSON_PAYLOAD AS NVARCHAR(MAX)), '$.created_at'),5,3),
            '-',
            RIGHT(JSON_VALUE(CAST(JSON_PAYLOAD AS NVARCHAR(MAX)) , '$.created_at'),4)
            )
    AS DATETIME) > (SELECT COALESCE(MAX(created_at), '2000-01-01 00:00:00.000') FROM [dbo].[ANALYZED_PRINCEHARRY])
"""   
df_raw_harry_new = pd.read_sql(harry_new_select, cnxn)

# Select new Andrew Tate Tweets
andrew_new_select = """SELECT * 
    FROM [dbo].[ANDREWTATE]
    WHERE CAST(
        concat(
            SUBSTRING(JSON_VALUE(CAST(JSON_PAYLOAD AS NVARCHAR(MAX)) ,'$.created_at'),9,2),
            '-',
            SUBSTRING(JSON_VALUE(CAST(JSON_PAYLOAD AS NVARCHAR(MAX)), '$.created_at'),5,3),
            '-',
            RIGHT(JSON_VALUE(CAST(JSON_PAYLOAD AS NVARCHAR(MAX)) , '$.created_at'),4)
            )
    AS DATE) > (SELECT COALESCE(MAX(created_at), '2000-01-01') FROM [dbo].[ANALYZED_ANDREWTATE])
"""   
df_raw_andrew_new = pd.read_sql(andrew_new_select, cnxn)

# Select new Dog Tweets
dog_new_select = """SELECT * 
    FROM [dbo].[DOG]
    WHERE CAST(
        concat(
            SUBSTRING(JSON_VALUE(CAST(JSON_PAYLOAD AS NVARCHAR(MAX)) ,'$.created_at'),9,2),
            '-',
            SUBSTRING(JSON_VALUE(CAST(JSON_PAYLOAD AS NVARCHAR(MAX)), '$.created_at'),5,3),
            '-',
            RIGHT(JSON_VALUE(CAST(JSON_PAYLOAD AS NVARCHAR(MAX)) , '$.created_at'),4)
            )
    AS DATE) > (SELECT COALESCE(MAX(created_at), '2000-01-01') FROM [dbo].[ANALYZED_DOG])
"""   
df_raw_dog_new = pd.read_sql(dog_new_select, cnxn)

# Convert every JSON table into a df with multiple colunms using json_normalize
json_list = []
for i in range(len(df_raw_harry_new)):
    json_list.append(json.loads(list(df_raw_harry_new.iloc[i])[0]))
df_norm_harry = pd.json_normalize(json_list)

json_list = []
for i in range(len(df_raw_dog_new)):
    json_list.append(json.loads(list(df_raw_dog_new.iloc[i])[0]))
df_norm_dog = pd.json_normalize(json_list)

json_list = []
for i in range(len(df_raw_andrew_new)):
    json_list.append(json.loads(list(df_raw_andrew_new.iloc[i])[0]))
df_norm_andrew = pd.json_normalize(json_list)

# Keep only the constant and important fields
selected_fields = ['created_at',
 'id',
 'text',
 'retweet_count',
 'favorite_count',
 'entities.hashtags',
 'entities.user_mentions',
 'metadata.iso_language_code',
 'user.id',
 'user.name',
 'user.screen_name',
 'user.location',
 'user.description',
 'user.followers_count',
 'user.friends_count',
 'user.listed_count',
 'user.created_at',
 'user.favourites_count',
 'user.geo_enabled',
 'user.statuses_count',
 'user.has_extended_profile',
 'user.default_profile']
df_norm_select_harry = df_norm_harry[selected_fields]
df_norm_select_dog = df_norm_dog[selected_fields]
df_norm_select_andrew = df_norm_andrew[selected_fields]

# =============================================================================
# PERFORM VADER SENTIMENT ANALYSIS
# =============================================================================
# Import sentiment functions    
from functions import sentiment_vader

# HARRY VADER NORMAL
df_sentiment_harry = sentiment_vader(df_norm_select_harry)

# ANDREW VADER NORMAL
df_sentiment_andrew = sentiment_vader(df_norm_select_andrew)

# DOG VADER NORMAL
df_sentiment_dog = sentiment_vader(df_norm_select_dog)

# =============================================================================
# PUSH DATA INTO SQL
# =============================================================================
input_dict = {'Harry':     [df_sentiment_harry, 'ANALYZED_PRINCEHARRY'],
              'Andrew':     [df_sentiment_andrew, 'ANALYZED_ANDREWTATE'],
              'Dog':        [df_sentiment_dog, 'ANALYZED_DOG']}

# insert all data into SQL database
for topic in list(input_dict.keys()):
    print(topic)
    df_input = input_dict[topic][0]
    table_name = input_dict[topic][1]
    for index, row in df_input.iterrows():
        str1 = "INSERT INTO " + table_name
        str2 = """ (
          created_at,
          id,
          tweet_text,
          retweet_count,
          favorite_count,
          entities_hashtags,
          entities_user_mentions,
          metadata_iso_language_code,
          tweet_user_id,
          tweet_user_name,
          user_screen_name,
          user_location,
          user_description,
          user_followers_count,
          user_friends_count,
          user_listed_count,
          user_created_at,
          user_favourites_count,
          user_geo_enabled,
          user_statuses_count,
          user_has_extended_profile,
          user_default_profile,
          tracked_location,
          score,
          polarity,
          final_text,
          final_text_without_icons,
          sentiment,
          stars,
          RT
          ) values(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)"""
        str_full = str1 + str2
        cursor.execute(str_full, (row.created_at,
                  row.id,
                  row.tweet_text,
                  row.retweet_count,
                  row.favorite_count,
                  row.entities_hashtags,
                  row.entities_user_mentions,
                  row.metadata_iso_language_code,
                  row.tweet_user_id,
                  row.tweet_user_name,
                  row.user_screen_name,
                  row.user_location,
                  row.user_description,
                  row.user_followers_count,
                  row.user_friends_count,
                  row.user_listed_count,
                  row.user_created_at,
                  row.user_favourites_count,
                  row.user_geo_enabled,
                  row.user_statuses_count,
                  row.user_has_extended_profile,
                  row.user_default_profile,
                  row.tracked_location,
                  row.score,
                  row.polarity,
                  row.final_text,
                  row.final_text_without_icons,
                  row.sentiment,
                  row.stars,
                  row.RT))
cnxn.commit()
cursor.close()






