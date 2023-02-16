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
nltk.download('vader_lexicon')
from rake_nltk import Rake
import os
import yaml
from better_profanity import profanity
sys.path.append('D:/projects/base/app/modules')       
from nltk.sentiment.vader import SentimentIntensityAnalyzer
sid = SentimentIntensityAnalyzer()
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut
from geopy.exc import GeocoderServiceError
import geopy
import math
# import locationtagger
# import spacy
from geopy.geocoders import Nominatim
# import folium
import time

# insert all data into SQL database
def insert_sql_table(topic, cnxn, cursor):
    df_input = topic[0]
    table_name = topic[1]
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

def sentiment_vader(df_input):

    df = df_input.copy()
    
    compound = 0
    positive = 0
    neutral = 0
    negative = 0
    
    df['created_at']= pd.to_datetime(df['created_at'])
    df['Tracked_Location'] = ''
    df['Score'] = ''
    df['Polarity'] = ''
    df['Final_Text'] = ''
    df['Final_Text_Without_Icons'] = ''
    df['Sentiment'] = ''
    df['Stars'] = ''
    df['RT'] = 'No'
    # tp = []
    
    # for row in range(len(df)):
    for row in range(len(df)):

        tweet = df.iloc[row,:]
        
        if tweet.text.startswith('RT'):
            df.loc[(df.id == tweet.id), 'RT'] = 'Yes'
        
        final_text = tweet.text.replace('RT', '')
    
        if final_text.startswith(' @'):
            position = final_text.index(':')
            final_text = final_text[position+2:]
        if final_text.startswith('@'):
            if ' ' in final_text:
                position = final_text.index(' ')
                final_text = final_text[position+2:]
            else: final_text = final_text
        
        final_text = final_text
        final_text_without_icons = process_tweet(final_text)

        analysis = sid.polarity_scores(final_text)
        tweet_compound = analysis['compound']
        if tweet_compound > 0:
            positive += 1
        elif tweet_compound < 0:
            negative += 1
        else:
            neutral += 1
        compound += tweet_compound
        df.loc[(df.id == tweet.id), 'Polarity'] = tweet_compound
        df.loc[(df.id == tweet.id), 'Score'] = str(analysis)
        df.loc[(df.id == tweet.id), 'Final_Text'] = final_text
        df.loc[(df.id == tweet.id), 'Final_Text_Without_Icons'] = final_text_without_icons
        
        # Detect location of user
        # loca = str(get_location(tweet['user.location']))
        loca = '.'
        if loca is not np.nan:
            df.loc[(df.id == tweet.id), 'Tracked_Location'] = str(loca)
        else:
            df.loc[(df.id == tweet.id), 'Tracked_Location'] = str('')
        
        if isinstance(tweet['entities.hashtags'], list):
            if len(tweet['entities.hashtags']) > 0:
                hashtags = []
                for j in tweet['entities.hashtags']:
                    if isinstance(j, dict):
                        hashtags.append(j['text'])
                hashtags = str(hashtags)
                hashtags = hashtags.replace("'", "")
            else:
                hashtags = str([])
        else:
            hashtags = str([])
        df.loc[(df.id == tweet.id), 'entities.hashtags'] = str(hashtags)
        
        if isinstance(tweet['entities.user_mentions'], list):
            if len(tweet['entities.user_mentions']) > 0:
                usermentions = []
                for j in tweet['entities.user_mentions']:
                    if isinstance(j, dict):
                        usermentions.append(j['screen_name'])
                usermentions = str(usermentions)
                usermentions = usermentions.replace("'", "")
            else:
                usermentions = []
        else:
            usermentions = []
        df.loc[(df.id == tweet.id), 'entities.user_mentions'] = str(usermentions)
        
    # Assign sentiment to df
    df.loc[df.Polarity < 0, 'Sentiment'] = 'negative'
    df.loc[(df.Polarity > 0), 'Sentiment'] = 'positive'
    df.loc[(df.Polarity == 0), 'Sentiment'] = 'neutral'
    
    # Assign number of stars to df
    df.loc[(df.Polarity <= -0.6), 'Stars'] = 1
    df.loc[(df.Polarity > -0.6) & (df.Polarity <= -0.2), 'Stars'] = 2
    df.loc[(df.Polarity > -0.2) & (df.Polarity <= 0.2), 'Stars'] = 3
    df.loc[(df.Polarity > 0.2) & (df.Polarity <= 0.6), 'Stars'] = 4
    df.loc[(df.Polarity > 0.6), 'Stars'] = 5
    
    # Display output
    # print(df.Polarity.max())
    # print(positive, neutral, negative)
    
    df = df[['created_at', 'id', 'text', 'retweet_count',
           'favorite_count', 'entities.hashtags', 'entities.user_mentions',
           'metadata.iso_language_code', 'user.id', 'user.name',
           'user.screen_name', 'user.location', 'user.description',
           'user.followers_count', 'user.friends_count', 'user.listed_count',
           'user.created_at', 'user.favourites_count', 'user.geo_enabled',
           'user.statuses_count', 'user.has_extended_profile',
           'user.default_profile', 'Tracked_Location', 'Score', 'Polarity', 'Final_Text', 'Final_Text_Without_Icons', 'Sentiment', 'Stars',
           'RT']]
    
    df.columns = ['created_at',
    'id',
    'tweet_text',
    'retweet_count',
    'favorite_count',
    'entities_hashtags',
    'entities_user_mentions',
    'metadata_iso_language_code',
    'tweet_user_id',
    'tweet_user_name',
    'user_screen_name',
    'user_location',
    'user_description',
    'user_followers_count',
    'user_friends_count',
    'user_listed_count',
    'user_created_at',
    'user_favourites_count',
    'user_geo_enabled',
    'user_statuses_count',
    'user_has_extended_profile',
    'user_default_profile',
    'tracked_location',
    'score',
    'polarity',
    'final_text',
    'final_text_without_icons',
    'sentiment',
    'stars',
    'RT']
    
    return df

def get_url_patern():
 return re.compile(r'(https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9][a-za-z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|https?:\/\/(?:www\.|(?!www))'
 r'[a-zA-Z0-9]\.[^\s]{2,}|www\.[a-zA-Z0-9]\.[^\s]{2,})')

def get_hashtags_pattern():
 return re.compile(r'#\w\*')

def get_single_letter_words_pattern():
 return re.compile(r'(?<![\w-])\w(?![\w-])')

def get_blank_spaces_pattern():
 return re.compile(r'\s{2,}|\t')

def get_twitter_reserved_words_pattern():
 return re.compile(r'(RT|rt|FAV|fav|VIA|via)')

def get_mentions_pattern():
 return re.compile(r'@\w\*')

def get_mentions_pattern():
 return re.compile(r'@\w\*')

def process_tweet(word):
    word=re.sub(pattern=get_url_patern(), repl="", string=word)
    word=re.sub(pattern=get_mentions_pattern(), repl="", string=word)
    word=re.sub(pattern=get_hashtags_pattern(), repl="", string=word)
    word=re.sub(pattern=get_twitter_reserved_words_pattern(), repl='', string=word)
    word=re.sub (r'http\S+', "", word) # remove http links
    word=re.sub(r'bit.ly/\S+', "", word) # remove bitly links
    word=word.strip('[link]') # remove [links]
    word=re.sub('(RT\s@[A-Za-z]+[A-Za-z0-9-*]+)', "", word) # remove retweet
    word=re.sub('(@[A-Za-z]+[A-Za-z0-9-_]+)', "", word) # remove tweeted at
    word=word.encode('ascii', 'ignore').decode('ascii')
    return word

# =============================================================================
# LOAD DATA
# =============================================================================

# read config file
path = os.getcwd()
os.chdir(path)

# config_path = path + '/config.yaml'
config_path = '/home/ubuntu/Documents/PythonProjects/TweepyAzureAnalysis/config.yaml'
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
cnxn = pyodbc.connect('DRIVER={ODBC Driver 18 for SQL Server};SERVER='+server+';DATABASE='+database+';ENCRYPT=yes;UID='+username+';PWD='+ password)
cursor = cnxn.cursor()

# Select new Harry Tweets
harry_new_select = """SELECT * 
    FROM [dbo].[PRINCEHARRY]
    WHERE 
		CONVERT(
			DATETIME
			,
			concat(
            SUBSTRING(JSON_VALUE(CAST(JSON_PAYLOAD AS NVARCHAR(MAX)) ,'$.created_at'),9,2),
            '-',
            SUBSTRING(JSON_VALUE(CAST(JSON_PAYLOAD AS NVARCHAR(MAX)), '$.created_at'),5,3),
            '-',
            RIGHT(JSON_VALUE(CAST(JSON_PAYLOAD AS NVARCHAR(MAX)) , '$.created_at'),4),
			' ',
			SUBSTRING(JSON_VALUE(CAST(JSON_PAYLOAD AS NVARCHAR(MAX)) ,'$.created_at'),12,8) 
            )
			, 106) > (SELECT COALESCE(MAX(created_at), '2000-01-01 00:00:00.000') FROM [dbo].[ANALYZED_PRINCEHARRY])
"""   
df_raw_harry_new = pd.read_sql(harry_new_select, cnxn)

# Select new Andrew Tate Tweets
andrew_new_select = """SELECT * 
    FROM [dbo].[ANDREWTATE]
    WHERE CONVERT(
			DATETIME
			,
			concat(
            SUBSTRING(JSON_VALUE(CAST(JSON_PAYLOAD AS NVARCHAR(MAX)) ,'$.created_at'),9,2),
            '-',
            SUBSTRING(JSON_VALUE(CAST(JSON_PAYLOAD AS NVARCHAR(MAX)), '$.created_at'),5,3),
            '-',
            RIGHT(JSON_VALUE(CAST(JSON_PAYLOAD AS NVARCHAR(MAX)) , '$.created_at'),4),
			' ',
			SUBSTRING(JSON_VALUE(CAST(JSON_PAYLOAD AS NVARCHAR(MAX)) ,'$.created_at'),12,8) 
            )
			, 106) > (SELECT COALESCE(MAX(created_at), '2000-01-01 00:00:00.000') FROM [dbo].[ANALYZED_ANDREWTATE])
"""   
df_raw_andrew_new = pd.read_sql(andrew_new_select, cnxn)

# Select new Dog Tweets
dog_new_select = """SELECT * 
    FROM [dbo].[DOG]
    WHERE CONVERT(
			DATETIME
			,
			concat(
            SUBSTRING(JSON_VALUE(CAST(JSON_PAYLOAD AS NVARCHAR(MAX)) ,'$.created_at'),9,2),
            '-',
            SUBSTRING(JSON_VALUE(CAST(JSON_PAYLOAD AS NVARCHAR(MAX)), '$.created_at'),5,3),
            '-',
            RIGHT(JSON_VALUE(CAST(JSON_PAYLOAD AS NVARCHAR(MAX)) , '$.created_at'),4),
			' ',
			SUBSTRING(JSON_VALUE(CAST(JSON_PAYLOAD AS NVARCHAR(MAX)) ,'$.created_at'),12,8) 
            )
			, 106) > (SELECT COALESCE(MAX(created_at), '2000-01-01 00:00:00.000') FROM [dbo].[ANALYZED_DOG])
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

# Per topic, get the dataframes, perform sentiment analysis and push the dat into the analyzed SQL table
if len(df_norm_harry)>0:
    df_norm_select_harry = df_norm_harry[selected_fields]
    df_sentiment_harry = sentiment_vader(df_norm_select_harry)
    insert_sql_table([df_sentiment_harry, 'ANALYZED_PRINCEHARRY'], cnxn, cursor)
else:
    print('No new Prince Harry Tweets')

if len(df_norm_andrew)>0:
    df_norm_select_andrew = df_norm_andrew[selected_fields]
    df_sentiment_andrew = sentiment_vader(df_norm_select_andrew)
    insert_sql_table([df_sentiment_andrew, 'ANALYZED_ANDREWTATE'], cnxn, cursor)
else:
    print('No new Andrew Tate Tweets')

if len(df_norm_dog)>0:
    df_norm_select_dog = df_norm_dog[selected_fields]
    df_sentiment_dog = sentiment_vader(df_norm_select_dog)
    insert_sql_table([df_sentiment_dog, 'ANALYZED_DOG'], cnxn, cursor)
else:
    print('No new Dog Tweets')

cnxn.commit()
cursor.close()




