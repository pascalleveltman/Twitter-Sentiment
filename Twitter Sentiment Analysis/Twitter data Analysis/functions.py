# -*- coding: utf-8 -*-
"""
Created on Fri Jan 20 11:03:59 2023

@author: PascalleVeltman
"""

# =============================================================================
# IMPORT
# =============================================================================
# Import Classes
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
import re
from better_profanity import profanity
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
from tqdm import tqdm
from textblob import TextBlob
from pattern.en import sentiment
from pattern.en import parse
from pattern.en import pprint
import spacy
from flair.models import TextClassifier
from flair.data import Sentence
# =============================================================================
# MAIN FUNCTIONS
# =============================================================================

def sentiment_all(df_input):
    
    # nlp = spacy.load("en_core_web_sm")
    
    
    df = df_input.copy()
    
    df['created_at']= pd.to_datetime(df['created_at'])
    df['Textblob_polarity'] = ''
    df['VADER_polarity'] = ''
    df['Flair_polarity'] = ''
    df['Pattern_polarity'] = ''
    df['VADER_score'] = ''
    df['Pattern_subjectivity'] = ''
    df['Final_Text'] = ''
    df['Final_Text_Without_Icons'] = ''
    # classifier = TextClassifier.load('en-sentiment')

    
    # for row in range(len(df)):
    for row in tqdm(range(len(df))):
        
        # print('new tweet')
        tweet = df.iloc[row,:]
        
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
        
        # Get vader sentiment
        analysis = sid.polarity_scores(final_text)
        tweet_compound = analysis['compound']
        
        # Get TextBob Sentiment
        textblob_polarity = TextBlob(final_text).sentiment.polarity
        
        pattern_sent = sentiment(final_text)
        pattern_polarity = pattern_sent[0]
        
        # sentence = Sentence(final_text)
        # classifier.predict(sentence)
        # flair_polarity = str(sentence.labels)
        
        # if tweet_compound > 0:
        #     positive += 1
        # elif tweet_compound < 0:
        #     negative += 1
        # else:
        #     neutral += 1
        # compound += tweet_compound
        
        df.loc[(df.id == tweet.id), 'VADER_polarity'] = tweet_compound
        df.loc[(df.id == tweet.id), 'VADER_score'] = str(analysis)
        df.loc[(df.id == tweet.id), 'Textblob_polarity'] = textblob_polarity
        # df.loc[(df.id == tweet.id), 'Flair_polarity'] = flair_polarity
        df.loc[(df.id == tweet.id), 'Pattern_subjectivity'] = pattern_sent[1]
        df.loc[(df.id == tweet.id), 'Pattern_polarity'] = pattern_sent[0]
        df.loc[(df.id == tweet.id), 'Final_Text'] = final_text
        df.loc[(df.id == tweet.id), 'Final_Text_Without_Icons'] = final_text_without_icons
    
    # Assign sentiment to df
    df.loc[df.Textblob_polarity < 0, 'Sentiment_textblob'] = 'negative'
    df.loc[(df.Textblob_polarity > 0), 'Sentiment_textblob'] = 'positive'
    df.loc[(df.Textblob_polarity == 0), 'Sentiment_textblob'] = 'neutral'
    
    # Assign sentiment to df
    df.loc[df.VADER_polarity < 0, 'Sentiment_VADER'] = 'negative'
    df.loc[(df.VADER_polarity > 0), 'Sentiment_VADER'] = 'positive'
    df.loc[(df.VADER_polarity == 0), 'Sentiment_VADER'] = 'neutral'
        
    return df
    

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
    for row in tqdm(range(len(df))):
        # print('new tweet')
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

def get_location(text1):

    if (text1 is not np.nan) & (text1 != 'nan') & (text1 != 'NaN') & (not text1.isspace()) & (text1 != ''):
        # extracting entities.
        # print('found:', text1)
        geolocator = Nominatim(user_agent="geoapiExercises")
        place_entity = locationtagger.find_locations(text = text1)
        geopy.geocoders.options.default_timeout = 10
        geopy.geocoders.options.default_retries = 10
        found = 0
        if (len(place_entity.countries) == 0) & (len(place_entity.regions) == 0) & (len(place_entity.cities) == 0) & (not place_entity.country_regions) & (not place_entity.country_cities) & (len(place_entity.other_countries) == 0) & (not place_entity.region_cities) & (len(place_entity.other_regions) == 0):
            found = 0
            # print('hi1')
        else:
            try:
                location = geolocator.geocode(text1)
                found = 1
                # print('hi2')
            except GeocoderServiceError as e:
                found = 0
                # print('hi3')
        if found == 1:
            # print('hi4')
            return location
        else:
            # print('hi5')
            return np.nan
    else: 
        # print('not found')
        return np.nan

def get_country(text1):

    if (text1 is not np.nan) & (text1 != 'nan') & (text1 != 'NaN') & (not text1.isspace()) & (text1 != ''):
        # extracting entities.
        # print('found:', text1)
        geolocator = Nominatim(user_agent="geoapiExercises")
        place_entity = locationtagger.find_locations(text = text1)
        geopy.geocoders.options.default_timeout = 10
        geopy.geocoders.options.default_retries = 10
        found = 0
        if (len(place_entity.countries) == 0) & (len(place_entity.regions) == 0) & (len(place_entity.cities) == 0) & (not place_entity.country_regions) & (not place_entity.country_cities) & (len(place_entity.other_countries) == 0) & (not place_entity.region_cities) & (len(place_entity.other_regions) == 0):
            found = 0
            # print('hi1')
        else:
            try:
                location = geolocator.geocode(text1)
                found = 1
                # print('hi2')
            except GeocoderServiceError as e:
                found = 0
                # print('hi3')
        if found == 1:
            # print('hi4')
            try: 
                country = location.address.split(",")[-1].strip()
                return country
            except:
                return np.nan
        else:
            # print('hi5')
            return np.nan
    else: 
        # print('not found')
        return np.nan
    
def get_country2(text1):

    if (text1 is not np.nan) & (text1 != 'nan') & (text1 != 'NaN') & (not text1.isspace()) & (text1 != ''):
        geolocator = Nominatim(user_agent="geoapiExercises")
        try:
            location = geolocator.geocode(text1)
            found = 1
            # print('hi2')
        except GeocoderServiceError as e:
            found = 0
            # print('hi3')
        if found == 1:
            # print('hi4')
            try: 
                country = location.address.split(",")[-1].strip()
                return country
            except:
                return np.nan
        else:
            # print('hi5')
            return np.nan
    else: 
        # print('not found')
        return np.nan
    
    
# def get_location2(text1):
    
#     geolocator = Nominatim(user_agent="geoapiExercises")
#     # place_entity = locationtagger.find_locations(text = text1)
#     geopy.geocoders.options.default_timeout = 10
#     geopy.geocoders.options.default_retries = 10
    
#     try:
#         location = geolocator.geocode(text1)
#         return location
#     except GeocoderServiceError as e:
#         found = 0
#         return np.nan
#     # if (text1 is not np.nan) & (text1 != 'nan') & (text1 != 'NaN') & (not text1.isspace()) & (text1 != ''):
#     #     geolocator = Nominatim(user_agent="geoapiExercises")
#     #     place_entity = locationtagger.find_locations(text = text1)
#     #     geopy.geocoders.options.default_timeout = 10
#     #     geopy.geocoders.options.default_retries = 10
    
