'''
## Assignment 2
Write a Python program that:
- Reads JSON objects of newsfeeds from the data file into a list or an array of Python dictionaries (or a Pandas dataframe)
- Prints the schema of the JSON object
- Prints the number of newsfeeds (JSON objects) in the collection
- Creates a set of unique newsfeeds by title and prints the new total collection size
- Prints the latest 100 article titles and urls
'''

import json
import pandas as pd
import os
from genson import SchemaBuilder
from datetime import datetime as dt

os.chdir("C:/Github/nlp-analytics/assignments/A2")
# Read in JSON object as a list of Python dictionaries

''' Read JSON object of newsfeeds and store as a list of feeds'''
# Better practice to use with so that once read in, the file is actually closed
with open('webhose_netflix.json') as f:
    data = f.readlines()
# Data will be a massive string object, don't print it lol
#data = open('webhose_netflix.json').readlines()

# Load each individual feed into a list of feeds
feeds = []
for feed in data:
    feeds.append(json.loads(feed))
# Each feed is now a dictionary
# type(feeds[0])

''' Print the JSON schema '''

# Use genson schema builder
proxy = feeds[0]
builder = SchemaBuilder()
builder.add_object(proxy)
builder.to_schema()

# Here's a more lightweight schema that doesn't dive deeper into the schema of dictionaries within a single key
keys = list(proxy.keys())
types = []
for key in keys:
    types.append(type(proxy[key]))

schema = dict(zip(keys, types))
schema

len(feeds)

'''Generate a list of unique feeds and print out the length of this unique list'''

# Create a new dictionary based on unique titles only.
unique_feeds = list({feed['title']: feed for feed in feeds}.values())
# Duplicates will just be rewritten over for the same key
len(unique_feeds) # 19514 unique feeds


'''Get latest 100 feeds with their article title and url'''

# Defining latest feeds as those with the latest PUBLISH DATE from the published key
latest_feeds = sorted(unique_feeds, key=lambda x: x['published'], reverse=True)
latest_100 = latest_feeds[:100]

for feed in latest_100:
    print("Feed from: " + feed['published'])
    print(feed['title'] + ": " + feed['url'] + "\n")