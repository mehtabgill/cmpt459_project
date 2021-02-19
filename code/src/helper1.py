import pandas as pd
import os
import numpy as np
import random
import json
from urllib.request import urlopen

def getplace(lat, lon):
    url = "http://maps.googleapis.com/maps/api/geocode/json?"
    url += "latlng=%s,%s&sensor=false" % (lat, lon)
    v = urlopen(url).read()
    j = json.loads(v)
    components = j['results'][0]['address_components']
    country = town = None
    for c in components:
        if "country" in c['types']:
            country = c['long_name']
        if "postal_town" in c['types']:
            town = c['long_name']
    return town, country

def transform_age(s):
    age_str = str(s)

    for i, ch in enumerate(age_str):
        if not ch.isdigit():
            if age_str[i+1:].startswith('m') or age_str[i+1:].startswith('M'):
                return 0
            else:
                return int(age_str[:i])
    return int(age_str)

# age1 = transform_age("10+")
# age2 = transform_age("10 - 25")
# age3 = transform_age("10-25")
# age4 = transform_age("10 month")
# age5 = transform_age("10 months")
# age6 = transform_age("10 Month")
# print(age1)
# print(age2)
# print(age3)
# print(age4)
# print(age5)
# print(age6)
SEX_CHOICE = ["male", "female"]
SEX_WEIGHTS = [0.55, 0.45]
def generate_sex():
    return random.choices(SEX_CHOICE, SEX_WEIGHTS)[0]


# datetime
def transform_datetime(date_str):
    for i, ch in enumerate(date_str):
        if ch == '-':
            return date_str[:i]
    return date_str


def generate_combined_key(row):
    if pd.isnull(row['province']):
        return row['country']
    else:
        return str(row['province']) + ', ' + row['country']