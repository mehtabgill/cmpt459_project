import pandas as pd
import os
import numpy as np
import random
import json
from urllib.request import urlopen
from requests import get, post, HTTPError
import http.client
from requests_oauthlib import OAuth1



def getplacehere(lat, lon):
    url = "https://revgeocode.search.hereapi.com/v1/revgeocode?"
    url += "at=%s,%s&lang=en-US" % (lat, lon)

    clientId = "_D6emiCPSlTUF4em-90Bcw"
    clientSecret = "Mc6o_3iosEwN9cxDVIOzRk0d2q4cucFL7ocarIjyGnQ9ddRxhWcIMFk6al3pO78NYSsgO7DjycZ6_I_-9pB8qg"
    data = {
        'grantType': 'client_credentials',
        'clientId': clientId,
        'clientSecret': clientSecret
        }

    response = post(
        url='https://account.api.here.com/oauth2/token',
        auth=OAuth1(clientId, client_secret=clientSecret) ,
        headers={'Content-type': 'application/json'},
        data=json.dumps(data)).json()

    token, token_type, expire_in = None, None, None
    try:
        token = response['accessToken']
        token_type = response['tokenType']
        expire_in = response['expiresIn']
    except KeyError as e:
        print(json.dumps(response, indent=2))
        exit(1)

    headers = {'Authorization': f'{token_type} {token}'}
    search_results = json.dumps(get(url, headers=headers).json(), indent=2)
    # print(f'lat:\n{lat}')
    # print(f'lon:\n{lon}')
    # print(f'results:\n{search_results}')
    # print("Debug: Trying to load json_result")
    json_result = json.loads(search_results)

    if len(json_result['items']) > 0:
        # print("Debug: Executed")
        # print(json_result['items'])
        if 'state' in json_result['items'][0]['address']:
            state_value = json_result['items'][0]['address']['state']
            return state_value
        else:
            return np.nan
    else:
        return np.nan

def get_province(row):
    return getplacehere(row['latitude'], row['longitude'])

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