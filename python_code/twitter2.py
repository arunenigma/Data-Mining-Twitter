# Fetching extended information about a Twitter user
import twitter
import json
screen_name = 'arunenigma'
t = twitter.Twitter(domain = 'api.twitter.com',api_version = '1')
response = t.users.show(screen_name = screen_name)
print json.dumps(response,sort_keys = True,indent = 4)
# Using OAuth to authenticate and grab some friend data
import sys
import time
import cPickle
consumer_key = 'fPnTbRpK2l6vIPw9agJA'
consumer_secret = 'RJdDoZ4ZTvSi4AggFV0OjSqkh9Z9i4AtS80QD2xb4'
SCREEN_NAME = sys.argv[1]
friends_limit = 1000

