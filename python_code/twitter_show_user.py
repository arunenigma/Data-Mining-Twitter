# -*- coding: utf-8 -*-
import sys
import twitter
import json
screen_name = sys.argv[1]
t = twitter.Twitter(domain='api.twitter.com', api_version='1')
response = t.users.show(screen_name=screen_name)
print json.dumps(response, sort_keys=True, indent=4)
