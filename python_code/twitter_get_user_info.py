# -*- coding: utf-8 -*-
# Author : Arunprasath Shankar
# axs918@case.edu

import sys
import json
import redis
from twitter__login import login

# retrieving user information for the given username passed as a command line argument
from twitter__util import getUserInfo

if __name__ == "__main__":
    screen_names = sys.argv[1:]

    t = login()
    r = redis.Redis()

    print json.dumps(
            getUserInfo(t, r, screen_names=screen_names),
            indent=4
          )
