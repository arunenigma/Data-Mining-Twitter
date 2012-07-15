# -*- -------------------------------------- -*-
# HEIRARCHIAL CLUSTERING TWITTER DATA
# Author : Arunprasath Shankar
# axs918@case.edu
# -*- -------------------------------------- -*-

import os
import re
import sys
import nltk
import twitter

twitter_api = twitter.Twitter(domain = "api.twitter.com",api_version = '1')
WORLD_WOE_ID = 1
# Paging through Twitter search results
search_results = []
for page in (1,12):
	search_results.append(twitter_api.search(q = sys.argv[1],rpp=100,page=page))
#print search_results
#print json.dumps(search_results,sort_keys=True,indent=1)
tweets = [r['text'] \
	for result in search_results \
		for r in result['results']]

def getwordcounts(tweet):
    wc={}    
    words = []
  
    words+= [w.encode('ascii','ignore') for w in tweet.split()]
    words+= [w for w in words if w.lower() not in nltk.corpus.stopwords.words('english')+[
    '.',
    ',',
    '--',
    '\'s',
    '?',
    ')',
    '(',
    ':',
    '\'',
    '\'re',
    '"',
    '-',
    '}',
    '{',
	'!',
	'.',
	'..',
	'...',
	'....',
	'.....',
	'&',
	'|'
	'1','2','3','4','5','6','7','8','9','0'
    ] and not w.startswith("http")]
    


    #print words
    for word in words:
        wc.setdefault(word,0)
        wc[word]+=1
    return tweet,wc

apcount={}
wordcounts={}
for tweet in tweets:
    tweet,wc = getwordcounts(tweet)
    wordcounts[tweet] = wc
    for word,count in wc.items():
        apcount.setdefault(word,0)
        if count>1:
            apcount[word]+=1

wordlist=[]
for w,bc in apcount.items():
    print w,bc
    #print len(tweets)
    frac=float(bc)/len(tweets)
    if frac>0 and frac<1: wordlist.append(w)

out=file('tweetdata.txt','w')
out.write(sys.argv[1])
for word in wordlist: out.write('\t%s' % word)
out.write('\n')
for tweet,wc in wordcounts.items():
    print tweet,wc
    #deal with unicode outside the ascii range
    tweet=tweet.encode('ascii','ignore')
    out.write(tweet)
    for word in wordlist:
        if word in wc: out.write('\t%d' % wc[word])
        else: out.write('\t0')
    out.write('\n')