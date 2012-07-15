# -*- coding: utf-8 -*-
# Author: Arunprasath Shankar
# axs918@case.edu

# calculating tf-idfimport sysfrom math import log10
QUERY_TERMS = sys.argv[1:]
def tf(term, tweet, normalize=True): 
	tweet = tweet.lower().split()	if normalize:		return tweet.count(term.lower()) / float(len(tweet)) 
	else:
		return tweet.count(term.lower()) / 1.0def idf(term, tweets):	num_texts_with_term = len([True for text in tweets if term.lower() in text.lower().split()])
	print num_texts_with_term
	print len(tweets)
# tf-idf calc involves multiplying against a tf value less than 0, so it's important # to return a value greater than 1 for consistent scoring, so I have normalized the idf values by adding 1 to the general formula. (Since multiplying two values # less than 1 returns a value less than each of them)	try:		return 1+(log10(float(len(tweets)) / num_texts_with_term))	except ZeroDivisionError: 
		return 1.0
		def tf_idf(term, tweet, tweets):	return tf(term, tweet) * idf(term, tweets)

# example casetweets = \{'a': 'Mr. Obama pushes the urge to keep low student loan rates President Mr. Obama has the power to change','b': 'Obama touting affordable education to keep low student loan','c': 'RT @BarackObama: Obama urges Congress to prevent student loan interest rates from doubling to save education'}
# Score queries by calculating cumulative tf_idf score for each term in queryquery_scores = {'a': 0, 'b': 0, 'c': 0}for term in [t.lower() for t in QUERY_TERMS]:	for tweet in sorted(tweets):		print 'TF(%s): %s' % (tweet, term), tf(term, tweets[tweet])		print 'IDF: %s' % (term, ), idf(term, tweets.values()) 
	print	for tweet in sorted(tweets):		score = tf_idf(term, tweets[tweet], tweets.values()) 

		print 'TF-IDF(%s): %s' % (tweet, term), score 
		query_scores[tweet] +=score	printprint "Overall TF-IDF scores for query '%s'" % (' '.join(QUERY_TERMS), ) 
for (tweet, score) in sorted(query_scores.items()):	print tweet, score, 

