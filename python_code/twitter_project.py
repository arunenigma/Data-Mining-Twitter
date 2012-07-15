# -*- -------------------------------------- -*-
# MINING, ANALYZING AND VISUALIZING TWITTER DATA
# Author : Arunprasath Shankar
# axs918@case.edu
# -*- -------------------------------------- -*-

import os
import re
import sys
import nltk
import json
import numpy
import shutil
import twitter
import cPickle
import graphplot
import graphutils
import webbrowser
import networkx as nx
import matplotlib.pyplot as plt
from operator import itemgetter

# saving output log to a file
class Logger(object):
	def __init__(self, filename="Default.log"):
		self.terminal = sys.stdout
		self.log = open(filename, "a")

	def write(self, message):
		self.terminal.write(message)
		self.log.write(message)

sys.stdout = Logger("/Users/arunprasathshankar/Desktop/twitter_project_results.txt")
#print 'RESULTS FOR THE SEARCH QUERIES  ' + sys.argv[1] + '  and  ' + sys.argv[2]

print 'RESULTS FOR THE SEARCH QUERY  ' + sys.argv[1] 
twitter_api = twitter.Twitter(domain = "api.twitter.com",api_version = '1')
WORLD_WOE_ID = 1
world_trends = twitter_api.trends._(WORLD_WOE_ID)
trends = [trend for trend in world_trends()[0]['trends']]
print
print 'CURRENT TWITTER TRENDING TOPIC '
print '______________________'
print
print trends
for t in trends:
	print str(t)
print
print
# Paging through Twitter search results
search_results = []
for page in (1,12):
	search_results.append(twitter_api.search(q = sys.argv[1],rpp=100,page=page))
#print search_results
print
print
#print json.dumps(search_results,sort_keys=True,indent=1)
tweets = [r['text'] \
	for result in search_results \
		for r in result['results']]
print 'TWEETS FOR THE QUERY  ' + sys.argv[1]
print '_________________________________'
print
#print tweets
for t in tweets:
	print t.encode('ascii', 'ignore')
	print '.........................................................................................'
print
print
words = []
for t in tweets:
	words+=[w for w in t.split()]
#print words
print 'WORDS'
print '_____'
print
for w in words:
	print w.encode('ascii', 'ignore')
print
print
print 'TOTAL WORDS = ',len(words) # total words
print
print
print 'UNIQUE WORDS = ',len(set(words)) # unique words
print
print
lex_div = 1.0 * len(set(words))/len(words) # lexical diversity
print 'LEXICAL DIVERSITY = ',lex_div
print
print
# Average words per tweets
avg_words = 1.0 * sum([len(t.split()) for t in tweets])/len(tweets) 
print 'AVERAGE WORDS = ',avg_words
print
print
# Pickling data
f = open("../python_code/myData.pickle",'wb')
cPickle.dump(words,f)
f.close()

# Using NLTK to perform basic frequency analysis
words = cPickle.load(open('../python_code/myData.pickle'))
freq_dist = nltk.FreqDist(words)
most_freq = []
least_freq = []
words_new = []
for w in words:
	w = w.encode('ascii','ignore')
	words_new.append(w)
text = nltk.Text(words_new)

fdist = text.vocab()
rank_list = []
fdist_word_list = []
print 'RANK     WORDS     FREQUENCY'
print '____________________________'
for rank, word in enumerate(fdist): 
	print rank+1, word.encode('ascii', 'ignore'), fdist[word]
	rank_list.append(rank)
	fdist_word_list.append(fdist[word])
# freq analysis - Histogram
#for rank, word in enumerate(fdist): 
	#print rank, word, fdist[word]
	#plt.plot(rank,fdist[word],'r_')
plt.clf()
fig = plt.figure()
plt.plot(rank_list,fdist_word_list,'b-',linewidth = 2.0)
plt.xlabel('Rank')
plt.ylabel('Frequency')
plt.title('Frequency Analysis(Not Filtered)')
plt.axis([0, max(rank_list), 0, max(fdist_word_list)])
plt.grid(True)

fig.savefig('/Users/arunprasathshankar/Desktop/freq_not_filtered.png',dpi=fig.dpi)
print
print
# 50 most frequent tokens
print '50 MOST FREQUENT TOKENS' 
print '_______________________'
print
most_freq = freq_dist.keys()[:50]
for mf in most_freq:
	print mf.encode('ascii', 'ignore'),
print
print
# 50 least frequent tokens
print '50 LEAST FREQUENT TOKENS'
print '________________________'
print
least_freq = freq_dist.keys()[-50:] 
for lf in least_freq:
	print lf.encode('ascii', 'ignore'),
print
print
# Using re module(regular expressions to find retweets)


rt_patterns = re.compile(r"(RT|via)((?:\b\W*@\w+)+)",re.IGNORECASE)
print 'RETWEETS'
print '________'
print
for t in tweets:
	print rt_patterns.findall(t)

g = nx.DiGraph()
all_tweets = [tweet for page in search_results for tweet in page["results"]]
print 
print

def get_rt_sources(tweet):
	rt_patterns = re.compile(r"(RT|via)((?:\b\W*@\w+)+)",re.IGNORECASE)
	return[source.strip() for tuple in rt_patterns.findall(tweet) for source in tuple if source not in ("RT","via")]

for tweet in all_tweets:
	rt_sources = get_rt_sources(tweet["text"])
	if not rt_sources: continue
	for rt_source in rt_sources:
		g.add_edge(rt_source,tweet["from_user"],{"tweet_id":tweet["id"]})
print 'NUMBER OF NODES = ', g.number_of_nodes()
print
print 'NUMBER OF EDGES = ',g.number_of_edges()
print
#print g.edges(data=True)[0]
print 'CONNECTED COMPONENTS = ', len(nx.connected_components(g.to_undirected()))
print
print 'DEGREE HISTOGRAM = ', nx.degree_histogram(g)
print
graphplot.save_histogram(g,"/Users/arunprasathshankar/Desktop/hist.png")
graphutils.save_graph(g,"/Users/arunprasathshankar/Desktop/spring_layout.png")
graphutils.save_graph2(g,"/Users/arunprasathshankar/Desktop/graphviz_layout.png")
graphutils.save_graph3(g,"/Users/arunprasathshankar/Desktop/fruchterman_reingold_layout.png")

all_content = " ".join([ p.encode('ascii', 'ignore') for p in tweets ])

tokens = all_content.split()
tokens = [w for w in tokens if w.lower() not in nltk.corpus.stopwords.words('english')+[
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
	'1','2','3','4','5','6','7','8','9','0'
    ]]
print 'WORDS EXCLUDING STOP WORDS/SYMBOLS/NUMBERS'
print '__________________________'
print
for w in tokens:
	print w
tokens = [w for w in tokens if len(w) < 15 and not w.startswith("http")]
print
print
print 'WORDS EXCLUDING HTTPs'
print '____________________'
print
for w in tokens:
	print w
print
print
freq_dist = nltk.FreqDist(tokens)
most_freq = []
least_freq = []
print
print
# 50 most frequent tokens
print '50 MOST FREQUENT TOKENS AFTER FILTERING OUT STOPWORDS/SYMBOLS/NUMBERS' 
print '_______________________'
print
most_freq = freq_dist.keys()[:50]
for mf in most_freq:
	print mf.encode('ascii', 'ignore'),
print
print
# 50 least frequent tokens
print '50 LEAST FREQUENT TOKENS AFTER FILTERING OUT STOPWORDS/SYMBOLS/NUMBERS'
print '________________________'
print
least_freq = freq_dist.keys()[-50:] 
for lf in least_freq:
	print lf.encode('ascii', 'ignore'),
print
print
text = nltk.Text(tokens)
fdist = text.vocab()
print 'COLLOCATIONS: ', text.collocations()
print '____________'
print
print
rank_list = []
fdist_word_list = []
print 'RANK     WORDS     FREQUENCY'
print '____________________________'
for rank, word in enumerate(fdist): 
	print rank+1, word.encode('ascii', 'ignore'), fdist[word]
	rank_list.append(rank)
	fdist_word_list.append(fdist[word])
# freq analysis - Histogram
#for rank, word in enumerate(fdist): 
	#print rank, word, fdist[word]
	#plt.plot(rank,fdist[word],'r_')
plt.clf()
fig = plt.figure()
plt.plot(rank_list,fdist_word_list,'r-',linewidth = 2.0)
plt.xlabel('Rank')
plt.ylabel('Frequency')
plt.title('Frequency Analysis (Filtered)')
plt.axis([0, max(rank_list), 0, max(fdist_word_list)])
plt.grid(True)

fig.savefig('/Users/arunprasathshankar/Desktop/freq_filtered.png',dpi=fig.dpi)

#####################################################################
# Querying Twitter Data with TF-IDF
QUERY_TERMS = sys.argv[2:]
# Provides tf/idf/tf_idf abstractions
tc = nltk.TextCollection(tweets)
relevant_tweets = []for idx in range(len(tweets)):
	score = 0

	for term in [t.lower() for t in QUERY_TERMS]:		score += tc.tf_idf(term, tweets[idx]) 
	if score > 0:		relevant_tweets.append({'score': score, 'title': tweets[idx]})# Sort by score and display resultsrelevant_tweets = sorted(relevant_tweets, key=lambda p: p['score'], reverse=True) 
#print relevant_tweets
for p in relevant_tweets:	print p['title'].encode('ascii', 'ignore')
	print '\tScore: %s' % (p['score'], )



print '..........................................................................................'

# Clustering Tweets using Cosine Similarity and Vector Space Model (Finding similar tweets)

td_matrix = {}for idx in range(len(tweets)):	tweet = tweets[idx]	fdist = nltk.FreqDist(tweet)	tweet_title = tweets[idx]
	td_matrix[(tweet_title)] = {}	for term in fdist.iterkeys():		td_matrix[(tweet_title)][term] = tc.tf_idf(term, tweet)
# Build vectors such that term scores are in the same positions...distances = {}
for (title1) in td_matrix.keys():	distances[(title1)] = {}
	(max_score, most_similar) = (0.0, (None, None)) 
	for (title2) in td_matrix.keys():# mutating the original data structures
# since we're in a loop and need the originals multiple times		terms1 = td_matrix[(title1)].copy() 
		terms2 = td_matrix[(title2)].copy()
				# Fill in "gaps" in each map so vectors of the same length can be computed		for term1 in terms1:			if term1 not in terms2:				terms2[term1] = 0		for term2 in terms2:			if term2 not in terms1:				terms1[term2] = 0
				# Create vectors from term maps		v1 = [score for (term, score) in sorted(terms1.items())] 
		v2 = [score for (term, score) in sorted(terms2.items())]
		#Compute similarity among tweets		distances[(title1)][(title2)] = nltk.cluster.util.cosine_distance(v1, v2)		if distances[(title1)][(title2)] > max_score:
			(max_score, most_similar) = (distances[(title1)][(title2)],(title2))
	print 'Most similar ---> %s\t\t\t%s\t\t\tscore %s' % (title1.encode('ascii', 'ignore'),most_similar.encode('ascii', 'ignore'), max_score)

# HTML templmates that we'll inject Protovis consumable data into
HTML_TEMPLATES = ['../web_code/protovis/matrix_diagram.html', 
                  '../web_code/protovis/arc_diagram.html']


# Compute the standard deviation for the distances as a basis of thresholding
std = numpy.std([distances[k1][k2] for k1 in distances for k2 in distances[k1]])

similar = []
keys = td_matrix.keys()

for k1 in keys:
    for k2 in keys:
        if k1 == k2:
            continue

        d = distances[k1][k2]
        if d < std / 2 and d > 0.000001:  # call them similar
            (title1) = k1
            (title2) = k2
            similar.append((k1, k2, distances[k1][k2]))
print similar

# Emit output expected by Protovis.

nodes = {}
node_idx = 0
edges = []
for s in similar:
    if s[0] not in nodes:
        nodes[s[0]] = node_idx
        node_idx += 1
    node0 = nodes[s[0]]

    if s[1] not in nodes:
        nodes[s[1]] = node_idx
        node_idx += 1
    node1 = nodes[s[1]]
    edges.append({'source': node0, 'target': node1, 'value': s[2]*30})


nodes = [{'nodeName': title} for ((title),idx) in
         sorted(nodes.items(), key=itemgetter(1))]

json_data = {'nodes': nodes, 'links': edges}

# json_data consumed by matrix_diagram.html
if not os.path.isdir('out'):
    os.mkdir('out')


shutil.rmtree('out/protovis-3.2', ignore_errors=True)

shutil.copytree('../web_code/protovis/protovis-3.2',
                'out/protovis-3.2')

for template in HTML_TEMPLATES:
    html = open(template).read() % (json.dumps(json_data),)
    f = open(os.path.join(os.getcwd(), 'out', os.path.basename(template)), 'w')
    f.write(html)
    f.close()

    print >> sys.stderr, 'Data file written to: %s' % f.name
    webbrowser.open('file://' + f.name)

# calculating lines of code of Project 
cur_path = os.getcwd()
ignore_set = set(["__init__.py", "count_sourcelines.py"])

loclist = []

for pydir, _, pyfiles in os.walk(cur_path):
	for pyfile in pyfiles:
		if pyfile.endswith(".py") and pyfile not in ignore_set:
			totalpath = os.path.join(pydir, pyfile)
			loclist.append( ( len(open(totalpath, "r").read().splitlines()),totalpath.split(cur_path)[1]) )

for linenumbercount, filename in loclist: 
    print "%05d lines in %s" % (linenumbercount, filename)

print "\nTotal: %s lines (%s)" %(sum([x[0] for x in loclist]), cur_path)


