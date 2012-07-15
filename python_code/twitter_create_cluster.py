import clusters
tweetnames,words,data=clusters.readfile('tweetdata.txt')
print data
print len(data)
rdata=clusters.rotatematrix(data)
print rdata

wordclust=clusters.hcluster(rdata)
#clusters.printclust(wordclust,labels=words)
clusters.drawdendrogram(wordclust,labels=words,jpeg='wordclust.jpg')
