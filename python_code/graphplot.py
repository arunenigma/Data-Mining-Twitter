import pylab
import networkx as nx

def make_histogram(aGraph):     
    fig = pylab.figure()
    pylab.title(aGraph.name)
    hist = nx.degree_histogram(aGraph)
    pylab.bar(range(len(hist)), hist, align = 'center')
    pylab.xlim((0, len(hist)))
    pylab.xlabel("Degree of node")
    pylab.ylabel("Number of nodes")
    return fig

def save_histogram(aGraph,filename):
    fig = make_histogram(aGraph)
    fig.savefig(filename)