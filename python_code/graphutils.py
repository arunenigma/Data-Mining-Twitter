import matplotlib.pylab as plt
import networkx as nx


def make_graph_fig(aGraph):
    fig = plt.figure(figsize=(35,35))
    plt.title(aGraph.name)
    pos = nx.spring_layout(aGraph)
    nx.draw(aGraph, pos, node_size = 500)
    return fig

def make_graph_fig2(aGraph):
    fig = plt.figure(figsize=(35,35))
    plt.title(aGraph.name)
    pos = nx.graphviz_layout(aGraph)
    nx.draw(aGraph, pos, node_size = 500)
    return fig

def make_graph_fig3(aGraph):
    fig = plt.figure(figsize=(35,35))
    plt.title(aGraph.name)
    pos = nx.fruchterman_reingold_layout(aGraph)
    nx.draw(aGraph, pos, node_size = 500)
    return fig

def save_graph(aGraph,filename):
    fig = make_graph_fig(aGraph)
    fig.savefig(filename)

def save_graph2(aGraph,filename):
    fig2 = make_graph_fig2(aGraph)
    fig2.savefig(filename)

def save_graph3(aGraph,filename):
    fig3 = make_graph_fig3(aGraph)
    fig3.savefig(filename)

def show_graph(aGraph):
    fig = make_graph_fig(aGraph)
    fig.show()
