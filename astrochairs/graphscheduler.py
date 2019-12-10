__all__ = ["make_graph", "add_edges", "Results",
           "find_solution", "add_edges_numerical", "find_solution_numerical"]

import numpy as np
import copy
import networkx as nx

def make_graph(node_ids, node_labels):
    """
    Make an undirected graph with the nodes specified
    in `node_ids`. Each node will have the attributes
    speficied in `node_labels`.

    Parameters
    ----------
    node_ids : iterable
        The list of node_ids. Can be a list of anything
        that networkx.Graph() can read and use.

    node_labels : dict
        A dictionary of the form {"label":[list of labels
        for all nodes], ...}.
        Defines all labels for all nodes.

    Returns
    -------
    G : networkx.Graph() instance
        The undirected graph containing the nodes and
        corresponding labels, but no edges yet.

    """
    for key in node_labels.keys():
        assert len(node_ids) == len(node_labels[key]), \
            "node_ids and topics for label %s "%key + \
            "are not of equal length, but they should be!"

    G=nx.Graph()
    for i, nid in enumerate(node_ids):
        G.add_node(nid)
        for k in node_labels.keys():
           G.nodes()[nid][k] = node_labels[k][i]

    return G

def add_edges(G, labels=None, hard_constraint=True, weights=None):
    """
    Add edges to the graph, with weights.
    Weights are determined by by the importance weights on
    each label.

    If no order of labels is
    specified, then the order of keys in the dictionary
    for each node will be used.

    TODO: Make hard_constraint an *index* rather than a bool

    Parameters
    ----------
    G : networkx.Graph() instance
        The graph without edges

    labels : list of strings
        A list of labels specifying the order of attributes on each
        node to use when calculating the weights.
        This list should be in descending order (with the most important
        label *first*).
        If none are specified, then the order of keywords in the
        dictionary of attributes for each node will be used.

    hard_constraint : bool
        Boolean flag determining whether hard constraints should be used.
        In this case, this means that for the first label specified in
        `labels`, no edges will be drawn when this label is the same
        for two nodes.

    weights : iterable of float (0, 1]
        The relative weights of each category. By default, the weight of an
        edge will be `weight=1`, adjusted by `weight[i]` for each pair of nodes
        where the labels in category `i` are the same. If `hard_constraints==True`,
        then edges between nodes for which labels in the first category are
        the same do not exist, and `weights` should have length `len(labels)-1`.
        If `hard_constraints == False`, then `len(weights)` should be `len(labels)`,
        where the first entry is used to set the weight of an edge between two
        nodes where `label[0]` has the same value.


    Returns
    -------
    G : networkx.Graph() instance
        The same input graph, but with edges.
    """

    # find the total number of labels
    nlabels = len(labels)

    if weights is not None:
        if hard_constraint:
            assert nlabels-1 == len(weights), "Number of weights must correspond" \
                                              "to the number of topics"
        else:
            assert nlabels == len(weights), "Number of weights must correspond" \
                                        "to the number of topics"

    else:
        weights = np.ones(nlabels)

    # the total number of nodes
    n_nodes = G.number_of_nodes()

    # list of nodes
    nodes = G.nodes()

    # get a list of lists of all node labels
    node_labels = []
    for l in labels:
        node_labels.append([G.nodes()[i][l] for i in G.nodes()])

    # TODO: Currently only works with two labels!
    # iterate over all the different possible labels
    for i, sl in enumerate(node_labels):
        # iterate over all nodes
        for k, n1 in enumerate(G.nodes()):
            for l, n2 in enumerate(G.nodes()):
                #print("n1: " + str(n1))
                #print("n2: " + str(n2))
                # if sessions have the same label,
                # either make no node (for first label),
                # or weaken a node that's already there
                if k == l:
                    #print("k == l, continuing")
                    continue
                if k > l:
                    #print("k > l, continuing")
                    continue

                if hard_constraint:
                    print("using hard constraints")
                    if i == 0:
                        #print("First label")
                        if G.nodes()[n1][labels[i]] == G.nodes()[n2][labels[i]]:
                            #print("Labels are the same, continuing")
                            continue
                        else:
                            #print("Making edge between %i and %i of weight %.2f"%(n1, n2, weights[i]))
                            G.add_edge(n1, n2, weight=1.0)
                    else:
                        #print("Second pass")
                        if G.nodes()[n1][labels[i]] == G.nodes()[n2][labels[i]]:
                            #print("Labels are the same.")
                            #print("Adjusting weight %.2f by %.2f"%(G[n1][n2]["weight"], weights[i]))
                            G[n1][n2]["weight"] *= weights[i-1]
                        else:
                            #print("labels are not the same. Not doing anything.")
                            continue
                else:
                    if i == 0:
                        if G.nodes()[n1][labels[i]] == G.nodes()[n2][labels[i]]:
                            G.add_edge(n1, n2, weight=weights[i])
                        else:
                            G.add_edge(n1, n2, weight=1.0)
                    else:
                        if G.nodes()[n1][labels[i]] == G.nodes()[n2][labels[i]]:
                            G[n1][n2]["weight"] *= weights[i]
                        else:
                            continue

    return G


def add_edges_numerical(G, edges, hard_constraint=True, threshold=0.5):
    """
    Add edges to the graph, with weights.
    Weights are determined by by the importance weights on
    each label.

    If no order of labels is
    specified, then the order of keys in the dictionary
    for each node will be used.

    Parameters
    ----------
    G : networkx.Graph() instance
        The graph without edges

    edges: iterable
        An iterable of edges, in order of (k, l) occurrence, i.e. starts at
        ((0,0), (0,1), (0,2), ..., (1, 2), (1, 3), ..., (n-1, n)).

    hard_constraint : bool
        Boolean flag determining whether hard constraints should be used.
        In this case, this means that for the first label specified in
        `labels`, no edges will be drawn when this label is the same
        for two nodes.

    threshold: float out of [0,1]
        The threshold of the dissimilarity measure describing how dissimilar
        two sessions are. If the threshold is small, the graph has a high tolerance
        to similar sessions. That is, it is more likely to schedule similar sessions
        at the same time. If it is small, then this tolerance is much smaller. There
        will be much fewer edges between nodes. This means that there is a much
        smaller chance that similar sessions will be scheduled at the same time.
        However, fewer edges also means that there's a greater chance that no
        solution will be found at all.


    Returns
    -------
    G : networkx.Graph() instance
        The same input graph, but with edges.
    """

    n_nodes = G.number_of_nodes()

    assert np.size(edges) == (n_nodes*(n_nodes - 1))/2.0, "Incorrect number of edges!"

    # counter for edges
    i = 0

    for k, n1 in enumerate(G.nodes()):
        for l, n2 in enumerate(G.nodes()):
            if k >= l:
                continue
            # if a hard constraint is set, make no edge
            # if the edge weight is smaller than `threshold`,
            # otherwise set the weight edge to 1
            if hard_constraint:
                if edges[i] < threshold:
                    i += 1
                    continue
                else:
                    G.add_edge(n1, n2, weight=edges[i])
            # otherwise just set all the weights the way they are
            else:
                G.add_edge(n1,n2,weight=edges[i])

            i += 1

    return G

def _sort_cliques_by_weights(G, cliques, n_elements):
    """
    Sort cliques by their weights.

    Parameters
    ----------
    G : networkx.Graph instance
        Undirected graph with nodes and edges.
        Edges must have attribute 'weight'

    cliques : iterable
        A list of lists; inner lists must have n_elements members

    n_elements : integer
        The number of elements in each clique

    Returns
    -------
    cliques : iterable
        All cliques sorted by weights in descending order

    summed_weights : iterable
        The list of summed weights, sorted in the
        same descending order as cliques

    """
    # compute summed weights for all cliques:

    cliques = np.asarray(cliques)

    summed_weights = []
    for cl in cliques:
        #print("cl: " + str(cl))
        if len(cl) != n_elements:
            ww = 0
        else:
            ww = 0
            for i in range(n_elements):
                #print("i: " + str(i))
                for j in range(n_elements):
                    #print("j: " + str(j))
                    if i >= j:
                        #print("i >= j, continuing")
                        continue
                    else:
                        ww += G[cl[i]][cl[j]]["weight"]
                        #print("ww_temp: " + str(ww))
        #print("weight: " + str(ww))
        summed_weights.append(ww)

    # sort cliques from highest weight to smallest
    sorted_cliques = cliques[np.argsort(summed_weights)[::-1]]
    # sort weights in the same way
    #print("summed_weights: " + str(summed_weights))

    summed_weights = np.sort(summed_weights)[::-1]
    return sorted_cliques[(summed_weights > 0)], summed_weights[(summed_weights > 0)]


class Results(object):

    def __init__(self, n_elements):
        self.n_elements = n_elements
        self.groups = []
        self.all_weights = []
        self.success = True
        self.weights_sum_total = 0

    def update_groups(self, groups):
        self.groups.append(groups)

    def update_weights(self, weights_sum_total):
        self.all_weights.append(weights_sum_total)
        self.weights_sum_total = np.sum(self.all_weights)

def find_solution(G, n_elements, n_unused=None, results=None):
    """
    Sort nodes in G into groups of n_elements members such that
    the total sum of weights is maximized.
    If the graph includes hard constraints on the relationship between
    nodes (i.e. missing edges), it is possible that no solution is found.

    In the case of a fully connected graph, the solution will be that
    which maximizes the weights. The weights are inherent attributes of
    the Graph and must be calculated beforehand (see `add_edges` for details).

    Parameters
    ----------
    G : networkx.Graph() instance
        Undirected graph with nodes and edges. The edges must have weights
        between 0 and 1, but edges can be missing if no relationship exists
        between nodes.

    groups : iterable
        A list of lists containing all groups of n_elements members fulfilling
        the connectivity constraints that maximize the sum of weights of all
        groups being used.
        Should be initialized with an empty list, will be modified during the
        recursion to be filled with the groups.

    n_elements : integer
        The number of elements per group. Must be an integer divisor of the
        total number of nodes in the graph.

    n_unused : integer
        The number of unused nodes in the graph at every recursion step.
        If None, then it will be initialized as the total number of nodes
        in the graph.

    weights_total_sum : list
        The total sum of weights of elements in `groups`.
        If None, then it will be initialized as an empty list to count
        the sum of weights for each individual group. Will be summed at
        the end before output into a float value.
        Note: DO NOT SET THIS AT THE BEGINNING OF THE RUN!

    Returns
    -------
    success : bool
        Flag indicating success or failure of the algorithm

    groups: iterable
        A list of lists containing all groups of n_elements members fulfilling
        the connectivity constraints that maximize the sum of weights of all
        groups being used.

    weights_total_sum : float
        The total sum of all weights of the output groups

    """

    assert G.number_of_nodes() % np.float(n_elements) == 0, "Number of sessions must be " + \
                                "an integer multiple of n_elements"


    ## initialize results object
    if results is None:
        results = Results(n_elements)

    if n_unused is None:
        n_unused = G.number_of_nodes()


    # base case
    if n_unused == 0:
        results.success = True
        return results

    # recursion
    else:
        ## find all cliques in the graph G
        cliques = list(nx.enumerate_all_cliques(G))

        ## find all cliques that have the required number of elements
        cliques = np.array([c for c in cliques if len(c)==n_elements])

        ## sort cliques by weights
        cliques, summed_weights = _sort_cliques_by_weights(G, cliques, n_elements)

        ## find the total number of cliques with n_elements members
        ncliques = len(cliques)

        ## loop over all cliques:
        for g,(cl,ww) in enumerate(zip(cliques, summed_weights)):
            cl_topics = [G.nodes()[c] for c in cl]

            ## add the new clique to the list of output groups
            results.update_groups(list(zip(cl, cl_topics)))

            ## add total weight of the clique:
            results.update_weights(ww)

            ## make a new deep copy for the next recursion step
            G_new = copy.deepcopy(G)

            ## remove clique from graph
            for n in cl:
                G_new.remove_node(n)

            ## compute new unused number of nodes
            n_unused = G_new.number_of_nodes()

            ## if no unused nodes are left, return the selected groups,
            ## otherwise recurse
            results = find_solution(G_new, n_elements, n_unused, results)
            if results is not None:
                if results.success:
                        return results

            ## backtrack
            else:
                results.success = False
                results.groups.pop(-1)
                results.all_weights.pop(-1)
                continue

    if len(results.groups) == 0:
        print("No solution found!")
        results.success = False
        return results

    else:
        results.groups.pop(-1)
        results.all_weights.pop(-1)

        results.success = False
        return results

def find_solution_numerical(G, n_elements, n_unused=None, results=None):
    """
    Sort nodes in G into groups of n_elements members such that
    the total sum of weights is maximized.
    If the graph includes hard constraints on the relationship between
    nodes (i.e. missing edges), it is possible that no solution is found.

    In the case of a fully connected graph, the solution will be that
    which maximizes the weights. The weights are inherent attributes of
    the Graph and must be calculated beforehand (see `add_edges` for details).

    Parameters
    ----------
    G : networkx.Graph() instance
        Undirected graph with nodes and edges. The edges must have weights
        between 0 and 1, but edges can be missing if no relationship exists
        between nodes.

    n_elements : integer
        The number of elements per group. Must be an integer divisor of the
        total number of nodes in the graph.

    n_unused : integer
        The number of unused nodes in the graph at every recursion step.
        If None, then it will be initialized as the total number of nodes
        in the graph.

    weights_total_sum : list
        The total sum of weights of elements in `groups`.
        If None, then it will be initialized as an empty list to count
        the sum of weights for each individual group. Will be summed at
        the end before output into a float value.
        Note: DO NOT SET THIS AT THE BEGINNING OF THE RUN!

    Returns
    -------
    success : bool
        Flag indicating success or failure of the algorithm

    groups: iterable
        A list of lists containing all groups of n_elements members fulfilling
        the connectivity constraints that maximize the sum of weights of all
        groups being used.

    weights_total_sum : float
        The total sum of all weights of the output groups

    """

    if G.number_of_nodes() % np.float(n_elements) == 0:
        print("Caution! Number of sessions is not an integer "
              "multiple of the number of parallel slots!")

    ## initialize results object
    if results is None:
        results = Results(n_elements)

    if n_unused is None:
        n_unused = G.number_of_nodes()


    ## base case
    if n_unused == 0:
        results.success = True
        return results

    ## recursion
    else:
        ## find all cliques in the graph G
        cliques = list(nx.enumerate_all_cliques(G))

        ## find all cliques that have the required number of elements
        cliques = np.array([c for c in cliques if len(c)==n_elements])

        ## sort cliques by weights
        cliques, summed_weights = _sort_cliques_by_weights(G, cliques, n_elements)

        ## find the total number of cliques with n_elements members
        ncliques = len(cliques)

        ## loop over all cliques:
        for g,(cl,ww) in enumerate(zip(cliques, summed_weights)):
            cl_topics = [G.nodes()[c] for c in cl]

            ## add the new clique to the list of output groups
            results.update_groups(list(zip(cl, cl_topics)))

            ## add total weight of the clique:
            results.update_weights(ww)

            ## make a new deep copy for the next recursion step
            G_new = copy.deepcopy(G)

            ## remove clique from graph
            for n in cl:
                G_new.remove_node(n)

            ## compute new unused number of nodes
            n_unused = G_new.number_of_nodes()

            ## if no unused nodes are left, return the selected groups,
            ## otherwise recurse
            results = find_solution(G_new, n_elements, n_unused, results)
            if results is not None:
                if results.success:
                        return results

            ## backtrack
            else:
                results.success = False
                results.groups.pop(-1)
                results.all_weights.pop(-1)
                continue

    # TODO: Need to add something here to figure out which sessions
    # have potentially been left out because the number of sessions wasn't
    # an integer multiple of the number of slots

    if len(results.groups) == 0:
        print("No solution found!")
        results.success = False
        return results

    else:
        results.groups.pop(-1)
        results.all_weights.pop(-1)

        results.success = False
        return results
