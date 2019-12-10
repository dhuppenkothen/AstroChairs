import numpy as np
import networkx

import pytest

from astrochairs import make_graph, add_edges, add_edges_numerical
from astrochairs.graphscheduler import _sort_cliques_by_weights
from astrochairs import Results

def test_make_graph_works():
    node_ids = [1, 2, 3]
    node_labels = {"topic1": ["black holes", "galaxies", "cosmology"]}

    G = make_graph(node_ids, node_labels)

def test_make_graph_fails_with_uneven_input_lengths():
    node_ids = [1, 2, 3]
    node_labels = {"topic1": ["black holes", "galaxies"]}

    with pytest.raises(AssertionError):
        G = make_graph(node_ids, node_labels)

def test_graph_is_set_up_correctly():
    node_ids = [1, 2, 3]

    key = "topic1"
    node_labels = {key: ["black holes", "galaxies", "cosmology"]}

    G = make_graph(node_ids, node_labels)

    assert list(G.nodes()) == node_ids
    assert G.number_of_nodes() == len(node_ids)
    assert [G.nodes()[i]["topic1"] for i in node_ids] == node_labels[key]
    assert list(G.nodes(data=True)[1].keys())[0] == key


def test_graph_works_with_random_node_ids():
    node_ids = [1, 56, 126]

    key = "topic1"
    node_labels = {key: ["black holes", "galaxies", "cosmology"]}

    G = make_graph(node_ids, node_labels)

    assert list(G.nodes()) == node_ids

def test_graph_works_with_two_topics():
    node_ids = [1, 2, 3]
    session_labels1 = ["black holes", "galaxies", "cosmology"]
    key1 = "topic1"
    session_labels2 = ["cosmology", "black holes", "galaxies"]
    key2 = "topic2"

    node_labels = {key1: session_labels1,
                   key2: session_labels2}

    G = make_graph(node_ids, node_labels)

    for i, s1, s2 in zip(node_ids, session_labels1, session_labels2):
        assert G.nodes[i] == {key1:s1, key2:s2}

def test_add_edges_initializes():
    node_ids = [1, 2, 3]
    key = "topic1"
    node_labels = {key: ["black holes", "galaxies", "cosmology"]}
    G = make_graph(node_ids, node_labels)

    G_new = add_edges(G, labels=["topic1"], hard_constraint=True)
    edges = list(G_new.edges())

    true_edges = [(1, 2), (1, 3), (2, 3)]

    assert np.all(edges == true_edges)

def test_add_edges_works_with_same_topic():
    node_ids = [1, 2, 3]
    key = "topic1"
    node_labels = {key: ["cosmology", "galaxies", "cosmology"]}
    G = make_graph(node_ids, node_labels)

    G_new = add_edges(G, labels=["topic1"], hard_constraint=True)
    edges = list(G_new.edges())

    assert (1, 3) not in edges

def test_add_edges_without_hard_constraint():
    node_ids = [1, 2, 3]
    key = "topic1"
    node_labels = {key: ["cosmology", "galaxies", "cosmology"]}
    G = make_graph(node_ids, node_labels)

    G_new = add_edges(G, labels=["topic1"], hard_constraint=False)
    edges = G_new.edges()

    assert edges()[1,3]["weight"] == 1.0

def test_add_edges_fail_with_wrong_labels():
    node_ids = [1, 2, 3]
    key = "topic1"
    node_labels = {key: ["cosmology", "galaxies", "cosmology"]}
    G = make_graph(node_ids, node_labels)

    with pytest.raises(KeyError):
        G_new = add_edges(G, labels=["topic2"], hard_constraint=False)

def test_add_edges_with_multiple_labels_hard_constraint():
    node_ids = [1, 2, 3]
    key1 = "topic1"
    key2 = "topic2"
    node_labels = {key1: ["cosmology", "galaxies", "cosmology"],
                   key2: ["galaxies", "galaxies", "black holes"]}

    G = make_graph(node_ids, node_labels)

    G_new = add_edges(G, labels=[key1, key2], hard_constraint=True,
                      weights=[0.1])
    edges = G_new.edges()

    assert (1, 3) not in list(edges)
    assert np.isclose(edges[1, 2]["weight"], 0.1, atol=1e-5, rtol=1e-5)

def test_add_edges_with_multiple_reversed_labels_hard_constraint():
    node_ids = [1, 2, 3]
    key1 = "topic2"
    key2 = "topic1"
    node_labels = {key1: ["cosmology", "galaxies", "cosmology"],
                   key2: ["galaxies", "galaxies", "black holes"]}

    G = make_graph(node_ids, node_labels)

    G_new = add_edges(G, labels=[key1, key2], hard_constraint=True,
                      weights=[0.1])
    edges = G_new.edges()

    assert (1, 3) not in list(edges)
    assert np.isclose(edges[1, 2]["weight"], 0.1, atol=1e-5, rtol=1e-5)


def test_add_edges_with_non_consecutive_node_ids():
    node_ids = [1, 23, 12]
    key1 = "topic2"
    key2 = "topic1"
    node_labels = {key1: ["cosmology", "galaxies", "cosmology"],
                   key2: ["galaxies", "galaxies", "black holes"]}

    G = make_graph(node_ids, node_labels)

    G_new = add_edges(G, labels=[key1, key2], hard_constraint=True,
                      weights=[0.1])
    edges = G_new.edges()

    assert (node_ids[0], node_ids[2]) not in list(edges)
    assert np.isclose(edges[node_ids[0], node_ids[1]]["weight"], 0.1, atol=1e-5, rtol=1e-5)


def test_add_edges_with_multiple_labels_no_hard_constraints():
    node_ids = [1, 2, 3, 4]
    key1 = "topic1"
    key2 = "topic2"
    node_labels = {key1: ["cosmology", "galaxies", "cosmology", "galaxies"],
                   key2: ["galaxies", "galaxies", "black holes", "galaxies"]}

    weights = [0.5, 0.1]
    G = make_graph(node_ids, node_labels)

    G_new = add_edges(G, labels=[key1, key2], hard_constraint=False,
                      weights=weights)
    edges = G_new.edges()

    assert edges[1, 3]["weight"]  == weights[0]
    assert np.isclose(edges[1, 2]["weight"], 0.1, atol=1e-5, rtol=1e-5)
    assert np.isclose(edges[2, 4]["weight"], weights[0]*weights[1], atol=1e-5)


def test_add_edges_numerical_runs():
    node_ids = [1, 2, 3]
    key = "topic1"
    node_labels = {key: ["cosmology", "galaxies", "cosmology"]}
    G = make_graph(node_ids, node_labels)

    # edges are: [(1,2), (1,3), (2,3)]
    edges = [1.0, 0.1, 0.9]

    G_new = add_edges_numerical(G, edges, hard_constraint=False)
    g_edges = G_new.edges()

    assert g_edges[1,2]["weight"] == edges[0]
    assert g_edges[1,3]["weight"] == edges[1]
    assert g_edges[2,3]["weight"] == edges[2]

def test_add_edges_with_hard_constraint():
    node_ids = [1, 2, 3]
    key = "topic1"
    node_labels = {key: ["cosmology", "galaxies", "cosmology"]}
    G = make_graph(node_ids, node_labels)

    # edges are: [(1,2), (1,3), (2,3)]
    edges = [1.0, 0.1, 0.9]

    G_new = add_edges_numerical(G, edges, hard_constraint=True)
    g_edges = G_new.edges()

    assert g_edges[1,2]["weight"] == edges[0]
    assert (1,3) not in list(g_edges)
    assert g_edges[2,3]["weight"] == edges[2]


def test_sort_cliques_by_weights():
    node_ids = [1, 2, 3]
    key = "topic1"
    node_labels = {key: ["cosmology", "galaxies", "cosmology"]}
    G = make_graph(node_ids, node_labels)

    # edges are: [(1,2), (1,3), (2,3)]
    edges = [1.0, 0.1, 0.9]

    G = add_edges_numerical(G, edges, hard_constraint=False)

    cliques = np.array([[1], [2], [3], [1, 2], [1, 3], [2, 3], [1, 2, 3]])

    n_elements = 2

    clique_weights_true = np.array([0, 0, 0,
                      edges[0], edges[1], edges[2], 0.0])

    sort_idx = np.argsort(clique_weights_true)[::-1]

    sorted_cliques_true = cliques[sort_idx]
    clique_weights_true = clique_weights_true[sort_idx]

    sorted_cliques_true = sorted_cliques_true[clique_weights_true> 0]
    clique_weights_true = clique_weights_true[clique_weights_true > 0]

    sorted_cliques, clique_weights = _sort_cliques_by_weights(G, cliques,
                                                              n_elements)

    assert np.all(sorted_cliques == sorted_cliques_true)
    assert np.all(clique_weights == clique_weights_true)

def test_sort_cliques_by_weights_with_more_elements():
    node_ids = [1, 2, 3, 4]

    key = "topic1"
    node_labels = {key: ["black holes", "galaxies", "cosmology", "agn"]}

    G = make_graph(node_ids, node_labels)

    # edges are: [(1,2), (1,3), (1,4), (2,3), (2,4), (3,4)]
    edges = [1.0, 0.1, 0.9, 0.5, 0.2, 0.3]

    G = add_edges_numerical(G, edges, hard_constraint=False)

    cliques = np.array([[1], [2], [3], [4], [1, 2], [1, 3], [1, 4], [2, 3],
                        [2, 4], [3, 4], [1, 2, 3], [1, 2, 4],[1, 3, 4],
                        [2, 3, 4],[1, 2, 3, 4]])

    n_elements = 3

    clique_weights_true = np.array([0,0,0,0,0,0,0,0,0,0,
                                    edges[0]+edges[1]+edges[3],
                                    edges[0]+edges[2]+edges[4],
                                    edges[1]+edges[2]+edges[5],
                                    edges[3]+edges[4]+edges[5],
                                    0])

    sort_idx = np.argsort(clique_weights_true)[::-1]

    sorted_cliques_true = cliques[sort_idx]
    clique_weights_true = clique_weights_true[sort_idx]


    sorted_cliques_true = sorted_cliques_true[clique_weights_true> 0]
    clique_weights_true = clique_weights_true[clique_weights_true > 0]

    sorted_cliques, clique_weights = _sort_cliques_by_weights(G, cliques,
                                                              n_elements)

    assert np.all(sorted_cliques == sorted_cliques_true)
    assert np.all(clique_weights == clique_weights_true)

def test_results_class_initializes():
    n_elements = 3
    res = Results(n_elements)

    assert res.n_elements == n_elements
    assert len(res.groups) == 0
    assert len(res.all_weights) == 0
    assert res.success is True
    assert res.weights_sum_total == 0

def test_results_class_update_groups():
    n_elements = 3
    res = Results(n_elements)

    groups = [1,2,3]

    res.update_groups(groups)

    assert res.groups == [groups]

def test_results_class_update_weights():
    n_elements = 3
    res = Results(n_elements)

    start_weight = 0.4
    res.all_weights.append(start_weight)

    add_weight = 0.5
    res.update_weights(add_weight)

    assert np.all(res.all_weights == np.array([start_weight, add_weight]))
    assert res.weights_sum_total == np.sum([start_weight, add_weight])

