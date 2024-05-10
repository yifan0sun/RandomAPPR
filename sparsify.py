
import numpy as np
import random
import networkx as nx
import scipy.sparse as sp
from scipy.sparse import eye
from scipy.sparse import csr_matrix, csgraph
from scipy.sparse.linalg import inv
from numpy.linalg import pinv

from scipy.sparse import triu, tril, csr_matrix, coo_matrix
 
import warnings
from collections import OrderedDict


def my_spmatrix_to_graph(adj_matrix):
    G = nx.Graph()
    row, col = adj_matrix.nonzero()
    values = adj_matrix.data
    
    num_nodes = adj_matrix.shape[0]
    
    
    
    for i in range(num_nodes):
        G.add_node(i) 
     
    for ii in range(len(row)):
        i,j,v = row[ii],col[ii],values[ii]
        G.add_edge(i, j, weight=v)
     
    return G
def my_graph_to_spmatrix(G): 
    rows = []
    cols = []
    data = []
    
    for u, v, weight in G.edges(data='weight', default=None):
        rows.append(u)
        cols.append(v)
        data.append(weight)
        
        # Include the symmetric counterpart for undirected graphs
        if u != v:
            rows.append(v)
            cols.append(u)
            data.append(weight)
    
    # Create the CSR matrix
    csr_mat = csr_matrix((data, (rows, cols)))
    
    # Return the CSR matrix
    return csr_mat


def uniform_sparsify(adjacency, p, reweight = True): 
    adjacency = triu(csr_matrix(adjacency))
    
    num_edges = adjacency.nnz
    orig_num_edges = num_edges + 0
    num_samples = int(num_edges * (1-p))
    
    actual_p = p + 0.
    
    while num_samples > 0:
        row, col = adjacency.nonzero()
        values = adjacency.data

        # Calculate probabilities for each edge based on edge weights
        probabilities = values / np.sum(values)

        # Sample edges based on probabilities
        
        sampled_indices = np.random.choice(np.arange(num_edges), size=num_samples, replace=False, p=probabilities)

        # Extract sampled edges and their weights
        sampled_row = row[sampled_indices]
        sampled_col = col[sampled_indices]
        

        #graph = nx.from_scipy_sparse_array(adjacency + 0.)
        graph = my_spmatrix_to_graph(adjacency)
        num_edges_not_removed = 0

        for e in range(len(sampled_row)): 
            edge = (sampled_row[e], sampled_col[e])

            if graph.has_edge(*edge):
                w = graph[edge[0]][edge[1]]['weight']
                graph.remove_edge(*edge)

                if not nx.is_connected(graph):
                    graph.add_edge(edge[0],edge[1],weight=w)  # Add the edge back
                    num_edges_not_removed += 1

        adjacency = my_graph_to_spmatrix(graph)
        #adjacency = nx.to_scipy_sparse_array(graph)
        adjacency = triu(adjacency)
        if num_edges_not_removed == num_samples: 
            warnings.warn(f"Warning: max samples to be removed while keeping graph connected. {num_samples} not yet removed.")
            break
        num_samples = num_edges_not_removed
        num_edges = adjacency.nnz
        if num_samples > 0:
            warnings.warn(f"Warning: num_samples = {num_samples}, nnz = {adjacency.nnz}")

    adjacency = adjacency + adjacency.transpose()
    
    if reweight:
        num_edges_removed = orig_num_edges - num_edges_not_removed
        actual_p = num_edges_removed / orig_num_edges
        adjacency = adjacency / actual_p
    return adjacency, actual_p
    
     
def resistive_sparsify(adjacency, p):
    """
    Sparsifies a graph based on the resistive distance between nodes. Edges are more likely
    to be retained if their resistive distance is smaller.
    
    Parameters:
    adjacency_matrix (csr_matrix): The input sparse adjacency matrix.
    p (float): The fraction of edges to retain.

    Returns:
    csr_matrix: The sparsified sparse adjacency matrix.
    """
    adjacency.eliminate_zeros()
    if not (0 <= p <= 1):
        raise ValueError("The fraction p must be between 0 and 1.")
    
    laplacian = csgraph.laplacian(adjacency, normed=False).toarray()
    pseudoinverse = pinv(laplacian)
    
    
    tadjacency = triu(adjacency)
    row, col = tadjacency.nonzero()
    values = tadjacency.data
    
    data = values * 0. 
    for e in range(len(row)):
        i = row[e]
        j = col[e]
        data[e] = pseudoinverse[i, i] + pseudoinverse[j, j] - 2 * pseudoinverse[i, j] 
    resistive_distances = sp.csr_matrix((data, (row, col)), shape=adjacency.shape)    
    resistive_distances.eliminate_zeros()
    
    new_adjacency, actual_p = uniform_sparsify(resistive_distances, p, reweight = False)
    
    
     
    for e in range(len(row)):
        i = row[e]
        j = col[e]
        if new_adjacency[i,j] == 0: continue
        new_adjacency[i,j] = adjacency[i,j] / actual_p
        new_adjacency[j,i] = adjacency[j,i] / actual_p
    return new_adjacency, actual_p




def influencer_sparsify(adjacency, qbar):
    #G = nx.from_scipy_sparse_array(adjacency)
    G = my_spmatrix_to_graph(adjacency)
    n = len(G.nodes)
    all_done = False
    while not all_done:
        total_removed = 0
        all_done = True
        rp = random.sample(range(n), n)
        nodes = list(G.nodes)
        nodes = [nodes[i] for i in rp]
        
        for ii, node in enumerate(nodes):
            remove = G.degree[node] - qbar
            du = G.degree[node] 
            if remove > 0:
                removed = 0
                neighbors = list(G.neighbors(node))
                rp = random.sample(range(len(neighbors)), len(neighbors))
                neighbors = [neighbors[i] for i in rp]
        
                for neighbor_node in neighbors:
                    w =  G.get_edge_data(node, neighbor_node)['weight']
                    G.remove_edge(node, neighbor_node)
                    if nx.is_connected(G):
                        removed += 1
                        total_removed += 1
                        if removed >= remove: break
                    else:
                        G.add_edge(node, neighbor_node, weight=w)
                if removed> 0:
                    for neighbor_node in G.neighbors(node):
                        w =  G.get_edge_data(node, neighbor_node)['weight']
                        G[node][neighbor_node]['weight'] =  (w / removed) * du
                
                if removed < remove:
                    all_done = False 
        if total_removed == 0:
            break
        else:
            print(total_removed)
    
    overflow = [G.degree[node] for node in G.nodes if G.degree[node] > qbar]
    
    if len(overflow) > 0:
        warnings.warn('was not able to reduce a row enough to achieve qbar')
        print(overflow)
            
    adjacency = my_graph_to_spmatrix(G)   
    return adjacency