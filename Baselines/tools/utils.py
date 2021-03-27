import json
import numpy as np
import pandas as pd
import networkx as nx
from tqdm import tqdm
from scipy import sparse
from texttable import Texttable

def preprocess_nxgraph(graph):
    node2idx = {}
    idx2node = []
    node_size = 0
    for node in graph.nodes():
        node2idx[node] = node_size
        idx2node.append(node)
        node_size += 1
    return idx2node, node2idx


def partition_dict(vertices, workers):
    batch_size = (len(vertices) - 1) // workers + 1
    part_list = []
    part = []
    count = 0
    for v1, nbs in vertices.items():
        part.append((v1, nbs))
        count += 1
        if count % batch_size == 0:
            part_list.append(part)
            part = []
    if len(part) > 0:
        part_list.append(part)
    return part_list


def partition_list(vertices, workers):
    batch_size = (len(vertices) - 1) // workers + 1
    part_list = []
    part = []
    count = 0
    for v1, nbs in enumerate(vertices):
        part.append((v1, nbs))
        count += 1
        if count % batch_size == 0:
            part_list.append(part)
            part = []
    if len(part) > 0:
        part_list.append(part)
    return part_list


def partition_num(num, workers):
    if num % workers == 0:
        return [num//workers]*workers
    else:
        return [num//workers]*workers + [num % workers]



def read_graph(graph_path):
    """
    Method to read graph and create a target matrix with pooled adjacency matrix powers up to the order.
    :param args: Arguments object.
    :return graph: graph.
    """
    print("\nTarget matrix creation started.\n")
    graph = nx.from_edgelist(pd.read_csv(graph_path).values.tolist())
    graph.remove_edges_from(graph.selfloop_edges())
    return graph

def tab_printer(args):
    """
    Function to print the logs in a nice tabular format.
    :param args: Parameters used for the model.
    """
    args = vars(args)
    keys = sorted(args.keys())
    t = Texttable() 
    t.add_rows([["Parameter", "Value"]] +  [[k.replace("_"," ").capitalize(), args[k]] for k in keys])
    print(t.draw())

def feature_calculator(args, graph):
    """
    Calculating the feature tensor.
    :param args: Arguments object.
    :param graph: NetworkX graph.
    :return target_matrices: Target tensor.
    """
    node2idx = {}
    index = 0
    for node in graph.nodes():
        node2idx[node] = index
        index += 1
    index_1 = [node2idx[edge[0]] for edge in graph.edges()] + [node2idx[edge[1]] for edge in graph.edges()]
    index_2 = [node2idx[edge[1]] for edge in graph.edges()] + [node2idx[edge[0]] for edge in graph.edges()]
    values = [1 for edge in index_1]
    #node_count = max(max(index_1)+1,max(index_2)+1)
    node_count = len(graph.nodes())
    adjacency_matrix = sparse.coo_matrix((values, (index_1,index_2)),shape=(node_count,node_count),dtype=np.float32)
    degrees = adjacency_matrix.sum(axis=0)[0].tolist()
    degs = sparse.diags(degrees, [0])
    normalized_adjacency_matrix = degs.dot(adjacency_matrix)
    target_matrices = [normalized_adjacency_matrix.todense()]
    powered_A = normalized_adjacency_matrix
    if args.window_size > 1:
        for power in tqdm(range(args.window_size-1), desc = "Adjacency matrix powers"):
            powered_A = powered_A.dot(normalized_adjacency_matrix)
            to_add = powered_A.todense()
            target_matrices.append(to_add)
    target_matrices = np.array(target_matrices)
    return target_matrices

def adjacency_opposite_calculator(graph):
    """
    Creating no edge indicator matrix.
    :param graph: NetworkX object.
    :return adjacency_matrix_opposite: Indicator matrix.
    """
    adjacency_matrix = sparse.csr_matrix(nx.adjacency_matrix(graph), dtype=np.float32).todense()
    adjacency_matrix_opposite = np.ones(adjacency_matrix.shape) - adjacency_matrix
    return adjacency_matrix_opposite


def create_inverse_degree_matrix(edges, node2idx):
    """
    Creating an inverse degree matrix from an edge list.
    :param edges: Edge list.
    :return D_1: Inverse degree matrix.
    """
    graph = nx.from_edgelist(edges)
    ind = []
    degs = []
    for node in graph.nodes():
        ind.append(node2idx[node])
        degs.append(1.0 / graph.degree(node))
    D_1 = sparse.coo_matrix((degs,(ind,ind)),shape=(graph.number_of_nodes(), graph.number_of_nodes()),dtype=np.float32)
    return D_1

def normalize_adjacency(edges, node2idx):
    """
    Method to calculate a sparse degree normalized adjacency matrix.
    :param edges: Edge list of graph.
    :return A: Normalized adjacency matrix.
    """
    D_1 = create_inverse_degree_matrix(edges, node2idx)
    index_1 = [node2idx[edge[0]] for edge in edges] + [node2idx[edge[1]] for edge in edges]
    index_2 = [node2idx[edge[1]] for edge in edges] + [node2idx[edge[0]] for edge in edges]
    values = [1.0 for edge in edges] + [1.0 for edge in edges]
    A = sparse.coo_matrix((values,(index_1, index_2)),shape=D_1.shape,dtype=np.float32)
    A = A.dot(D_1)
    return A

def read_graph(edge_path):
    """
    Method to read graph and create a target matrix.
    :param edge_path: Path to the ege list.
    :return A: Target matrix.
    """
    edges = pd.read_csv(edge_path).values.tolist()
    A = normalize_adjacency(edges)
    return A

def tab_printer(args):
    """
    Function to print the logs in a nice tabular format.
    :param args: Parameters used for the model.
    """
    args = vars(args)
    t = Texttable() 
    t.add_rows([["Parameter", "Value"]] +  [[k.replace("_"," ").capitalize(),v] for k,v in args.items()])
    print(t.draw())

def merge_graphs(graph):
    '''
    A big graph usually have several disconnected components 
    this function is to connect those components as a big connected graph
    Input A networkx graph may be not connected
    return A connected graph
    '''
    from random import choice
    G = nx.Graph()
    subgraphs_nodes = nx.connected_components(graph)
    subgraphs = []
    for c in sorted(subgraphs_nodes, key=len,reverse=True):
        subgraphs.append(graph.subgraph(c))
    G.add_edges_from(subgraphs[0].edges())
    if len(subgraphs) == 1:
        return G
    index = 0
    for sub_graph in subgraphs:
        if index == 0:
            index += 1
            continue
        else:
            head = choice(list(G.nodes()))
            tail = choice(list(sub_graph.nodes()))
            G.add_edge(head, tail)
            G.add_edges_from(sub_graph.edges())
            index += 1        
    print('new graph nodes {}'.format(len(G.nodes())))
    print('new graph edges {}'.format(len(G.edges())))
    return G


def conn_graphs(G_list):
    from random import choice
    if len(G_list) == 1:
        return G_list[0]  
    G = nx.Graph()
    if len(G_list) == 0:
        print('insert empty list')   
    G.add_nodes_from(G_list[0].nodes())
    G.add_edges_from(G_list[0].edges())
     
    index = 0
    for sub_graph in G_list:
        if index == 0:
            index += 1
            continue
        else:
            head = choice(list(G.nodes()))
            tail = choice(list(sub_graph.nodes()))
            G.add_edge(head, tail)
            G.add_edges_from(sub_graph.edges())
            index += 1        
    return G

def get_uniform_node_number_subgraphs(G):
    import community
    subgraphs_nodes = nx.connected_components(G)
    subgraphs = []
    for c in sorted(subgraphs_nodes, key=len,reverse=True):
        ap_graph = nx.Graph()
        ap_graph.add_nodes_from(c)
        ap_graph.add_edges_from(G.subgraph(c).edges())
        subgraphs.append(ap_graph)
    new_subgraphs = []
    new_subgraphs = new_subgraphs + subgraphs[1:]
    partitions = community.best_partition(subgraphs[0])
    for com in set(partitions.values()):
        list_nodes = [nodes for nodes in partitions.keys() if partitions[nodes] == com]
        ap_graph = nx.Graph()
        ap_graph.add_nodes_from(list_nodes)
        ap_graph.add_edges_from(subgraphs[0].subgraph(list_nodes).edges())
        new_subgraphs.append(ap_graph)
    new_subgraphs = sorted(new_subgraphs, key=len)


    print('total subgraphs ', len(new_subgraphs))
    result_subgraphs = []
    insert_subgraphs = []
    total_nodes = 0
    for g in new_subgraphs:
        insert_subgraphs.append(g)
        total_nodes += g.number_of_nodes()
        if total_nodes > 1000:
            result_subgraphs.append(conn_graphs(insert_subgraphs))
            total_nodes = 0
            insert_subgraphs = []
    if total_nodes > 0 and total_nodes <= 1000:
        result_subgraphs.append(conn_graphs(insert_subgraphs))
    result_subgraphs = sorted(result_subgraphs, key=len, reverse=True)
    total_num = 0
    for gr in result_subgraphs:
        total_num += gr.number_of_nodes()
    print('result nodes, ', total_num)
    print('final get subgraphs ', len(result_subgraphs))
    return result_subgraphs


def get_uniform_node_number_subgraphs_shaanxi(G):
    import community
    subgraphs_nodes = nx.connected_components(G)
    new_subgraphs = []
    if nx.number_connected_components(G) == 1:
        partitions = community.best_partition(G)
    else:
        subgraphs = []
        for c in sorted(subgraphs_nodes, key=len,reverse=True):
            ap_graph = nx.Graph()
            ap_graph.add_nodes_from(c)
            ap_graph.add_edges_from(G.subgraph(c).edges())
            subgraphs.append(ap_graph)
        new_subgraphs = new_subgraphs + subgraphs[1:]
        partitions = community.best_partition(subgraphs[0])
    for com in set(partitions.values()):
        list_nodes = [nodes for nodes in partitions.keys() if partitions[nodes] == com]
        ap_graph = nx.Graph()
        ap_graph.add_nodes_from(list_nodes)
        ap_graph.add_edges_from(G.subgraph(list_nodes).edges())
        new_subgraphs.append(ap_graph)
    new_subgraphs = sorted(new_subgraphs, key=len)


    print('total subgraphs ', len(new_subgraphs))
    result_subgraphs = []
    insert_subgraphs = []
    total_nodes = 0
    index = 0
    for g in new_subgraphs:
        if index == 0:
            insert_subgraphs.append(g)
            total_nodes += g.number_of_nodes()
            index += 1
            continue
        if total_nodes > 1000:
            result_subgraphs.append(conn_graphs(insert_subgraphs))
            total_nodes = 0
            insert_subgraphs = []
            index += 1
            insert_subgraphs.append(g)
        else:
            insert_subgraphs.append(g)
            total_nodes += g.number_of_nodes()
            index += 1
    if total_nodes > 0:
        result_subgraphs.append(conn_graphs(insert_subgraphs))
    result_subgraphs = sorted(result_subgraphs, key=len)
    return_subgraphs = []
    for gr in result_subgraphs:
        if gr.number_of_nodes() > 20000:
            return_subgraphs += get_uniform_node_number_subgraphs(gr)
        else: return_subgraphs += [gr]
    total_num = 0
    return_subgraphs = sorted(return_subgraphs, key=len, reverse=True)
    for gr in return_subgraphs:
        total_num += gr.number_of_nodes()
        print(' sub graph nodes :', gr.number_of_nodes())
    return_subgraphs = sorted(return_subgraphs, key=len)
    print('result nodes, ', total_num)
    print('final get subgraphs ', len(return_subgraphs))
    return return_subgraphs