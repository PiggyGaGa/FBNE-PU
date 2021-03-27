'''
构建网络
@Time:2019/07/03
@Author :PiggyGaGa
'''

import time
import numpy as np
import pandas as pd
import networkx as nx
from pandas.core.frame import DataFrame

class GraphConstruct(object):
    def __init__(self):
        print(' ' * 20)
        print(' !---- 初始化完成  ---!')
    
    def merge_pickle(self, pickle_name_list, new_pickle=''):
        '''
        多个pickle 融合成一个pickle
        '''
        pickle_list = []
        for name in pickle_name_list:
            data = pd.read_pickle(name)
            pickle_list.append(data)
        result = pd.concat(pickle_list, axis=0)
        result.drop_duplicates(inplace=True)
        if new_pickle:
            self.save_pickle(result, new_pickle + '.pickle')
        print('和1525家虚开企业与10000家未标记的企业的相关的发票数未：{}'.format(len(result)))
        return result
    def get_all_edges_from_pickle(self, pickle_name):
        df = pd.read_pickle(pickle_name)
        edges = df[['xfnsrdzdah', 'gfnsrdzdah']]
        return edges.drop_duplicates()
    def get_all_edges_from_dataframe(self, dataframe):
        edges = dataframe[['xfnsrdzdah', 'gfnsrdzdah']]
        return edges.drop_duplicates()


    def construct_network(self, edges):
        '''
        通过网络的交易信息获取整个网络结构
        用networkx 存储
        '''
        print('begin to construct network ')
        G = nx.Graph()
        G.add_edges_from(edges)
        print('. '*10)
        print('!--- construct network success ----!')
        print('. '*10)
        return G
    def get_max_disjoint_sub_graph(self, graph):
        _, sub_graph_edges = self.get_nodes_edges(graph)
        sub_graph = nx.Graph()
        sub_graph.add_edges_from(sub_graph_edges.values)
        print('sub graph nodes number: {}'.format(len(sub_graph.nodes())))
        print('sub graph edges number: {}'.format(len(sub_graph.edges())))
        return sub_graph

    def get_nodes_edges(self, graph):
        '''
        从网络中将节点和边存储下来，
        只提取最大联通字图的边和节点
        '''
        #edges = graph.edges()
        res = [c for c in sorted(nx.connected_components(graph), key=len, reverse=True)]
        nodes = res[0]
        edges = np.array(graph.edges())
        sub_graph_edges = edges[np.isin(edges[:, 0], nodes) == np.isin(edges[:, 1], nodes)]
        sub_graph_edges = DataFrame(sub_graph_edges)
        sub_graph_edges.drop_duplicates(inplace=True)
        return nodes, sub_graph_edges

 
    def save_pickle(self, df, name):
        '''
        save dataframe to pickle
        '''
        pd.to_pickle(df, name)
    def __read_excel__(self, filepath):
        excel_pd = pd.read_excel(filepath, 'Sheet1')
        return excel_pd
   

    def get_enterprises_from_nsrjbxx(self, pickle_name):
        '''
        从 纳税人基本信息中提取出所有的纳税人电子档案号信息，作为网络节点
        '''
        data = pd.read_pickle(pickle_name)
        data = data['nsrdzdah']
        data.drop_duplicates(inplace=True)

        return data.values.tolist()

'''
if __name__ == "__main__"


    VAT_object = GraphConstruct()
    #print('空id', VAT_object.get_HY(''))
    Positive_namelist = ['../data/zzszyfp_xf.pickle', '../data/zzszyfp_gf.pickle']
    Unlabeled_namelist = ['../data/zzszyfp_xf_unlabeled.pickle', '../data/zzszyfp_gf_unlabeled.pickle']

    edges = VAT_object.get_all_edges_from_dataframe(VAT_object.merge_pickle(['../data/zzszyfp_xf.pickle', '../data/zzszyfp_gf.pickle', '../data/zzszyfp_xf_unlabeled.pickle', '../data/zzszyfp_gf_unlabeled.pickle']))
    print('total edes num {}'.format(len(edges.drop_duplicates())))

    graph = VAT_object.construct_network(edges.values)
'''