import pickle
import networkx as nx
import numpy as np
import csv
import pandas as pd
from scipy.sparse import csr_matrix,block_diag
import os
import sys
import random

'''
从一个稀疏邻接矩阵中恢复图结构并提取子图
'''

# G：输入的图对象。
# degreeThres：度数阈值，默认为 0
def get_subgraph(G,degreeThres=0):
    # 从输入图 G 中去除度数小于等于 degreeThres 的节点，并提取剩余部分中最大的连通子图

    # print(nx.is_connected(G))
    # connected_components = list(nx.connected_components(G))
    # largest_component = max(connected_components,key=len)
    # largest_subgraph = G.subgraph(largest_component)
    # file_name = 'largest_conponent_subgraph'
    # with open(file_name,'wb') as f:
    #     pickle.dump(largest_subgraph,f,protocol=2)

    # 打印图是否连通
    print("图是否连通: ",nx.is_connected(G)) 
    print('')
    # 删除低度节点
    nodes_to_remove = [node for node,degree in dict(G.degree()).items() if degree <= degreeThres] 
    subgraph = nx.Graph(G)
    # 得到删除低度节点的图 subgraph
    subgraph.remove_nodes_from(nodes_to_remove)


    # 开始找最大连通分量 
    print("图中的连通分量个数为: ",nx.number_connected_components(subgraph))
    # 所有的连通分量list
    connected_components = list(nx.connected_components(subgraph))
    print("所有的连通分量list", connected_components)
    # 最大连通分量
    largest_component = max(connected_components,key=len)
    largest_subgraph = subgraph.subgraph(largest_component)
    print("largest_component为: ",largest_component)
    print("largest_subgraph为: ",largest_subgraph)


    # 打印最大连通分量的节点数和边数
    print('最大连通分量 nodes:{}'.format(largest_subgraph.number_of_nodes()))
    print('最大连通分量 edges:{}'.format(largest_subgraph.number_of_edges()))
    # 保存为 .pkl 文件
    file_name = 'D:/weak_tie/subgraph_deleteDegree{}_test.pkl'.format(degreeThres) # 保存为 .pkl 文件
    with open(file_name, 'wb') as f:
        pickle.dump(largest_subgraph, f, protocol=2)
    
    return largest_subgraph


if __name__ == '__main__':
    # degree = 45
#     # file_name = 'D:/weak_tie/subgraph_deleteDegree{}.pkl'.format(degree)
#     # with open(file_name, 'rb') as f:
#     #     G = pickle.load(f)
#     # adj = nx.to_scipy_sparse_matrix(G, format='csr')
#     # data_file = 'D:/data-processed/subgraph_deleteDegree{}-adj.pkl'.format(degree)
#     # with open(data_file, "wb") as f:
#     #     pickle.dump(adj, f)


    dataSet = 'Twitter'
    data_file = '../autodl-tmp/data-processed/{}-adj.pkl'.format(dataSet)
    with open(data_file, 'rb') as f:
        adj = pickle.load(f)
    print(adj.shape[0])
    print(adj.nnz//2)

    # G = nx.from_scipy_sparse_matrix(adj)
    # subgraph = get_subgraph(G, degreeThres=0)
