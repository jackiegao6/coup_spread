import networkx as nx
import os
import pickle

if __name__=="__main__":
    # 1. 创建一个示例图 (例如，一个 Barabási-Albert 模型图，它具有幂律度分布)
    n_nodes = 1000
    m_edges = 4
    G = nx.barabasi_albert_graph(n_nodes, m_edges)
    adj = nx.adjacency_matrix(G).toarray()
    output_file = "/home/wen/pythonspace/data-test/datasets/BA-adj.pkl"
    with open(output_file, 'wb') as f:
        pickle.dump(adj, f)
