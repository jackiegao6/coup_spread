import logging
import numpy as np
import networkx as nx
import random

import scipy.sparse as sp
from typing import List, Dict


# 随机策略
def deliverers_random(n: int, m: int) -> list:
    logging.info(f"数据集节点总数:  n = {n}, m = {m}")
    if m > n:
        raise ValueError(f"m={m} 不能大于 n={n}")
    selected_nodes = random.sample(range(n), k=m)
    logging.info(f"选择 {m}  {selected_nodes}\n")
    return selected_nodes


# 度中心性策略 认为认识人越多的人影响力越大
def deliverers_degreeTopM(adj, m: int) -> list:
    """
    选择度数最高的 m 个节点。
    """
    logging.info("依据: 选择连接数（度数）最多的节点。")
    
    # .A 属性将稀疏矩阵转为密集 NumPy 数组
    degrees = np.asarray(adj.sum(axis=1)).flatten()
    
    # 使用 argsort 获取排序后的索引
    sorted_indexes = np.argsort(degrees)[::-1] # 从大到小排序
    
    top_m_indexes = sorted_indexes[:m]
    
    logging.info("选择的 Top M 节点及其度数:")
    selected_nodes_with_values = []
    for index in top_m_indexes:
        value = degrees[index]
        selected_nodes_with_values.append((index, value))
    logging.info("") # 打印一个空行
        
    return [node for node, value in selected_nodes_with_values] # 保持原始返回类型

# PageRank 策略
def deliverers_pageRank2(adj, m: int, tranProMatrix) -> list:
    """
    依据实际转移概率 (tranProMatrix) 来选 PageRank Top-m，
    保证选出的节点在 tranProMatrix 中有出概率 (out_prob > 0)。
    """
    logging.info("--- Running: PageRank (on tranProMatrix) Top M ---")


    # 确保 tranProMatrix 是 (n x n)，tranProMatrix[i,j] 表示 j -> i 的概率
    # networkx 需要 A[u,v] 表示 u->v 的权重，因此传入它的是 tranProMatrix.T
    if sp.issparse(tranProMatrix):
        B = tranProMatrix.T.tocsr()
        G = nx.from_scipy_sparse_array(B, create_using=nx.DiGraph)
    else:
        B = np.asarray(tranProMatrix).T
        G = nx.from_numpy_array(B, create_using=nx.DiGraph)

    # 计算 PageRank
    pagerank_scores = nx.pagerank(G, weight='weight')

    # 计算每个节点在 tranProMatrix 中的出概率（列和）
    if sp.issparse(tranProMatrix):
        out_prob = np.array(tranProMatrix.sum(axis=0)).ravel()
    else:
        out_prob = np.asarray(tranProMatrix).sum(axis=0)


    # 在有出概率的节点中按 pagerank 排序并取前 m
    ranked = sorted(pagerank_scores.items(), key=lambda x: x[1], reverse=True)
    selected = []
    for node, score in ranked:
        if out_prob[node] > 0:
            selected.append(node)
            if len(selected) >= m:
                break

    # 如果不足 m，补上出度最多的节点（作为回退）
    if len(selected) < m:
        logging.warning("PageRank-selected nodes with out_prob>0 fewer than m, fallback to out-prob sorting")
        candidates = np.argsort(-out_prob)  # indices sorted by descending out_prob
        for node in candidates:
            if node not in selected and out_prob[node] > 0:
                selected.append(int(node))
                if len(selected) >= m:
                    break

    logging.info("Selected PageRank seeds: %s", selected)
    return selected

# 忽略图的结构，选择alpha值最高的k个节点作为种子
def deliverers_alpha_sort(
        adj: sp.csr_matrix,
        tranProMatrix: np.ndarray,
        seeds_num: int,
        alpha: Dict[int, float]
) -> list:
    """
    Args:
        adj (sp.csr_matrix): 邻接矩阵 未使用
        tranProMatrix (np.ndarray): 转移概率矩阵 未使用。
        seeds_num (int): 要选择的种子数量 (k)。
        alpha (Dict[int, float]): 包含每个节点领券概率的字典。

    Returns:
        list: 包含k个最优种子节点ID的列表，按alpha值从高到低排序。
    """

    num_nodes = adj.shape[0]
    if seeds_num > num_nodes:
        logging.warning(f"种子数 ({seeds_num}) 大于图中节点数 ({num_nodes})")
        seeds_num = num_nodes

    # 1. 将alpha字典转换为 (节点ID, alpha值) 的元组列表 便于排序
    alpha_items = list(alpha.items())

    # 2. 降序排序 按照元组的第二个元素（即alpha值）进行排序
    alpha_items.sort(key=lambda item: item[1], reverse=True)
    logging.info(f"{len(alpha_items)} 个节点的alpha值降序排序 done")
    logging.info(f"==============>> {alpha_items[:10]}")

    # 3. 提取排序后前k个元组
    selected_items = alpha_items[:seeds_num]
    selected_seeds = [item[0] for item in selected_items]

    logging.info(f"\n选择的种子集 (按alpha排序): {selected_seeds}\n")
    return selected_seeds


# 重要性混合策略
def deliverers_importance_sort(
        adj: sp.csr_matrix,
        tranProMatrix: np.ndarray,
        seeds_num: int,
        alpha: Dict[int, float],
        trans: Dict[int, float],
) -> list:
    """
    Importance_i = alpha_i + p_i * (sum of alpha_j for all relevant neighbors j)

    Args:
        adj (sp.csr_matrix): 邻接矩阵。
        tranProMatrix (np.ndarray): 转移概率矩阵, tranProMatrix[v, u] = p(u->v)。
        seeds_num (int): 要选择的种子数量 (k)。
        alpha (Dict[int, float]): 包含每个节点领券概率的字典。

    Returns:
        list: 包含k个最优种子节点ID的列表，按Importance值从高到低排序。
    """

    num_nodes = adj.shape[0]
    if seeds_num > num_nodes:
        logging.warning(f"种子数 ({seeds_num}) 大于图中节点数 ({num_nodes})")
        seeds_num = num_nodes

    # 计算所有节点的Importance值
    node_importance = {}

    for cur_node in range(num_nodes):
        neighbor_alphas_sum = 0.0 # 初始化
        trans_value = 0.0 # 初始化

        # 找到所有 i 指向的邻居 j
        # 找到第i行的非零行索引 (col_indices, row_indices)
        out_neighbors = adj.getcol(cur_node).nonzero()[1]
        for out in out_neighbors:
            neighbor_alphas_sum += alpha.get(out, 0.0)
            trans_value = tranProMatrix[out, cur_node]

        # 计算最终的 Importance 值
        importance = alpha.get(cur_node, 0.0) + trans_value * neighbor_alphas_sum
        node_importance[cur_node] = importance

    # 排序
    importance_items = list(node_importance.items())
    importance_items.sort(key=lambda item: item[1], reverse=True)
    logging.info(f"{len(importance_items)} 个节点的Importance值降序排序 done")

    # 前k个种子节点
    selected_seeds = [item[0] for item in importance_items[:seeds_num]]
    logging.info(f"\n选择的种子集 (按Importance排序): {selected_seeds}\n")

    return selected_seeds


# 一阶邻居影响力策略
def deliverers_1_neighbor(succ_distribution,init_tranProMatrix,m) -> list:
    logging.info("--- Running: 1-Hop Neighbor Influence Top M ---")
    logging.info("依据: 选择其所有邻居的成功使用概率之和最大的节点。")

    # 确保邻接矩阵是 NumPy 数组
    adj = init_tranProMatrix
    adj_array = adj.toarray() if hasattr(adj, 'toarray') else np.asarray(adj)
    
    # 计算每个节点的邻居影响力
    # adj_array.T[i] 是一个向量，表示哪些节点是 i 的邻居
    # succ_distribution 是所有节点的成功概率向量
    # 点积操作高效地计算了每个节点的所有邻居的成功概率之和
    one_hop_influence = adj_array.T.dot(succ_distribution)
    
    sorted_indexes = np.argsort(one_hop_influence)[::-1]
    
    top_m_indexes = sorted_indexes[:m]
    
    logging.info("选择的 Top M 节点及其一步邻居影响力:")
    selected_nodes_with_values = []
    for index in top_m_indexes:
        value = one_hop_influence[index]
        logging.info(f"  - 节点 {index}: 邻居影响力 = {value:.4f}")
        selected_nodes_with_values.append((index, value))
    logging.info("")
        
    return [node for node, value in selected_nodes_with_values]
