import numpy as np
import networkx as nx
import copy

import scipy.sparse as sp


def getTranProMatrix(adj, fixed_prob=0.5) -> np.ndarray:
    """
    生成转移概率矩阵（固定边概率）。
    核心思想：体现 Hub 的优势。Hub 的出边多，总能量就大。
    
    Args:
        adj: 邻接矩阵
        fixed_prob: 每条边的基础转发概率 (建议 0.1 ~ 0.2)
    """

    # 1. 转为稀疏矩阵
    if isinstance(adj, np.ndarray):
        adj_sparse = sp.csr_matrix(adj)
    else:
        adj_sparse = adj

    # 2. 转置 (j->i)
    adj_t = adj_sparse.T

    # 3. 直接赋值
    # 所有的非零元素（边），都赋值为 fixed_prob
    # 不再除以 degree！
    
    # 这里的 adj_t 是 j->i 的矩阵。
    # 我们直接创建一个新的矩阵，结构与 adj_t 相同，但值全是 fixed_prob
    
    # 获取非零元素的索引
    rows, cols = adj_t.nonzero()
    data = np.full_like(rows, fixed_prob, dtype=float)
    
    # 构造新的概率矩阵
    tran_matrix_sparse = sp.csr_matrix((data, (rows, cols)), shape=adj_t.shape)

    # 4. (可选) 稍微加一点随机扰动或者衰减，防止完全一样，但没必要太复杂
    
    return tran_matrix_sparse.toarray()
