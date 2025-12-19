import numpy as np

import scipy.sparse as sp

# [i,j] = i -> j
def getTranProMatrix(adj) -> np.ndarray:
    """
    生成转移概率矩阵（列归一化）。
    这里假设 adj 为标准邻接矩阵：adj[i, j]=1 表示 i->j (Row=Source, Col=Target)。
    返回的矩阵将转换为：M[i, j] 表示 j->i (Col=Source, Row=Target)，以适配你的蒙特卡洛逻辑。
    """

    # 1. 统一处理为稀疏矩阵 (CSR格式适合算术运算)
    if isinstance(adj, np.ndarray):
        adj_sparse = sp.csr_matrix(adj)
    else:
        adj_sparse = adj

    # 2. 转置矩阵
    # 你的逻辑需要 Col=Source (j->i)，但通常输入是 Row=Source (i->j)。
    # 如果你的输入已经是 j->i，请注释掉下面这行转置。
    adj_t = adj_sparse.T

    # 此时 adj_t[i, j] 表示 j -> i (列 j 指向 行 i)

    # 3. 计算出度 (按列求和 -> 也就是压缩行，因为现在每列代表一个源节点的所有出边)
    # 对于稀疏矩阵，sum(axis=0) 返回的是一个 matrix 对象 (1, N)
    out_degree = np.array(adj_t.sum(axis=0)).flatten()

    # 4. 防止除以 0 (对于孤立节点，度为 0，倒数设为 0)
    with np.errstate(divide='ignore', invalid='ignore'):
        degree_inv = 1.0 / out_degree
    degree_inv[np.isinf(degree_inv)] = 0.0

    # 5. 利用稀疏矩阵乘法进行列归一化
    # 每一列 j 的所有元素都需要乘以 1/degree[j]
    # 在线性代数中，这等于： Matrix * Diagonal_Matrix(1/D)
    D_inv_diag = sp.diags(degree_inv)

    # 结果矩阵 M，其中 M[:, j] sum 为 1 (如果度>0)
    tran_matrix_sparse = adj_t.dot(D_inv_diag)

    # 6. 如果节点数不大，转回稠密矩阵方便后续索引；如果节点数很大，建议直接返回稀疏矩阵
    # 考虑到你的蒙特卡洛也是用的 numpy 索引，这里转回 dense
    return tran_matrix_sparse.toarray()