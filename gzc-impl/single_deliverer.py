import numpy as np
import networkx as nx
import copy

import scipy.sparse as sp

# [i,j] = i -> j
def getTranProMatrix(adj, scale_factor=3.0) -> np.ndarray:
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
        degree_inv = scale_factor / out_degree
    degree_inv = np.minimum(degree_inv, 0.9)
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


# [i,j] = i -> j
def getTranProMatrix_with_div_degree(adj) -> np.ndarray:
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


# deprecated ============================================
def getBestSingleDeliverer(tranProMatrix, succ_distribution, users_useAndDis):
    # 保持不变，但注意这里的 N 计算可能受矩阵转置影响
    # 如果 tranProMatrix 现在是 P(u->v)，那么 (I - M) 的逆是正确的
    n = tranProMatrix.shape[0]
    I = np.eye(n)
    curr_succ_distribution = copy.deepcopy(succ_distribution)

    # 注意：如果 M[u,v] 是 u->v，那么影响力传播是 vec * M
    try:
        N = np.linalg.inv(I - tranProMatrix)
    except np.linalg.LinAlgError:
        # 如果矩阵不可逆（通常不会发生，除非有概率为1的死循环），加一点阻尼
        N = np.linalg.inv(I - 0.999 * tranProMatrix)

    # 计算期望收益
    # 这里的数学含义：每个节点作为源点，最终流向哪里并被succ接收
    succ_nodes_for_every_node = np.dot(N, curr_succ_distribution.T)
    # 注意：如果是 M[u,v]=u->v, 应该看 N 的行和(加权succ)
    # 原代码 np.dot(dist, dot(N, M)) 比较复杂，简化为直接看吸收概率:
    # 期望收益 = (I-M)^-1 * Succ_Vector

    # 简化版计算：
    expected_rewards = np.dot(N, curr_succ_distribution)

    return np.argmax(expected_rewards)



def getTranProMatrix_old(adj, tran_distribution: np.ndarray) -> np.ndarray:
    """
    根据邻接矩阵和节点转发概率，生成转移概率矩阵。

    Args:
        adj (np.ndarray | sp.spmatrix): 图的邻接矩阵（稠密或稀疏）。
        tran_distribution (np.ndarray): 每个节点的转发概率向量，形状 (n,)。

    Returns:
        np.ndarray: 转移概率矩阵 M，M[i, j] 表示节点 j 向节点 i 成功转发的概率。

    M[i, j] = 1 表示 j → i 这条边存在（j 指向 i）
    axis=0（列求和） axis=1（行求和）

    """

    if not isinstance(adj, np.ndarray):
        A_graph = adj.toarray().astype(float)
    else:
        A_graph = adj.astype(float)

    # 保证分布是 1D 向量
    tran_distribution_list = np.array(tran_distribution).flatten()

    # 每个节点的度 (出度)
    D = np.sum(A_graph, axis=0)

    prob_per_neighbor = np.zeros_like(D, dtype=float)
    non_isolated_nodes = D > 0

    # 仅对非孤立节点计算其到每个邻居的转发概率
    prob_per_neighbor[non_isolated_nodes] = tran_distribution_list[non_isolated_nodes] / D[non_isolated_nodes]

    tran_matrix = A_graph * prob_per_neighbor
    return tran_matrix


def getBestSingleDeliverer_old(tranProMatrix, succ_distribution, users_useAndDis):
    """
    寻找在当前概率模型下最佳的单个投放节点 即能带来最多优惠券使用量的初始节点
    :param tranProMatrix: 转移概率矩阵
    :param succ_distribution: 每个节点的成功使用概率
    :param users_useAndDis: 包含已经被使用过优惠券的用户节点的列表
    :return:
    """
    n = tranProMatrix.shape[0]
    # 使用 马尔可夫链中的吸收概率求解公式（矩阵逆）来模拟从每个节点开始投放时最终“成功”的概率
    I = np.eye(n)

    curr_succ_distribution = copy.deepcopy(succ_distribution)
    # if len(users_useAndDis) > 0:
    #     for user in users_useAndDis:
    #         curr_succ_distribution[int(user)] = 0

    N = np.linalg.inv(I - tranProMatrix)

    succ_nodes_for_every_node = np.dot(curr_succ_distribution, np.dot(N, tranProMatrix)) + curr_succ_distribution

    root_node_index = np.argmax(succ_nodes_for_every_node)  # 找出影响力最大的节点(当每个节点被选为种子节点时， 比较他们各自的激活成功的节点数（包括自己）)
    return root_node_index


# def getBestSingleDeliverer_theroy(init_tranProMatrix,  # 初始的、未受影响时的转移矩阵
#                                   succ_distribution,
#                                   Q,  # 引入理论吸收模型优化 Q 向量 Q[i] 代表节点 i 已经被影响的程度（从0到1）
#                                   tranProMatrix):  # 节点完全被影响后（比如已经收到过优惠券）的转移矩阵
#     '''动态模型。它引入了一个状态向量 Q，模拟网络状态因为影响力传播而发生改变的过程'''
#     # 一个理论增强版本：使用向量 Q（代表某节点是否已被影响），并将其用于调整转发概率矩阵 W，迭代更新吸收/传播模型
#
#     n = init_tranProMatrix.shape[0]
#     I = np.eye(n)
#
#     W = (1 - Q) * init_tranProMatrix + Q * tranProMatrix  # 根据当前 Q 向量混合两个状态的传播模型
#
#     R = np.dot(np.linalg.inv(I - W), W)
#     curr_succ_distribution = (1 - Q) * succ_distribution  # 如果一个节点已经被影响了 (Q[i]>0)，那么它再次因为新的传播而“首次成功使用”的概率就会降低
#     succ_pros = np.dot(curr_succ_distribution,
#                        R) + curr_succ_distribution  # 计算边际影响力增益。即在当前网络状态 Q下，再选择一个新节点投放，能带来多大的额外期望使用量
#     max_column_index = succ_pros.argmax()
#     R[max_column_index][max_column_index] += 1  # 这一步是为了将 R 的一列转换回基本矩阵 N 的一列
#     Q_increment = np.multiply(curr_succ_distribution, R[:, max_column_index].reshape(
#         -1))  # 计算本次选择 max_column_index 后，对网络中所有节点的影响力增量。R[:, max_column_index] 是从最佳投放点出发的期望访问次数，乘以每个节点的边际成功率，就得到了对每个节点的“影响程度增量”
#
#     Q = np.minimum(Q + Q_increment, 1.0)  # 更新状态向量 Q。将影响力增量加到 Q 上
#     return max_column_index, Q
