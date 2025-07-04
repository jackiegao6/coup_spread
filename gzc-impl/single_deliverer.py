import numpy as np
import networkx as nx
import copy

def getTranProMatrix(adj, tran_distribution):
    """
    根据邻接矩阵和初始转发概率，生成一个转发概率矩阵。
    核心假设：一个节点 i 会将其总的转发概率 tran_distribution[i] 平均分配给它的所有邻居。
    
    矩阵中的元素 A[j, i] 代表优惠券从节点 i 转发到节点 j 的概率。

    Args:
        adj (scipy.sparse.csr_matrix): NetworkX 生成的图的稀疏邻接矩阵。
        tran_distribution (np.ndarray): 一个一维数组，其中 tran_distribution[i] 是
                                        节点 i 将优惠券转发出去的总概率。

    Returns:
        tuple: 一个包含以下两个元素的元组：
            - tranProMatrix (np.ndarray): n x n 的转发概率矩阵。
            - D (np.ndarray): 一个包含每个节点度数的一维数组。
    """
    # 1. 将稀疏邻接矩阵转换为标准的、浮点类型的 NumPy 矩阵
    A_graph = adj.toarray().astype(float)
    n = A_graph.shape[0]

    # 确保 tran_distribution 是一个扁平的一维数组，便于后续广播操作
    tran_distribution = np.array(tran_distribution).flatten()

    # 2. 计算每个节点的度（即每列的和）
    # 在无向图中，按行求和或按列求和结果一样
    D = np.sum(A_graph, axis=0)

    # 3. 计算分配给每个邻居的概率，并优雅地处理除以零的情况
    # 首先创建一个全为零的数组
    prob_per_neighbor = np.zeros_like(D, dtype=float)
    
    # 创建一个布尔掩码，用于标记度大于0的节点（非孤立节点）
    non_isolated_nodes = D > 0
    
    # 仅对非孤立节点计算其到每个邻居的转发概率
    prob_per_neighbor[non_isolated_nodes] = tran_distribution[non_isolated_nodes] / D[non_isolated_nodes]

    # 4. 使用向量化（广播机制）高效地创建转发概率矩阵
    # A_graph 是一个 (n, n) 的 0-1 矩阵
    # prob_per_neighbor 是一个 (n,) 的一维数组 [p0, p1, p2, ...]
    # A_graph * prob_per_neighbor 这个操作会自动将 prob_per_neighbor 广播到 A_graph 的每一列
    # 效果等同于：A_graph 的第 i 列所有元素都乘以 prob_per_neighbor[i]
    tranProMatrix = A_graph * prob_per_neighbor
    
    return tranProMatrix, D


def getBestSingleDeliverer(tranProMatrix,succ_distribution,users_useAndDis): #寻找在当前概率模型下最佳的单个投放节点 即能带来最多优惠券使用量的初始节点
    n = tranProMatrix.shape[0]

    # 使用 马尔可夫链中的吸收概率求解公式（矩阵逆）来模拟从每个节点开始投放时最终“成功”的概率
    I = np.eye(n)

    curr_succ_distribution = copy.deepcopy(succ_distribution)
    # todo gzc: 二维还是一维
    # if len(users_useAndDis) > 0:
    #     curr_succ_distribution[:, users_useAndDis] = 0

    N = np.linalg.inv(I-tranProMatrix)

    # `succ_pros` 现在是一个一维数组，其中 `succ_pros[j]` 就是从节点 `j` 开始投放优惠券，最终带来的总期望使用量
    succ_pros = np.dot(curr_succ_distribution,np.dot(N,tranProMatrix))+curr_succ_distribution


    max_column_index = np.argmax(succ_pros) # 找出影响力最大的节点
    return max_column_index

def getBestSingleDeliverer_theroy(init_tranProMatrix,succ_distribution,Q,tranProMatrix):#引入理论吸收模型优化 Q 向量，寻找最佳单投放点
    #一个理论增强版本：使用向量 Q（代表某节点是否已被影响），并将其用于调整转发概率矩阵 W，迭代更新吸收/传播模型

    n = init_tranProMatrix.shape[0]
    I = np.eye(n)
    # W = (1-Q)[:,np.newaxis]*init_tranProMatrix+Q[:,np.newaxis]*tranProMatrix
    # W = np.multiply((1-Q),init_tranProMatrix)+np.multiply(Q,tranProMatrix)
    # temp4 = (1-Q).reshape(-1,1)*init_tranProMatrix
    # temp5 = (1-Q)*init_tranProMatrix


    W = (1-Q)*init_tranProMatrix+Q*tranProMatrix #根据当前 Q 向量混合两个状态的传播模型

    R = np.dot(np.linalg.inv(I-W),W)
    curr_succ_distribution = (1-Q)*succ_distribution
    succ_pros = np.dot(curr_succ_distribution,R)+curr_succ_distribution
    max_column_index = succ_pros.argmax()
    # temp = np.sum(R,axis=0)
    # temp1 = np.multiply((1-Q),succ_distribution)
    # temp2 = np.multiply(temp1,temp)
    R[max_column_index][max_column_index] += 1
    Q_increment = np.multiply(np.multiply((1-Q),curr_succ_distribution),R[:,max_column_index].reshape(-1))
    Q = Q+Q_increment
    # temp3 = np.multiply(Q_increment, (1 - constantFactor_distribution))
    # tran_increment = np.multiply(Q_increment, (1 - constantFactor_distribution))/D.reshape(-1)

    return max_column_index,Q
