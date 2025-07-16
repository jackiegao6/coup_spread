import numpy as np
import networkx as nx
import copy

def getTranProMatrix(adj, tran_distribution_list):
    """
    根据邻接矩阵和初始转发概率，生成一个转发概率矩阵。
    核心假设：一个节点 i 会将其总的转发概率 tran_distribution_list[i] 平均分配给它的所有邻居。
    
    矩阵中的元素 A[j, i] 代表优惠券从节点 i 转发到节点 j 的概率。

    Args:
        adj (scipy.sparse.csr_matrix): NetworkX 生成的图的稀疏邻接矩阵。
        tran_distribution_list (np.ndarray): 一个一维数组，其中 tran_distribution_list[i] 是 节点 i 将优惠券转发出去的总概率。

    Returns:
        tuple: 一个包含以下两个元素的元组：
            - tranProMatrix (np.ndarray): n x n 的转发概率矩阵。
            - D (np.ndarray): 一个包含每个节点度数的一维数组。
    """
    A_graph = adj.toarray().astype(float)
    tran_distribution_list = np.array(tran_distribution_list).flatten()
    D = np.sum(A_graph, axis=0)# 计算每个节点的度
    prob_per_neighbor = np.zeros_like(D, dtype=float)
    non_isolated_nodes = D > 0
    
    # 仅对非孤立节点计算其到每个邻居的转发概率
    prob_per_neighbor[non_isolated_nodes] = tran_distribution_list[non_isolated_nodes] / D[non_isolated_nodes]

    # 4. 使用向量化（广播机制）高效地创建转发概率矩阵
    # A_graph 是一个 (n, n) 的 0-1 矩阵
    # prob_per_neighbor 是一个 (n,) 的一维数组 [p0, p1, p2, ...]
    # A_graph * prob_per_neighbor 这个操作会自动将 prob_per_neighbor 广播到 A_graph 的每一列
    # 效果等同于：A_graph 的第 i 列所有元素都乘以 prob_per_neighbor[i]
    tranProMatrix = A_graph * prob_per_neighbor
    
    return tranProMatrix, D


def getBestSingleDeliverer(tranProMatrix,succ_distribution,users_useAndDis): #寻找在当前概率模型下最佳的单个投放节点 即能带来最多优惠券使用量的初始节点
    '''第一种理论计算方法，一个静态模型。假设网络参数（如成功率）在一次投放中是不变的'''
    n = tranProMatrix.shape[0]
    # 使用 马尔可夫链中的吸收概率求解公式（矩阵逆）来模拟从每个节点开始投放时最终“成功”的概率
    I = np.eye(n)

    curr_succ_distribution = copy.deepcopy(succ_distribution)
    # if len(users_useAndDis) > 0:
    #     for user in users_useAndDis:
    #         curr_succ_distribution[int(user)] = 0

    '''
    在马尔可夫链理论中，如果 P 是转移概率矩阵，那么 (I - P)^-1 这个矩阵 N 的元素 N[i, j] 有一个非常重要的物理意义：它代表从状态 j 出发，在被吸收（即流程结束）之前，预期访问状态 i 的平均次数。
    N[i, j] 就代表如果从用户 j 开始投放优惠券，优惠券平均会到达（或经过）用户 i 多少次。
    '''
    N = np.linalg.inv(I-tranProMatrix)

    # `succ_pros` 现在是一个一维数组，其中 `succ_pros[j]` 就是从节点 `j` 开始投放优惠券，整个网络中（包括它自己和所有被转发到的节点）产生的总的期望使用量
    '''
    计算从每个节点开始投放，最终产生的总期望使用量。
    np.dot(N, tranProMatrix): 计算了从任意一个节点出发，经过一次转发后，对所有节点的预期访问次数。
    np.dot(curr_succ_distribution, ...): 这是一个点积运算。它将上一步算出的“预期访问次数”与每个节点的“成功使用概率”相乘并求和。这部分计算的是所有通过转发而产生的总期望使用量。
    + curr_succ_distribution: 加上了初始投放就直接被使用的期望。例如，从节点 j 开始投放，它本身就有 curr_succ_distribution[j] 的概率会直接使用。
    '''
    succ_pros = np.dot(curr_succ_distribution,np.dot(N,tranProMatrix))+curr_succ_distribution

    max_column_index = np.argmax(succ_pros) # 找出影响力最大的节点
    return max_column_index

def getBestSingleDeliverer_theroy(init_tranProMatrix,   # 初始的、未受影响时的转移矩阵
                                  succ_distribution,
                                  Q,                    # 引入理论吸收模型优化 Q 向量 Q[i] 代表节点 i 已经被影响的程度（从0到1）
                                  tranProMatrix):       # 节点完全被影响后（比如已经收到过优惠券）的转移矩阵
    '''动态模型。它引入了一个状态向量 Q，模拟网络状态因为影响力传播而发生改变的过程'''
    #一个理论增强版本：使用向量 Q（代表某节点是否已被影响），并将其用于调整转发概率矩阵 W，迭代更新吸收/传播模型

    n = init_tranProMatrix.shape[0]
    I = np.eye(n)

    '''
    如果节点 `i` 完全未被影响 (`Q[i] = 0`)，那么从它出发的转发行为完全由 `init_tranProMatrix` 的第 `i` 列决定。
    如果节点 `i` 完全被影响 (`Q[i] = 1`)，那么它的转发行为完全由 `tranProMatrix` 的第 `i` 列决定。
    如果 `Q[i]` 是 0.5，那么它的转发行为是两种模型的平均。
    '''
    W = (1-Q)*init_tranProMatrix+Q*tranProMatrix #根据当前 Q 向量混合两个状态的传播模型

    R = np.dot(np.linalg.inv(I-W),W)
    curr_succ_distribution = (1-Q)*succ_distribution # 如果一个节点已经被影响了 (Q[i]>0)，那么它再次因为新的传播而“首次成功使用”的概率就会降低
    succ_pros = np.dot(curr_succ_distribution,R)+curr_succ_distribution # 计算边际影响力增益。即在当前网络状态 Q下，再选择一个新节点投放，能带来多大的额外期望使用量
    max_column_index = succ_pros.argmax()

    '''
    状态更新部分
    **`R[max_column_index][max_column_index] += 1`**: 这一步是将 `R[:][max_column_index]` 变回 `N` 的对应列。
    因为 `N[j, j]` 代表从 `j` 出发访问 `j` 自身的期望次数（至少为1，即初始那次），所以这里加上了1。
    现在 `R[:, max_column_index]` 实际上是基本矩阵 `N` 的第 `max_column_index` 列，代表从最佳投放点出发，对网络中所有节点的期望访问次数。
    '''
    R[max_column_index][max_column_index] += 1 # 这一步是为了将 R 的一列转换回基本矩阵 N 的一列
    Q_increment = np.multiply(curr_succ_distribution,R[:,max_column_index].reshape(-1)) # 计算本次选择 max_column_index 后，对网络中所有节点的影响力增量。R[:, max_column_index] 是从最佳投放点出发的期望访问次数，乘以每个节点的边际成功率，就得到了对每个节点的“影响程度增量”


    Q = np.minimum(Q + Q_increment, 1.0) # 更新状态向量 Q。将影响力增量加到 Q 上
    return max_column_index,Q
