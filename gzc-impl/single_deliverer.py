import numpy as np
import networkx as nx
import copy


def getTranProMatrix(adj, tran_distribution):
    """
    :param adj: 原始的图
    :param tran_distribution_list: 每个节点的转发概率
    :return: 返回转移概率矩阵 tranProMatrix[i, j] 代表了优惠券从节点 j 成功转发到节点 i 的概率
    """
    A_graph = adj.toarray().astype(float)
    tran_distribution_list = np.array(tran_distribution).flatten()
    D = np.sum(A_graph, axis=0)  # 计算每个节点的度
    prob_per_neighbor = np.zeros_like(D, dtype=float)
    non_isolated_nodes = D > 0

    # 仅对非孤立节点计算其到每个邻居的转发概率
    prob_per_neighbor[non_isolated_nodes] = tran_distribution_list[non_isolated_nodes] / D[non_isolated_nodes]
    tranProMatrix = A_graph * prob_per_neighbor
    return tranProMatrix, D


def getBestSingleDeliverer(tranProMatrix, succ_distribution, users_useAndDis):
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
