import pickle
import numpy as np
import networkx as nx
import pandas as pd
import copy

'''
模拟优惠券投放策略优化模型
在社交网络中找出最优的单点投放节点
整体思路是：建模转发/使用行为为一个概率传播矩阵，通过矩阵运算分析“单点投放”效果
'''


#给定网络，获得对应转发概率矩阵
#user_pro 节点使用优惠券概率
#absorb 节点吸收概率（不使用也不转发)
#Vij表示节点j在节点i的概率（投放者第一步投放的状态，用户第一步状态）

def getTranProMatrix(adj,tran_distribution): #生成转发概率矩阵（每个节点将优惠券传给邻居的概率）
    # 给定邻接矩阵 adj 和初始投放概率 tran_distribution，生成转发概率矩阵 A 和每列度向量 D

    # G = nx.Graph()
    # G.add_nodes_from([1,2,3,4,5])
    # G.add_edges_from([(1,3),(1,4),(2,3),(2,5),(3,5)])
    # adj = nx.adjacency_matrix(G)

    # 将稀疏邻接矩阵 adj 转换为普通的 numpy 矩阵 A
    A = adj.toarray().astype(float)
    n = A.shape[0]
    # n = A.shape[1]

    # 计算每列的度（邻居数）
    D = np.sum(A, axis=0).reshape(-1,1)
    # 将 tran_distribution 分配到每个邻居，即每个节点将优惠券平均投给它的邻居
    # 例如，如果一个节点有 3 个邻居，初始投放概率为 0.6，那么每个邻居收到优惠券的概率就是 0.6 / 3 = 0.2
    tran_distribution = tran_distribution.reshape(-1,1)/D  # todo 分配到每个邻居

    # temp = np.tile(tran_distribution,(n,1))
    # tranProMatrix = np.where(A!=0,temp,A)

    # 将每一列中非零位置（邻居）赋值为 tran_distribution[i] 即某个节点将优惠券平均投给它的邻居
    for i in range(n): 
        A[A[:,i]!=0,i] = tran_distribution[i]

    # A[indices] = tran_distribution[indices[1]]
    # A[A!=0] = tran_distribution[:n]
    # tranProMatrix = np.transpose(np.where(A!=0,A/D,A))
    return A,D


def getBestSingleDeliverer(tranProMatrix,succ_distribution,users_useAndDis): #寻找在当前概率模型下最佳的单个投放节点 即能带来最多优惠券使用量的初始节点
    n = tranProMatrix.shape[0]

    # todo 使用了马尔可夫链中的吸收概率求解公式（矩阵逆）来模拟从每个节点开始投放时最终“成功”的概率
    # 创建单位矩阵 I
    I = np.eye(n)
    # tranProMatrix = np.dot(tranProMatrix,tranProMatrix)
    curr_succ_distribution = copy.deepcopy(succ_distribution)
    if len(users_useAndDis) > 0:
        curr_succ_distribution[:, users_useAndDis] = 0
    inverse_matrix = np.linalg.inv(I-tranProMatrix)
    # succ_pros = np.dot(np.dot(inverse_matrix,tranProMatrix),neighbor_having)
    # succ_pros = use_pro+np.dot(use_pros,succ_pros)
    succ_pros = np.dot(curr_succ_distribution,np.dot(inverse_matrix,tranProMatrix))+curr_succ_distribution


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

if __name__ == '__main__':
    data_file = 'D:/data-processed/{}-adj.pkl'.format('Facebook') #加载 Facebook 图的邻接矩阵
    with open(data_file, 'rb') as f:
        adj = pickle.load(f)
    use_pro = 0.4 # 设置优惠券使用概率为 0.4
    dis_pro = 0.2 # 丢弃概率

    #计算转发矩阵
    tranProMatrix,neighbor_having = getTranProMatrix(adj,use_pro,dis_pro)
    max_column_index = getBestSingleDeliverer(tranProMatrix,use_pro,neighbor_having)#找到当前最适合单点投放的用户编号（max_column_index）
    '''
    这份脚本的作用是：

    在社交网络中模拟优惠券从一个节点开始传播的过程

    根据传播矩阵和用户行为建模找到最有效的单点投放者

    支持理论优化策略，用于更复杂的投放策略分析
    '''