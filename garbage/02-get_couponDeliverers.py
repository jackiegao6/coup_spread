import pickle
import numpy as np
import networkx as nx
import pandas as pd
import random
import single_deliverer
import copy
import getCouponUsers
import networkx as nx
'''
整个优惠券投放模拟系统的策略选择模块
根据不同策略选择 m 个投放节点deliverers,以最大化优惠券的使用或传播效果
'''



#蒙特卡洛模拟 + 贪心策略 选择m张券的投放源节点 模拟传播结果，迭代选择最有效节点
def deliverers_monteCarlo(dataSet,m,init_tranProMatrix,succ_distribution,dis_distribution,constantFactor_distribution,L,personalization):

    deliverers = []
    deliverer = single_deliverer.getBestSingleDeliverer(init_tranProMatrix, succ_distribution,users) # 初始选择第一个节点


    deliverers.append(deliverer)
    new_deliverer = deliverer
    print("first deliverer:",deliverer)

    for i in range(m-1): 
        deliverers,new_deliverer = getCouponUsers.monteCarloSimulation(init_tranProMatrix,deliverers,new_deliverer,L,succ_distribution,dis_distribution,constantFactor_distribution,personalization)
        print(i, " deliverer:",new_deliverer)
    print("all deliverers",deliverers)
    return deliverers



# 使用理论吸收模型 + 投放增量迭代选择节点
def deliverers_theroy(dataSet, m, init_tranProMatrix,succ_distribution, dis_distribution,constantFactor_distribution,personalization,D):
    # 使用理论模型中的传播矩阵 W，迭代更新影响向量 Q，模拟用户“被影响”的过程

    deliverers = []
    n = init_tranProMatrix.shape[0]
    Q = np.full((1, n), 0,dtype=np.float64) #初始化影响向量 Q 为全 0，表示无人受影响


    tranProMatrix = copy.deepcopy(init_tranProMatrix)
    temp1 = np.multiply(succ_distribution, (1 - constantFactor_distribution))
    tran_increment = np.multiply(succ_distribution, (1 - constantFactor_distribution)).reshape((1,-1)) / D.reshape(-1)

    for i in range(n):
        column_indices = np.nonzero(tranProMatrix[:,i])[0]
        temp = tran_increment[0][i]
        tranProMatrix[column_indices,i] += tran_increment[0][i]# todo 更新转发概率矩阵

    for i in range(m):
       deliverer, Q = single_deliverer.getBestSingleDeliverer_theroy(init_tranProMatrix,succ_distribution,Q,tranProMatrix) # 每轮调用理论传播模型
       deliverers.append(deliverer)
       print(deliverer)
    print(deliverers)
    return deliverers # 最终输出 m 个理论最优投放者




#随机选择m张券的投放源节点 随机选 m 个节点
def deliverers_random(dataSet,m):
    data_file = 'D:/data-processed/{}-adj.pkl'.format(dataSet)
    with open(data_file, 'rb') as f:
        adj = pickle.load(f)
    n = adj.shape[0]
    return random.choices(range(n),k=m)# todo 只选了初始 没进行策略？


# 选邻居最多的 m 个节点
def deliverers_degreeTopM(dataSet,m):
    data_file = 'D:/data-processed/{}-adj.pkl'.format(dataSet)
    with open(data_file, 'rb') as f:
        adj = pickle.load(f)
    degrees = np.asarray(adj.sum(axis=1)).flatten()
    top_m_indexes = np.argsort(degrees)[-m:][::-1]
    return top_m_indexes.tolist() # 根据邻接矩阵计算每个节点的度数，选出度最大的前 m 个节点




#PageRank 值前 m 高	使用图的 PageRank 值选节点
def deliverers_pageRank(dataSet,m):
    # 构建 networkx 图，从中计算 PageRank 值，然后选前 m 高的节点
    data_file = 'D:/data-processed/{}-adj.pkl'.format(dataSet)
    with open(data_file, 'rb') as f:
        adj = pickle.load(f)
    G = nx.from_scipy_sparse_matrix(adj,nx.Graph)
    scores = nx.pagerank(G)
    top_m = sorted(scores,key=scores.get,reverse=True)[:m]
    return top_m




#优惠券使用率最大的m个节点作为投放源节点 使用概率最大前 m 个节点	选择优惠券使用意愿最大的用户
def deliverers_succPro(dataSet,m,succ_distribution):
    # 根据 succ_distribution（每个用户使用优惠券的概率），直接选出前 m 高
    succPro_indexes = np.argsort(succ_distribution)[::-1][:m]
    return succPro_indexes.tolist()




# 	邻居影响力最大	根据用户能“直接影响”邻居的总使用概率选节点
def deliverers_1_neighbor(dataSet,m,succ_distribution,tran_distribution,init_tranProMatrix):
    # 计算每个节点直接影响其邻居成功使用优惠券的概率总和
    temp1 = succ_distribution.reshape(-1,1)
    temp2 = np.multiply(init_tranProMatrix, succ_distribution.reshape(-1,1))
    one_neighbor_pros = np.sum(np.multiply(init_tranProMatrix,succ_distribution.reshape(-1,1)),axis=0)
    one_neighbor_pro_indexes = np.argsort(one_neighbor_pros)[::-1][:m]
    return one_neighbor_pro_indexes

if __name__ == '__main__':
    dataSet = '/root/autodl-tmp/data-processed/facebook-adj.pkl'
    m = 5
    use_pro = 0.4
    dis_pro = 0.2
    L = 100
    constantFactor = 0.5
    adj = None

    G = nx.Graph()
    G.add_nodes_from([1,2,3,4,5])
    G.add_edges_from([(1,3),(1,4),(2,3),(2,5),(3,5)])
    adj = nx.adjacency_matrix(G)

    tran_distribution = np.array([0.5,0.2,0.3,0.5,0.5])# 转发概率
    succ_distribution = np.array([0.3,0.3,0.4,0.2,0.1])# 使用概率
    dis_distribution = np.array([0.2,0.5,0.3,0.3,0.4])# 放弃概率
    constantFactor_distribution = np.array([0.5,0.2,0.3,0.3,0.4])# 调整因子（理论模型用）
    
    personalization = None
    init_tranProMatrix,D = single_deliverer.getTranProMatrix(adj,tran_distribution)
    deliverers = deliverers_threoy(dataSet, m, init_tranProMatrix,succ_distribution, dis_distribution,constantFactor_distribution,personalization,D)
    deliverers = deliverers_monteCarlo(dataSet,m,init_tranProMatrix,succ_distribution,dis_distribution,constantFactor_distribution,L,personalization)
    # deliverers = deliverers_1_neighbor(dataSet,m,succ_distribution,tran_distribution,init_tranProMatrix)
    print(deliverers)
    # deliverers_monteCarlo(dataSet,m,init_tranProMatrix,dis_pro,use_pro,L,constantFactor)