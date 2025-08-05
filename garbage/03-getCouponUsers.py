import pickle
import numpy as np
import networkx as nx
import pandas as pd
import random
import single_deliverer
import copy
#给定投放源预测哪些用户使用了优惠券

#使用蒙特卡洛模拟L轮，根据使用期望排序选取userNum个使用了优惠券的用户
#indexes投放源节点下标
# 用户的使用概率、丢弃概率和转发概率，这些概率分别由succ_distribution、dis_distribution和tranProMatrix表示
def monteCarloSimulation(tranProMatrix,indexes,index,L,succ_distribution,dis_distribution,constantFactor_distribution,personalization):
    n = tranProMatrix.shape[0]
    S = np.zeros((1,n))
    avg_succ_pros = np.full((1, n), 0,dtype=np.float64)

    temp_tranProMatrix = copy.deepcopy(tranProMatrix)# 转发概率
    temp_succ_distribution = copy.deepcopy(succ_distribution)# 用户的使用概率
    temp_dis_distribution = copy.deepcopy(dis_distribution)# 丢弃概率

    for i in range(L): # 进行L轮蒙特卡洛模拟，记录每轮中用户使用优惠券的情况

        if personalization == None:
            avg_succ_pros += monteCarlo_singleTime(temp_tranProMatrix,indexes,temp_succ_distribution,temp_dis_distribution,constantFactor_distribution)
        elif personalization == 'firstUnused':
            avg_succ_pros += monteCarlo_singleTime_firstUnused(temp_tranProMatrix,indexes,temp_succ_distribution,temp_dis_distribution,constantFactor_distribution)
        elif personalization =='firstDiscard':
            avg_succ_pros += monteCarlo_singleTime_firstDiscard(temp_tranProMatrix,indexes,temp_succ_distribution,temp_dis_distribution,constantFactor_distribution)


    avg_succ_pros /= L # 计算平均使用概率
    highest_deliverer = np.argmax(avg_succ_pros)
    # if highest_user not in users:
    #     modify_tranProMatrix_singleUser(tranProMatrix,highest_user,constantFactor,use_pro)
    indexes.append(highest_deliverer)
    return indexes,highest_deliverer

#单轮蒙特卡洛模拟，返回这轮中哪些节点使用了券，哪些废弃了券
#constantFactor常数因子
def monteCarlo_singleTime(tranProMatrix,indexes,succ_distribution,dis_distribution,constantFactor_distribution):
    n = tranProMatrix.shape[0]
    users_useAndDis = []
    # succ_pros = np.full((1, n), 0,dtype=np.float64)
    for index in indexes:
        random_pro = np.random.rand() # 生成 [0, 1) 之间的随机数，用于模拟用户的行为
        next_node = index

    
        if next_node not in users_useAndDis:
            # 根据生成的随机数和用户的概率分布，模拟优惠券的传播过程
            if random_pro < succ_distribution[next_node]: # 用户使用优惠券
                users_useAndDis.append(next_node)
                # todo 
                modify_tranProMatrix_singleUser(tranProMatrix, next_node, constantFactor_distribution[next_node],succ_distribution[next_node])
                continue
            elif random_pro < succ_distribution[next_node] + dis_distribution[next_node]: #用户丢弃优惠券
                continue
            else: # 否则，用户将优惠券转发给邻居
                neighbors = [index for index, value in enumerate(tranProMatrix[:, next_node]) if value != 0]
                if len(neighbors)>0:
                    neighbors_pro = [value for index, value in enumerate(tranProMatrix[:, next_node]) if value != 0]
                    neighbors_pro = neighbors_pro / np.sum(neighbors_pro)
                    next_node = np.random.choice(neighbors, p=neighbors_pro)
        
        
        else:
            if random_pro < dis_distribution[next_node] + constantFactor_distribution[next_node] * succ_distribution[next_node]:
                continue
            else:
                neighbors = [index for index, value in enumerate(tranProMatrix[:, next_node]) if value != 0]
                if len(neighbors) > 0:
                    neighbors_pro = [value for index, value in enumerate(tranProMatrix[:, next_node]) if
                                     value != 0]
                    neighbors_pro = neighbors_pro / np.sum(neighbors_pro)
                    next_node = np.random.choice(neighbors, p=neighbors_pro)
                else:
                    continue

        while (True):
            random_pro = np.random.rand()
            if next_node not in users_useAndDis:
                if random_pro<succ_distribution[next_node]:
                    users_useAndDis.append(next_node)
                    modify_tranProMatrix_singleUser(tranProMatrix, next_node, constantFactor_distribution[next_node], succ_distribution[next_node])
                    break
                elif random_pro < succ_distribution[next_node]+dis_distribution[next_node]:
                    break
                else:
                    neighbors = [index for index, value in enumerate(tranProMatrix[:,next_node]) if value != 0]
                    if len(neighbors) > 0:
                        neighbors_pro = [value for index, value in enumerate(tranProMatrix[:,next_node]) if value != 0]
                        neighbors_pro = neighbors_pro / np.sum(neighbors_pro)
                        next_node = np.random.choice(neighbors, p=neighbors_pro)

            else:
                if random_pro < dis_distribution[next_node]+constantFactor_distribution[next_node]*succ_distribution[next_node]:
                    break
                else:
                    neighbors = [index for index, value in enumerate(tranProMatrix[:,next_node]) if value != 0]
                    if len(neighbors) > 0:
                        neighbors_pro = [value for index, value in enumerate(tranProMatrix[:,next_node]) if value != 0]
                        neighbors_pro = neighbors_pro / np.sum(neighbors_pro)
                        next_node = np.random.choice(neighbors, p=neighbors_pro)
                    else:
                        break
            
    # **结束条件**: 一个随机游走分支在以下情况结束：节点决定“使用”或“丢弃”。节点尝试转发，但没有邻居可供转发。                    
    succ_pros = get_succPros(tranProMatrix, users_useAndDis, succ_distribution)
    return succ_pros

def monteCarlo_singleTime_firstUnused(tranProMatrix,indexes,succ_distribution,dis_distribution,constantFactor_distribution):
    n = tranProMatrix.shape[0]
    users_useAndDis = []
    firstUnused = []
    # succ_pros = np.full((1, n), 0,dtype=np.float64)
    for index in indexes:
        random_pro = np.random.rand()
        next_node = index
        if next_node not in users_useAndDis and next_node not in firstUnused:
            if random_pro < succ_distribution[next_node]:
                users_useAndDis.append(next_node)
                modify_tranProMatrix_singleUser(tranProMatrix, next_node, constantFactor_distribution[next_node],
                                                succ_distribution[next_node])
                continue
            elif random_pro < succ_distribution[next_node] + dis_distribution[next_node]:
                firstUnused.append(next_node)
                modify_tranProMatrix_singleUser(tranProMatrix, next_node, constantFactor_distribution[next_node],
                                                succ_distribution[next_node])
                continue
            else:
                neighbors = [index for index, value in enumerate(tranProMatrix[:, next_node]) if value != 0]
                if len(neighbors) > 0:
                    firstUnused.append(next_node)
                    neighbors_pro = [value for index, value in enumerate(tranProMatrix[:, next_node]) if value != 0]
                    neighbors_pro = neighbors_pro / np.sum(neighbors_pro)
                    next_node = np.random.choice(neighbors, p=neighbors_pro)

        else:
            if random_pro < dis_distribution[next_node] + constantFactor_distribution[next_node] * succ_distribution[
                next_node]:
                continue
            else:
                neighbors = [index for index, value in enumerate(tranProMatrix[:, next_node]) if value != 0]
                if len(neighbors) > 0:
                    neighbors_pro = [value for index, value in enumerate(tranProMatrix[:, next_node]) if
                                     value != 0]
                    neighbors_pro = neighbors_pro / np.sum(neighbors_pro)
                    next_node = np.random.choice(neighbors, p=neighbors_pro)
                else:
                    continue

        while (True):
            random_pro = np.random.rand()
            if next_node not in users_useAndDis and next_node not in firstUnused:
                if random_pro < succ_distribution[next_node]:
                    users_useAndDis.append(next_node)
                    modify_tranProMatrix_singleUser(tranProMatrix, next_node, constantFactor_distribution[next_node],
                                                    succ_distribution[next_node])
                    break
                elif random_pro < succ_distribution[next_node] + dis_distribution[next_node]:
                    firstUnused.append(next_node)
                    modify_tranProMatrix_singleUser(tranProMatrix, next_node, constantFactor_distribution[next_node],
                                                    succ_distribution[next_node])
                    break
                else:
                    neighbors = [index for index, value in enumerate(tranProMatrix[:, next_node]) if value != 0]
                    if len(neighbors) > 0:
                        firstUnused.append(next_node)
                        neighbors_pro = [value for index, value in enumerate(tranProMatrix[:, next_node]) if value != 0]
                        neighbors_pro = neighbors_pro / np.sum(neighbors_pro)
                        next_node = np.random.choice(neighbors, p=neighbors_pro)



            else:
                if random_pro < dis_distribution[next_node] + constantFactor_distribution[next_node] * \
                        succ_distribution[next_node]:
                    break
                else:
                    neighbors = [index for index, value in enumerate(tranProMatrix[:, next_node]) if value != 0]
                    if len(neighbors) > 0:
                        neighbors_pro = [value for index, value in enumerate(tranProMatrix[:, next_node]) if value != 0]
                        neighbors_pro = neighbors_pro / np.sum(neighbors_pro)
                        next_node = np.random.choice(neighbors, p=neighbors_pro)
                    else:
                        break
    if len(firstUnused)>0:
       users_useAndDis.extend(firstUnused)
    succ_pros = get_succPros(tranProMatrix, users_useAndDis, succ_distribution)
    return succ_pros

def monteCarlo_singleTime_firstDiscard(tranProMatrix,indexes,succ_distribution,dis_distribution,constantFactor_distribution):
    n = tranProMatrix.shape[0]
    users_useAndDis = []
    firstdiscard = []
    # succ_pros = np.full((1, n), 0,dtype=np.float64)
    for index in indexes:
        random_pro = np.random.rand()
        next_node = index
        if next_node not in users_useAndDis and next_node not in firstdiscard:
            if random_pro < succ_distribution[next_node]:
                users_useAndDis.append(next_node)
                modify_tranProMatrix_singleUser(tranProMatrix, next_node, constantFactor_distribution[next_node],
                                                succ_distribution[next_node])
                continue
            elif random_pro < succ_distribution[next_node] + dis_distribution[next_node]:
                firstdiscard.append(next_node)
                modify_tranProMatrix_singleUser(tranProMatrix, next_node, constantFactor_distribution[next_node],
                                                succ_distribution[next_node])
                continue
            else:
                neighbors = [index for index, value in enumerate(tranProMatrix[:, next_node]) if value != 0]
                if len(neighbors) > 0:
                    neighbors_pro = [value for index, value in enumerate(tranProMatrix[:, next_node]) if value != 0]
                    neighbors_pro = neighbors_pro / np.sum(neighbors_pro)
                    next_node = np.random.choice(neighbors, p=neighbors_pro)

        else:
            if random_pro < dis_distribution[next_node] + constantFactor_distribution[next_node] * succ_distribution[
                next_node]:
                continue
            else:
                neighbors = [index for index, value in enumerate(tranProMatrix[:, next_node]) if value != 0]
                if len(neighbors) > 0:
                    neighbors_pro = [value for index, value in enumerate(tranProMatrix[:, next_node]) if
                                     value != 0]
                    neighbors_pro = neighbors_pro / np.sum(neighbors_pro)
                    next_node = np.random.choice(neighbors, p=neighbors_pro)
                else:
                    continue

        while (True):
            random_pro = np.random.rand()
            if next_node not in users_useAndDis and next_node not in firstdiscard:
                if random_pro < succ_distribution[next_node]:
                    users_useAndDis.append(next_node)
                    modify_tranProMatrix_singleUser(tranProMatrix, next_node, constantFactor_distribution[next_node],
                                                    succ_distribution[next_node])
                    break
                elif random_pro < succ_distribution[next_node] + dis_distribution[next_node]:
                    firstdiscard.append(next_node)
                    modify_tranProMatrix_singleUser(tranProMatrix, next_node, constantFactor_distribution[next_node],
                                                    succ_distribution[next_node])
                    break
                else:
                    neighbors = [index for index, value in enumerate(tranProMatrix[:, next_node]) if value != 0]
                    if len(neighbors) > 0:
                        neighbors_pro = [value for index, value in enumerate(tranProMatrix[:, next_node]) if value != 0]
                        neighbors_pro = neighbors_pro / np.sum(neighbors_pro)
                        next_node = np.random.choice(neighbors, p=neighbors_pro)

            else:
                if random_pro < dis_distribution[next_node] + constantFactor_distribution[next_node] * \
                        succ_distribution[next_node]:
                    if next_node not in firstdiscard:
                        firstdiscard.append(next_node)
                        modify_tranProMatrix_singleUser(tranProMatrix, next_node,
                                                        constantFactor_distribution[next_node],
                                                        succ_distribution[next_node])
                    break
                else:
                    neighbors = [index for index, value in enumerate(tranProMatrix[:, next_node]) if value != 0]
                    if len(neighbors) > 0:
                        neighbors_pro = [value for index, value in enumerate(tranProMatrix[:, next_node]) if value != 0]
                        neighbors_pro = neighbors_pro / np.sum(neighbors_pro)
                        next_node = np.random.choice(neighbors, p=neighbors_pro)
                    else:
                        break
    if len(firstdiscard)>0:
       users_useAndDis.extend(firstdiscard)
    succ_pros = get_succPros(tranProMatrix, users_useAndDis, succ_distribution)
    return succ_pros

def modify_tranProMatrix_singleUser(tranProMatrix,index,constantFactor,use_pro):
    if np.count_nonzero(tranProMatrix[:,index])>0:
        tranProMatrix[:, index] += np.where(tranProMatrix[:, index] != 0,
                                           (1 - constantFactor) / np.count_nonzero(tranProMatrix[:, index]) * use_pro, 0)
    return tranProMatrix

def modify_tranProMatrix_Users(tranProMatrix,users,constantFactor,use_pro):
    unique_users = list(set(users))
    for user in unique_users:
      tranProMatrix[:,user] += np.where(tranProMatrix[:,user]!=0,(1-constantFactor)/np.count_nonzero(tranProMatrix[:,user])*use_pro,0)
    return tranProMatrix

def get_succPros(tranProMatrix,users_useAndDis,succ_distribution):
    n = tranProMatrix.shape[0]
    I = np.eye(n)
    inverse_matrix = np.linalg.inv(I - tranProMatrix)

    # deliverers_neighbors_usePro = use_pros = np.full((1,n),use_pro)

    if len(users_useAndDis)>0:
        succ_distribution[users_useAndDis] = 0
    # succ_pros = np.dot(np.dot(inverse_matrix, tranProMatrix),neighbor_having)
    # succ_pros = np.dot(use_pros,succ_pros)
    succ_pros = np.dot(succ_distribution,np.dot(inverse_matrix, tranProMatrix))+succ_distribution
    return succ_pros



if __name__ == '__main__':
    data_file = 'D:/data-processed/{}-adj.pkl'.format('Amherst')
    with open(data_file, 'rb') as f:
        adj = pickle.load(f)
    n = adj.shape[0]
    use_pro = 0.4
    dis_pro = 0.2
    L = 5
    constantFactor = 0.5
    users = []
    tranProMatrix, neighbor_having = single_deliverer.getTranProMatrix(adj,use_pro,dis_pro)
    max_column_index = single_deliverer.getBestSingleDeliverer(tranProMatrix,use_pro,neighbor_having)
    indexes = [max_column_index]
    tranProMatrix, users = monteCarloSimulation(tranProMatrix,indexes,L,dis_pro,use_pro,constantFactor,users)