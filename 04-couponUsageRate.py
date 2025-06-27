import pickle
import numpy as np
import networkx as nx
import pandas as pd
import random
import single_deliverer
import copy
import single_deliverer
import getCouponUsers
import networkx as nx
import get_couponDeliverers
import os
import time
import random
from scipy.stats import truncnorm
#dataSet 数据集
#times 模拟轮数
#methods 使用的方法集
#m 投放源节点个数
#monteCarlo_L 蒙特卡洛模拟轮数
def couponUsageRate(dataSet,times,methods,seedNum_list,monteCarlo_L,distribution,constantFactorDistri,personalization,dataFile_prefix,method_type):
    print(dataSet)
    data_file = 'D:/data-processed/{}-adj.pkl'.format(dataSet)
    with open(data_file, 'rb') as f:
        adj = pickle.load(f)
    n = adj.shape[0]
    m = seedNum_list[-1]
    distribution_file = 'D:/{}/distribution{}_distri{}_constantFactor{}_seedNum{}.pkl' \
        .format(dataFile_prefix,dataSet, distribution, constantFactorDistri, m)
    distribution_list = get_distribution(distribution_file,distribution,n)
    succ_distribution, dis_distribution, tran_distribution, constantFactor_distribution = distribution_list
    init_tranProMatrix,D = single_deliverer.getTranProMatrix(adj,tran_distribution)

    method_deliverers = []
    method_deliverers_file = 'D:/{}/deliverers_{}_distri{}_constantFactor{}_monteCarloL{}_seedNum{}.txt'\
        .format(dataFile_prefix,dataSet,distribution,constantFactorDistri,monteCarlo_L,m,personalization)
    methods_temp = []
    method2runningTime = {key:0 for key in methods}

    if method_type == None:
        if os.path.exists(method_deliverers_file):
            with open(method_deliverers_file, 'r') as f:
                for i in range(len(methods)):
                    line = f.readline().strip()
                    parts = line.strip().split(':')
                    methods_temp.append(parts[0])
                    deliverers = eval(parts[1])
                    method_deliverers.append(deliverers)
            methods = methods_temp
        else:
            print(methods)
            for method in methods:
                start_time = time.time()
                if method == 'deliverers_theroy':
                    deliverers = get_couponDeliverers.deliverers_theroy(dataSet, m, init_tranProMatrix,
                                                                            succ_distribution, dis_distribution,
                                                                            constantFactor_distribution,
                                                                            personalization,D)
                    method_deliverers.append(deliverers)
                elif method == 'monteCarlo':
                    deliverers = get_couponDeliverers.deliverers_monteCarlo(dataSet,m,init_tranProMatrix,
                                                                            succ_distribution,dis_distribution,
                                                                            constantFactor_distribution,monteCarlo_L,
                                                                            personalization)
                    method_deliverers.append(deliverers)
                elif method == 'random':
                    deliverers = get_couponDeliverers.deliverers_random(dataSet,m)
                    method_deliverers.append(deliverers)
                elif method == 'degreeTopM':
                    deliverers = get_couponDeliverers.deliverers_degreeTopM(dataSet,m)
                    method_deliverers.append(deliverers)
                elif method == 'pageRank':
                    deliverers = get_couponDeliverers.deliverers_pageRank(dataSet,m)
                    method_deliverers.append(deliverers)
                elif method == 'succPro':
                    deliverers = get_couponDeliverers.deliverers_succPro(dataSet,m,succ_distribution)
                    method_deliverers.append(deliverers)
                elif method == '1_neighbor':
                    deliverers = get_couponDeliverers.deliverers_1_neighbor(dataSet,m,succ_distribution,tran_distribution,init_tranProMatrix)
                    method_deliverers.append(deliverers)
                end_time = time.time()

                method2runningTime[method] = end_time-start_time
            res_list = list(zip(methods, method_deliverers))
            with open(method_deliverers_file, 'a+') as file:
                for item in res_list:
                    file.write(f'{item[0]}:{item[1]}\n')
                for key, value in method2runningTime.items():
                    file.write(f'{key}:{value}\n')

    elif method_type == 'new':
        print( methods)
        for method in methods:
            if method == 'DeepIM_IC':
                deliverers = []
                DeepIM_IC_file = 'D:/res_coupon_deliverers/outcome/DeepIM/{}_IC_extracted_0_2.0%.txt'.format(dataSet)
                with open(DeepIM_IC_file, 'rb') as f:
                    for node in f:
                        deliverers.append(int(node))
                method_deliverers.append(deliverers)
            elif method == 'DeepIM_LT':
                deliverers = []
                DeepIM_LT_file = 'D:/res_coupon_deliverers/outcome/DeepIM/{}_IC_extracted_0_2.0%.txt'.format(dataSet)
                with open(DeepIM_LT_file, 'rb') as f:
                    for node in f:
                        deliverers.append(int(node))
                method_deliverers.append(deliverers)
            elif method == 'OPIM_IC':
                deliverers = []
                OPIM_IC_file = 'D:/res_coupon_deliverers\outcome/OPIM/{}/{}_IC_seedsize_2%.txt'.format(dataSet,dataSet)
                with open(OPIM_IC_file, 'rb') as f:
                    for node in f:
                        deliverers.append(int(node))
                method_deliverers.append(deliverers)
            elif method == 'OPIM_LT':
                deliverers = []
                OPIM_LT_file = 'D:/res_coupon_deliverers\outcome/OPIM/{}/{}_LT_seedsize_2%.txt'.format(dataSet,dataSet)
                with open(OPIM_LT_file, 'rb') as f:
                    for node in f:
                        deliverers.append(int(node))
                method_deliverers.append(deliverers)

    if personalization == 'None':
        usageRate_file = 'D:/{}/usageRate_{}_distri{}_constantFactor{}_monteCarloL{}_testTimes{}_seedNum{}_{}.txt' \
            .format(dataFile_prefix,dataSet, distribution, constantFactorDistri, monteCarlo_L, times, m,personalization)
        with open(usageRate_file, 'a+') as file:
            file.write(f'times:{times}\n')
        file.close()
        simulation(methods,method_deliverers,init_tranProMatrix,usageRate_file,distribution_list,seedNum_list,)
    elif personalization == 'firstUnused':
        usageRate_file = 'D:/{}/usageRate_{}_distri{}_constantFactor{}_monteCarloL{}_testTimes{}_seedNum{}_{}.txt' \
            .format(dataFile_prefix,dataSet, distribution, constantFactorDistri, monteCarlo_L, times, m,personalization)
        with open(usageRate_file, 'a+') as file:
            file.write(f'times:{times}\n')
        file.close()
        simulation_firstUnused(methods, method_deliverers, init_tranProMatrix, usageRate_file, distribution_list, seedNum_list)
    elif personalization == 'firstDiscard':
        usageRate_file = 'D:/{}/usageRate_{}_distri{}_constantFactor{}_monteCarloL{}_testTimes{}_seedNum{}_{}.txt' \
            .format(dataFile_prefix,dataSet, distribution, constantFactorDistri, monteCarlo_L, times, m,personalization)
        with open(usageRate_file, 'a+') as file:
            file.write(f'times:{times}\n')
        file.close()
        simulation_firstDiscard(methods, method_deliverers, init_tranProMatrix, usageRate_file, distribution_list, seedNum_list)
    # method_usageRate_list = list(zip(methods, usageRate_list))
    # with open(usageRate_file, 'w') as file:
    #     for item in method_usageRate_list:
    #         file.write(f'{item[0]}:{item[1]}\n')



def get_distribution(distribution_file,distribution,n):
    if os.path.exists(distribution_file):
        with open(distribution_file, 'rb') as f:
            dis_dict = pickle.load(f)
            succ_distribution = dis_dict['succ_distribution']
            dis_distribution = dis_dict['dis_distribution']
            tran_distribution = dis_dict['tran_distribution']
            # test_distribution = succ_distribution+dis_distribution+tran_distribution
            constantFactor_distribution = dis_dict['constantFactor_distribution']
    else:
        if distribution == 'random':
            tran_distribution = 0.5 + 0.2 * np.random.rand(n)
            succ_distribution = np.random.uniform(0.2, 0.3, n)
            # dis_distribution = np.random.uniform(0.1,0.1,n)
            # dis_distribution = dis_distribution+(1-tran_distribution-dis_distribution)*np.random.rand(n)
            # succ_distribution = 1 - tran_distribution - dis_distribution
            # succ_distribution = (1-tran_distribution)*np.random.rand(n)
            # succ_distribution = 1 - tran_distribution - dis_distribution
            # succ_distribution = np.random.uniform(0.2,0.3,n)
            # tran_distribution = 0.5+0.2*np.random.rand(n)
            dis_distribution = 1-tran_distribution-succ_distribution
            # tran_distribution = 1-succ_distribution-dis_distribution
            total = succ_distribution + dis_distribution + tran_distribution
            # succ_distribution = succ_distribution / total
            # dis_distribution = dis_distribution / total
            # tran_distribution = tran_distribution / total
            constantFactor_distribution = np.full(n,1.0)
            # constantFactor_distribution = 0.0 * np.random.rand(n)
            # test = succ_distribution+dis_distribution+tran_distribution
        elif distribution == 'poisson':
            lambda_1 = np.random.uniform(1, 10)
            lambda_2 = np.random.uniform(1, 10)
            lambda_3 = np.random.uniform(1, 10)
            lambda_4 = np.random.uniform(1, 10)
            succ_distribution = np.random.poisson(lambda_1,n).astype(float)
            dis_distribution = np.random.poisson(lambda_2,n).astype(float)
            tran_distribution = np.random.poisson(lambda_3,n).astype(float)
            # tran_distribution = np.clip(tran_distribution,0,1)
            total = (succ_distribution + dis_distribution + tran_distribution).astype(float)
            succ_distribution = np.divide(succ_distribution,total,out=np.zeros_like(succ_distribution),where=(total!=0))
            dis_distribution = np.divide(dis_distribution,total,out=np.zeros_like(dis_distribution),where=(total!=0))
            # tran_distribution = np.divide(tran_distribution,total,out=np.ones_like(tran_distribution),where=(total!=0)) # todo gzc modify
            tran_distribution = np.divide(tran_distribution,total,out=np.zeros_like(tran_distribution),where=(total!=0))
            constantFactor_distribution = np.random.poisson(lambda_4,n).astype(float)
            constantFactor_distribution = (constantFactor_distribution-np.min(constantFactor_distribution))\
                                          /(np.max(constantFactor_distribution)-np.min(constantFactor_distribution))
            # test = succ_distribution+dis_distribution+tran_distribution
        elif distribution == 'normal':
            succ_distribution = truncnorm.rvs(0,np.inf,loc=1,scale=1,size=n)
            dis_distribution = truncnorm.rvs(0,np.inf,loc=1,scale=1,size=n)
            tran_distribution = truncnorm.rvs(0,np.inf,loc=1,scale=1,size=n)
            total = succ_distribution + dis_distribution + tran_distribution
            succ_distribution = np.divide(succ_distribution, total, out=np.zeros_like(succ_distribution),
                                          where=(total != 0))
            dis_distribution = np.divide(dis_distribution, total, out=np.zeros_like(dis_distribution),
                                         where=(total != 0))
            # tran_distribution = np.divide(tran_distribution, total, out=np.ones_like(tran_distribution), # todo gzc modify
            #                               where=(total != 0))
            tran_distribution = np.divide(tran_distribution, total, out=np.zeros_like(tran_distribution), 
                                          where=(total != 0))
            constantFactor_distribution = truncnorm.rvs(0,np.inf,loc=0,scale=1,size=n)
            constantFactor_distribution = (constantFactor_distribution - np.min(constantFactor_distribution)) \
                                          / (np.max(constantFactor_distribution) - np.min(constantFactor_distribution))
            # test = succ_distribution+dis_distribution+tran_distribution
        with open(distribution_file, 'wb') as f:
            dis_dict = {'succ_distribution':succ_distribution,'dis_distribution':dis_distribution,
                        'tran_distribution':tran_distribution,'constantFactor_distribution':constantFactor_distribution}
            pickle.dump(dis_dict,f)
    return succ_distribution,dis_distribution,tran_distribution,constantFactor_distribution


#模拟一组优惠券传播，得到券使用率，methods投放源计算方法，method_deliverers投放源计算方法对应的投放源节点
def simulation(methods,method_deliverers,init_tranProMatrix,usageRate_file,distribution_list,seedNum_list):
    succ_distribution, dis_distribution, tran_distribution, constantFactor_distribution = distribution_list
    m = seedNum_list[-1]
    lastSeedSet2usageNum = {key: 0 for key in methods}
    for k in range(len(seedNum_list)):
        seedNum = seedNum_list[k]

        with open(usageRate_file, 'a+') as file:
            file.write(f'seedNum:{seedNum}\n')
            
        for i in range(len(methods)):
            if k == 0:
                deliverers = method_deliverers[i][:seedNum]
            else:
                deliverers = method_deliverers[i][seedNum_list[k-1]:seedNum_list[k]]

            method = methods[i]
            usageNum = 0
            count = 0
            avg_usageRate_list = []
            usageNum_list = []
            for j in range(times[-1]):
                tranProMatrix = copy.deepcopy(init_tranProMatrix)
                users_useAndDis = []
                for index in deliverers:
                    random_pro = np.random.rand()
                    next_node = index
                    if next_node not in users_useAndDis:
                        if random_pro < tran_distribution[next_node]:
                            neighbors = [index for index, value in enumerate(tranProMatrix[:, next_node]) if value != 0]
                            if len(neighbors) > 0:
                                neighbors_pro = [value for index, value in enumerate(tranProMatrix[:, next_node]) if
                                                 value != 0]
                                neighbors_pro = neighbors_pro / np.sum(neighbors_pro)
                                next_node = np.random.choice(neighbors, p=neighbors_pro)
                        elif random_pro < tran_distribution[next_node]+dis_distribution[next_node]:
                            continue
                        else:
                            users_useAndDis.append(next_node)
                            tranProMatrix = getCouponUsers.modify_tranProMatrix_singleUser(tranProMatrix, next_node,
                                                                                           constantFactor_distribution[
                                                                                               next_node],
                                                                                           succ_distribution[next_node])
                            usageNum += 1
                            continue
                        # if random_pro < succ_distribution[next_node]:
                        #     users_useAndDis.append(next_node)
                        #     tranProMatrix = getCouponUsers.modify_tranProMatrix_singleUser(tranProMatrix, next_node, constantFactor_distribution[next_node],
                        #                                                    succ_distribution[next_node])
                        #     usageNum += 1
                        #     continue
                        # elif random_pro < succ_distribution[next_node] + dis_distribution[next_node]:
                        #     continue
                        # else:
                        #     neighbors = [index for index, value in enumerate(tranProMatrix[:, next_node]) if value != 0]
                        #     if len(neighbors) > 0:
                        #         neighbors_pro = [value for index, value in enumerate(tranProMatrix[:, next_node]) if value != 0]
                        #         neighbors_pro = neighbors_pro / np.sum(neighbors_pro)
                        #         next_node = np.random.choice(neighbors, p=neighbors_pro)
                    else:
                        if random_pro < tran_distribution[next_node] + (1-constantFactor_distribution[next_node]) * succ_distribution[next_node]:
                            neighbors = [index for index, value in enumerate(tranProMatrix[:, next_node]) if value != 0]
                            if len(neighbors) > 0:
                                neighbors_pro = [value for index, value in enumerate(tranProMatrix[:, next_node]) if
                                                 value != 0]
                                neighbors_pro = neighbors_pro / np.sum(neighbors_pro)
                                next_node = np.random.choice(neighbors, p=neighbors_pro)
                            else:
                                continue
                        else:
                            continue
                        # if random_pro < dis_distribution[next_node] + constantFactor_distribution[next_node] * succ_distribution[next_node]:
                        #     continue
                        # else:
                        #     neighbors = [index for index, value in enumerate(tranProMatrix[:, next_node]) if value != 0]
                        #     if len(neighbors) > 0:
                        #         neighbors_pro = [value for index, value in enumerate(tranProMatrix[:, next_node]) if
                        #                          value != 0]
                        #         neighbors_pro = neighbors_pro / np.sum(neighbors_pro)
                        #         next_node = np.random.choice(neighbors, p=neighbors_pro)
                        #     else:
                        #         continue
                    while (True):
                        random_pro = np.random.rand()
                        if next_node not in users_useAndDis:
                            if random_pro < tran_distribution[next_node]:
                                neighbors = [index for index, value in enumerate(tranProMatrix[:, next_node]) if
                                             value != 0]
                                if len(neighbors) > 0:
                                    neighbors_pro = [value for index, value in enumerate(tranProMatrix[:, next_node]) if
                                                     value != 0]
                                    neighbors_pro = neighbors_pro / np.sum(neighbors_pro)
                                    next_node = np.random.choice(neighbors, p=neighbors_pro)
                            elif random_pro < tran_distribution[next_node] + dis_distribution[next_node]:
                                break
                            else:
                                users_useAndDis.append(next_node)
                                tranProMatrix = getCouponUsers.modify_tranProMatrix_singleUser(tranProMatrix, next_node,
                                                                                               constantFactor_distribution[
                                                                                                   next_node],
                                                                                               succ_distribution[
                                                                                                   next_node])
                                usageNum += 1
                                break
                        # if next_node not in users_useAndDis:
                        #     if random_pro < succ_distribution[next_node]:
                        #         users_useAndDis.append(next_node)
                        #         tranProMatrix = getCouponUsers.modify_tranProMatrix_singleUser(tranProMatrix, next_node, constantFactor_distribution[next_node], succ_distribution[next_node])
                        #         usageNum += 1
                        #         break
                        #     elif random_pro < succ_distribution[next_node]+dis_distribution[next_node]:
                        #         break
                        #     else:
                        #         neighbors = [index for index, value in enumerate(tranProMatrix[:, next_node]) if value != 0]
                        #         if len(neighbors) > 0:
                        #             neighbors_pro = [value for index, value in enumerate(tranProMatrix[:, next_node]) if
                        #                              value != 0]
                        #             neighbors_pro = neighbors_pro / np.sum(neighbors_pro)
                        #             next_node = np.random.choice(neighbors, p=neighbors_pro)

                        else:
                            if random_pro < tran_distribution[next_node] + (
                                    1 - constantFactor_distribution[next_node]) * succ_distribution[next_node]:
                                neighbors = [index for index, value in enumerate(tranProMatrix[:, next_node]) if
                                             value != 0]
                                if len(neighbors) > 0:
                                    neighbors_pro = [value for index, value in enumerate(tranProMatrix[:, next_node]) if
                                                     value != 0]
                                    neighbors_pro = neighbors_pro / np.sum(neighbors_pro)
                                    next_node = np.random.choice(neighbors, p=neighbors_pro)
                                else:
                                    continue
                            else:
                                break
                            # if random_pro < dis_distribution[next_node] + constantFactor_distribution[next_node] * succ_distribution[next_node]:
                            #     break
                            # else:
                            #     neighbors = [index for index, value in enumerate(tranProMatrix[:, next_node]) if value != 0]
                            #     if len(neighbors) > 0:
                            #         neighbors_pro = [value for index, value in enumerate(tranProMatrix[:, next_node]) if
                            #                          value != 0]
                            #         neighbors_pro = neighbors_pro / np.sum(neighbors_pro)
                            #         next_node = np.random.choice(neighbors, p=neighbors_pro)
                            #     else:
                            #         break
                if j == times[count]-1:
                    if k == 0:
                        avg_usageRate = usageNum/times[count]/seedNum
                        avg_usageRate_list.append(avg_usageRate)
                        last_usageNum = 0
                    else:
                        last_usageNum = lastSeedSet2usageNum[method][count]
                        avg_usageRate = (last_usageNum+usageNum)/times[count]/seedNum
                        avg_usageRate_list.append(avg_usageRate)
                    usageNum_list.append(last_usageNum+usageNum)
                    print('seedNum='+str(seedNum)+' '+method + ' avgUsageRate = ' + str(avg_usageRate)+' times:'+ str(times[count]))
                    count += 1
                    
            with open(usageRate_file, 'a+') as file:
                file.write(f'{method}:{avg_usageRate_list}\n')
                lastSeedSet2usageNum[method] = usageNum_list
    return None

#在simulation基础上：对于首次取得券未使用的用户后续使用概率为0
def simulation_firstUnused(methods, method_deliverers, init_tranProMatrix, usageRate_file, distribution_list, seedNum_list):
    succ_distribution, dis_distribution, tran_distribution, constantFactor_distribution = distribution_list
    m = seedNum_list[-1]
    lastSeedSet2usageNum = {key: 0 for key in methods}
    for k in range(len(seedNum_list)):
        seedNum = seedNum_list[k]
        with open(usageRate_file, 'a+') as file:
            file.write(f'seedNum:{seedNum}\n')
        for i in range(len(methods)):
            if k == 0:
                deliverers = method_deliverers[i][:seedNum]
            else:
                deliverers = method_deliverers[i][seedNum_list[k-1]:seedNum_list[k]]
            method = methods[i]
            usageNum = 0
            count = 0
            avg_usageRate_list = []
            usageNum_list = []
            for j in range(times[-1]):
                tranProMatrix = copy.deepcopy(init_tranProMatrix)
                users_useAndDis = []
                firstUnused_list = []
                for index in deliverers:
                    random_pro = np.random.rand()
                    next_node = index
                    if next_node not in users_useAndDis and next_node not in firstUnused_list:
                        tranProMatrix = getCouponUsers.modify_tranProMatrix_singleUser(tranProMatrix, next_node,
                                                                                       constantFactor_distribution[
                                                                                           next_node],
                                                                                       succ_distribution[next_node])
                        if random_pro < succ_distribution[next_node]:
                            users_useAndDis.append(next_node)
                            usageNum += 1
                            continue
                        elif random_pro < succ_distribution[next_node] + dis_distribution[next_node] :
                            firstUnused_list.append(next_node)
                            continue
                        else:
                            firstUnused_list.append(next_node)
                            neighbors = [index for index, value in enumerate(tranProMatrix[:, next_node]) if value != 0]
                            if len(neighbors) > 0:
                                neighbors_pro = [value for index, value in enumerate(tranProMatrix[:, next_node]) if
                                                 value != 0]
                                neighbors_pro = neighbors_pro / np.sum(neighbors_pro)
                                next_node = np.random.choice(neighbors, p=neighbors_pro)
                    else:
                        if random_pro < dis_distribution[next_node] + constantFactor_distribution[next_node] * \
                                succ_distribution[next_node]:
                            continue
                        else:
                            neighbors = [index for index, value in enumerate(tranProMatrix[:, next_node]) if value != 0]
                            if len(neighbors) > 0:
                                neighbors_pro = [value for index, value in enumerate(tranProMatrix[:, next_node]) if
                                                 value != 0]
                                neighbors_pro = neighbors_pro / np.sum(neighbors_pro)
                                next_node = np.random.choice(neighbors, p=neighbors_pro)
                            else:
                                break
                    while (True):
                        random_pro = np.random.rand()
                        if next_node not in users_useAndDis and next_node not in firstUnused_list:
                            tranProMatrix = getCouponUsers.modify_tranProMatrix_singleUser(tranProMatrix, next_node,
                                                                                           constantFactor_distribution[
                                                                                               next_node],
                                                                                           succ_distribution[next_node])
                            if random_pro < succ_distribution[next_node]:
                                users_useAndDis.append(next_node)
                                usageNum += 1
                                break
                            elif random_pro < succ_distribution[next_node] + dis_distribution[next_node]:
                                firstUnused_list.append(next_node)
                                break
                            else:
                                firstUnused_list.append(next_node)
                                neighbors = [index for index, value in enumerate(tranProMatrix[:, next_node]) if value != 0]
                                if len(neighbors) > 0:
                                    neighbors_pro = [value for index, value in enumerate(tranProMatrix[:, next_node]) if
                                                     value != 0]
                                    neighbors_pro = neighbors_pro / np.sum(neighbors_pro)
                                    next_node = np.random.choice(neighbors, p=neighbors_pro)


                        else:
                            if random_pro < dis_distribution[next_node] + constantFactor_distribution[next_node] * \
                                    succ_distribution[next_node]:
                                break
                            else:
                                neighbors = [index for index, value in enumerate(tranProMatrix[:, next_node]) if value != 0]
                                if len(neighbors) > 0:
                                    neighbors_pro = [value for index, value in enumerate(tranProMatrix[:, next_node]) if
                                                     value != 0]
                                    neighbors_pro = neighbors_pro / np.sum(neighbors_pro)
                                    next_node = np.random.choice(neighbors, p=neighbors_pro)
                                else:
                                    break
                if j == times[count] - 1:
                    if k == 0:
                        avg_usageRate = usageNum / times[count] / seedNum
                        avg_usageRate_list.append(avg_usageRate)
                        last_usageNum = 0
                    else:
                        last_usageNum = lastSeedSet2usageNum[method][count]
                        avg_usageRate = (last_usageNum + usageNum) / times[count] / seedNum
                        avg_usageRate_list.append(avg_usageRate)
                    usageNum_list.append(last_usageNum + usageNum)
                    print('seedNum=' + str(seedNum) + ' ' + method + ' avgUsageRate = ' + str(
                        avg_usageRate) + ' times:' + str(times[count]))
                    count += 1
            with open(usageRate_file, 'a+') as file:
                file.write(f'{method}:{avg_usageRate_list}\n')
                lastSeedSet2usageNum[method] = usageNum_list
                
#在simulation基础上：对于存在丢弃券的用户后续使用券的概率为0
def simulation_firstDiscard(methods, method_deliverers, init_tranProMatrix, usageRate_file, distribution_list, seedNum_list):
    succ_distribution, dis_distribution, tran_distribution, constantFactor_distribution = distribution_list
    m = seedNum_list[-1]
    lastSeedSet2usageNum = {key: 0 for key in methods}
    for k in range(len(seedNum_list)):
        seedNum = seedNum_list[k]
        with open(usageRate_file, 'a+') as file:
            file.write(f'seedNum:{seedNum}\n')
        for i in range(len(methods)):
            if k == 0:
                deliverers = method_deliverers[i][:seedNum]
            else:
                deliverers = method_deliverers[i][seedNum_list[k - 1]:seedNum_list[k]]
            method = methods[i]
            usageNum = 0
            count = 0
            avg_usageRate_list = []
            usageNum_list = []
            for j in range(times[-1]):
                tranProMatrix = copy.deepcopy(init_tranProMatrix)
                users_useAndDis = []
                firstDiscard_list = []
                for index in deliverers:
                    random_pro = np.random.rand()
                    next_node = index
                    if next_node not in users_useAndDis and next_node not in firstDiscard_list:
                        if random_pro < succ_distribution[next_node]:
                            users_useAndDis.append(next_node)
                            tranProMatrix = getCouponUsers.modify_tranProMatrix_singleUser(tranProMatrix, next_node,
                                                                                           constantFactor_distribution[
                                                                                               next_node],
                                                                                           succ_distribution[next_node])
                            usageNum += 1
                            continue
                        elif random_pro < succ_distribution[next_node] + dis_distribution[next_node]:
                            firstDiscard_list.append(next_node)
                            tranProMatrix = getCouponUsers.modify_tranProMatrix_singleUser(tranProMatrix, next_node,
                                                                                           constantFactor_distribution[
                                                                                               next_node],
                                                                                           succ_distribution[next_node])
                            continue
                        else:
                            neighbors = [index for index, value in enumerate(tranProMatrix[:, next_node]) if value != 0]
                            if len(neighbors) > 0:
                                neighbors_pro = [value for index, value in enumerate(tranProMatrix[:, next_node]) if
                                                 value != 0]
                                neighbors_pro = neighbors_pro / np.sum(neighbors_pro)
                                next_node = np.random.choice(neighbors, p=neighbors_pro)
                    else:
                        if random_pro < dis_distribution[next_node] + constantFactor_distribution[next_node] * \
                                succ_distribution[next_node]:
                            firstDiscard_list.append(next_node)
                            continue
                        else:
                            neighbors = [index for index, value in enumerate(tranProMatrix[:, next_node]) if value != 0]
                            if len(neighbors) > 0:
                                neighbors_pro = [value for index, value in enumerate(tranProMatrix[:, next_node]) if
                                                 value != 0]
                                neighbors_pro = neighbors_pro / np.sum(neighbors_pro)
                                next_node = np.random.choice(neighbors, p=neighbors_pro)
                            else:
                                break
                    while (True):
                        random_pro = np.random.rand()
                        if next_node not in users_useAndDis and next_node not in firstDiscard_list:
                            if random_pro < succ_distribution[next_node]:
                                users_useAndDis.append(next_node)
                                tranProMatrix = getCouponUsers.modify_tranProMatrix_singleUser(tranProMatrix, next_node,
                                                                                               constantFactor_distribution[
                                                                                                   next_node],
                                                                                               succ_distribution[next_node])
                                usageNum += 1
                                break
                            elif random_pro < succ_distribution[next_node] + dis_distribution[next_node]:
                                firstDiscard_list.append(next_node)
                                tranProMatrix = getCouponUsers.modify_tranProMatrix_singleUser(tranProMatrix, next_node,
                                                                                               constantFactor_distribution[
                                                                                                   next_node],
                                                                                               succ_distribution[
                                                                                                   next_node])
                                break
                            else:
                                neighbors = [index for index, value in enumerate(tranProMatrix[:, next_node]) if value != 0]
                                if len(neighbors) > 0:
                                    neighbors_pro = [value for index, value in enumerate(tranProMatrix[:, next_node]) if
                                                     value != 0]
                                    neighbors_pro = neighbors_pro / np.sum(neighbors_pro)
                                    next_node = np.random.choice(neighbors, p=neighbors_pro)
                        else:
                            if random_pro < dis_distribution[next_node] + constantFactor_distribution[next_node] * \
                                    succ_distribution[next_node]:
                                if next_node not in firstDiscard_list:
                                   firstDiscard_list.append(next_node)
                                break
                            else:
                                neighbors = [index for index, value in enumerate(tranProMatrix[:, next_node]) if value != 0]
                                if len(neighbors) > 0:
                                    neighbors_pro = [value for index, value in enumerate(tranProMatrix[:, next_node]) if
                                                     value != 0]
                                    neighbors_pro = neighbors_pro / np.sum(neighbors_pro)
                                    next_node = np.random.choice(neighbors, p=neighbors_pro)
                                else:
                                    break
                if j == times[count] - 1:
                    if k == 0:
                        avg_usageRate = usageNum / times[count] / seedNum
                        avg_usageRate_list.append(avg_usageRate)
                        last_usageNum = 0
                    else:
                        last_usageNum = lastSeedSet2usageNum[method][count]
                        avg_usageRate = (last_usageNum + usageNum) / times[count] / seedNum
                        avg_usageRate_list.append(avg_usageRate)
                    usageNum_list.append(last_usageNum + usageNum)
                    print('seedNum=' + str(seedNum) + ' ' + method + ' avgUsageRate = ' + str(
                        avg_usageRate) + ' times:' + str(times[count]))
                    count += 1
            with open(usageRate_file, 'a+') as file:
                file.write(f'{method}:{avg_usageRate_list}\n')
                lastSeedSet2usageNum[method] = usageNum_list

#num组投放源节点，每组之间按相同间隔
#seedNum_percent种子节点个数百分比，1000表示从0.1%取到1%
def get_seedNumList(dataSet,num,seedNum_percent):
    # data_file = 'D:/data-processed/{}-adj.pkl'.format(dataSet)
    data_file = 'D:/data-processed/{}-adj.pkl'.format(dataSet)
    if not os.path.exists(data_file):
        load_edges(dataSet)
    with open(data_file, 'rb') as f:
        adj = pickle.load(f)
    n = adj.shape[0]
    seedNumList = [round(n*i/seedNum_percent) for i in range(1,num+1)]
    return seedNumList

def load_edges(FILE_NAME):
    network = nx.Graph()
    edge_file = open('D:/data-processed/{}.edges'.format(FILE_NAME),"r")
    for line in edge_file:
        split = [x for x in line.split()]
        node_from = int(split[0])
        node_to = int(split[1])
        network.add_edge(node_from, node_to)
    adj = nx.adjacency_matrix(network)
    file = 'D:/data-processed/{}-adj.pkl'.format(FILE_NAME)
    with open(file, "wb") as f:
        pickle.dump(adj, f)

if __name__ == '__main__':
    # dataSets = ['pb','cora','ca-GrQc','soc-anybeat']
    dataSets = ['ca-GrQc']
    for dataSet in dataSets:
        times = [1000,1500]
        # methods = ['succPro','monteCarlo','random','degreeTopM','pageRank']
        methods = ['deliverers_theroy','succPro','degreeTopM','pageRank','1_neighbor']
        # methods = ['DeepIM_IC','DeepIM_LT','OPIM_IC','OPIM_LT']
        # methods = ['deliverers_theroy','succPro']
        seedNum_percent = 1000
        seedNum_list = get_seedNumList(dataSet,10,seedNum_percent)
        couponUsageRate(dataSet,times,methods,seedNum_list,monteCarlo_L = 1000,distribution='random',constantFactorDistri='random',personalization ='None',dataFile_prefix='/大论文/第一个点实验/res_coupon_deliverers_theroy_test_succPro0.2_0.3_constantFactor0.4',method_type=None)
        # couponUsageRate(dataSet,times,methods,seedNum_list,monteCarlo_L = 1000,distribution='random',constantFactorDistri='random',personalization ='None',dataFile_prefix='res_coupon_deliverers_theroy_test_succPro0.2_0.3',method_type=None)