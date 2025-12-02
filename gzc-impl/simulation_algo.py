from typing import Optional

import numpy as np
import logging
import single_deliverer
import pickle
import networkx as nx
import random

def find_next_best_deliverer(
    current_deliverers: list,
    tranProMatrix: np.ndarray,
    L: int,
    succ_distribution: np.ndarray,
    dis_distribution: np.ndarray,
    constantFactor_distribution: np.ndarray,
    personalization: str
) -> int:

    n = tranProMatrix.shape[0]
    candidate_nodes = [node for node in range(n) if node not in current_deliverers]

    if not candidate_nodes:
        logging.warning("===> 没有候选节点了，无法选择。")
        return None

    # 1. 计算当前集合的基础影响力
    base_influence = _run_full_simulation(
        L, current_deliverers, tranProMatrix, succ_distribution, 
        dis_distribution, constantFactor_distribution, personalization
    )
    print(f"当前投放者 {current_deliverers} 的基础影响力: {base_influence:.4f}")

    best_next_deliverers = []
    max_marginal_gain = -1.0

    # 2. 遍历所有候选节点，计算每个节点的边际增益
    for candidate in candidate_nodes:
        # 构造临时投放集合进行测试
        test_deliverer_set = current_deliverers + [candidate]
        
        # 计算加入候选节点后的新影响力
        new_influence = _run_full_simulation(
            L, test_deliverer_set, tranProMatrix, succ_distribution,
            dis_distribution, constantFactor_distribution, personalization
        )
        
        marginal_gain = new_influence - base_influence
        
        print(f"  测试候选节点 {candidate}: 新影响力={new_influence:.4f}, 边际增益={marginal_gain:.4f}")

        if marginal_gain >= max_marginal_gain:
            max_marginal_gain = marginal_gain
            best_next_deliverers.append(candidate)

    best_next_deliverer = random.choice(best_next_deliverers)
    print(f"选择的最优新投放者: {best_next_deliverer} (最大边际增益: {max_marginal_gain:.4f})")
    
    return best_next_deliverer


def _run_full_simulation(
    L: int,
    deliverer_set: list,
    tranProMatrix: np.ndarray,
    succ_distribution: np.ndarray,
    dis_distribution: np.ndarray,
    constantFactor_distribution: np.ndarray,
    personalization: str
) -> float:

    n = tranProMatrix.shape[0]
    total_influence_accumulator = 0.0

    # 根据个性化策略选择正确的模拟函数
    if personalization == 'firstUnused':
        single_simulation_func = monteCarlo_singleTime_improved2
    elif personalization == 'firstDiscard':
        single_simulation_func = monteCarlo_singleTime_improved2
    else: # 默认或None
        single_simulation_func = monteCarlo_singleTime_improved2

    for _ in range(L):
        # 它应该返回一个(1, n)或(n,)的数组，代表此轮模拟中各节点的成功状态(0或1)
        success_vector = single_simulation_func(
            tranProMatrix,
            deliverer_set,
            succ_distribution,
            dis_distribution,
            constantFactor_distribution
        )
        # 累加本轮模拟的总成功人数
        total_influence_accumulator += np.sum(success_vector)

    # 返回平均总影响力
    return total_influence_accumulator / L

# AgainReJudge
def monteCarlo_singleTime_improved2(
    tranProMatrix: np.ndarray,
    initial_deliverers: list,
    succ_distribution: np.ndarray,
    dis_distribution: np.ndarray,
    constantFactor_distribution: np.ndarray
) -> np.ndarray:

    n = tranProMatrix.shape[0]
    activatedUsers = set()
    activated_list = []

    for start_user in initial_deliverers: # 为每个初始投放者启动一个独立的随机游走

        current_user = start_user

        # 模拟单张优惠券的随机游走过程
        while True:
            rand_pro = np.random.rand()
            # 检查当前节点是否已经做出过决定
            if current_user in activatedUsers:
                # 做出过决定 再次接触优惠券的逻辑
                if rand_pro < succ_distribution[current_user]:
                    # 继续用
                    break
                elif rand_pro < (succ_distribution[current_user] + dis_distribution[current_user]):
                    # 继续丢弃
                    break
                else:
                    continue

            else:
                # 首次接触优惠券的逻辑
                if rand_pro < succ_distribution[current_user]:
                    # 决定“使用”
                    activatedUsers.add(current_user)
                    activated_list.append(current_user)
                    # 游走在此中断，因为优惠券被使用了
                    break
                elif rand_pro < (succ_distribution[current_user] + dis_distribution[current_user]):
                    # 决定“丢弃”
                    break # 游走在此中断

            # 如果没有中断，则意味着节点决定“转发”
            next_node = _select_next_neighbor(current_user, tranProMatrix)

            if next_node is None:
                # 没有邻居可转发，游走中断
                break
            else:
                # 更新当前节点，继续游走
                current_user = next_node

    # 将最终成功使用的节点集合转换为0/1向量
    success_vector = np.zeros(n, dtype=int)
    success_vector[activated_list] = 1

    return success_vector

def monteCarlo_singleTime_improved2_AgainContinue(
    tranProMatrix: np.ndarray,
    initial_deliverers: list,
    succ_distribution: np.ndarray,
    dis_distribution: np.ndarray,
    constantFactor_distribution: np.ndarray
) -> np.ndarray:
    '''
    简单实现 纯全局变量实现
    '''

    n = tranProMatrix.shape[0]
    activatedUsers = set()
    activated_list = []

    # 为每个初始投放者启动一个独立的随机游走
    for start_user in initial_deliverers:
        # logging.info(f"\t\t\t当前模拟的起始节点: {start_user}")
        current_user = start_user

        # 模拟单张优惠券的随机游走过程
        while True:
            rand_pro = np.random.rand()

            # 检查当前节点是否已经做出过决定
            if current_user not in activatedUsers:
                # 首次接触优惠券的逻辑
                if rand_pro < succ_distribution[current_user]:
                    # 决定“使用”
                    activatedUsers.add(current_user)
                    activated_list.append(current_user)
                    # 游走在此中断，因为优惠券被使用了
                    break
                elif rand_pro < (succ_distribution[current_user] + dis_distribution[current_user]):
                    # 决定“丢弃”
                    break # 游走在此中断

            # 做出过决定 再次接触优惠券的逻辑 直接转发 || 如果没有中断，则意味着节点决定“转发”
            next_node = _select_next_neighbor(current_user, tranProMatrix)

            if next_node is None:
                # 没有邻居可转发，游走中断
                break
            else:
                # 更新当前节点，继续游走
                current_user = next_node


    # 将最终成功使用的节点集合转换为0/1向量
    success_vector = np.zeros(n, dtype=int)

    success_vector[activated_list] = 1

    return success_vector


def _select_next_neighbor(
        current_user: int,
        tranProMatrix: np.ndarray
) -> Optional[int]:
    """
    Args:
        current_user: 当前节点 u
        tranProMatrix: M[u, v] 表示 u -> v 的概率
    """
    # 修正：现在我们应该看第 current_user 行 (u -> ?)
    # 之前的代码看的是列 [:, current_user]，那是错误的

    # 获取 current_user 这一行所有非零元素的列索引
    neighbors = np.flatnonzero(tranProMatrix[current_user, :])

    if neighbors.size == 0:
        return None

    # 获取对应的概率
    probabilities = tranProMatrix[current_user, neighbors]
    prob_sum = np.sum(probabilities)

    if prob_sum <= 0:
        return np.random.choice(neighbors)

    normalized_probs = probabilities / prob_sum
    return np.random.choice(neighbors, p=normalized_probs)

def _select_next_neighbor_old(
    current_user: int,
    tranProMatrix: np.ndarray
) -> Optional[int]:
    """
    从当前节点的邻居中，根据转发概率矩阵选择下一个节点。
    Args:
        current_user: 当前节点编号
        tranProMatrix: 转移概率矩阵 (n x n)，tranProMatrix[i, j] 表示 j -> i 的转发概率
    Returns:
        邻居节点编号
    """
    # 找到邻居及其对应的转发概率
    neighbors = np.flatnonzero(tranProMatrix[:, current_user])
    if neighbors.size == 0:
        return None

    probabilities = tranProMatrix[neighbors, current_user]
    prob_sum = np.sum(probabilities)
    
    if prob_sum <= 0:
        # 如果概率和为0（或负数，异常情况），则均匀选择一个邻居
        return np.random.choice(neighbors)
    
    # 归一化概率并选择
    normalized_probs = probabilities / prob_sum
    return np.random.choice(neighbors, p=normalized_probs)


def monteCarloSimulation(tranProMatrix,indexes,L,succ_distribution,dis_distribution,constantFactor_distribution,personalization):
    best_next_deliverer = find_next_best_deliverer(current_deliverers=indexes, tranProMatrix=tranProMatrix,L=L,
                             succ_distribution=succ_distribution, dis_distribution=dis_distribution, constantFactor_distribution=constantFactor_distribution,
                             personalization=personalization)
    return best_next_deliverer


if __name__ == '__main__':

    G = nx.Graph()
    G.add_nodes_from([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]) 
    G.add_edges_from([(0, 2), (0, 3), (1, 2), (1, 4), (2, 4), (4,5), (4,6), (4,7),(4,8),(4,9), (5,6),(5,9)])
    adj = nx.adjacency_matrix(G)

    n = adj.shape[0]
    use_pro = [0.01,0.2,0.3,0.4,0.3,
               0.1,0.3,0.5,0.2,0.2]
    dis_pro = [0.04,0.2,0.2,0.1,0.1,
               0.4,0.3,0.2,0.1,0.3]
    L = 5
    constantFactor = [1,1,1,1,1,
                      1,1,1,1,1]
    users = []
    initial_tran_distribution = np.array([0.001, 0.5, 0.9, 0.2, 0.7, 0.8, 0.5, 0.9, 0.2, 0.7])

    tranProMatrix, neighbor_having = single_deliverer.getTranProMatrix(adj,initial_tran_distribution)
    bestSingleDeliverer = single_deliverer.getBestSingleDeliverer(tranProMatrix,use_pro,neighbor_having)
    indexes = [bestSingleDeliverer]
    best_next_deliverer = monteCarloSimulation(tranProMatrix,indexes,L,
                                               succ_distribution=use_pro,
                                               dis_distribution=dis_pro,
                                               constantFactor_distribution=constantFactor,
                                               personalization=users)
    print("best_next_deliverer is: ",best_next_deliverer)