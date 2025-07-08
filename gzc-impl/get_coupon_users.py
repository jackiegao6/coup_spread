import numpy as np
import logging
import single_deliverer
import pickle
import networkx as nx
import random


def monteCarloSimulation(tranProMatrix,indexes,L,succ_distribution,dis_distribution,constantFactor_distribution,personalization):
    best_next_deliverer = find_next_best_deliverer(current_deliverers=indexes, tranProMatrix=tranProMatrix,L=L,
                             succ_distribution=succ_distribution, dis_distribution=dis_distribution, constantFactor_distribution=constantFactor_distribution,
                             personalization=personalization)
    return best_next_deliverer

def monteCarlo_singleTime_improved(
    tranProMatrix: np.ndarray,
    initial_deliverers: list,
    succ_distribution: np.ndarray,
    dis_distribution: np.ndarray,
    constantFactor_distribution: np.ndarray
) -> np.ndarray:
    """
    执行单次蒙特卡洛模拟，模拟从初始投放者开始的多路随机游走。
    此函数是“纯”的，不会修改任何输入参数。

    Args:
        tranProMatrix (np.ndarray): 描述转发概率的矩阵 (只读)。
        initial_deliverers (list): 初始投放者（种子节点）的列表。
        ... (其他概率分布)

    Returns:
        np.ndarray: 一个n维的0/1向量，1表示该节点在本轮模拟中成功使用了优惠券。
    """
    n = tranProMatrix.shape[0]
    users_useAndDis = set()# 使用集合(set)来高效存储已做出决定的节点，查询时间复杂度为O(1)
    
    # 为每个初始投放者启动一个独立的随机游走
    for start_user in initial_deliverers:
        current_user = start_user
        
        # 模拟单张优惠券的随机游走过程
        while True:
            rand_pro = np.random.rand()
            # 检查当前节点是否已经做出过决定
            if current_user in users_useAndDis:
                # 做出过决定 再次接触优惠券的逻辑
                # 免疫/饱和效应：节点再次收到后更容易使用或丢弃
                if rand_pro < (dis_distribution[current_user] + 
                              constantFactor_distribution[current_user] * succ_distribution[current_user]):
                    break # 游走在此中断
            else:
                # 首次接触优惠券的逻辑
                if rand_pro < succ_distribution[current_user]:
                    # 决定“使用”
                    users_useAndDis.add(current_user)
                    # 游走在此中断，因为优惠券被使用了
                    break 
                elif rand_pro < (succ_distribution[current_user] + dis_distribution[current_user]):
                    # 决定“丢弃”
                    # todo gzc: 丢弃也算成功使用 吸收态
                    users_useAndDis.add(current_user) # 即使丢弃，也算“处理过”
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
    if users_useAndDis: # 如果集合不为空
        activated_list = list(users_useAndDis)
        success_vector[activated_list] = 1
        
    return success_vector


def _select_next_neighbor(current_user: int, tran_matrix: np.ndarray) -> int or None:
    """
    从当前节点的邻居中，根据转发概率矩阵选择下一个节点。
    """
    # 找到邻居及其对应的转发概率
    neighbors = np.nonzero(tran_matrix[:, current_user])[0]
    
    if len(neighbors) == 0:
        return None  # 没有邻居可供转发
        
    probabilities = tran_matrix[neighbors, current_user]
    prob_sum = np.sum(probabilities)
    
    if prob_sum <= 0:
        # 如果概率和为0（或负数，异常情况），则均匀选择一个邻居
        return np.random.choice(neighbors)
    
    # 归一化概率并选择
    normalized_probs = probabilities / prob_sum
    return np.random.choice(neighbors, p=normalized_probs)

def find_next_best_deliverer(
    current_deliverers: list,
    tranProMatrix: np.ndarray,
    L: int,
    succ_distribution: np.ndarray,
    dis_distribution: np.ndarray,
    constantFactor_distribution: np.ndarray,
    personalization: str
) -> int:
    """
    通过蒙特卡洛模拟计算边际增益，找到下一个最优的投放者。
    这个函数取代了你原始的 monteCarloSimulation。

    Args:
        current_deliverers (list): 当前已经选定的投放者集合。
        ... (其他参数与之前相同)

    Returns:
        int: 下一个最优投放者的节点ID。
    """
    n = tranProMatrix.shape[0]
    candidate_nodes = [node for node in range(n) if node not in current_deliverers]

    if not candidate_nodes:
        logging.warning("没有候选节点了，无法选择。")
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
    """
    对给定的投放者集合，运行L次蒙特卡洛模拟，并返回网络中的总平均影响力。
    总影响力 = 平均每个节点成功使用的概率之和。
    这是一个辅助函数，封装了完整的L轮模拟。
    """
    n = tranProMatrix.shape[0]
    total_influence_accumulator = 0.0

    # 根据个性化策略选择正确的模拟函数
    # 这样做可以避免在循环内部重复进行if/elif判断
    if personalization == 'firstUnused':
        # single_simulation_func = monteCarlo_singleTime_firstUnused
        single_simulation_func = monteCarlo_singleTime_improved
    elif personalization == 'firstDiscard':
        # single_simulation_func = monteCarlo_singleTime_firstDiscard
        single_simulation_func = monteCarlo_singleTime_improved
    else: # 默认或None
        single_simulation_func = monteCarlo_singleTime_improved

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