from typing import Optional

import numpy as np
import logging
import single_deliverer
import pickle
import networkx as nx
import random

def monteCarlo_singleTime_improved2(
    tranProMatrix: np.ndarray,
    initial_deliverers: list,
    succ_distribution: np.ndarray,
    dis_distribution: np.ndarray,
    constantFactor_distribution: np.ndarray # 这个参数似乎没用到，保留占位
) -> np.ndarray:

    n = tranProMatrix.shape[0]
    activatedUsers = set()

    for start_user in initial_deliverers: 
        current_user = start_user

        while True:
            # 1. 节点行为判定 (使用 / 丢弃 / 转发)
            rand_action = np.random.rand()
            p_succ = succ_distribution[current_user]
            p_dis = dis_distribution[current_user]
            
            # 行为判定阈值
            if current_user in activatedUsers:
                # 已激活用户再次拿到券：
                # 逻辑可自定义，这里假设他如果"再次使用"就消耗掉券，"丢弃"也消耗
                if rand_action < (p_succ + p_dis):
                    break # 券被消耗或丢弃，停止
            else:
                # 未激活用户
                if rand_action < p_succ:
                    activatedUsers.add(current_user)
                    break # 成功使用，券被消耗，停止
                elif rand_action < (p_succ + p_dis):
                    break # 丢弃，停止

            # --- 决定转发 ---
            # 能走到这里，说明 rand_action >= p_succ + p_dis
            # 此时需要选择下一个邻居。
            
            # 【重要修改】：传入一个新的随机数，用于选择邻居和判定由边导致的失败
            rand_neighbor = np.random.rand()
            
            # 使用新的选择函数，它可能返回 None (表示转发失败/丢失)
            next_node = _select_next_neighbor_lossy(current_user, tranProMatrix, rand_neighbor)

            if next_node is None:
                # 转发失败（比如边权不够，或者是个孤立点），游走结束
                break
            else:
                current_user = next_node

    # 结果转换
    success_vector = np.zeros(n, dtype=int)
    if activatedUsers:
        success_vector[list(activatedUsers)] = 1
        
    return success_vector



def _select_next_neighbor_lossy(
    current_user: int,
    tranProMatrix: np.ndarray,
    rand_val: float
) -> Optional[int]:
    """
    有损的邻居选择：
    1. 不再强制归一化到 1.0。
    2. 如果所有邻居的概率之和 sum < 1.0，则有 (1-sum) 的概率返回 None (丢失)。
    3. 如果 sum > 1.0 (Hub节点)，则截断为 1.0 (必然转发)，并按权重分配。
    """
    neighbors = np.flatnonzero(tranProMatrix[:, current_user])
    if neighbors.size == 0:
        return None

    # 获取原始概率 (Raw Probabilities)
    probabilities = tranProMatrix[neighbors, current_user]
    prob_sum = np.sum(probabilities)

    if prob_sum == 0:
        return None
    
    # 逻辑分叉：
    # 情况 A: 总能量足够大 (Hub)，必然转发，按比例分配
    if prob_sum >= 1.0:
        normalized_probs = probabilities / prob_sum
        # 标准轮盘赌
        cumulative = np.cumsum(normalized_probs)
        idx = np.searchsorted(cumulative, rand_val)
        if idx >= len(neighbors): idx = len(neighbors) - 1
        return neighbors[idx]
    
    # 情况 B: 总能量不足 (Leaf)，可能丢失
    else:
        # 这里不归一化 probabilities，而是直接看 rand_val 落在哪个区间
        # 区间分布: [Neighbor1 (0.1)] [Neighbor2 (0.2)] ... [Void/Loss (1 - 0.3)]
        
        cumulative = np.cumsum(probabilities)
        
        # 如果随机数落在 累积概率 之外，说明掉进了 "丢失区"
        if rand_val >= cumulative[-1]:
            return None
        
        # 否则，选中了某个邻居
        idx = np.searchsorted(cumulative, rand_val)
        return neighbors[idx]


def _select_next_neighbor_old(
    current_user: int,
    tranProMatrix: np.ndarray,
    rand_pro
) -> Optional[int]:
    """
    从当前节点的邻居中，根据转发概率矩阵 和 rand_pro参数 选择邻居
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

    if prob_sum == 0:
        return None
    
    # 归一化概率
    normalized_probs = probabilities / prob_sum
    # 计算累积概率分布
    cumulative_probs = np.cumsum(normalized_probs)
    selected_index = np.searchsorted(cumulative_probs, rand_pro)
    if selected_index >= len(neighbors):
        selected_index = len(neighbors) - 1
    return neighbors[selected_index]


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


def monteCarloSimulation(tranProMatrix,indexes,L,succ_distribution,dis_distribution,constantFactor_distribution,personalization):
    best_next_deliverer = find_next_best_deliverer(current_deliverers=indexes, tranProMatrix=tranProMatrix,L=L,
                             succ_distribution=succ_distribution, dis_distribution=dis_distribution, constantFactor_distribution=constantFactor_distribution,
                             personalization=personalization)
    return best_next_deliverer


# if __name__ == '__main__':
#
#     G = nx.Graph()
#     G.add_nodes_from([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
#     G.add_edges_from([(0, 2), (0, 3), (1, 2), (1, 4), (2, 4), (4,5), (4,6), (4,7),(4,8),(4,9), (5,6),(5,9)])
#     adj = nx.adjacency_matrix(G)
#
#     n = adj.shape[0]
#     use_pro = [0.01,0.2,0.3,0.4,0.3,
#                0.1,0.3,0.5,0.2,0.2]
#     dis_pro = [0.04,0.2,0.2,0.1,0.1,
#                0.4,0.3,0.2,0.1,0.3]
#     L = 5
#     constantFactor = [1,1,1,1,1,
#                       1,1,1,1,1]
#     users = []
#     initial_tran_distribution = np.array([0.001, 0.5, 0.9, 0.2, 0.7, 0.8, 0.5, 0.9, 0.2, 0.7])
#
#     tranProMatrix, neighbor_having = single_deliverer.getTranProMatrix(adj,initial_tran_distribution)
#     bestSingleDeliverer = single_deliverer.getBestSingleDeliverer(tranProMatrix,use_pro,neighbor_having)
#     indexes = [bestSingleDeliverer]
#     best_next_deliverer = monteCarloSimulation(tranProMatrix,indexes,L,
#                                                succ_distribution=use_pro,
#                                                dis_distribution=dis_pro,
#                                                constantFactor_distribution=constantFactor,
#                                                personalization=users)
#     print("best_next_deliverer is: ",best_next_deliverer)