import logging
import numpy as np
import networkx as nx
import random

import single_deliverer
import get_coupon_users_multi_thread

def deliverers_monteCarlo(
    m: int,
    init_tranProMatrix: np.ndarray,
    succ_distribution: np.ndarray,
    dis_distribution: np.ndarray,
    constantFactor_distribution: np.ndarray,
    L: int,
    personalization: dict,
) -> list:
    """
    使用贪心策略和蒙特卡洛模拟，从网络中选择m个最优的优惠券投放者。

    Args:
        m (int): 需要选择的投放者总数。
        init_tranProMatrix (np.ndarray): 初始的转发概率矩阵。
        succ_distribution (np.ndarray): 节点成功接收优惠券的概率分布。
        dis_distribution (np.ndarray): 节点对优惠券不感兴趣的概率分布。
        constantFactor_distribution (np.ndarray): 其他影响因子分布。
        L (int): 蒙特卡洛模拟的迭代次数。
        personalization (dict): 个性化参数，可能用于影响模拟。

    Returns:
        list: 包含m个最优投放者节点ID的列表。
    """
    if m <= 0:
        return []

    # 1. 寻找第一个最优投放者
    best_first_deliverer = single_deliverer.getBestSingleDeliverer(
        init_tranProMatrix, succ_distribution, users_useAndDis=[]
    )
    
    deliverers = [best_first_deliverer]
    print(f"第 1 个投放者选择完毕: {best_first_deliverer}")

    # 2. 迭代寻找剩余的 m-1 个投放者
    for i in range(m - 1):
        # 此处调用一个职责更清晰的函数，它只负责寻找下一个最优节点
        # 它需要知道当前已经选择了哪些节点，以便计算边际增益
        next_best_deliverer = get_coupon_users_multi_thread.find_next_best_deliverer(
            current_deliverers=deliverers,  # 传入当前已选集合
            tranProMatrix=init_tranProMatrix,
            L=L,
            succ_distribution=succ_distribution,
            dis_distribution=dis_distribution,
            constantFactor_distribution=constantFactor_distribution,
            personalization=personalization
        )
        
        if next_best_deliverer is None:
            logging.warning("===> 未能找到下一个有效的投放者，提前终止。")
            break
            
        deliverers.append(next_best_deliverer)
        print(f"第 {i + 2} 个投放者选择完毕: {next_best_deliverer}")

    print(f"最终选择的投放者集合: {deliverers}")
    return deliverers




# 随机选择m张券的投放源节点 随机选 m 个节点
def deliverers_random(n, m):
    return random.choices(range(n),k=m)


# 选邻居最多的 m 个节点
def deliverers_degreeTopM(adj, m):
    degrees = np.asarray(adj.sum(axis=1)).flatten()
    top_m_indexes = np.argsort(degrees)[-m:][::-1]
    return top_m_indexes.tolist() # 根据邻接矩阵计算每个节点的度数，选出度最大的前 m 个节点

#PageRank 值前 m 高	使用图的 PageRank 值选节点
def deliverers_pageRank(adj,m):
    # 构建 networkx 图，从中计算 PageRank 值，然后选前 m 高的节点
    G = nx.from_scipy_sparse_matrix(adj,nx.Graph)
    scores = nx.pagerank(G)
    top_m = sorted(scores,key=scores.get,reverse=True)[:m]
    return top_m


#优惠券使用率最大的m个节点作为投放源节点 使用概率最大前 m 个节点	选择优惠券使用意愿最大的用户
def deliverers_succPro(m,succ_distribution):
    # 根据 succ_distribution（每个用户使用优惠券的概率），直接选出前 m 高
    succPro_indexes = np.argsort(succ_distribution)[::-1][:m]
    return succPro_indexes.tolist()


# 	邻居影响力最大	根据用户能“直接影响”邻居的总使用概率选节点
def deliverers_1_neighbor(succ_distribution,init_tranProMatrix,m):
    # 计算每个节点直接影响其邻居成功使用优惠券的概率总和
    one_neighbor_pros = np.sum(np.multiply(init_tranProMatrix,succ_distribution.reshape(-1,1)),axis=0)
    one_neighbor_pro_indexes = np.argsort(one_neighbor_pros)[::-1][:m]
    return one_neighbor_pro_indexes



def _create_enhanced_tran_matrix(
    base_matrix: np.ndarray,
    succ_dist: np.ndarray,
    const_factor_dist: np.ndarray,
    degrees: np.ndarray
) -> np.ndarray:
    """
    根据理论模型创建一个增强的转发概率矩阵。
    这是一个辅助函数，将复杂的矩阵更新逻辑封装起来。
    """
    n = base_matrix.shape[0]
    
    # 1. 计算转发增量，并处理除零问题
    tran_increment_numerator = succ_dist * (1 - const_factor_dist)
    
    # 初始化增量数组
    tran_increment_per_edge = np.zeros_like(degrees, dtype=float)
    
    # 定位非孤立节点
    non_isolated_nodes = degrees > 0
    
    # 仅为非孤立节点计算增量
    tran_increment_per_edge[non_isolated_nodes] = \
        tran_increment_numerator[non_isolated_nodes] / degrees[non_isolated_nodes]

    # 2. 使用广播机制高效地将增量应用到矩阵上
    # base_matrix 是 0-1 矩阵，乘以增量后，只有存在边的地方才有值
    increment_matrix = base_matrix * tran_increment_per_edge.reshape(1, -1)
    
    # 3. 创建增强矩阵
    enhanced_matrix = base_matrix + increment_matrix
    
    # [重要] 检查并修正概率值，确保它们不会超过1.0
    # 这是对原始逻辑的一个重要修正，防止概率错误。
    np.clip(enhanced_matrix, 0, 1.0, out=enhanced_matrix)
    
    return enhanced_matrix


def select_deliverers_theory_improved(
    m: int,
    base_tran_matrix: np.ndarray,
    succ_distribution: np.ndarray,
    dis_distribution: np.ndarray,
    constantFactor_distribution: np.ndarray,
    D: np.ndarray,
    # dataset 和 personalization 在原始代码中未使用，如果需要请添加
) -> list:
    """
    使用基于理论传播模型的贪心算法，选择m个最优投放者。

    Args:
        m (int): 需要选择的投放者总数。
        base_tran_matrix (np.ndarray): 基础的0-1转发关系矩阵（邻接矩阵）。
        succ_distribution (np.ndarray): 成功接收概率分布。
        dis_distribution (np.ndarray): 不感兴趣概率分布。
        constantFactor_distribution (np.ndarray): 影响因子分布。
        D (np.ndarray): 每个节点的度向量。

    Returns:
        list: 包含m个最优投放者节点ID的列表。
    """
    if m <= 0:
        return []

    n = base_tran_matrix.shape[0]
    
    # 1. 根据模型理论，创建增强后的转发概率矩阵 W
    # 注意：这里的 base_tran_matrix 应该是原始的、未修改的0-1邻接矩阵
    enhanced_tran_matrix = _create_enhanced_tran_matrix(
        base_tran_matrix,
        succ_distribution,
        constantFactor_distribution,
        D.flatten() # 确保D是一维的
    )

    # 2. 初始化影响向量 Q，代表每个用户已被影响的累积概率
    # 使用一维向量更符合NumPy的习惯
    Q_cumulative_influence = np.zeros(n, dtype=np.float64)
    
    deliverers = []
    
    # 3. 迭代 m 次，每次选择一个能带来最大边际增益的投放者
    for i in range(m):
        # `getBestSingleDeliverer_theory` 的职责是：
        # 在 `deliverers` 之外的所有节点中，找到能使 Q 增益最大的那个节点。
        new_deliverer, Q_cumulative_influence = single_deliverer.getBestSingleDeliverer_theory(
            enhanced_tran_matrix,
            succ_distribution, # 可能需要传入，用于计算初始激活
            Q_cumulative_influence,
            current_deliverers=deliverers # 传递已选节点，以避免重复选择
        )
        
        if new_deliverer is None:
            logging.warning("未能找到下一个有效的投放者，提前终止。")
            break
            
        deliverers.append(new_deliverer)
        logging.info(f"第 {i + 1} 个理论最优投放者选择完毕: {new_deliverer}")

    logging.info(f"最终选择的理论投放者集合: {deliverers}")
    return deliverers



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


    deliverers = deliverers_monteCarlo(3, tranProMatrix, succ_distribution=use_pro,
                               dis_distribution=dis_pro,
                               constantFactor_distribution=constantFactor,
                               L=5,
                               personalization=users)