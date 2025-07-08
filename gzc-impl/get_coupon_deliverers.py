import logging
import numpy as np
import networkx as nx

import single_deliverer
import get_coupon_users

def deliverers_monteCarlo(
    m: int,
    init_tranProMatrix: np.ndarray,
    succ_distribution: np.ndarray,
    dis_distribution: np.ndarray,
    constantFactor_distribution: np.ndarray,
    L: int,
    personalization: dict,
    # dataset 似乎在你的原始代码中未使用，如果需要请添加
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
        next_best_deliverer = get_coupon_users.find_next_best_deliverer(
            current_deliverers=deliverers,  # 传入当前已选集合
            tranProMatrix=init_tranProMatrix,
            L=L,
            succ_distribution=succ_distribution,
            dis_distribution=dis_distribution,
            constantFactor_distribution=constantFactor_distribution,
            personalization=personalization
        )
        
        if next_best_deliverer is None:
            logging.warning("未能找到下一个有效的投放者，提前终止。")
            break
            
        deliverers.append(next_best_deliverer)
        print(f"第 {i + 2} 个投放者选择完毕: {next_best_deliverer}")

    print(f"最终选择的投放者集合: {deliverers}")
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