import logging
import numpy as np
import networkx as nx
import random
from collections import defaultdict

import single_deliverer
import get_coupon_users_multi_thread

import scipy.sparse as sp

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


def deliverers_random(n: int, m: int) -> list:
    """
    随机选择 m 个节点。
    """
    print("--- Running: Random ---")
    print(f"依据: 完全随机选择。")
    selected_nodes = random.sample(range(n), k=m)
    print(f"选择了 {m} 个节点: {selected_nodes}\n")
    return selected_nodes # 随机选择没有评价值，只返回节点列表

def deliverers_degreeTopM(adj, m: int) -> list:
    """
    选择度数最高的 m 个节点。
    """
    print("--- Running: Degree Top M ---")
    print("依据: 选择连接数（度数）最多的节点。")
    
    # .A 属性将稀疏矩阵转为密集 NumPy 数组
    degrees = np.asarray(adj.sum(axis=1)).flatten()
    
    # 使用 argsort 获取排序后的索引
    sorted_indexes = np.argsort(degrees)[::-1] # 从大到小排序
    
    top_m_indexes = sorted_indexes[:m]
    
    print("选择的 Top M 节点及其度数:")
    selected_nodes_with_values = []
    for index in top_m_indexes:
        value = degrees[index]
        print(f"  - 节点 {index}: 度数 = {int(value)}")
        selected_nodes_with_values.append((index, value))
    print("") # 打印一个空行
        
    return [node for node, value in selected_nodes_with_values] # 保持原始返回类型

def deliverers_pageRank(adj, m: int) -> list:
    """
    选择 PageRank 分数最高的 m 个节点。
    """
    print("--- Running: PageRank Top M ---")
    print("依据: 选择 PageRank 分数最高的节点（综合考虑数量和质量）。")
    
    # from_scipy_sparse_array 在新版 networkx 中是推荐用法
    # 如果 adj 是对称的，可以用 nx.Graph，如果是有向的，用 nx.DiGraph
    if isinstance(adj, np.ndarray):
        adj = sp.csr_matrix(adj)
    G = nx.from_scipy_sparse_array(adj, create_using=nx.DiGraph)

    # 计算 PageRank
    pagerank_scores = nx.pagerank(G)
    
    # 对字典按值进行排序
    sorted_nodes = sorted(pagerank_scores.items(), key=lambda item: item[1], reverse=True)
    
    top_m_nodes = sorted_nodes[:m]
    
    print("选择的 Top M 节点及其 PageRank 分数:")
    for node, score in top_m_nodes:
        print(f"  - 节点 {node}: PageRank = {score:.4f}")
    print("")
        
    return [node for node, score in top_m_nodes]

def deliverers_succPro(m: int, succ_distribution: np.ndarray) -> list:
    """
    选择自身成功使用概率最高的 m 个节点。
    """
    print("--- Running: Success Probability Top M ---")
    print("依据: 选择自身使用优惠券意愿（成功概率）最强的节点。")
    
    sorted_indexes = np.argsort(succ_distribution)[::-1] # 从大到小排序
    
    top_m_indexes = sorted_indexes[:m]
    
    print("选择的 Top M 节点及其成功概率:")
    selected_nodes_with_values = []
    for index in top_m_indexes:
        value = succ_distribution[index]
        print(f"  - 节点 {index}: 成功概率 = {value:.4f}")
        selected_nodes_with_values.append((index, value))
    print("")
        
    return [node for node, value in selected_nodes_with_values]

def deliverers_1_neighbor(succ_distribution,init_tranProMatrix,m) -> list:
    print("--- Running: 1-Hop Neighbor Influence Top M ---")
    print("依据: 选择其所有邻居的成功使用概率之和最大的节点。")

    # 确保邻接矩阵是 NumPy 数组
    adj = init_tranProMatrix
    adj_array = adj.toarray() if hasattr(adj, 'toarray') else np.asarray(adj)
    
    # 计算每个节点的邻居影响力
    # adj_array.T[i] 是一个向量，表示哪些节点是 i 的邻居
    # succ_distribution 是所有节点的成功概率向量
    # 点积操作高效地计算了每个节点的所有邻居的成功概率之和
    one_hop_influence = adj_array.T.dot(succ_distribution)
    
    sorted_indexes = np.argsort(one_hop_influence)[::-1]
    
    top_m_indexes = sorted_indexes[:m]
    
    print("选择的 Top M 节点及其一步邻居影响力:")
    selected_nodes_with_values = []
    for index in top_m_indexes:
        value = one_hop_influence[index]
        print(f"  - 节点 {index}: 邻居影响力 = {value:.4f}")
        selected_nodes_with_values.append((index, value))
    print("")
        
    return [node for node, value in selected_nodes_with_values]


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


def _generate_single_rr_set(
    n: int, 
    graph: nx.DiGraph
) -> set:

    # 1. 在整个网络中随机选择一个起始节点
    start_node = random.randrange(n)
    
    rr_set = {start_node}
    queue = [start_node]
    
    # 使用 visited 集合来防止在一次生成中重复访问同一个节点
    visited = {start_node}

    head = 0
    while head < len(queue):
        current_node = queue[head]
        head += 1
        
        # 2. 遍历所有指向 current_node 的“入边”
        for predecessor in graph.predecessors(current_node):
            if predecessor not in visited:
                visited.add(predecessor)
                
                # 3. 以传播概率 p 进行“反向穿越”
                edge_prob = graph.edges[predecessor, current_node].get('p', 0.1) # 默认概率0.1
                if random.random() < edge_prob:
                    # 如果成功，则将前驱节点加入样本和队列
                    rr_set.add(predecessor)
                    queue.append(predecessor)
                    
    return rr_set


def _generate_rr_sets(
    n: int, 
    graph: nx.DiGraph, 
    num_RR: int
) -> list:
    print(f"--- generating numbers of {num_RR} (RR-Sets) ---")
    return [_generate_single_rr_set(n, graph) for _ in range(num_RR)]


def deliverers_ris_coverage(
    adj, # 原始邻接矩阵，用于确定图的结构
    tranProMatrix, # 包含边特定概率的转移矩阵
    m: int,
    num_samples: int = 50000 
) -> list:
    print("--- Running: Reverse Reachable Set (RIS) Coverage ---")

    if isinstance(adj, np.ndarray):
        adj = sp.csr_matrix(adj)

    n = adj.shape[0]
    
    G = nx.from_scipy_sparse_array(adj, create_using=nx.DiGraph)
    
    num_edges_processed = 0
    for u, v in G.edges():
        probability = tranProMatrix[v, u]
        
        G.edges[u, v]['p'] = probability
        num_edges_processed += 1
        
    print(f"图构建完成，共处理了 {num_edges_processed} 条边。")
    
    rr_sets = _generate_rr_sets(n, G, num_samples)
    
    selected_seeds = []
    
    # 创建一个从节点到其所在RR-Set索引的映射
    node_to_rr_indices = defaultdict(list)
    for i, rr_set in enumerate(rr_sets):
        for node in rr_set:
            node_to_rr_indices[node].append(i)

    is_rr_set_covered = np.zeros(num_samples, dtype=bool)# 计算边际增益
    
    for i in range(m):
        candidate_gains = defaultdict(int)
        for node_id, rr_indices in node_to_rr_indices.items():
            if node_id in selected_seeds:
                continue
            
            gain = np.sum(~is_rr_set_covered[rr_indices])# 计算这个节点覆盖的RR-Set数量
            candidate_gains[node_id] = gain

        if not candidate_gains:
            print("no more uncovered nodes, process finish!")
            break

        best_candidate = max(candidate_gains, key=candidate_gains.get)
        selected_seeds.append(best_candidate)
        
        for rr_index in node_to_rr_indices[best_candidate]:
            is_rr_set_covered[rr_index] = True
            
        print(f"  - 第 {i+1} 个种子: 节点 {best_candidate} (新增覆盖了 {candidate_gains[best_candidate]} 个 RR-Set)")

    print(f"\n种子集合: {selected_seeds}\n")
    return selected_seeds

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


    deliverers1 = deliverers_monteCarlo(3, tranProMatrix, succ_distribution=use_pro,
                               dis_distribution=dis_pro,
                               constantFactor_distribution=constantFactor,
                               L=5,
                               personalization=users)
    print(deliverers1)

    deliverers = deliverers_ris_coverage(adj=adj,
                                         m=3,
                                         tranProMatrix=tranProMatrix
                                         )
    print(deliverers)