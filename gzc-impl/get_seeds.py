import logging
import numpy as np
import networkx as nx
import random
import scipy.sparse as sp
from typing import Dict, List, Any

def deliverers_monteCarlo(
        n: int, 
        m: int, 
        tranProMatrix: np.ndarray, 
        succ_distribution: np.ndarray, 
        dis_distribution: np.ndarray, 
        constantFactor_distribution: np.ndarray,
        simulation_algo_func, # 传入你的模拟函数，如 AgainContinue
        L: int = 100          # 每个节点的模拟次数 默认100次
) -> list:
    """
    蒙特卡洛模拟策略：通过模拟 L 次随机游走，计算每个节点作为种子的平均激活能力。
    
    Args:
        n: 节点总数
        m: 需要选出的种子数
        L: 每个节点的模拟次数（L越高越准，但越慢）
        simulation_algo_func: 之前定义的单次模拟逻辑函数
    """
    logging.info(f"--- Running: Monte Carlo Selection (L={L}) ---")
    logging.info(f"依据: 对每个节点进行 {L} 次模拟实验，评估其平均激活贡献。")
    
    node_scores = np.zeros(n)
    
    # 为了加快速度，我们可以只评估一部分有潜力的节点（可选）
    # 这里我们演示全量评估
    for i in range(n):
        if i % 1000 == 0:
            logging.info(f"正在模拟评估第 {i}/{n} 个节点...")
            
        success_count = 0
        for _ in range(L):
            # 这里的 initial_deliverers 只传入当前这一个节点 [i]
            # 返回的是一个 0/1 向量
            res_vector = simulation_algo_func(
                tranProMatrix, 
                [i], 
                succ_distribution, 
                dis_distribution, 
                constantFactor_distribution
            )
            success_count += np.sum(res_vector)
            
        node_scores[i] = success_count / L

    # 排序获取 Top M
    sorted_indexes = np.argsort(node_scores)[::-1]
    selected_seeds = sorted_indexes[:m].tolist()
    
    logging.info(f"蒙特卡洛评估完成。最高期望激活值为: {node_scores[selected_seeds[0]]:.4f}")
    logging.info(f"选择的种子集: {selected_seeds[:10]} ...")
    
    return [int(node) for node in selected_seeds]

# 随机策略
def deliverers_random(n: int, m: int) -> list:
    logging.info(f"数据集节点总数:  n = {n}, m = {m}")
    if m > n:
        raise ValueError(f"m={m} 不能大于 n={n}")
    selected_nodes = random.sample(range(n), k=m)
    logging.info(f"选择 {m}  {selected_nodes}\n")
    return selected_nodes


# 度中心性策略 认为认识人越多的人影响力越大
def deliverers_degreeTopM(adj, m: int) -> list:
    """
    选择度数最高的 m 个节点。
    """
    logging.info("依据: 选择连接数（度数）最多的节点。")
    
    # .A 属性将稀疏矩阵转为密集 NumPy 数组
    degrees = np.asarray(adj.sum(axis=1)).flatten()
    
    # 使用 argsort 获取排序后的索引
    sorted_indexes = np.argsort(degrees)[::-1] # 从大到小排序
    
    top_m_indexes = sorted_indexes[:m]
    
    logging.info("选择的 Top M 节点及其度数:")
    selected_nodes_with_values = []
    for index in top_m_indexes:
        value = degrees[index]
        selected_nodes_with_values.append((index, value))
    logging.info("") # 打印一个空行
        
    return [node for node, value in selected_nodes_with_values] # 保持原始返回类型

# PageRank 策略
def deliverers_pageRank2(adj, m: int, tranProMatrix) -> list:
    """
    依据实际转移概率 (tranProMatrix) 来选 PageRank Top-m，
    保证选出的节点在 tranProMatrix 中有出概率 (out_prob > 0)。
    """
    logging.info("--- Running: PageRank (on tranProMatrix) Top M ---")


    # 确保 tranProMatrix 是 (n x n)，tranProMatrix[i,j] 表示 j -> i 的概率
    # networkx 需要 A[u,v] 表示 u->v 的权重，因此传入它的是 tranProMatrix.T
    if sp.issparse(tranProMatrix):
        B = tranProMatrix.T.tocsr()
        G = nx.from_scipy_sparse_array(B, create_using=nx.DiGraph)
    else:
        B = np.asarray(tranProMatrix).T
        G = nx.from_numpy_array(B, create_using=nx.DiGraph)

    # 计算 PageRank
    pagerank_scores = nx.pagerank(G, weight='weight')

    # 计算每个节点在 tranProMatrix 中的出概率（列和）
    if sp.issparse(tranProMatrix):
        out_prob = np.array(tranProMatrix.sum(axis=0)).ravel()
    else:
        out_prob = np.asarray(tranProMatrix).sum(axis=0)


    # 在有出概率的节点中按 pagerank 排序并取前 m
    ranked = sorted(pagerank_scores.items(), key=lambda x: x[1], reverse=True)
    selected = []
    for node, score in ranked:
        if out_prob[node] > 0:
            selected.append(node)
            if len(selected) >= m:
                break

    # 如果不足 m，补上出度最多的节点（作为回退）
    if len(selected) < m:
        logging.warning("PageRank-selected nodes with out_prob>0 fewer than m, fallback to out-prob sorting")
        candidates = np.argsort(-out_prob)  # indices sorted by descending out_prob
        for node in candidates:
            if node not in selected and out_prob[node] > 0:
                selected.append(int(node))
                if len(selected) >= m:
                    break

    logging.info("Selected PageRank seeds: %s", selected)
    return selected

# 忽略图的结构，选择alpha值最高的k个节点作为种子
def deliverers_alpha_sort(
        adj: sp.csr_matrix,
        tranProMatrix: np.ndarray,
        seeds_num: int,
        alpha: Dict[int, float]
) -> list:
    """
    Args:
        adj (sp.csr_matrix): 邻接矩阵 未使用
        tranProMatrix (np.ndarray): 转移概率矩阵 未使用。
        seeds_num (int): 要选择的种子数量 (k)。
        alpha (Dict[int, float]): 包含每个节点领券概率的字典。

    Returns:
        list: 包含k个最优种子节点ID的列表，按alpha值从高到低排序。
    """

    num_nodes = adj.shape[0]
    if seeds_num > num_nodes:
        logging.warning(f"种子数 ({seeds_num}) 大于图中节点数 ({num_nodes})")
        seeds_num = num_nodes

    # 1. 将alpha字典转换为 (节点ID, alpha值) 的元组列表 便于排序
    alpha_items = list(alpha.items())

    # 2. 降序排序 按照元组的第二个元素（即alpha值）进行排序
    alpha_items.sort(key=lambda item: item[1], reverse=True)
    logging.info(f"{len(alpha_items)} 个节点的alpha值降序排序 done")
    logging.info(f"==============>> {alpha_items[:10]}")

    # 3. 提取排序后前k个元组
    selected_items = alpha_items[:seeds_num]
    selected_seeds = [item[0] for item in selected_items]

    logging.info(f"\n选择的种子集 (按alpha排序): {selected_seeds}\n")
    return selected_seeds


# 重要性混合策略
def deliverers_importance_sort(
        adj: sp.csr_matrix,
        tranProMatrix: np.ndarray,
        seeds_num: int,
        alpha: Dict[int, float],
        trans: Dict[int, float],
) -> list:
    """
    Importance_i = alpha_i + p_i * (sum of alpha_j for all relevant neighbors j)

    Args:
        adj (sp.csr_matrix): 邻接矩阵。
        tranProMatrix (np.ndarray): 转移概率矩阵, tranProMatrix[v, u] = p(u->v)。
        seeds_num (int): 要选择的种子数量 (k)。
        alpha (Dict[int, float]): 包含每个节点领券概率的字典。

    Returns:
        list: 包含k个最优种子节点ID的列表，按Importance值从高到低排序。
    """

    num_nodes = adj.shape[0]
    if seeds_num > num_nodes:
        logging.warning(f"种子数 ({seeds_num}) 大于图中节点数 ({num_nodes})")
        seeds_num = num_nodes

    # 计算所有节点的Importance值
    node_importance = {}

    for cur_node in range(num_nodes):
        neighbor_alphas_sum = 0.0 # 初始化
        trans_value = 0.0 # 初始化

        # 找到所有 i 指向的邻居 j
        # 找到第i行的非零行索引 (col_indices, row_indices)
        out_neighbors = adj.getcol(cur_node).nonzero()[1]
        for out in out_neighbors:
            neighbor_alphas_sum += alpha.get(out, 0.0)
            trans_value = tranProMatrix[out, cur_node]

        # 计算最终的 Importance 值
        importance = alpha.get(cur_node, 0.0) + trans_value * neighbor_alphas_sum
        node_importance[cur_node] = importance

    # 排序
    importance_items = list(node_importance.items())
    importance_items.sort(key=lambda item: item[1], reverse=True)
    logging.info(f"{len(importance_items)} 个节点的Importance值降序排序 done")

    # 前k个种子节点
    selected_seeds = [item[0] for item in importance_items[:seeds_num]]
    logging.info(f"\n选择的种子集 (按Importance排序): {selected_seeds}\n")

    return selected_seeds


# 一阶邻居影响力策略
def deliverers_1_neighbor(succ_distribution,init_tranProMatrix,m) -> list:
    logging.info("--- Running: 1-Hop Neighbor Influence Top M ---")
    logging.info("依据: 选择其所有邻居的成功使用概率之和最大的节点。")

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
    
    logging.info("选择的 Top M 节点及其一步邻居影响力:")
    selected_nodes_with_values = []
    for index in top_m_indexes:
        value = one_hop_influence[index]
        logging.info(f"  - 节点 {index}: 邻居影响力 = {value:.4f}")
        selected_nodes_with_values.append((index, value))
    logging.info("")
        
    return [node for node, value in selected_nodes_with_values]


def deliverers_monteCarlo_spread_aware(
        n: int, 
        m: int, 
        tranProMatrix: np.ndarray, 
        succ_distribution: np.ndarray, 
        dis_distribution: np.ndarray, 
        constantFactor_distribution: np.ndarray,
        simulation_algo_func, # 传入你的模拟函数，如 AgainContinue
        L: int = 100          # 每个节点的模拟次数 默认100次
) -> List[int]:
    """
    基于传播潜力的蒙特卡洛策略 (Propagation-Focused MC)
    
    核心逻辑：使用 AgainContinue 模拟，计算每个种子能产生的【平均触达覆盖数】。
    """
    logging.info(f"--- Running: Monte Carlo Spread-Aware (L={L}) ---")
    
    # 存储每个节点的传播得分
    spread_scores = np.zeros(n)

    # 预先处理邻居索引，加速模拟过程
    # 注意：如果 n 很大，可以只对度数 > 0 的节点进行评估
    for seed_candidate in range(n):
        if seed_candidate % 1000 == 0:
            logging.info(f"正在评估节点传播能力: {seed_candidate}/{n}...")

        total_reach_count = 0
        
        for _ in range(L):
            # --- 模拟单次随机游走 (AgainContinue 逻辑) ---
            current_user = seed_candidate
            visited_nodes = {current_user} # 记录这张券走过的所有人
            
            while True:
                rand_pro = np.random.rand()
                p_succ = succ_distribution[current_user]
                p_dis = dis_distribution[current_user]
                
                # 判定：是【使用】还是【丢弃】
                # 在传播潜力评估中，我们关注它在“终止”前能走多远
                if rand_pro < (p_succ + p_dis):
                    # 券在这里被核销或扔掉，游走终止
                    break
                
                # 否则判定为【转发】
                # 寻找下一个邻居
                neighbors = np.flatnonzero(tranProMatrix[:, current_user])
                if neighbors.size == 0:
                    break # 死胡同
                
                # 简单实现：按转发概率选下一个
                probs = tranProMatrix[neighbors, current_user]
                probs /= np.sum(probs)
                current_user = np.random.choice(neighbors, p=probs)
                
                visited_nodes.add(current_user)
                
                # 防止由于逻辑问题导致无限循环（TTL 保护）
                if len(visited_nodes) > 1000: 
                    break
            
            total_reach_count += len(visited_nodes)
        
        # 该节点的得分 = 平均每张券能让多少人“看到”
        spread_scores[seed_candidate] = total_reach_count / L

    # 按传播得分降序排列
    sorted_indexes = np.argsort(spread_scores)[::-1]
    selected_seeds = sorted_indexes[:m].tolist()
    
    logging.info(f"评估完成。最高触达得分: {spread_scores[selected_seeds[0]]:.2f}")
    return [int(node) for node in selected_seeds]