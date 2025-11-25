from typing import List, Dict
import numpy as np
import scipy.sparse as sp
import random
import logging

def deliverers_alpha_sort(
        adj: sp.csr_matrix,
        tranProMatrix: np.ndarray,
        seeds_num: int,
        alpha: Dict[int, float]
) -> list:
    """
    忽略图的结构，选择alpha值最高的k个节点作为种子

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
    print(f"{len(importance_items)} 个节点的Importance值降序排序 done")

    # 前k个种子节点
    selected_seeds = [item[0] for item in importance_items[:seeds_num]]
    logging.info(f"\n选择的种子集 (按Importance排序): {selected_seeds}\n")

    return selected_seeds


if __name__ == "__main__":
    NUM_NODES = 50
    SEEDS_TO_SELECT = 5

    # adj 和 tranProMatrix 虽然在此函数中不被使用，但为了接口兼容性，我们仍然创建它们
    adj_matrix = sp.csr_matrix((NUM_NODES, NUM_NODES))
    trans_prob_matrix = np.zeros((NUM_NODES, NUM_NODES))

    custom_alpha_probs = {node: random.uniform(0.05, 0.3) for node in range(NUM_NODES)}
    custom_alpha_probs[10] = 0.95  # 节点10的alpha最高
    custom_alpha_probs[32] = 0.88  # 节点32的alpha次高
    custom_alpha_probs[5] = 0.85
    custom_alpha_probs[41] = 0.79
    custom_alpha_probs[22] = 0.75

    top_alpha_seeds = deliverers_alpha_sort(
        adj=adj_matrix,
        tranProMatrix=trans_prob_matrix,
        seeds_num=SEEDS_TO_SELECT,
        alpha=custom_alpha_probs
    )
