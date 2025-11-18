import random
import time
from multiprocessing import Pool, cpu_count
from collections import defaultdict
from typing import List, Dict, Set, Tuple, Any
import numpy as np
import scipy.sparse as sp
import logging
from collections import deque
from typing import List, Set, Tuple

# 一次完整的SSR抽样过程
def run_single_ssr_generation3(
        args: Tuple,
        max_path_length: int = 100
) -> List[Set[int]]:
    """
    根据新要求，通过生成独立的反向路径来构建RR-set。
    **此版本经过内存优化，只存储无法再扩展的“叶子”路径。**

    Args:
        args: 一个元组，包含 (nodes, in_neighbors_array, alpha, k)
        max_path_length (int): 为防止无限循环，限制单条路径的最大深度。

    Returns:
        List[Set[int]]: 一个SSR，即包含k个RR-set的列表。
    """
    nodes, in_neighbors_array, alpha, k = args

    # 1. 随机采样一个节点 v
    root_node_v = random.choice(nodes)
    ssr = []  # 当前采样节点的 SSR，包含k个rr-set

    # 2. 为k张券生成k 个RR-set
    for _ in range(k):
        # 检查v是否会领取这张券
        if random.random() <= alpha[root_node_v]:

            # 1. 使用高效的双端队列 (deque) 来管理当前正在扩展的路径
            path_queue = deque([[root_node_v]])

            # 2. 初始化一个列表，专门用来存储最终的“叶子路径”
            leaf_paths = []

            while path_queue:
                current_path = path_queue.popleft()
                leaf_node = current_path[-1]

                # 安全阀：如果路径太长，则将其视为叶子路径，不再扩展
                if len(current_path) >= max_path_length:
                    leaf_paths.append(current_path)
                    continue

                # 标志位，用于判断当前路径是否成功扩展过
                was_extended = False

                # 遍历末端节点的所有入邻居
                for in_neighbor, probability in in_neighbors_array.get(leaf_node, {}).items():
                    if in_neighbor not in current_path:
                        # 模拟边(in_neighbor -> leaf_node)的激活
                        if random.random() <= probability:
                            # 如果成功，创建一条新路径并加入队列
                            new_path = current_path + [in_neighbor]
                            path_queue.append(new_path)
                            # 标记当前路径已被成功扩展
                            was_extended = True

                # 3. 关键优化：如果当前路径遍历完所有邻居后都未能扩展，
                #    说明它是一条“叶子路径”，我们将其保留。
                if not was_extended:
                    leaf_paths.append(current_path)

            # 4. 最终只合并所有“叶子路径”中的节点。
            #    由于任何非叶子路径都是某条叶子路径的前缀，
            #    所以这样做可以得到与之前完全相同的结果，但内存占用大大减少。
            final_rr_set = set()
            for path in leaf_paths:
                final_rr_set.update(path)

            ssr.append(final_rr_set)

        else:
            # 否则，此券的RR-set为空
            ssr.append(set())

    return ssr


# 一次完整的SSR抽样过程
def run_single_ssr_generation2(
        args: Tuple,
        max_path_length: int = 100
) -> List[Set[int]]:
    """
    Args:
        args: 一个元组，包含 (nodes, in_neighbors_array, alpha, k, max_path_length)
              nodes (List[int]): 所有节点的列表。
              in_neighbors_array (Dict[int, Dict[int, float]]): 反向图。
              alpha (Dict[int, float]): 节点的领券概率。
              k (int): 优惠券的数量。
              max_path_length (int): 为防止无限循环，限制单条路径的最大深度。

    Returns:
        List[Set[int]]: 一个SSR，即包含k个RR-set的列表。
    """
    nodes, in_neighbors_array, alpha, k = args

    # 1. 随机采样一个节点 v
    root_node_v = random.choice(nodes)
    ssr = []  # 当前采样节点的 SSR，包含k个rr-set

    # 2. 为k张券生成k 个RR-set
    for _ in range(k):
        # 检查v是否会领取这张券
        if random.random() <= alpha[root_node_v]:

            # 1. 初始化一个列表来保存所有成功生成的反向路径
            all_paths = []

            # 2. 使用一个队列来进行非递归的路径展开，队列中每个元素是一条路径
            #    初始时，队列中只有一条仅包含根节点的路径
            path_queue = [[root_node_v]]

            head = 0
            while head < len(path_queue):
                current_path = path_queue[head]
                head += 1

                # 当前路径的末端节点，是我们下一步要“反向”扩展的节点
                leaf_node = current_path[-1]

                # 安全阀：如果路径太长，则停止在这条路径上继续探索
                if len(current_path) > max_path_length:
                    continue

                # 遍历末端节点的所有入邻居（即反向图中的邻居）
                for in_neighbor, probability in in_neighbors_array.get(leaf_node, {}).items():
                    # 3. 关键改动：我们不再检查 in_neighbor 是否在全局 rr_set 中，
                    #    而是只检查它是否已经存在于“当前路径”中，以防止单路径内的死循环 (e.g., A->B->A)。
                    if in_neighbor not in current_path:
                        # 模拟边(in_neighbor -> leaf_node)的激活
                        if random.random() <= probability:
                            # 如果成功，创建一条新路径，并将其加入队列等待继续扩展
                            new_path = current_path + [in_neighbor]
                            path_queue.append(new_path)

            # 4. 此时 path_queue 中保存了所有从根节点出发的、成功的、雪花状的反向路径
            #    将所有路径中的节点合并，得到最终的RR-set
            final_rr_set = set()
            for path in path_queue:
                final_rr_set.update(path)

            ssr.append(final_rr_set)

        else:
            # 否则，此券的RR-set为空
            ssr.append(set())

    return ssr


# 一次完整的SSR抽样过程
def run_single_ssr_generation1(args: Tuple) -> List[Set[int]]:
    """
    Args:
        args: 一个元组，包含 (nodes, in_neighbors_array, alpha, k)
              nodes (List[int]): 所有节点的列表。
              in_neighbors_array (Dict[int, Dict[int, float]]): 反向图。
              alpha (Dict[int, float]): 节点的领券概率。
              k (int): 优惠券的数量。

    Returns:
        List[Set[int]]: 一个SSR，即包含k个RR-set的列表。
    """
    nodes, in_neighbors_array, alpha, k = args

    # 1. 随机采样一个节点 v
    root_node_v = random.choice(nodes)
    ssr = [] # 当前采样节点的 SSR，包含k个rr-set

    # 2. 为k张券生成k 个RR-set
    for _ in range(k):
        # 检查v是否会领取这张券
        if random.random() <= alpha[root_node_v]:
            # 如果领券，则通过反向BFS生成RR-set
            rr_set = {root_node_v}
            queue = [root_node_v]
            head = 0
            while head < len(queue):
                current_node = queue[head]
                head += 1
                for in_neighbor, probability in in_neighbors_array.get(current_node, {}).items():
                    if in_neighbor not in rr_set:
                        # 模拟边(in_neighbor -> current_node)的激活
                        if random.random() <= probability:
                            rr_set.add(in_neighbor)
                            queue.append(in_neighbor)
            ssr.append(rr_set)
        else:
            # 否则，此券的RR-set为空
            ssr.append(set())
    return ssr



class CouponInfluenceMaximizer:


    def __init__(self, adj: sp.csr_matrix, tranProMatrix: np.ndarray, alpha: Dict[int, float], k: int):
        self.k = k # 优惠券（种子节点）的数量。
        self.alpha = alpha # 每个节点的领券概率。

        # 直接从 adj 获取节点信息
        self.num_nodes = adj.shape[0]
        self.nodes = list(range(self.num_nodes))

        # 使用 tranProMatrix 构建反向图
        self.in_neighbors_array = self._build_in_neighbors_array(adj, tranProMatrix)
        self.all_ssrs: List[List[Set[int]]] = []
        logging.info(f"图初始化完成，包含 {self.num_nodes} 个节点，选择 {k} 个种子。")

    def _build_in_neighbors_array(self, adj: sp.csr_matrix, tranProMatrix: np.ndarray) -> Dict[int, Dict[int, float]]:
        """根据 adj 和 tranProMatrix 构建反向图 利用二维数组"""
        in_neighbors_array = defaultdict(dict)
        rows, cols = adj.nonzero()
        num_edges_processed = 0
        for u, v in zip(rows, cols):
            # tranProMatrix[v, u] 存储的是 u->v 的概率
            probability = tranProMatrix[v, u]
            if probability > 0:
                in_neighbors_array[v][u] = probability
                num_edges_processed += 1
        logging.info(f"反向图构建完成，处理了 {num_edges_processed} 条边。")
        return dict(in_neighbors_array)

    def generate_rr_sets_parallel(self, N: int, workers: int = 2):

        logging.info(f"\n采样次数: {N}, 开始生成 {N} 组SSR，使用 {workers} 个并行进程...")
        start_time = time.time()
        args_list = [(self.nodes, self.in_neighbors_array, self.alpha, self.k) for _ in range(N)]
        with Pool(processes=workers) as pool:
            # todo 在这里选取生成SSR的方法
            results = pool.map(run_single_ssr_generation3, args_list)
        self.all_ssrs = results
        end_time = time.time()
        logging.info(f"{N} 个SSR生成完毕。耗时: {end_time - start_time:.2f} 秒。")

    def select_seeds_new(self) -> Tuple[List[int], float]:
        """
        按列（优惠券）依次选择种子（边际增益）（种子节点不能重复）
        """
        logging.info("\n开始选择最优的种子节点 ...")
        start_time = time.time()
        # 1. 构建倒排索引
        inverted_index = defaultdict(list)
        for ssr_idx, ssr in enumerate(self.all_ssrs):
            for coupon_j, rr_set in enumerate(ssr):
                for node in rr_set:
                    # 每个node对应哪个（第i个SSR，第j张券）
                    inverted_index[node].append((ssr_idx, coupon_j))

        # 2. 贪心选择k个不重复的种子
        selected_seeds = []
        covered_ssr_indices = set()  # 记录已覆盖的SSR（行）
        for i in range(self.k): # 外层循环：迭代k次，为每张优惠券（每一列）选择一个种子
            
            coupon_to_assign = i # 当前要分配的优惠券列索引
            max_marginal_gain = -1
            best_candidate_node = None

            # 内层循环：遍历所有节点，寻找当前列的最佳种子
            for node in self.nodes:
                if node in selected_seeds:
                    continue # # 种子不能重复

                # 1.找到该节点在当前优惠券列中出现的所有SSR
                ssrs_for_this_combo = {
                    ssr_idx for (ssr_idx, coupon_j) in inverted_index.get(node, [])
                    if coupon_j == coupon_to_assign
                }
                # 计算边际增益
                marginal_gain = len(ssrs_for_this_combo - covered_ssr_indices)
                if marginal_gain > max_marginal_gain:
                    max_marginal_gain = marginal_gain
                    best_candidate_node = node

            selected_seeds.append(best_candidate_node)
            print(f"  - 第 {i + 1}/{self.k} 个种子 (为券 {coupon_to_assign + 1}): "
                  f"节点 {best_candidate_node}，新增覆盖 {max_marginal_gain} 个样本。")
            ssrs_covered_by_new_seed = {
                ssr_idx for (ssr_idx, coupon_j) in inverted_index.get(best_candidate_node, [])
                if coupon_j == coupon_to_assign
            }
            covered_ssr_indices.update(ssrs_covered_by_new_seed)

        estimated_influence = (len(covered_ssr_indices) / len(self.all_ssrs)) * self.num_nodes
        end_time = time.time()
        logging.info(f"种子节点选择完毕。耗时: {end_time - start_time:.2f} 秒。")
        return selected_seeds, estimated_influence

def deliverers_ris_coverage(
        adj: sp.csr_matrix,  # 原始邻接矩阵
        tranProMatrix: np.ndarray,  # 转移概率矩阵
        seeds_num: int,
        num_samples: int = 100,
        alpha: Dict[int, float] = None  # 新增一个可选参数alpha
) -> list:
    """
    Args:
        adj (sp.csr_matrix): 邻接矩阵。
        tranProMatrix (np.ndarray): 转移概率矩阵, tranProMatrix[v, u] = p(u->v)。
        seeds_num (int): 要选择的种子数量 (对应k)。
        num_samples (int, optional): RR-set/SSR的抽样次数 (对应N)。默认为 100。
        alpha (Dict[int, float], optional): 节点的领券概率。
                                             如果为 None，则所有节点默认使用 0.1。

    Returns:
        list: 包含k个最优种子节点ID的列表。
    """
    # 1. 处理缺失的 alpha 参数（test）
    if alpha is None:
        num_nodes = adj.shape[0]
        alpha = {node: 0.1 for node in range(num_nodes)}

    # 2. 实例化核心实现类
    maximizer = CouponInfluenceMaximizer(
        adj=adj,
        tranProMatrix=tranProMatrix,
        alpha=alpha,
        k=seeds_num
    )

    # 3. 执行核心逻辑
    maximizer.generate_rr_sets_parallel(N=num_samples)
    selected_seeds, estimated_influence = maximizer.select_seeds_new()

    print(f"\n估算的最大影响力: {estimated_influence:.2f}")
    logging.info(f"最终选择的种子集: {selected_seeds}\n")
    return selected_seeds

if __name__ == "__main__":


    NUM_NODES = 13
    SEEDS_TO_SELECT = 3
    NUM_SAMPLES_FOR_RUN = 30

    adj_matrix = sp.lil_matrix((NUM_NODES, NUM_NODES))
    trans_prob_matrix = np.zeros((NUM_NODES, NUM_NODES))
    p_edge = 0.8
    for i in range(NUM_NODES):
        for j in range(NUM_NODES):
            if i != j and random.random() < p_edge:
                prob = random.uniform(0.4, 0.6)
                adj_matrix[i, j] = 1
                trans_prob_matrix[j, i] = prob

    adj_matrix_csr = adj_matrix.tocsr()


    custom_alpha_probs = {node: random.uniform(0.05, 0.3) for node in range(NUM_NODES)}
    seeds2 = deliverers_ris_coverage(
        adj=adj_matrix_csr,
        tranProMatrix=trans_prob_matrix,
        seeds_num=SEEDS_TO_SELECT,
        num_samples=NUM_SAMPLES_FOR_RUN,
        alpha=custom_alpha_probs
    )