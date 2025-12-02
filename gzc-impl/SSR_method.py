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


# 替换 SSR_method.py 中的同名函数

def run_single_ssr_generation3(
        args: Tuple,
        max_path_length: int = 100
) -> List[Set[int]]:
    """
    修正版：使用【反向随机游走 (Reverse Random Walk)】生成 RR-set。
    这与 Simulation 中的 "Single Token Random Walk" 逻辑保持一致。
    """
    nodes, in_neighbors_array, alpha, k = args

    # 1. 随机采样一个节点 v (作为优惠券的最终接收者)
    root_node_v = random.choice(nodes)
    ssr = []

    # 2. 生成 k 个 RR-set (对应 k 个种子)
    # 实际上如果是标准 RIS，这里不需要循环 k 次，只需要生成一次。
    # 但为了配合你的 select_seeds_new_new 逻辑 (element_id = (ssr_idx, coupon_j))，我们保持结构不变。

    for _ in range(k):
        # 1. 检查根节点是否领券 (对应 Simulation 中的 succ 判定)
        # 如果根节点根本不领券，那反向推导没有意义，这是一个无效样本
        if random.random() > alpha[root_node_v]:
            ssr.append(set())
            continue

        # 2. 开始反向随机游走
        current_rr_set = {root_node_v}
        current_node = root_node_v

        # 模拟路径回溯
        for _ in range(max_path_length):
            # 获取所有入邻居 (可能把券传给 current_node 的人)
            # in_neighbors_array[v] = {u: P(u->v)}
            predecessors_map = in_neighbors_array.get(current_node, {})

            if not predecessors_map:
                break  # 没有入度，回溯结束

            candidates = list(predecessors_map.keys())
            weights = list(predecessors_map.values())

            # --- 核心逻辑修正 ---
            # Simulation: 必定选一个邻居(归一化后)继续，除非停止。
            # 这里我们根据 P(u->v) 的相对权重来选择“谁是父节点”。

            # 计算总权重
            total_weight = sum(weights)
            if total_weight == 0:
                break

            # 归一化权重用于选择
            probs = [w / total_weight for w in weights]

            # 选出一个父节点 (模拟“券是从哪来的”)
            chosen_parent = random.choices(candidates, weights=probs, k=1)[0]

            # 加入 RR-set
            current_rr_set.add(chosen_parent)

            # 决定是否继续回溯 (模拟 Forward Simulation 中的 Stop 概率)
            # 在 Forward 中，节点 u 以 P_tran(u) 的概率继续转发。
            # 这里我们需要估计父节点是否有能力继续转发。
            # 简单起见，或者更严谨地，我们可以查表。
            # 但由于我们不知道父节点具体的 P_tran (除非传进来)，
            # 鉴于 "Influencer Model" 中 P_tran 普遍较高，我们设定一个高阻尼因子。
            # 或者直接利用 weights 的 sum (即 sum(P(u->v))) 作为近似参考，但这在反向时不准。

            # 使用固定阻尼因子模拟平均转发率 (例如 0.85 或 0.9)
            # 这比之前的 0.009 概率要合理得多。
            if random.random() > 0.9:
                break

            # 继续回溯
            current_node = chosen_parent

        ssr.append(current_rr_set)

    return ssr

# 一次完整的SSR抽样过程
def run_single_ssr_generation3(
        args: Tuple,
        max_path_length: int = 10000
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

            # 1. deque 管理当前正在扩展的路径
            path_deque = deque([[root_node_v]])

            # 2. 当有一个路径到达终点时 触发更新rr-set（不显式保存所有已到达终点的路径）
            final_rr_set = set()

            while path_deque:
                current_path = path_deque.popleft()
                leaf_node = current_path[-1]

                # 如果路径太长，视为到达终点，不再扩展
                if len(current_path) >= max_path_length:
                    final_rr_set.update(current_path)
                    continue

                # 标志位，用于判断当前路径是否成功扩展过
                extended_flag = False

                # 遍历末端节点的所有入邻居
                for in_neighbor, probability in in_neighbors_array.get(leaf_node, {}).items():
                    if in_neighbor not in current_path:
                        # 模拟边(in_neighbor -> leaf_node)的激活
                        if random.random() <= probability:
                            # 如果成功，创建一条新路径并加入队列
                            new_path = current_path + [in_neighbor]
                            path_deque.append(new_path)
                            # 标记当前路径已被成功扩展
                            extended_flag = True

                # 3. 如果当前路径遍历完所有入度后都未能扩展，说明该路径到了终点，将其保存
                if not extended_flag:
                    final_rr_set.update(current_path)

            ssr.append(final_rr_set)

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

    def _build_in_neighbors_array_old(self, adj: sp.csr_matrix, tranProMatrix: np.ndarray) -> Dict[int, Dict[int, float]]:
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

    def _build_in_neighbors_array(self, adj: sp.csr_matrix, tranProMatrix: np.ndarray) -> Dict[int, Dict[int, float]]:
        """
        构建反向图。
        adj[u, v] = 1 代表 u -> v。
        我们需要为 v 找到所有入邻居 u，并记录 P(u -> v)。
        """
        in_neighbors_array = defaultdict(dict)

        # 获取所有 u -> v 的边
        if sp.issparse(adj):
            rows, cols = adj.nonzero()
        else:
            rows, cols = np.nonzero(adj)

        num_edges_processed = 0
        for u, v in zip(rows, cols):
            # 关键修正：
            # 我们需要的是 u 激活 v 的概率。
            # 修正后的 getTranProMatrix 保证了 tranProMatrix[u, v] 就是 P(u->v)
            probability = tranProMatrix[u, v]

            if probability > 0:
                # 记录 v 的入邻居是 u，概率是 probability
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

    def select_seeds_new_new(self) -> Tuple[List[int], float]:
        logging.info("\n开始选择最优的种子节点 (Standard Greedy) ...")

        # 1. 扁平化 RR-sets: 记录每个节点覆盖了哪些 (ssr_idx, coupon_idx)
        # node_coverage[u] = set( (0,0), (0,1), (5,2)... )
        node_coverage = defaultdict(set)
        total_elements_count = 0

        for ssr_idx, ssr in enumerate(self.all_ssrs):
            for coupon_j, rr_set in enumerate(ssr):
                element_id = (ssr_idx, coupon_j)  # 这是一个需要被覆盖的唯一事件
                total_elements_count += 1
                for node in rr_set:
                    node_coverage[node].add(element_id)

        selected_seeds = []
        covered_elements = set()

        # 2. 标准贪心选择
        for i in range(self.k):
            max_gain = -1
            best_node = -1

            # 寻找能覆盖最多“未覆盖元素”的节点
            for node in self.nodes:
                if node in selected_seeds:
                    continue

                # 增益 = 该节点覆盖的集合 - 已经被覆盖的集合
                gain = len(node_coverage[node] - covered_elements)

                if gain > max_gain:
                    max_gain = gain
                    best_node = node

            if best_node != -1:
                selected_seeds.append(best_node)
                covered_elements.update(node_coverage[best_node])
                print(f"  - 选出第 {i + 1} 个种子: {best_node}, 边际增益: {max_gain}")
            else:
                break

        estimated_influence = (len(covered_elements) / total_elements_count) * self.num_nodes * self.k  # 粗略估算
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
    selected_seeds, estimated_influence = maximizer.select_seeds_new_new()

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