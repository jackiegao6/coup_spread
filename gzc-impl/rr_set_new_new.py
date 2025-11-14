import random
import time
from multiprocessing import Pool, cpu_count
from collections import defaultdict
from typing import List, Dict, Set, Tuple, Any
import scipy.sparse as sp
import numpy as np



# 为了并行化，这个函数需要定义在类的外部（或作为静态方法）
# 它执行一次完整的SSR抽样过程
def run_single_ssr_generation(args: Tuple) -> List[Set[int]]:
    """
    为并行池执行单次SSR（Super Set of RR-sets）的生成。

    Args:
        args: 一个元组，包含 (nodes, reversed_graph, alpha, k)
              nodes (List[int]): 所有节点的列表。
              reversed_graph (Dict[int, Dict[int, float]]): 反向图。
              alpha (Dict[int, float]): 节点的领券概率。
              k (int): 优惠券的数量。

    Returns:
        List[Set[int]]: 一个SSR，即包含k个RR-set的列表。
    """
    nodes, reversed_graph, alpha, k = args

    # 1. 随机选择一个根节点 v
    root_node_v = random.choice(nodes)
    ssr = [] # 当前的 "SSR"，包含k个rr-set

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

                for in_neighbor, probability in reversed_graph.get(current_node, {}).items():
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
    """
    使用反向可达集(RR-set)方法，寻找k个种子节点以最大化优惠券激活数。
    支持并行化生成RR-set。
    """
    def __init__(self, edges: List[Tuple[int, int, float]], alpha: Dict[int, float], k: int):
        """
        初始化影响力最大化器。

        Args:
            edges (List[Tuple[int, int, float]]): 图的边列表，格式为 (u, v, p)，p为u->v的传播概率。
            alpha (Dict[int, float]): 每个节点的领券概率。
            k (int): 优惠券（种子节点）的数量。
        """
        self.k = k
        self.alpha = alpha
        self.nodes = list(alpha.keys())
        self.num_nodes = len(self.nodes)
        self.reversed_graph = self._build_reversed_graph(edges)
        self.all_ssrs: List[List[Set[int]]] = []
        print(f"图已初始化，包含 {self.num_nodes} 个节点，将选择 {k} 个种子。")

    def _build_reversed_graph(self, edges: List[Tuple[int, int, float]]) -> Dict[int, Dict[int, float]]:
        """从边列表构建反向图，用于高效的反向传播。"""
        reversed_graph = defaultdict(dict)
        for u, v, p in edges:
            if u in self.nodes and v in self.nodes:
                reversed_graph[v][u] = p
        return dict(reversed_graph)

    def generate_rr_sets_parallel(self, N: int, workers: int = -1):
        """
        通过并行计算，生成N组SSR (Super Set of RR-sets)。

        Args:
            N (int): 抽样总次数。
            workers (int): 使用的CPU核心数。-1表示使用所有可用核心。
        """
        if N <= 0:
            raise ValueError("抽样次数N必须为正整数。")
        if workers == -1:
            workers = 8
        print(f"\n开始生成 {N} 组SSR样本，使用 {workers} 个并行进程...")
        start_time = time.time()
        args_list = [(self.nodes, self.reversed_graph, self.alpha, self.k) for _ in range(N)] # 为每个进程准备参数
        with Pool(processes=workers) as pool:
            results = pool.map(run_single_ssr_generation, args_list)
        self.all_ssrs = results
        end_time = time.time()
        print(f"SSR样本生成完毕。耗时: {end_time - start_time:.2f} 秒。")


    def select_seeds_new(self) -> Tuple[List[int], float]:
        """
        根据新要求实现的贪心算法：
        1. 按列（优惠券）依次选择种子。
        2. 种子节点不能重复。
        3. 严格计算边际增益。

        Returns:
            Tuple[List[int], float]:
                - 一个列表，包含k个不重复的种子节点。
                - 估算出的最大影响力。
        """

        print("\n开始选择最优的种子节点 ...")
        start_time = time.time()

        # 1. 构建倒排索引，与之前相同，这是最高效的预处理方式
        inverted_index = defaultdict(list)
        for ssr_idx, ssr in enumerate(self.all_ssrs):
            for coupon_j, rr_set in enumerate(ssr):
                for node in rr_set:
                    inverted_index[node].append((ssr_idx, coupon_j))

        # 2. 贪心选择k个不重复的种子
        selected_seeds = []
        covered_ssr_indices = set()  # 使用集合跟踪已覆盖的SSR（行）

        # 外层循环：迭代k次，为每张优惠券（每一列）选择一个种子
        for i in range(self.k):
            coupon_to_assign = i  # 当前要分配的优惠券列索引
            max_marginal_gain = -1
            best_candidate_node = None

            # 内层循环：遍历所有节点，寻找当前列的最佳种子
            for node in self.nodes:
                # 要求3：种子不能重复
                if node in selected_seeds:
                    continue

                # 找到该节点在当前优惠券列中出现的所有SSR
                ssrs_for_this_combo = {
                    ssr_idx for (ssr_idx, coupon_j) in inverted_index.get(node, [])
                    if coupon_j == coupon_to_assign
                }

                # 要求2：计算边际增益（新增的覆盖数）
                marginal_gain = len(ssrs_for_this_combo - covered_ssr_indices)

                if marginal_gain > max_marginal_gain:
                    max_marginal_gain = marginal_gain
                    best_candidate_node = node

            # 如果找不到任何有增益的种子，提前结束
            if best_candidate_node is None:
                print(f"警告：在为优惠券 {coupon_to_assign + 1} 选择种子时，没有找到任何有增益的节点。")
                break

            # 选定本轮的种子
            selected_seeds.append(best_candidate_node)
            print(f"  - 第 {i + 1}/{self.k} 个种子 (为券 {coupon_to_assign + 1}): "
                  f"节点 {best_candidate_node}，新增覆盖 {max_marginal_gain} 个样本。")

            # 更新已覆盖的SSR集合
            ssrs_covered_by_new_seed = {
                ssr_idx for (ssr_idx, coupon_j) in inverted_index.get(best_candidate_node, [])
                if coupon_j == coupon_to_assign
            }
            covered_ssr_indices.update(ssrs_covered_by_new_seed)

        # 3. 计算最终的期望影响力
        estimated_influence = (len(covered_ssr_indices) / len(self.all_ssrs)) * self.num_nodes

        end_time = time.time()
        print(f"种子节点选择完毕。耗时: {end_time - start_time:.2f} 秒。")

        return selected_seeds, estimated_influence


def deliverers_ris_coverage(
        adj,  # 原始邻接矩阵
        tranProMatrix: np.ndarray,  # 转移概率矩阵
        seeds_num: int,
        num_samples: int,
        dis: tuple,
        # alpha: Dict[int, float] = None,  # 新增一个可选参数alpha
) -> list:
    """
    使用多优惠券并行RIS模型计算影响力最大的种子节点。
    此函数封装了CouponInfluenceMaximizer类的复杂逻辑，以匹配原始接口。

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
    if isinstance(adj, np.ndarray):
        adj = sp.csr_matrix(adj)

    succ_dis = dis[0]

    # 1. 处理缺失的 alpha 参数
    num_nodes = adj.shape[0]
    alpha = {node: succ_dis[node] for node in range(num_nodes)}

    # 2. 实例化核心实现类
    # 参数直接映射: seeds_num -> k
    maximizer = CouponInfluenceMaximizer(
        adj=adj,
        tranProMatrix=tranProMatrix,
        alpha=alpha,
        k=seeds_num
    )

    # 3. 执行核心逻辑
    # 参数直接映射: num_samples -> N
    maximizer.generate_rr_sets_parallel(N=num_samples)
    selected_seeds, estimated_influence = maximizer.select_seeds_new()

    # 4. 格式化并返回输出
    print(f"\n估算的最大影响力: {estimated_influence:.2f}")
    print(f"最终选择的种子集: {selected_seeds}\n")
    return selected_seeds



if __name__ == "__main__":

    # 1. 准备和您接口一致的输入数据
    NUM_NODES = 50
    SEEDS_TO_SELECT = 5
    NUM_SAMPLES_FOR_RUN = 20000

    adj_matrix = sp.lil_matrix((NUM_NODES, NUM_NODES))
    trans_prob_matrix = np.zeros((NUM_NODES, NUM_NODES))
    p_edge = 0.1
    for i in range(NUM_NODES):
        for j in range(NUM_NODES):
            if i != j and random.random() < p_edge:
                prob = random.uniform(0.01, 0.1)
                adj_matrix[i, j] = 1
                trans_prob_matrix[j, i] = prob

    adj_matrix_csr = adj_matrix.tocsr()

    # 2. (场景一) 调用您的原始接口，不提供alpha
    print("=" * 50)
    print("场景一: 调用原始接口 (使用默认alpha)")
    print("=" * 50)
    seeds1 = deliverers_ris_coverage(
        adj=adj_matrix_csr,
        tranProMatrix=trans_prob_matrix,
        seeds_num=SEEDS_TO_SELECT,
        num_samples=NUM_SAMPLES_FOR_RUN
    )

    # 3. (场景二) 调用新接口，提供自定义的alpha值
    print("\n" + "=" * 50)
    print("场景二: 调用新接口 (提供自定义alpha)")
    print("=" * 50)
    custom_alpha_probs = {node: random.uniform(0.05, 0.3) for node in range(NUM_NODES)}
    seeds2 = deliverers_ris_coverage(
        adj=adj_matrix_csr,
        tranProMatrix=trans_prob_matrix,
        seeds_num=SEEDS_TO_SELECT,
        num_samples=NUM_SAMPLES_FOR_RUN,
        alpha=custom_alpha_probs
    )