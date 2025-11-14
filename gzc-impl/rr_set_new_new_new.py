import random
import time
from multiprocessing import Pool, cpu_count
from collections import defaultdict
from typing import List, Dict, Set, Tuple, Any

# 导入您接口所需的库
import numpy as np
import scipy.sparse as sp


# ==============================================================================
# 核心实现：CouponInfluenceMaximizer 类 (来自我们之前的讨论)
# 这部分是内部逻辑，将被下面的接口函数调用。
# ==============================================================================

def run_single_ssr_generation(args: Tuple) -> List[Set[int]]:
    """并行工作单元，无需修改。"""
    nodes, reversed_graph, alpha, k = args
    root_node_v = random.choice(nodes)
    ssr = []
    for _ in range(k):
        if random.random() <= alpha[root_node_v]:
            rr_set = {root_node_v}
            queue = [root_node_v]
            head = 0
            while head < len(queue):
                current_node = queue[head]
                head += 1
                for in_neighbor, probability in reversed_graph.get(current_node, {}).items():
                    if in_neighbor not in rr_set and random.random() <= probability:
                        rr_set.add(in_neighbor)
                        queue.append(in_neighbor)
            ssr.append(rr_set)
        else:
            ssr.append(set())
    return ssr


class CouponInfluenceMaximizer:
    """
    核心实现类，封装了所有复杂逻辑。
    它的 __init__ 方法被修改为可以直接接收 adj 和 tranProMatrix。
    """

    def __init__(self, adj: sp.csr_matrix, tranProMatrix: np.ndarray, alpha: Dict[int, float], k: int):
        self.k = k
        self.alpha = alpha

        # 直接从 adj 获取节点信息
        self.num_nodes = adj.shape[0]
        self.nodes = list(range(self.num_nodes))

        # 使用 tranProMatrix 构建反向图
        self.reversed_graph = self._build_reversed_graph(adj, tranProMatrix)
        self.all_ssrs: List[List[Set[int]]] = []
        print(f"图已初始化，包含 {self.num_nodes} 个节点，将选择 {k} 个种子。")

    def _build_reversed_graph(self, adj: sp.csr_matrix, tranProMatrix: np.ndarray) -> Dict[int, Dict[int, float]]:
        """根据 adj 和 tranProMatrix 构建反向图字典。"""
        reversed_graph = defaultdict(dict)
        rows, cols = adj.nonzero()
        num_edges_processed = 0
        for u, v in zip(rows, cols):
            # tranProMatrix[v, u] 存储的是 u->v 的概率
            probability = tranProMatrix[v, u]
            if probability > 0:
                reversed_graph[v][u] = probability
                num_edges_processed += 1
        print(f"反向图构建完成，处理了 {num_edges_processed} 条边。")
        return dict(reversed_graph)

    def generate_rr_sets_parallel(self, N: int, workers: int = -1):
        if N <= 0: raise ValueError("抽样次数N必须为正整数。")
        if workers == -1: workers = 8
        print(f"\n开始生成 {N} 组SSR样本，使用 {workers} 个并行进程...")
        start_time = time.time()
        args_list = [(self.nodes, self.reversed_graph, self.alpha, self.k) for _ in range(N)]
        with Pool(processes=workers) as pool:
            results = pool.map(run_single_ssr_generation, args_list)
        self.all_ssrs = results
        end_time = time.time()
        print(f"SSR样本生成完毕。耗时: {end_time - start_time:.2f} 秒。")

    def select_seeds_new(self) -> Tuple[List[int], float]:
        """最新的种子选择策略，无需修改。"""
        if not self.all_ssrs: raise RuntimeError("请先调用 generate_rr_sets_parallel()。")
        print("\n开始选择最优的种子节点 (新策略)...")
        start_time = time.time()
        inverted_index = defaultdict(list)
        for ssr_idx, ssr in enumerate(self.all_ssrs):
            for coupon_j, rr_set in enumerate(ssr):
                for node in rr_set:
                    inverted_index[node].append((ssr_idx, coupon_j))

        selected_seeds = []
        covered_ssr_indices = set()
        for i in range(self.k):
            coupon_to_assign = i
            max_marginal_gain = -1
            best_candidate_node = None
            for node in self.nodes:
                if node in selected_seeds: continue
                ssrs_for_this_combo = {
                    ssr_idx for (ssr_idx, coupon_j) in inverted_index.get(node, [])
                    if coupon_j == coupon_to_assign
                }
                marginal_gain = len(ssrs_for_this_combo - covered_ssr_indices)
                if marginal_gain > max_marginal_gain:
                    max_marginal_gain = marginal_gain
                    best_candidate_node = node

            if best_candidate_node is None:
                print(f"警告：为券 {coupon_to_assign + 1} 选择种子时无增益。")
                break

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
        print(f"种子节点选择完毕。耗时: {end_time - start_time:.2f} 秒。")
        return selected_seeds, estimated_influence


# ==============================================================================
# 您的接口函数：现在它是一个调用核心实现的包装器
# ==============================================================================
def deliverers_ris_coverage(
        adj: sp.csr_matrix,  # 原始邻接矩阵
        tranProMatrix: np.ndarray,  # 转移概率矩阵
        seeds_num: int,
        num_samples: int = 100,
        alpha: Dict[int, float] = None  # 新增一个可选参数alpha
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
    print("--- 正在运行新版 deliverers_ris_coverage (多优惠券并行模型) ---")

    # 1. 处理缺失的 alpha 参数
    if alpha is None:
        num_nodes = adj.shape[0]
        print(f"警告: 未提供 alpha (节点领券概率)。将为所有 {num_nodes} 个节点使用默认值 0.1。")
        alpha = {node: 0.1 for node in range(num_nodes)}

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


# ==========================
# 主执行模块：演示如何调用您的接口
# ==========================
if __name__ == "__main__":

    # 1. 准备和您接口一致的输入数据
    NUM_NODES = 500
    SEEDS_TO_SELECT = 10
    NUM_SAMPLES_FOR_RUN = 200

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

    # 3. 提供自定义的alpha值
    custom_alpha_probs = {node: random.uniform(0.05, 0.3) for node in range(NUM_NODES)}
    seeds2 = deliverers_ris_coverage(
        adj=adj_matrix_csr,
        tranProMatrix=trans_prob_matrix,
        seeds_num=SEEDS_TO_SELECT,
        num_samples=NUM_SAMPLES_FOR_RUN,
        alpha=custom_alpha_probs
    )