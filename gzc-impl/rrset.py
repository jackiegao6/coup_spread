import random
import time
from multiprocessing import Pool, cpu_count
from collections import defaultdict
from typing import List, Dict, Set, Tuple, Any


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

    ssr = []  # 当前的 "SSR"，包含k个rr-set

    # 2. & 3. 为每张券尝试生成RR-set
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

                # 遍历所有入邻居
                for in_neighbor, probability in reversed_graph.get(current_node, {}).items():
                    if in_neighbor not in rr_set:
                        # 模拟边(in_neighbor -> current_node)的激活 todo probability
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

        # 为每个进程准备参数
        args_list = [(self.nodes, self.reversed_graph, self.alpha, self.k) for _ in range(N)]

        with Pool(processes=workers) as pool:
            results = pool.map(run_single_ssr_generation, args_list)

        self.all_ssrs = results
        end_time = time.time()
        print(f"SSR样本生成完毕。耗时: {end_time - start_time:.2f} 秒。")
        print(f"共生成 {len(self.all_ssrs)} 个SSR，每个SSR包含 {self.k} 个RR-set。")

    def select_seeds(self) -> Tuple[List[Tuple[int, int]], float]:
        """
        使用贪心算法从生成的SSR中选择k个最优的种子节点。

        Returns:
            Tuple[List[Tuple[int, int]], float]:
                - 一个元组列表，每个元组 (seed_node, coupon_slot) 代表选择的种子及其分配的券。
                - 估算出的最大影响力（激活的用户期望数）。
        """
        if not self.all_ssrs:
            raise RuntimeError("请先调用 generate_rr_sets_parallel() 生成RR-set样本。")

        print("\n开始选择最优的种子节点...")
        start_time = time.time()

        # 1. 构建倒排索引: node -> List[(ssr_idx, coupon_idx)]
        # 这个索引告诉我们，一个节点出现在了哪些SSR的哪个优惠券的RR-set中
        inverted_index = defaultdict(list)
        for ssr_idx, ssr in enumerate(self.all_ssrs):
            for coupon_j, rr_set in enumerate(ssr):
                for node in rr_set:
                    inverted_index[node].append((ssr_idx, coupon_j))

        # 2. 贪心选择k个种子
        seeds = []
        covered_ssr_indices = set()  # 记录已被覆盖的SSR的索引

        # 临时存储每个节点和券组合能覆盖的SSR列表，避免重复计算
        # coverage_cache: (node, coupon_slot) -> Set[ssr_idx]
        coverage_cache = defaultdict(set)
        for node, appearances in inverted_index.items():
            for ssr_idx, coupon_j in appearances:
                coverage_cache[(node, coupon_j)].add(ssr_idx)

        for i in range(self.k):
            max_marginal_gain = -1
            best_seed_combo = None  # (node, coupon_slot)

            # 寻找能带来最大边际增益的 (node, coupon_slot) 组合
            for (node, coupon_slot), covers in coverage_cache.items():
                # 边际增益 = 新覆盖的SSR数量
                marginal_gain = len(covers - covered_ssr_indices)

                if marginal_gain > max_marginal_gain:
                    max_marginal_gain = marginal_gain
                    best_seed_combo = (node, coupon_slot)

            # 如果找不到任何可以增加覆盖的种子（不太可能发生，除非图是空的）
            if best_seed_combo is None:
                print(f"警告：在第 {i + 1} 轮没有找到有增益的种子。")
                break

            # 选定本轮最优的种子组合，并更新覆盖集
            seeds.append(best_seed_combo)
            newly_covered = coverage_cache[best_seed_combo]
            covered_ssr_indices.update(newly_covered)

            # 从cache中移除已选择的组合，避免重复选择
            # 注意：这里我们允许同一个节点被选为不同券的种子
            # 如果要禁止，需要更复杂的逻辑来移除所有与该节点或券槽相关的条目

            print(f"  第 {i + 1}/{self.k} 个种子选定: 节点 {best_seed_combo[0]} (用于优惠券 {best_seed_combo[1] + 1})，"
                  f"新增覆盖 {max_marginal_gain} 个样本。")

        # 3. 计算最终的期望影响力
        # 影响力 = (被覆盖的SSR样本比例) * (网络总节点数)
        estimated_influence = (len(covered_ssr_indices) / len(self.all_ssrs)) * self.num_nodes

        end_time = time.time()
        print(f"种子节点选择完毕。耗时: {end_time - start_time:.2f} 秒。")

        return seeds, estimated_influence


# ==========================
# 主执行模块
# ==========================
if __name__ == "__main__":

    # 1. 定义图和参数
    NUM_NODES = 50
    K_COUPONS = 5  # 种子/优惠券数量
    NUM_SAMPLES = 20000  # RR-set抽样次数，数量越大结果越准，计算时间越长

    # 创建一个随机图 (Erdos-Renyi G(n,p) model)
    nodes_list = list(range(NUM_NODES))
    edge_list = []
    p_edge = 0.1  # 节点间存在边的概率
    for i in nodes_list:
        for j in nodes_list:
            if i != j and random.random() < p_edge:
                # (u, v, propagation_probability)
                edge_list.append((i, j, random.uniform(0.01, 0.1)))  # 随机设置较小的传播概率

    # 为每个节点设置随机的领券概率
    alpha_probs = {node: random.uniform(0.1, 0.5) for node in nodes_list}

    print("=" * 40)
    print("优惠券影响力最大化模拟")
    print(f"网络规模: {NUM_NODES} 节点, {len(edge_list)} 条边")
    print(f"参数: k={K_COUPONS}, N={NUM_SAMPLES}")
    print("=" * 40)

    # 2. 初始化并运行算法
    try:
        # 实例化
        maximizer = CouponInfluenceMaximizer(edges=edge_list, alpha=alpha_probs, k=K_COUPONS)

        # 第一步：并行生成RR-set样本
        maximizer.generate_rr_sets_parallel(N=NUM_SAMPLES)

        # 第二步：选择种子节点
        selected_seeds, influence = maximizer.select_seeds()

        # 3. 打印最终结果
        print("\n" + "=" * 40)
        print("最终结果")
        print("=" * 40)
        print(f"估算出的最大用户激活期望数: {influence:.2f}")
        print("选择的最优种子组合 (节点, 分配的优惠券编号):")
        for seed, coupon_slot in selected_seeds:
            print(f"  - 节点 {seed} <-- 优惠券 {coupon_slot + 1}")
        print("=" * 40)

    except Exception as e:
        print(f"\n程序运行出错: {e}")