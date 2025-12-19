import random
import time
from multiprocessing import Pool, cpu_count
from collections import defaultdict
from typing import List, Dict, Set, Tuple, Any
import numpy as np
import scipy.sparse as sp
import logging
from typing import List, Set, Tuple

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
    selected_seeds, estimated_influence = maximizer.select_seeds()

    print(f"\n估算的最大影响力: {estimated_influence:.2f}")
    logging.info(f"最终选择的种子集: {selected_seeds}\n")
    return selected_seeds



def run_single_ssr_generation_without_path(args: Tuple) -> List[Set[int]]:
    """
    基于 CSR 矩阵的高性能 RR-Set 生成器。
    原理：反向独立级联 (Reverse IC) 模型，不再存储路径，只存储可达节点集合。
    """
    # 解包参数：注意这里直接接收 CSR 的三个核心数组，避免传递整个对象带来的开销
    num_nodes, indices, indptr, data, alpha, k = args

    # 1. 随机采样一个目标节点 v (我们希望覆盖的用户)
    root_node_v = random.randint(0, num_nodes - 1)

    ssr_list = []  # 存储 k 张券的 RR-Set

    for _ in range(k):
        # 保持你的业务逻辑：先判断该节点是否有意愿(alpha)接收这张券
        # 如果 alpha 是 dict: alpha.get(root_node_v, 0)
        # 如果 alpha 是 list/array: alpha[root_node_v]
        p_accept = alpha[root_node_v] if isinstance(alpha, (list, np.ndarray)) else alpha.get(root_node_v, 0)

        if random.random() > p_accept:
            ssr_list.append(set())
            continue

        # --- 开始反向 BFS 遍历 ---

        # RR-Set 也就是“如果谁有了券，能传给 root_node_v”的节点集合
        rr_set = {root_node_v}
        queue = [root_node_v]  # BFS 队列

        # 使用指针遍历队列
        idx = 0
        while idx < len(queue):
            curr_node = queue[idx]
            idx += 1

            # 【核心优化】：直接从 CSR 结构读取入邻居
            # 这里的 curr_node 对应矩阵的 行(Row)
            # 因为 tranProMatrix[i, j] = j->i，所以 Row i 里的元素就是 j
            start_ptr = indptr[curr_node]
            end_ptr = indptr[curr_node + 1]

            # 如果没有入邻居，跳过
            if start_ptr == end_ptr:
                continue

            # 获取邻居ID和对应的概率
            # matrix_indices 里的就是 j (源节点)
            neighbor_indices = indices[start_ptr:end_ptr]
            neighbor_probs = data[start_ptr:end_ptr]

            # 【向量化随机判定】：一次性生成所有随机数，比循环快得多
            rand_vals = np.random.rand(len(neighbor_indices))

            # 找出成功激活的边
            # 逻辑：如果 rand <= prob，说明反向边存在 (即 j 能够激活 curr)
            success_mask = rand_vals <= neighbor_probs

            # 得到成功反向传播的邻居
            active_neighbors = neighbor_indices[success_mask]

            for neighbor in active_neighbors:
                if neighbor not in rr_set:
                    rr_set.add(neighbor)
                    queue.append(neighbor)

        ssr_list.append(rr_set)

    return ssr_list


class CouponInfluenceMaximizer:

    def __init__(self, adj: sp.csr_matrix, tranProMatrix: np.ndarray, alpha: Dict[int, float], k: int):
        self.k = k # 优惠券（种子节点）的数量。
        self.alpha = alpha # 每个节点的领券概率。

        # 直接从 adj 获取节点信息
        self.num_nodes = adj.shape[0]
        self.nodes = list(range(self.num_nodes))

        # tranProMatrix 转为 CSR 格式
        if isinstance(tranProMatrix, np.ndarray):
            # 将稠密矩阵转为 CSR
            self.tran_matrix_csr = sp.csr_matrix(tranProMatrix)
        else:
            self.tran_matrix_csr = tranProMatrix

        # 预先提取 CSR 的底层数组，方便传递给多进程，减少序列化开销
        self.csr_indices = self.tran_matrix_csr.indices
        self.csr_indptr = self.tran_matrix_csr.indptr
        self.csr_data = self.tran_matrix_csr.data

        self.all_ssrs: List[List[Set[int]]] = []
        logging.info(f"图初始化完成，节点数: {self.num_nodes}，CSR 矩阵准备就绪。")

    def generate_rr_sets_parallel(self, N: int, workers: int = 8):
        logging.info(f"\n开始生成 {N} 组 SSR (并行进程: {workers})...")
        start_time = time.time()

        # 构造参数包：只传底层数组和必要参数
        # 注意：不要传 self，否则会序列化整个类实例
        args_tuple = (
            self.num_nodes,
            self.csr_indices,
            self.csr_indptr,
            self.csr_data,
            self.alpha,
            self.k
        )
        # 复制 N 份参数
        args_list = [args_tuple] * N

        with Pool(processes=workers) as pool:
            self.all_ssrs = pool.map(run_single_ssr_generation_without_path, args_list)

        end_time = time.time()
        logging.info(f"生成完毕。耗时: {end_time - start_time:.2f} 秒。")

    def select_seeds(self) -> Tuple[List[int], float]:
        """
        按列（优惠券）依次选择种子。
        优化：只搜索在相关 RR-Set 中出现过的节点（剪枝）。
        """
        logging.info("\n开始选择最优种子 (Optimized Greedy) ...")
        start_time = time.time()

        # 1. 构建按优惠券分类的倒排索引
        # coupon_node_map[coupon_idx][node_id] = {ssr_idx1, ssr_idx2, ...}
        # 这样我们在选第 i 张券时，只需要遍历 coupon_node_map[i].keys()
        coupon_node_map = defaultdict(lambda: defaultdict(set))

        total_samples = len(self.all_ssrs)

        for ssr_idx, ssr_list in enumerate(self.all_ssrs):
            for coupon_j, rr_set in enumerate(ssr_list):
                if not rr_set: continue
                for node in rr_set:
                    coupon_node_map[coupon_j][node].add(ssr_idx)

        selected_seeds = []
        covered_ssr_indices = set()  # 记录已覆盖的SSR索引 (逻辑上假设不同券覆盖的样本是同一个池子)

        # 2. 贪心选择
        for i in range(self.k):
            coupon_to_assign = i

            best_node = -1
            max_gain = -1

            # 【优化2】：只遍历“能覆盖当前券”的节点
            # 如果一个节点从未在第 i 张券的 RR-Set 中出现，它对此券的增益必为 0
            candidate_nodes = coupon_node_map[coupon_to_assign].keys()

            # 如果 candidate_nodes 为空（说明这张券太难发了，或者 alpha太低，没人能反向激活），则随机选或者跳过
            if not candidate_nodes:
                logging.warning(f"券 {i} 无法通过反向传播到达任何节点，随机选择备选。")
                # 简单处理：选一个不在 selected_seeds 里的节点
                remaining = list(set(self.nodes) - set(selected_seeds))
                best_node = remaining[0] if remaining else 0
                max_gain = 0
            else:
                for node in candidate_nodes:
                    if node in selected_seeds:
                        continue  # 种子不能重复

                    # 获取该节点覆盖的 SSR 集合
                    covered_by_node = coupon_node_map[coupon_to_assign][node]

                    # 计算边际增益：(新覆盖的 - 已经覆盖的)
                    # 这里的逻辑是：如果 SSR_X 已经被上一张券覆盖了，这一张券再覆盖它就不算增益了
                    # (这取决于你的业务逻辑：是追求“总覆盖人数”还是“总发券量”。这里沿用你代码的逻辑：总覆盖人数)
                    current_gain = len(covered_by_node - covered_ssr_indices)

                    if current_gain > max_gain:
                        max_gain = current_gain
                        best_node = node

            # 处理一种情况：所有候选节点都在 selected_seeds 里了，或者 gain 都是 0
            if best_node == -1:
                remaining = list(set(self.nodes) - set(selected_seeds))
                best_node = remaining[0] if remaining else 0

            selected_seeds.append(best_node)

            # 更新覆盖状态
            newly_covered = coupon_node_map[coupon_to_assign][best_node]
            covered_ssr_indices.update(newly_covered)

            logging.info(f"  - 券 {i + 1}/{self.k} 选中节点 {best_node}, 新增覆盖 {max_gain}")

        # 计算影响力
        estimated_influence = (len(covered_ssr_indices) / total_samples) * self.num_nodes
        end_time = time.time()
        logging.info(f"选种完毕。耗时: {end_time - start_time:.2f} 秒。")
        return selected_seeds, estimated_influence
