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
# deprecated
def run_single_ssr_generation_with_path(
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


def run_single_ssr_generation_without_path(args: Tuple) -> List[Set[int]]:
    """
    基于 CSR 矩阵的高性能 RR-Set 生成器。
    【核心优化】：引入“联合概率判定”，利用同一个随机数同时校验“边权重”和“邻居转发意愿”。
    """
    # 解包参数：包含 succ_dist 和 dis_dist
    num_nodes, indices, indptr, data, alpha, k, succ_dist, dis_dist = args

    # 1. 随机采样一个目标节点 v
    root_node_v = random.randint(0, num_nodes - 1)

    ssr_list = []  # 存储 k 张券的 RR-Set

    for _ in range(k):
        # 0. 目标节点本身的领券意愿 check
        p_accept = alpha[root_node_v] if isinstance(alpha, (list, np.ndarray)) else alpha.get(root_node_v, 0)

        if random.random() > p_accept:
            ssr_list.append(set())
            continue

        # --- 开始反向 BFS 遍历 ---
        rr_set = {root_node_v}
        queue = [root_node_v]  # BFS 队列

        idx = 0
        while idx < len(queue):
            curr_node = queue[idx]
            idx += 1

            # CSR 切片：获取所有指向 curr_node 的上游邻居 j
            start_ptr = indptr[curr_node]
            end_ptr = indptr[curr_node + 1]

            if start_ptr == end_ptr:
                continue

            # neighbor_indices 里的就是 j (源节点)
            neighbor_indices = indices[start_ptr:end_ptr]
            edge_probs = data[start_ptr:end_ptr]  # 边 j->i 的概率 M[j, i]

            # ==========================================================
            # 【核心优化点】：实现“凭同一个随机数进行接收/转发判断”
            # ==========================================================

            # 1. 获取邻居 j 的行为分布
            n_succ = succ_dist[neighbor_indices]
            n_dis = dis_dist[neighbor_indices]

            # 2. 计算邻居 j 的“转发意愿” (Forwarding Probability)
            # 只有当 j 不使用(succ) 且 不丢弃(dis) 时，它才可能作为上游节点
            n_fwd_willingness = 1.0 - n_succ - n_dis

            # 边界保护：防止浮点误差导致负数
            n_fwd_willingness = np.maximum(n_fwd_willingness, 0.0)

            # 3. 计算“联合有效概率” (Joint Effective Probability)
            # 这一步体现了“同一个随机数”的逻辑：
            # 我们生成一个 r，它必须足够小，才能同时满足 "r < 转发意愿" 和 "r < 边概率"
            # 数学上，这等价于 r < (边概率 * 转发意愿)
            effective_probs = edge_probs * n_fwd_willingness

            # 4. 生成随机数并判定
            rand_vals = np.random.rand(len(neighbor_indices))

            # 只有当随机数小于联合概率时，才认为该邻居有效
            success_mask = rand_vals <= effective_probs

            # 获取通过筛选的邻居
            active_neighbors = neighbor_indices[success_mask]

            for neighbor in active_neighbors:
                if neighbor not in rr_set:
                    rr_set.add(neighbor)
                    queue.append(neighbor)

        ssr_list.append(rr_set)

    return ssr_list


class CouponInfluenceMaximizer:
    def __init__(self,
                 adj: sp.csr_matrix,
                 tranProMatrix: np.ndarray,
                 alpha: Dict[int, float],
                 distributions: Tuple[np.ndarray, np.ndarray, Any, Any],  # 新增：接收分布元组
                 k: int):

        self.k = k
        self.alpha = alpha

        # 解包分布数据 (succ, dis, ...)
        # 假设 distributions 顺序是 (succ, dis, constant, ...)
        self.succ_dist = distributions[0]
        self.dis_dist = distributions[1]

        # 确保它们是 numpy array 以便 Worker 进行花式索引 (Fancy Indexing)
        if not isinstance(self.succ_dist, np.ndarray):
            self.succ_dist = np.array(self.succ_dist)
        if not isinstance(self.dis_dist, np.ndarray):
            self.dis_dist = np.array(self.dis_dist)

        self.num_nodes = adj.shape[0]
        self.nodes = list(range(self.num_nodes))

        # CSR 处理
        if isinstance(tranProMatrix, np.ndarray):
            self.tran_matrix_csr = sp.csr_matrix(tranProMatrix)
        else:
            self.tran_matrix_csr = tranProMatrix

        self.csr_indices = self.tran_matrix_csr.indices
        self.csr_indptr = self.tran_matrix_csr.indptr
        self.csr_data = self.tran_matrix_csr.data

        self.all_ssrs = []
        logging.info(f"图初始化完成，CSR 矩阵及概率分布准备就绪。")

    def generate_rr_sets_parallel(self, N: int, workers: int = 4):
        logging.info(f"\n开始生成 {N} 组 SSR (并行进程: {workers})...")
        start_time = time.time()

        # 构造参数包：加入 succ_dist 和 dis_dist
        args_tuple = (
            self.num_nodes,
            self.csr_indices,
            self.csr_indptr,
            self.csr_data,
            self.alpha,
            self.k,
            self.succ_dist,  # 传进去
            self.dis_dist  # 传进去
        )

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

    #deprecated
    def select_seeds_IM(self) -> Tuple[List[int], float]:
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
        alpha: Dict[int, float] = None,  # 新增一个可选参数alpha
        distributions = None
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
        distributions = distributions,
        k=seeds_num
    )

    # 3. 执行核心逻辑
    maximizer.generate_rr_sets_parallel(N=num_samples)
    selected_seeds, estimated_influence = maximizer.select_seeds()

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