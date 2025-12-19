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


# ========== 全局变量存储区 ==========
GLOBAL_DATA = {}

def init_worker(num_nodes, indices, indptr, data, alpha, k, succ_dist, dis_dist):
    """每个 worker 启动时运行，将大对象存入全局变量，避免 pickle 传输。"""
    global GLOBAL_DATA
    GLOBAL_DATA["num_nodes"] = num_nodes
    GLOBAL_DATA["indices"] = indices
    GLOBAL_DATA["indptr"] = indptr
    GLOBAL_DATA["data"] = data
    GLOBAL_DATA["alpha"] = alpha
    GLOBAL_DATA["k"] = k
    GLOBAL_DATA["succ_dist"] = succ_dist
    GLOBAL_DATA["dis_dist"] = dis_dist

# RIS 应该优先选择那些“能通向高价值用户”的种子，而不是“能通向很多用户”的种子
def run_single_ssr_generation_without_path(args: Tuple) -> List[Tuple[float, Set[int]]]:

    GD = GLOBAL_DATA
    num_nodes = GD["num_nodes"]
    indices = GD["indices"]
    indptr = GD["indptr"]
    data = GD["data"]
    alpha = GD["alpha"]
    k = GD["k"]
    succ_dist = GD["succ_dist"]
    dis_dist = GD["dis_dist"]

    # 1. 随机采样一个目标节点 v
    root_node_v = random.randint(0, num_nodes - 1)

    ssr_list = []  # 存储 k 张券的 RR-Set

    # 获取该 root 节点的潜在价值 (即 alpha 值)
    # 这将作为该 RR-Set 的权重
    root_value = alpha[root_node_v] if isinstance(alpha, (list, np.ndarray)) else alpha.get(root_node_v, 0)

    for _ in range(k):
        # 0. 目标节点本身的领券意愿 check
        p_accept = alpha[root_node_v] if isinstance(alpha, (list, np.ndarray)) else alpha.get(root_node_v, 0)

        if random.random() > p_accept:
            ssr_list.append((0.0, set()))
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

        # 修改返回值：不仅返回集合，还返回这个集合的"价值"
        # 也就是 root_node_v 的 alpha 值
        ssr_list.append((root_value, rr_set))

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

    
    def generate_rr_sets_parallel(self, N: int, workers: int = 4): # 默认值改为 None
        
        # 1. 动态获取合理的进程数
        # 如果用户没有指定 workers，或者指定得太离谱，自动调整为 CPU 核心数
        if workers is None or workers > cpu_count():
            workers = cpu_count()
            # 在某些云环境中 cpu_count 可能很大但被限制，保险起见可以设个上限，比如 32
            # workers = min(workers, 32) 
        
        logging.info(f"\n开始生成 {N} 组 SSR (并行进程: {workers})...")
        start_time = time.time()

        with Pool(
            processes=workers,
            initializer=init_worker,
            initargs=(
                self.num_nodes,
                self.csr_indices,
                self.csr_indptr,
                self.csr_data,
                self.alpha,
                self.k,
                self.succ_dist,
                self.dis_dist,
            )
        ) as pool:
            # 2. 使用 chunksize 优化任务分发
            # 如果 N 很大（如 10000），不要让 chunksize 默认为 1，否则通信开销大
            # 自动计算 chunksize，让每个进程分摊到合适的任务量
            chunk_size = max(1, N // (workers * 4))
            
            self.all_ssrs = pool.map(
                run_single_ssr_generation_without_path, 
                range(N), 
                chunksize=chunk_size
            )

        end_time = time.time()
        logging.info(f"生成完毕。耗时: {end_time - start_time:.2f} 秒。")


    def select_seeds(self) -> Tuple[List[int], float]:
        logging.info("\n开始选择最优种子 (Weighted Greedy) ...")
        
        # 1. 构建倒排索引，同时存储权重
        # coupon_node_map[coupon_idx][node_id] = { (ssr_idx, weight), ... }
        coupon_node_map = defaultdict(lambda: defaultdict(list))
        
        # 记录每个 SSR 的权重，方便后续查找
        # ssr_weights[ssr_idx] = weight
        # 注意：这里需要重新理一下结构，因为现在 ssr_list 里的结构变了
        
        # 假设 self.all_ssrs 是 N 个 list，每个 list 包含 k 个 (weight, set)
        
        total_utility_potential = 0.0
        
        # 扁平化处理，给每个最小单元 (sample_idx, coupon_idx) 一个唯一 ID
        # 并记录其权重
        element_weights = {} # (sample_i, coupon_j) -> weight
        
        for sample_i, ssr_k_list in enumerate(self.all_ssrs):
            for coupon_j, (weight, rr_set) in enumerate(ssr_k_list):
                if not rr_set: continue
                
                # 记录这个 RR-Set 的价值
                element_id = (sample_i, coupon_j)
                element_weights[element_id] = weight
                total_utility_potential += weight
                
                for node in rr_set:
                    coupon_node_map[coupon_j][node].append(element_id)

        selected_seeds = []
        covered_elements = set() # 记录已经覆盖了哪些 (sample_i, coupon_j)
        current_total_gain = 0.0

        for i in range(self.k):
            coupon_to_assign = i
            best_node = -1
            max_gain = -1.0

            candidate_nodes = coupon_node_map[coupon_to_assign].keys()
            
            if not candidate_nodes:
                 # ... 随机填充逻辑 ...
                 pass
            else:
                for node in candidate_nodes:
                    if node in selected_seeds: continue
                    
                    # 获取该节点能覆盖的所有 SSR ID
                    covered_element_ids = coupon_node_map[coupon_to_assign][node]
                    
                    # 计算增益：新覆盖的 SSR 的权重之和
                    gain = 0.0
                    for eid in covered_element_ids:
                        if eid not in covered_elements:
                            gain += element_weights[eid]
                    
                    if gain > max_gain:
                        max_gain = gain
                        best_node = node
            
            # ... 选中 best_node ...
            selected_seeds.append(best_node)
            
            # 更新覆盖状态
            if best_node != -1:
                new_covered = coupon_node_map[coupon_to_assign][best_node]
                for eid in new_covered:
                    if eid not in covered_elements:
                        covered_elements.add(eid)
                        current_total_gain += element_weights[eid]

            logging.info(f"  - 券 {i+1} 选中 {best_node}, 增益: {max_gain:.4f}")

        return selected_seeds, current_total_gain




#接口函数
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