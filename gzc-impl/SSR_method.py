import random
import time
from multiprocessing import Pool
from collections import defaultdict
from typing import List, Dict, Set, Tuple
import numpy as np
import scipy.sparse as sp
import logging

def run_single_ssr_generation_worker(args: Tuple) -> List[Set[int]]:
    """
    单次多重 RR-Set (可能世界) 生成，修复了转移逻辑和条件采样。
    """
    (num_nodes, in_indices, in_indptr, in_data, 
     out_indices, out_indptr, out_data, 
     alpha, beta, k, root_node_v, is_optimized) = args

    # 获取当前目标节点的采纳率
    p_accept = alpha[root_node_v] if isinstance(alpha, (list, np.ndarray)) else alpha.get(root_node_v, 0)

    adoptions =[False] * k
    if is_optimized:
        # 优化采样：按论文纲要，拒绝 "全假" 事件，强制至少有一次采纳
        while not any(adoptions):
            adoptions =[random.random() < p_accept for _ in range(k)]
    else:
        # 朴素独立采样：允许出现全部为空的无用样本
        adoptions =[random.random() < p_accept for _ in range(k)]

    ssr_list =[]
    
    for j in range(k):
        if not adoptions[j]:
            ssr_list.append(set())
            continue

        rr_set = {root_node_v}
        queue = [root_node_v]
        # w_choices 记录节点 w 在当前第 j 个可能世界中的出度选择结果（惰性计算）
        w_choices = {}  

        idx = 0
        while idx < len(queue):
            curr_node = queue[idx] # 这里相当于论文中的 u
            idx += 1

            # 遍历指向当前节点的所有入邻居 w (w -> u)
            start_ptr = in_indptr[curr_node]
            end_ptr = in_indptr[curr_node + 1]

            for i in range(start_ptr, end_ptr):
                w = in_indices[i]
                
                # 在第 j 个世界中，w 尚未做出转移决策，我们为其模拟一次
                if w not in w_choices:
                    alpha_w = alpha[w] if isinstance(alpha, (list, np.ndarray)) else alpha.get(w, 0)
                    beta_w = beta[w] if isinstance(beta, (list, np.ndarray)) else beta.get(w, 0)
                    
                    # 按照模型，w 可能先自用或丢弃，这两者都会导致转移终止
                    if random.random() < (alpha_w + beta_w):
                        w_choices[w] = -1 # 不转移给任何人
                    else:
                        # 如果要转移，从出边中按照比例选择【唯一】一个邻居
                        w_start = out_indptr[w]
                        w_end = out_indptr[w+1]
                        if w_start == w_end:
                            w_choices[w] = -1
                        else:
                            choice_rand = random.random()
                            cumulative = 0.0
                            chosen = -1
                            for out_idx in range(w_start, w_end):
                                cumulative += out_data[out_idx]
                                if choice_rand <= cumulative:
                                    chosen = out_indices[out_idx]
                                    break
                            w_choices[w] = chosen

                # 如果 w 在这个世界中刚好选择了当前节点 curr_node (u)，则将其加入反向可达集
                if w_choices[w] == curr_node:
                    if w not in rr_set:
                        rr_set.add(w)
                        queue.append(w)

        ssr_list.append(rr_set)

    return ssr_list


class CouponInfluenceMaximizer:
    def __init__(self, adj: sp.csr_matrix, tranProMatrix: np.ndarray, alpha: np.ndarray, beta: np.ndarray, k: int, is_optimized: bool = True):
        self.k = k
        self.alpha = alpha
        self.beta = beta
        self.is_optimized = is_optimized

        self.num_nodes = adj.shape[0]
        self.nodes = list(range(self.num_nodes))

        # 转换为CSR以提高切片性能
        self.tran_matrix_csr = sp.csr_matrix(tranProMatrix) if isinstance(tranProMatrix, np.ndarray) else tranProMatrix

        # 【重点】构造入边矩阵 (in_csr) 和 出边矩阵 (out_csr)
        # tranProMatrix[i, j] 表示 j -> i，因此行提取代表入边，转置后代表出边
        self.in_csr = self.tran_matrix_csr.tocsr()
        self.in_indices, self.in_indptr, self.in_data = self.in_csr.indices, self.in_csr.indptr, self.in_csr.data

        self.out_csr = self.tran_matrix_csr.T.tocsr()
        self.out_indices, self.out_indptr, self.out_data = self.out_csr.indices, self.out_csr.indptr, self.out_csr.data

        self.all_ssrs: List[List[Set[int]]] =[]

        # 计算论文中的理论分布调整
        if self.is_optimized:
            self.W_v = np.zeros(self.num_nodes)
            for v in range(self.num_nodes):
                p_acc = self.alpha[v] if isinstance(self.alpha, (list, np.ndarray)) else self.alpha.get(v, 0)
                # 事件至少有一次为真的概率
                self.W_v[v] = 1.0 - (1.0 - p_acc)**self.k 
            self.W_total = np.sum(self.W_v)
            self.v_probs = self.W_v / self.W_total if self.W_total > 0 else (np.ones(self.num_nodes) / self.num_nodes)
        else:
            self.W_total = self.num_nodes # 朴素采样的总权重即为节点数

        logging.info(f"图初始化完成，节点数: {self.num_nodes}，当前模式: {'优化采样' if is_optimized else '朴素采样'}")

    def generate_rr_sets_parallel(self, N: int, workers: int = 8):
        logging.info(f"开始生成 {N} 组多重RR-Set...")
        start_time = time.time()

        # 根据是否优化，决定根节点 v 的采样方式
        if self.is_optimized:
            sampled_vs = random.choices(range(self.num_nodes), weights=self.v_probs, k=N)
        else:
            sampled_vs =[random.randint(0, self.num_nodes - 1) for _ in range(N)]

        args_list =[
            (self.num_nodes, self.in_indices, self.in_indptr, self.in_data,
             self.out_indices, self.out_indptr, self.out_data,
             self.alpha, self.beta, self.k, sampled_vs[i], self.is_optimized)
            for i in range(N)
        ]

        with Pool(processes=workers) as pool:
            self.all_ssrs = pool.map(run_single_ssr_generation_worker, args_list)

        logging.info(f"生成完毕。耗时: {time.time() - start_time:.2f} 秒。")

    def select_seeds(self) -> Tuple[List[int], float]:
        logging.info("开始基于多重RR集贪心选择最优种子...")
        start_time = time.time()
        coupon_node_map = defaultdict(lambda: defaultdict(set))
        total_samples = len(self.all_ssrs)

        # 倒排索引：快速找到节点覆盖的采样编号
        for ssr_idx, ssr_list in enumerate(self.all_ssrs):
            for coupon_j, rr_set in enumerate(ssr_list):
                if not rr_set: continue
                for node in rr_set:
                    coupon_node_map[coupon_j][node].add(ssr_idx)

        selected_seeds =[]
        covered_ssr_indices = set()

        for i in range(self.k):
            coupon_to_assign = i
            best_node, max_gain = -1, -1
            candidate_nodes = coupon_node_map[coupon_to_assign].keys()

            if candidate_nodes:
                for node in candidate_nodes:
                    if node in selected_seeds: continue
                    covered_by_node = coupon_node_map[coupon_to_assign][node]
                    current_gain = len(covered_by_node - covered_ssr_indices)
                    if current_gain > max_gain:
                        max_gain = current_gain
                        best_node = node

            if best_node == -1:
                remaining = list(set(self.nodes) - set(selected_seeds))
                best_node = remaining[0] if remaining else 0

            selected_seeds.append(best_node)
            covered_ssr_indices.update(coupon_node_map[coupon_to_assign][best_node])
            # logging.info(f"  - 券 {i + 1}/{self.k} 选中节点 {best_node}, 边际增益 {max_gain}")

        # 影响力的期望估计：由于有偏向采样，乘子变为 W_total
        estimated_influence = (len(covered_ssr_indices) / total_samples) * self.W_total
        logging.info(f"选种完毕。耗时: {time.time() - start_time:.2f} 秒。估算范围: {estimated_influence:.2f}")
        return selected_seeds, estimated_influence

def deliverers_ris_coverage(adj: sp.csr_matrix, tranProMatrix: np.ndarray, seeds_num: int, num_samples: int = 100, alpha: np.ndarray = None, beta: np.ndarray = None, is_optimized: bool = True) -> list:
    maximizer = CouponInfluenceMaximizer(adj, tranProMatrix, alpha, beta, k=seeds_num, is_optimized=is_optimized)
    maximizer.generate_rr_sets_parallel(N=num_samples)
    selected_seeds, _ = maximizer.select_seeds()
    return selected_seeds