import logging
import os
import random
import time
from multiprocessing import Pool
from typing import List, Set, Tuple
from tqdm import tqdm
import numpy as np
import scipy.sparse as sp


def _to_numpy_prob(x, n: int, name: str) -> np.ndarray:
    """将 dict/list/ndarray 统一转成 float64 numpy 向量。"""
    if x is None:
        raise ValueError(f"{name} 不能为空")
    if isinstance(x, np.ndarray):
        arr = x.astype(np.float64, copy=False)
    elif isinstance(x, list):
        arr = np.asarray(x, dtype=np.float64)
    elif isinstance(x, dict):
        arr = np.zeros(n, dtype=np.float64)
        for idx, value in x.items():
            arr[int(idx)] = float(value)
    else:
        raise TypeError(f"不支持的 {name} 类型: {type(x)}")

    if arr.shape[0] != n:
        raise ValueError(f"{name} 长度 {arr.shape[0]} 与节点数 {n} 不一致")
    return np.clip(arr, 0.0, 1.0)


def compute_path_success_probabilities(
    tranProMatrix: np.ndarray,
    alpha: np.ndarray,
    beta: np.ndarray,
    max_iter: int = 100,
    tol: float = 1e-10,
) -> np.ndarray:
    """
    计算 path-aware 成功概率 q。

    q[v] 表示：一张券从 v 出发，在当前“自用 / 丢弃 / 唯一转发”模型下，
    最终产生一次成功核销的概率。

    递推式：
        q[v] = alpha[v] + (1 - alpha[v] - beta[v]) * sum_u P(u | v) * q[u]

    其中 tranProMatrix[i, j] 表示 j -> i 的转移概率。
    因此对每个源点 v，需要取 tranProMatrix 的第 v 列；
    用矩阵形式就是 out_csr.dot(q)，其中 out_csr = tranProMatrix.T。
    """
    n = tranProMatrix.shape[0]
    alpha = _to_numpy_prob(alpha, n, "alpha")
    beta = _to_numpy_prob(beta, n, "beta")
    cont = np.clip(1.0 - alpha - beta, 0.0, 1.0)

    tran_csr = sp.csr_matrix(tranProMatrix) if not sp.issparse(tranProMatrix) else tranProMatrix.tocsr()
    out_csr = tran_csr.T.tocsr()  # out_csr[v] 给出 v 的出边分布

    q = alpha.copy()
    for _ in range(max_iter):
        next_q = alpha + cont * out_csr.dot(q)
        next_q = np.clip(next_q, 0.0, 1.0)
        if np.max(np.abs(next_q - q)) < tol:
            q = next_q
            break
        q = next_q

    return q

def _sample_root_success_flags(p_success: float, k: int, is_optimized: bool) -> List[bool]:
    """为根节点的 k 张券采样“这张券最终是否会产生一次成功核销”事件。"""
    p_success = float(np.clip(p_success, 0.0, 1.0))

    if p_success <= 0.0:
        return [False] * k

    if not is_optimized:
        return [random.random() < p_success for _ in range(k)]

    flags = [False] * k
    while not any(flags):
        flags = [random.random() < p_success for _ in range(k)]
    return flags


def run_single_ssr_generation_worker(args: Tuple) -> List[Set[int]]:
    """
    单次多重 SSR 生成。

    path-aware 版本的关键变化：
    - 老版本根事件概率只看 alpha[root]（根节点立刻自用）
    - 新版本可改为 q[root]（从根出发最终发生一次核销）

    一旦根事件成功，该样本的反向可达集仍然是“哪些节点能把券送到这个 root”。
    因为从 seed 到 root 的前缀，和从 root 往后的成功后缀是条件独立的。
    """
    (
        num_nodes,
        in_indices,
        in_indptr,
        out_indices,
        out_indptr,
        out_data,
        alpha,
        beta,
        root_event_probs,
        k,
        root_node_v,
        is_optimized,
    ) = args

    p_root_success = root_event_probs[root_node_v]
    success_flags = _sample_root_success_flags(p_root_success, k, is_optimized)

    ssr_list: List[Set[int]] = []

    for j in range(k):
        if not success_flags[j]:
            ssr_list.append(set())
            continue

        rr_set = {root_node_v}
        queue = [root_node_v]
        w_choices = {}

        idx = 0
        while idx < len(queue):
            curr_node = queue[idx]
            idx += 1

            start_ptr = in_indptr[curr_node]
            end_ptr = in_indptr[curr_node + 1]

            for i in range(start_ptr, end_ptr):
                w = in_indices[i]

                if w not in w_choices:
                    alpha_w = alpha[w]
                    beta_w = beta[w]

                    # w 如果自用或丢弃，则不可能继续把券传到 curr_node
                    if random.random() < (alpha_w + beta_w):
                        w_choices[w] = -1
                    else:
                        w_start = out_indptr[w]
                        w_end = out_indptr[w + 1]
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
                            # 数值误差兜底：若 cumulative < 1 且没命中，取最后一个邻居
                            if chosen == -1:
                                chosen = out_indices[w_end - 1]
                            w_choices[w] = chosen

                if w_choices[w] == curr_node and w not in rr_set:
                    rr_set.add(w)
                    queue.append(w)

        ssr_list.append(rr_set)

    return ssr_list

class CouponInfluenceMaximizer:
    def __init__(
        self,
        adj: sp.csr_matrix,
        tranProMatrix: np.ndarray,
        alpha: np.ndarray,
        beta: np.ndarray,
        k: int,
        is_optimized: bool = True,
        root_event_mode: str = "alpha",
        path_max_iter: int = 100,
        path_tol: float = 1e-10,
    ):
        self.k = k
        self.is_optimized = is_optimized
        self.root_event_mode = root_event_mode

        self.num_nodes = adj.shape[0]
        self.nodes = list(range(self.num_nodes))
        self.alpha = _to_numpy_prob(alpha, self.num_nodes, "alpha")
        self.beta = _to_numpy_prob(beta, self.num_nodes, "beta")

        self.tran_matrix_csr = sp.csr_matrix(tranProMatrix) if not sp.issparse(tranProMatrix) else tranProMatrix.tocsr()
        self.in_csr = self.tran_matrix_csr.tocsr()
        self.in_indices = self.in_csr.indices
        self.in_indptr = self.in_csr.indptr
        self.out_csr = self.tran_matrix_csr.T.tocsr()
        self.out_indices = self.out_csr.indices
        self.out_indptr = self.out_csr.indptr
        self.out_data = self.out_csr.data

        self.node_coverage = [[[] for _ in range(self.num_nodes)] for _ in range(self.k)]
        self.total_samples = 0

        self.root_event_probs = self._build_root_event_probs(
            mode=root_event_mode,
            path_max_iter=path_max_iter,
            path_tol=path_tol,
        )

        if self.is_optimized:
            self.W_v = 1.0 - np.power(1.0 - self.root_event_probs, self.k)
            self.W_total = float(np.sum(self.W_v))
            if self.W_total > 0:
                self.v_probs = self.W_v / self.W_total
            else:
                self.v_probs = np.ones(self.num_nodes, dtype=np.float64) / self.num_nodes
        else:
            self.W_v = np.ones(self.num_nodes, dtype=np.float64)
            self.W_total = float(self.num_nodes)
            self.v_probs = np.ones(self.num_nodes, dtype=np.float64) / self.num_nodes

        logging.info(
            "RIS 初始化完成 | nodes=%d | k=%d | optimized=%s | root_event_mode=%s | avg_root_prob=%.6f",
            self.num_nodes,
            self.k,
            self.is_optimized,
            self.root_event_mode,
            float(np.mean(self.root_event_probs)),
        )

    def _build_root_event_probs(self, mode: str, path_max_iter: int, path_tol: float) -> np.ndarray:
        mode = mode.lower()
        if mode in {"alpha", "legacy", "immediate"}:
            root_probs = self.alpha.copy()

        elif mode in {"1hop", "onehop"}:
            cont = np.clip(1.0 - self.alpha - self.beta, 0.0, 1.0)
            root_probs = self.alpha + cont * self.out_csr.dot(self.alpha)
            root_probs = np.clip(root_probs, 0.0, 1.0)

        elif mode in {"path", "path_aware", "pathaware", "multi_hop"}:
            root_probs = compute_path_success_probabilities(
                tranProMatrix=self.tran_matrix_csr,
                alpha=self.alpha,
                beta=self.beta,
                max_iter=path_max_iter,
                tol=path_tol,
            )

        else:
            raise ValueError(f"未知 root_event_mode: {mode}")

        return np.clip(root_probs, 0.0, 1.0)

    def generate_rr_sets_parallel(self, N: int, workers: int = 16, chunksize: int = 512):
        logging.info("开始生成 %d 组多重 SSR...", N)
        start_time = time.time()
        self.total_samples = N

        if self.is_optimized:
            sampled_vs = random.choices(range(self.num_nodes), weights=self.v_probs, k=N)
        else:
            sampled_vs = [random.randint(0, self.num_nodes - 1) for _ in range(N)]

        def args_generator():
            for i in range(N):
                yield (
                    self.num_nodes,
                    self.in_indices,
                    self.in_indptr,
                    self.out_indices,
                    self.out_indptr,
                    self.out_data,
                    self.alpha,
                    self.beta,
                    self.root_event_probs,
                    self.k,
                    sampled_vs[i],
                    self.is_optimized,
                )

        with Pool(processes=workers) as pool:
            # 用 tqdm 包装 imap_unordered，实时显示进度
            iterator = pool.imap_unordered(
                run_single_ssr_generation_worker, 
                args_generator(), 
                chunksize=chunksize
            )
            for ssr_idx, ssr_list in enumerate(
                tqdm(
                    iterator,
                    total=N,
                    desc="生成多重 SSR",
                    unit="组",
                    ncols=80,           # 可选：进度条宽度
                    leave=True
                )
            ):
                for coupon_j, rr_set in enumerate(ssr_list):
                    for node in rr_set:
                        self.node_coverage[coupon_j][node].append(ssr_idx)
                del ssr_list  # 及时释放内存

        for j in range(self.k):
            for v in range(self.num_nodes):
                if self.node_coverage[j][v]:
                    self.node_coverage[j][v] = np.asarray(self.node_coverage[j][v], dtype=np.int32)
                else:
                    self.node_coverage[j][v] = np.asarray([], dtype=np.int32)

        logging.info("SSR 生成完成。耗时: %.2f 秒。", time.time() - start_time)

    def select_seeds(self) -> Tuple[List[int], float]:
        logging.info("开始基于 path-aware SSR 的贪心选种...")
        start_time = time.time()

        selected_seeds: List[int] = []
        selected_set = set()
        covered_flags = np.zeros(self.total_samples, dtype=bool)

        for coupon_idx in range(self.k):
            best_node = -1
            max_gain = -1

            for node in range(self.num_nodes):
                if node in selected_set:
                    continue

                node_ssrs = self.node_coverage[coupon_idx][node]
                if node_ssrs.size == 0:
                    continue

                gain = int(np.count_nonzero(~covered_flags[node_ssrs]))
                if gain > max_gain:
                    max_gain = gain
                    best_node = node

            if best_node == -1:
                remaining = [node for node in self.nodes if node not in selected_set]
                best_node = remaining[0] if remaining else 0
                max_gain = 0

            selected_seeds.append(best_node)
            selected_set.add(best_node)

            best_node_ssrs = self.node_coverage[coupon_idx][best_node]
            if best_node_ssrs.size > 0:
                covered_flags[best_node_ssrs] = True

            logging.info(
                "  - coupon %d/%d 选中节点 %d, 边际增益 %d",
                coupon_idx + 1,
                self.k,
                best_node,
                max_gain,
            )

        total_covered = int(np.count_nonzero(covered_flags))
        estimated_influence = (total_covered / max(self.total_samples, 1)) * self.W_total
        logging.info(
            "贪心选种完成。耗时: %.2f 秒。估计影响力: %.4f",
            time.time() - start_time,
            estimated_influence,
        )
        return selected_seeds, estimated_influence
    

def deliverers_ris_coverage(
    adj: sp.csr_matrix,
    tranProMatrix: np.ndarray,
    seeds_num: int,
    num_samples: int = 100,
    alpha: np.ndarray | None = None,
    beta: np.ndarray | None = None,
    is_optimized: bool = True,
    root_event_mode: str = "alpha",
    path_max_iter: int = 100,
    path_tol: float = 1e-10,
    workers: int | None = None,
    chunksize: int = 512,
) -> list:
    maximizer = CouponInfluenceMaximizer(
        adj=adj,
        tranProMatrix=tranProMatrix,
        alpha=alpha,
        beta=beta,
        k=seeds_num,
        is_optimized=is_optimized,
        root_event_mode=root_event_mode,
        path_max_iter=path_max_iter,
        path_tol=path_tol,
    )
    maximizer.generate_rr_sets_parallel(N=num_samples, workers=workers, chunksize=chunksize)
    selected_seeds, _ = maximizer.select_seeds()
    return selected_seeds


def deliverers_ris_path_aware(
    adj: sp.csr_matrix,
    tranProMatrix: np.ndarray,
    seeds_num: int,
    num_samples: int = 100,
    alpha: np.ndarray | None = None,
    beta: np.ndarray | None = None,
    is_optimized: bool = True,
    path_max_iter: int = 100,
    path_tol: float = 1e-10,
    workers: int | None = None,
    chunksize: int = 512,
) -> list:
    """便捷别名：直接调用 path-aware root 事件版本。"""
    return deliverers_ris_coverage(
        adj=adj,
        tranProMatrix=tranProMatrix,
        seeds_num=seeds_num,
        num_samples=num_samples,
        alpha=alpha,
        beta=beta,
        is_optimized=is_optimized,
        root_event_mode="path_aware",
        path_max_iter=path_max_iter,
        path_tol=path_tol,
        workers=workers,
        chunksize=chunksize,
    )