import os
import pickle
import numpy as np
from scipy.stats import truncnorm, gamma, expon
import logging
import networkx as nx  # 引入 networkx 来处理图和度
import scipy.sparse as sp
from config import ExperimentConfig
from typing import Dict


def _normalize_triplet(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> tuple:
    total = a + b + c
    out_a, out_b, out_c = np.zeros_like(a), np.zeros_like(b), np.zeros_like(c)
    mask = total != 0
    out_a[mask] = a[mask] / total[mask]
    out_b[mask] = b[mask] / total[mask]
    out_c[mask] = c[mask] / total[mask]
    return out_a, out_b, out_c


def _min_max_scale(v: np.ndarray) -> np.ndarray:
    min_val, max_val = np.min(v), np.max(v)
    range_val = max_val - min_val
    if range_val == 0:
        return np.clip(v, 0, 1)
    return (v - min_val) / range_val


def _generate_powerlaw_distributions_degree_aware(
        n: int,
        degrees: np.ndarray,
        config: "ExperimentConfig",
) -> Dict[str, np.ndarray]:

    """
    生成与节点度数相关的、呈幂律分布的三种行为概率。
    1. 每个行为（接受、抛弃、转发）的“基础潜力”与节点的度数成幂函数关系。
       potential = (degree + 1)^gamma
       - gamma > 0: 度数越高的节点，潜力越大（正相关）。
       - gamma < 0: (degree+1)^gamma 会随着 degree 的增加而减小，建立负相关关系
            degree_exponent_succ = -0.1，意味着度数越高的节点 不太可能亲自 接受信息，更倾向于传递信息
    """
    logging.info("===> Generating 'Power-law Degree-Aware' distributions...")

    rng = config.rng
    gamma_succ = config.succ_degree_influence_factor
    gamma_dis = config.dis_degree_influence_factor
    gamma_tran = config.tran_degree_influence_factor
    random_factor = config.randomness_factor# 随机扰动因子

    # 避免度数为0的节点产生问题
    degrees_smoothed = degrees.astype(float) + 1.0

    # 根据幂函数计算每个行为的潜力 -> 概率
    succ_potential = np.power(degrees_smoothed, gamma_succ)
    dis_potential = np.power(degrees_smoothed, gamma_dis)
    tran_potential = np.power(degrees_smoothed, gamma_tran)

    # 为每个节点的每个潜力值乘随机扰动因子
    succ_potential *= rng.uniform(1 - random_factor, 1 + random_factor, size=n)
    dis_potential *= rng.uniform(1 - random_factor, 1 + random_factor, size=n)
    tran_potential *= rng.uniform(1 - random_factor, 1 + random_factor, size=n)

    # 确保>0
    succ_potential = np.maximum(succ_potential, 1e-9)
    dis_potential = np.maximum(dis_potential, 1e-9)
    tran_potential = np.maximum(tran_potential, 1e-9)

    # 归一化
    potentials = np.vstack([succ_potential, dis_potential, tran_potential]).T  # (n, 3)
    total_potential_per_node = potentials.sum(axis=1, keepdims=True)
    normalized_probs = potentials / total_potential_per_node
    succ_norm = normalized_probs[:, 0]
    dis_norm = normalized_probs[:, 1]
    tran_norm = normalized_probs[:, 2]

    # 常数因子可以继续使用其他分布 不重要
    const = gamma.rvs(a=2, scale=1, size=n, random_state=rng)
    const_norm = _min_max_scale(const)

    return {
        'succ_distribution': succ_norm,
        'dis_distribution': dis_norm,
        'tran_distribution': tran_norm,
        'constantFactor_distribution': const_norm
    }


def _generate_random_distributions(
        n: int,
        degrees: np.ndarray,
        config: ExperimentConfig
) -> Dict[str, np.ndarray]:
    # 1. 读取基础参数
    dirichlet_alpha = config.random_dirichlet
    
    # 读取度数影响因子 (关键修改点)
    # 如果 config 里没写，默认给 0 (即不影响)
    s_factor = getattr(config, 'succ_degree_influence_factor', 0.0)
    d_factor = getattr(config, 'dis_degree_influence_factor', 0.0)
    t_factor = getattr(config, 'tran_degree_influence_factor', 0.0)

    logging.info(f"===> Generating 'Dirichlet' with Degree Bias...")
    logging.info(f"     Alpha: {dirichlet_alpha}")
    logging.info(f"     Factors -> Succ: {s_factor}, Dis: {d_factor}, Tran: {t_factor}")

    # 2. 生成基础概率 (Base Probabilities)
    # 比如 [500, 500, 500] 生成出来的就是 [0.33, 0.33, 0.33]
    base_probs = np.random.dirichlet(dirichlet_alpha, size=n) # Shape: (n, 3)
    
    p_succ_base = base_probs[:, 0]
    p_dis_base  = base_probs[:, 1]
    p_tran_base = base_probs[:, 2]

    # 3. 计算度数权重 (Degree Weights)
    # 使用幂律函数：Weight = (Degree + 1) ^ factor
    # +1 是为了防止度数为0时出现数学错误
    degrees_safe = degrees.astype(float) + 1.0
    
    w_succ = np.power(degrees_safe, s_factor)
    w_dis  = np.power(degrees_safe, d_factor)
    w_tran = np.power(degrees_safe, t_factor)

    # 4. 将基础概率与权重相乘得到“潜力值” (Potentials)
    # 逻辑：如果 factor > 0，度数越大，潜力越大；如果 factor < 0，度数越大，潜力越小
    potential_succ = p_succ_base * w_succ
    potential_dis  = p_dis_base  * w_dis
    potential_tran = p_tran_base * w_tran

    # 5. 重新归一化 (Re-normalization)
    # 因为乘了权重后，三者之和不再是 1，必须除以总和
    total_potential = potential_succ + potential_dis + potential_tran
    
    # 防止除以0 (极少数情况)
    total_potential[total_potential == 0] = 1e-9

    succ_distribution = potential_succ / total_potential
    dis_distribution  = potential_dis  / total_potential
    tran_distribution = potential_tran / total_potential

    # 6. 打印统计信息，确保改动生效 (调试用)
    # 检查度数最高的节点发生了什么变化
    max_deg_idx = np.argmax(degrees)
    logging.info(f"--- check node (Max Degree={degrees[max_deg_idx]}) ---")
    logging.info(f"    Base Prob: {base_probs[max_deg_idx]}")
    logging.info(f"    Final Prob: Succ={succ_distribution[max_deg_idx]:.4f}, Dis={dis_distribution[max_deg_idx]:.4f}, Tran={tran_distribution[max_deg_idx]:.4f}")

    # 7. 格式化返回
    constantFactor_distribution = np.ones(n, dtype=float)

    return {
        'succ_distribution': succ_distribution,
        'dis_distribution': dis_distribution,
        'tran_distribution': tran_distribution,
        'constantFactor_distribution': constantFactor_distribution
    }

def get_distribution_degree_aware(
        distribution_file: str,
        distribution_type: str,
        adj: nx.Graph,  # 直接传入图对象或邻接矩阵
        config: ExperimentConfig,
) -> tuple:
    """
    生成与节点度相关的概率分布
    """

    # 读取矩阵：从图或矩阵中获取节点数和度向量
    if isinstance(adj, np.ndarray):
        n = adj.shape[0]
        degrees = adj.sum(axis=1)
    elif isinstance(adj, nx.Graph):
        n = adj.number_of_nodes()
        degrees = np.array([d for n, d in adj.degree()])
    elif sp.issparse(adj):  # 添加对稀疏矩阵的支持
        n = adj.shape[0]
        degrees = np.array(adj.sum(axis=1)).flatten()
    else:
        raise TypeError("矩阵类型不支持")



    # 先找缓存
    if os.path.exists(distribution_file):
        logging.info(f"复用概率分布 文件位置: {distribution_file}")
        with open(distribution_file, 'rb') as f:
            dis_dict = pickle.load(f)
        return tuple(dis_dict.values())

    logging.info(f"首次生成度相关概率分布 分布类型: '{distribution_type}'.")


    # 生成函数注册表
    generator_registry = {
        'powerlaw': _generate_powerlaw_distributions_degree_aware,
        'random': _generate_random_distributions
        # 可以按同样模式添加 'normal', 'exponential' 等
    }

    if distribution_type not in generator_registry:
        raise ValueError(f"该分布类型暂不支持: '{distribution_type}'.")

    generator_func = generator_registry[distribution_type]
    # 将 n 和 degrees 都传递给生成函数
    dis_dict = generator_func(n, degrees, config)

    os.makedirs(os.path.dirname(distribution_file), exist_ok=True)
    with open(distribution_file, 'wb') as f:
        pickle.dump(dis_dict, f)
    logging.info(f"保存概率分布 文件位置: {distribution_file}")

    return tuple(dis_dict.values())