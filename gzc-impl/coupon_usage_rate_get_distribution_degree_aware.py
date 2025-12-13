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


def _map_degree_to_behavior(
        scaled_degrees: np.ndarray,
        base_value: float,
        degree_influence_factor: float,
        min_value: float = 1e-6
) -> np.ndarray:
    """
    将缩放后的度映射为行为倾向参数，用于后续分布的生成。

    Args:
        scaled_degrees (np.ndarray): 缩放后的度向量 (通常 ∈ [0, 1])。
        base_value (float): 基础值 (低度节点的初始倾向)。
        degree_influence_factor (float): 度的影响系数，正值表示正相关，负值表示负相关。
        min_value (float, optional): 行为倾向的下限，避免 0 或负数导致无效分布参数。

    Returns:
        np.ndarray: 行为倾向参数数组 (>= min_value)。
    """

    # 计算基础倾向值
    behavior_tendency = base_value + degree_influence_factor * scaled_degrees

    # 将结果裁剪到(0, inf)以作为后续随机分布的参数
    return np.clip(behavior_tendency, min_value, None)


def _generate_poisson_distributions_degree_aware(n: int, degrees: np.ndarray, config: ExperimentConfig) -> dict:
    logging.info("===> Generating 'poisson' distributions (degree-aware)...")

    # 对度进行对数缩放，以减弱超级节点的影响
    scaled_degrees = _min_max_scale(np.log1p(degrees))

    # 设定场景：影响者模型 (高阶节点更爱转发，更少自己使用)
    # 转发概率：与度正相关
    tran_tendency = _map_degree_to_behavior(scaled_degrees, base_value=2.0, degree_influence_factor=8.0)
    # 成功概率：与度负相关
    succ_tendency = _map_degree_to_behavior(scaled_degrees, base_value=5.0, degree_influence_factor=-4.0)
    # 丢弃概率：与度负相关 (高阶节点更活跃，更少丢弃)
    dis_tendency = _map_degree_to_behavior(scaled_degrees, base_value=3.0, degree_influence_factor=-2.0)

    # 将倾向值作为泊松分布的 lambda 参数，并加入随机性
    tran = np.random.poisson(tran_tendency, n).astype(float)
    succ = np.random.poisson(succ_tendency, n).astype(float)
    dis = np.random.poisson(dis_tendency, n).astype(float)

    const = np.random.poisson(np.random.uniform(1, 10), n).astype(float)

    succ_norm, dis_norm, tran_norm = _normalize_triplet(succ, dis, tran)
    const_norm = _min_max_scale(const)

    return {
        'succ_distribution': succ_norm,
        'dis_distribution': dis_norm,
        'tran_distribution': tran_norm,
        'constantFactor_distribution': const_norm
    }


def _generate_gamma_distributions_degree_aware(n: int, degrees: np.ndarray, config: ExperimentConfig) -> dict:
    logging.info("===> Generating 'gamma' distributions (degree-aware)...")

    scaled_degrees = _min_max_scale(np.log1p(degrees))

    # 设定场景：活跃用户模型 (高阶节点在各方面都更活跃)
    # 形状参数k受度影响
    shape_tran = _map_degree_to_behavior(scaled_degrees, base_value=1.5, degree_influence_factor=3.0)
    shape_succ = _map_degree_to_behavior(scaled_degrees, base_value=1.0, degree_influence_factor=2.0)
    shape_dis = _map_degree_to_behavior(scaled_degrees, base_value=2.0, degree_influence_factor=-1.5)

    # 使用伽马分布生成
    tran = gamma.rvs(a=shape_tran, scale=1.5, size=n)
    succ = gamma.rvs(a=shape_succ, scale=1.0, size=n)
    dis = gamma.rvs(a=shape_dis, scale=2.0, size=n)
    const = gamma.rvs(a=2, scale=1, size=n)

    succ_norm, dis_norm, tran_norm = _normalize_triplet(succ, dis, tran)
    const_norm = _min_max_scale(const)

    return {
        'succ_distribution': succ_norm,
        'dis_distribution': dis_norm,
        'tran_distribution': tran_norm,
        'constantFactor_distribution': const_norm
    }


def _generate_powerlaw_distributions_degree_aware_new(
        n: int,
        degrees: np.ndarray,
        config: "ExperimentConfig",
) -> Dict[str, np.ndarray]:
    logging.info("===> Generating 'Influencer Model' (Power-law) distributions...")

    rng = config.rng
    # 读取配置
    gamma_succ = config.succ_degree_influence_factor # 强负相关
    gamma_dis = config.dis_degree_influence_factor # 强正相关
    gamma_tran = config.tran_degree_influence_factor

    # 平滑度数
    degrees_smoothed = degrees.astype(float) + 1.0

    # 1. 计算 Succ (接收概率) - 度数越高，Succ 越急剧下降
    # 归一化度数到 [0, 1]
    deg_norm = (degrees_smoothed - degrees_smoothed.min()) / (degrees_smoothed.max() - degrees_smoothed.min())
    # 使用反向幂函数：Base prob * (1 - deg_norm)^alpha
    # 边缘节点接近 Base，大V接近 0
    base_succ_prob = 0.3  # 普通人最高 30% 概率领券
    succ_raw = base_succ_prob * np.power((1.0 - deg_norm), 2.0)  # 2.0次幂衰减

    # 2. 计算 Tran (转发概率) - 度数越高，Tran 越高
    # 大V 接近 0.95，边缘节点 接近 0.1
    base_tran_prob = 0.95
    min_tran_prob = 0.1
    tran_raw = min_tran_prob + (base_tran_prob - min_tran_prob) * np.power(deg_norm, 0.5)

    # 3. 计算 Dis (丢弃概率)
    # 剩余的概率分配给 Dis，或者设定一个基础值
    # 为了简单，先生成 Succ 和 Tran，Dis = 1 - Succ - Tran (如果不为负)
    # 但为了更稳健，我们生成三个势能然后归一化

    # 重新构建势能：
    potential_succ = succ_raw
    potential_tran = tran_raw
    # 丢弃势能设为一个较小的常数，或者稍微随度数变化
    potential_dis = np.full(n, 0.05)

    # 加入随机扰动
    r_factor = config.randomness_factor
    potential_succ *= rng.uniform(1 - r_factor, 1 + r_factor, size=n)
    potential_tran *= rng.uniform(1 - r_factor, 1 + r_factor, size=n)
    potential_dis *= rng.uniform(1 - r_factor, 1 + r_factor, size=n)

    # 堆叠并归一化
    potentials = np.vstack([potential_succ, potential_dis, potential_tran]).T
    total_potentials = potentials.sum(axis=1, keepdims=True)
    probs = potentials / total_potentials

    return {
        'succ_distribution': probs[:, 0],
        'dis_distribution': probs[:, 1],
        'tran_distribution': probs[:, 2],
        'constantFactor_distribution': np.ones(n)  # 简化影响因子
    }


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
    succ_potential *= np.random.uniform(1 - random_factor, 1 + random_factor, size=n)
    dis_potential *= np.random.uniform(1 - random_factor, 1 + random_factor, size=n)
    tran_potential *= np.random.uniform(1 - random_factor, 1 + random_factor, size=n)

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

    # 常数因子可以继续使用其他分布
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
        config: ExperimentConfig  # 使用字典代替类，更通用
) -> Dict[str, np.ndarray]:
    # 从配置中读取狄利克雷分布的alpha参数 [2,2,5]
    dirichlet_alpha = config.random_dirichlet
    logging.info(f"===> Generating 'Dirichlet' distributions with alpha={dirichlet_alpha}...")

    prob = np.random.dirichlet(dirichlet_alpha, size=n)

    succ_distribution = np.round(prob[:, 0], 7)
    dis_distribution = np.round(prob[:, 1], 7)
    tran_distribution = np.round(prob[:, 2], 7)
    constantFactor_distribution = np.ones(n, dtype=float)

    return {
        'succ_distribution': succ_distribution,
        'dis_distribution': dis_distribution,
        'tran_distribution': tran_distribution,
        'constantFactor_distribution': constantFactor_distribution
    }

def _generate_powerlaw_distributions_degree_aware_old(
    n: int,
    degrees: np.ndarray,
    config: "ExperimentConfig",
) -> Dict[str, np.ndarray]:

    logging.info("===> 使用幂律分布生成四种分布")
    rng = config.rng

    # 将度数缩放一下
    scaled_degrees = _min_max_scale(np.log1p(degrees))

    # 转发(tran): 度越高，指数a越小，生成的值越倾向于1。
    # 成功(succ) & 丢弃(dis): 度越高，指数a越大，生成的值越倾向于0。

    # 指数 a 必须 > 0。
    # degree_influence 为正，实现正向关系
    tran_a_vector = _map_degree_to_behavior(scaled_degrees, base_value=config.tran_base_value, degree_influence_factor=config.tran_degree_influence_factor)
    succ_a_vector = _map_degree_to_behavior(scaled_degrees, base_value=config.succ_base_value, degree_influence_factor=config.succ_degree_influence_factor)
    dis_a_vector = _map_degree_to_behavior(scaled_degrees, base_value=config.dis_base_value, degree_influence_factor=config.dis_degree_influence_factor)

    # 使用 np.random.power 生成 [0, 1) 区间的原始倾向值
    # 传入：a指数向量 + 节点数量
    tran = rng.power(tran_a_vector, n)
    succ = rng.power(succ_a_vector, n)
    dis = rng.power(dis_a_vector, n)

    # 常数因子可以继续使用其他分布
    const = gamma.rvs(a=2, scale=1, size=n, random_state=rng)

    # 归一化原始倾向值，使其成为概率
    succ_norm, dis_norm, tran_norm = _normalize_triplet(succ, dis, tran)
    const_norm = _min_max_scale(const)

    return {
        'succ_distribution': succ_norm,
        'dis_distribution': dis_norm,
        'tran_distribution': tran_norm,
        'constantFactor_distribution': const_norm
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
        'poisson': _generate_poisson_distributions_degree_aware,
        'gamma': _generate_gamma_distributions_degree_aware,
        'powerlaw': _generate_powerlaw_distributions_degree_aware,
        'powerlaw-old': _generate_powerlaw_distributions_degree_aware_old,
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