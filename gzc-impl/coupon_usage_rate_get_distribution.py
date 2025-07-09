import os
import pickle
import numpy as np
from scipy.stats import truncnorm
import logging

def _normalize_triplet(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> tuple:
    """将三个向量归一化，使得它们的元素和为1。"""
    total = a + b + c
    # 修正逻辑：如果total为0，所有分量都应为0。
    # 我们创建一个默认的输出，然后只在total!=0的地方进行除法。
    out_a = np.zeros_like(a, dtype=float)
    out_b = np.zeros_like(b, dtype=float)
    out_c = np.zeros_like(c, dtype=float)
    
    mask = total != 0
    
    out_a[mask] = a[mask] / total[mask]
    out_b[mask] = b[mask] / total[mask]
    out_c[mask] = c[mask] / total[mask]
    
    return out_a, out_b, out_c

def _min_max_scale(v: np.ndarray) -> np.ndarray:
    """将向量进行Min-Max归一化到[0, 1]区间。"""
    min_val = np.min(v)
    max_val = np.max(v)
    range_val = max_val - min_val
    
    if range_val == 0:
        return np.clip(v, 0, 1)
        
    return (v - min_val) / range_val

def _generate_random_distributions(n: int) -> dict:
    """生成基于 'random' 策略的分布。"""
    logging.info("Generating 'random' distributions...")
    tran_distribution = 0.5 + 0.2 * np.random.rand(n)
    succ_distribution = np.random.uniform(0.2, 0.3, n)
    dis_distribution = 1.0 - tran_distribution - succ_distribution # 确保和为1
    np.clip(dis_distribution, 0, None, out=dis_distribution) # 确保丢弃概率不为负
    
    constantFactor_distribution = np.ones(n, dtype=float)
    
    return {
        'succ_distribution': succ_distribution,
        'dis_distribution': dis_distribution,
        'tran_distribution': tran_distribution,
        'constantFactor_distribution': constantFactor_distribution
    }

def _generate_poisson_distributions(n: int) -> dict:
    """生成基于 'poisson' 策略的分布。"""
    logging.info("Generating 'poisson' distributions...")
    # 使用随机的lambda值
    lambdas = np.random.uniform(1, 10, 4)
    
    succ = np.random.poisson(lambdas[0], n).astype(float)
    dis = np.random.poisson(lambdas[1], n).astype(float)
    tran = np.random.poisson(lambdas[2], n).astype(float)
    const = np.random.poisson(lambdas[3], n).astype(float)
    
    succ_norm, dis_norm, tran_norm = _normalize_triplet(succ, dis, tran)
    const_norm = _min_max_scale(const)
    
    return {
        'succ_distribution': succ_norm,
        'dis_distribution': dis_norm,
        'tran_distribution': tran_norm,
        'constantFactor_distribution': const_norm
    }

def _generate_normal_distributions(n: int) -> dict:
    """生成基于 'normal' 策略的分布。"""
    logging.info("Generating 'normal' distributions...")
    # 从截断正态分布中采样
    succ = truncnorm.rvs(0, np.inf, loc=1, scale=1, size=n)
    dis = truncnorm.rvs(0, np.inf, loc=1, scale=1, size=n)
    tran = truncnorm.rvs(0, np.inf, loc=1, scale=1, size=n)
    const = truncnorm.rvs(0, np.inf, loc=0, scale=1, size=n)
    
    succ_norm, dis_norm, tran_norm = _normalize_triplet(succ, dis, tran)
    const_norm = _min_max_scale(const)
    
    return {
        'succ_distribution': succ_norm,
        'dis_distribution': dis_norm,
        'tran_distribution': tran_norm,
        'constantFactor_distribution': const_norm
    }


def get_distribution(distribution_file: str, distribution_type: str, n: int) -> tuple:
    """
    加载或生成实验所需的概率分布。
    这是一个职责清晰、可扩展的版本。
    """
    # 1. 优先从缓存加载
    if os.path.exists(distribution_file):
        logging.info(f"Loading distributions from cache: {distribution_file}")
        with open(distribution_file, 'rb') as f:
            dis_dict = pickle.load(f)
        return tuple(dis_dict.values())

    # 2. 如果缓存不存在，使用注册表模式生成
    logging.info(f"Cache not found. Generating new distributions of type '{distribution_type}'.")
    generator_registry = {
        'random': _generate_random_distributions,
        'poisson': _generate_poisson_distributions,
        'normal': _generate_normal_distributions,
    }

    if distribution_type not in generator_registry:
        raise ValueError(f"Unknown distribution type: '{distribution_type}'. "
                         f"Available types are: {list(generator_registry.keys())}")
    
    # 调用正确的生成函数
    generator_func = generator_registry[distribution_type]
    dis_dict = generator_func(n)

    # 3. 保存到缓存以备后用
    logging.info(f"Saving newly generated distributions to: {distribution_file}")
    with open(distribution_file, 'wb') as f:
        pickle.dump(dis_dict, f)
        # f"{self.data_prefix}/distribution-{self.data_set}/distri-{self.distribution}_constantFactor{self.constant_factor_distri}_seedNum{m}.pkl"

    return tuple(dis_dict.values())
