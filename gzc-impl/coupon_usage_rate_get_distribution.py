import os
import pickle
import numpy as np
from scipy.stats import truncnorm
import logging


def get_distribution(distribution_file: str, distribution_type: str, n: int) -> tuple:

    # 先找缓存
    if os.path.exists(distribution_file):
        logging.info(f"===> Loading distributions from cache: {distribution_file}")
        with open(distribution_file, 'rb') as f:
            dis_dict = pickle.load(f)
        return tuple(dis_dict.values())

    logging.info(f"===> Cache not found. Generating new distributions of type '{distribution_type}'.")
    generator_registry = {
        'random': _generate_random_distributions,
        'poisson': _generate_poisson_distributions,
        'normal': _generate_normal_distributions,
        'exponential': _generate_exponential_distributions,
        'gamma': _generate_gamma_distributions,  
    }

    if distribution_type not in generator_registry:
        raise ValueError(f"Unknown distribution type: '{distribution_type}'. "f"Available types are: {list(generator_registry.keys())}")

    
    generator_func = generator_registry[distribution_type]
    dis_dict = generator_func(n)

    logging.info(f"===> Saving newly generated distributions to: {distribution_file}")
    os.makedirs(os.path.dirname(distribution_file), exist_ok=True)
    with open(distribution_file, 'wb') as f:
        pickle.dump(dis_dict, f)

    return tuple(dis_dict.values())


def _generate_random_distributions(n: int) -> dict:
    # n 个节点
    logging.info("===> Generating 'random' distributions...")
    tran_distribution = 0.5 + 0.2 * np.random.rand(n)
    succ_distribution = np.random.uniform(0.2, 0.3, n)
    dis_distribution = 1.0 - tran_distribution - succ_distribution # 确保和为1
    np.clip(dis_distribution, 0, None, out=dis_distribution) #  确保计算出的丢弃概率不会因为浮点数误差而变成负数
    
    constantFactor_distribution = np.ones(n, dtype=float)
    
    return {
        'succ_distribution': succ_distribution,
        'dis_distribution': dis_distribution,
        'tran_distribution': tran_distribution,
        'constantFactor_distribution': constantFactor_distribution
    }

def _generate_poisson_distributions(n: int) -> dict:
    logging.info("===> Generating 'poisson' distributions...")
    # 使用随机的lambda值 不同的lambda值 分布图不同
    lambdas = np.random.uniform(1, 10, 4)
    
    succ = np.random.poisson(lambdas[0], n).astype(float)
    dis = np.random.poisson(lambdas[1], n).astype(float)
    tran = np.random.poisson(lambdas[2], n).astype(float)
    const = np.random.poisson(lambdas[3], n).astype(float)
    
    succ_norm, dis_norm, tran_norm = _normalize_triplet(succ, dis, tran)
    const_norm = _min_max_scale(const) # 缩放常数因子 可以作为一个折扣系数来使用，例如，用户的最终影响力 = 基础影响力 * constantFactor
    
    return {
        'succ_distribution': succ_norm,
        'dis_distribution': dis_norm,
        'tran_distribution': tran_norm,
        'constantFactor_distribution': const_norm
    }

def _generate_normal_distributions(n: int) -> dict:
    logging.info("===> Generating 'normal' distributions...")
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


def _generate_exponential_distributions(n: int) -> dict:
    logging.info("===> Generating 'exponential' distributions...")
    # 指数分布的 scale 参数 (β) 是速率 (λ) 的倒数。我们为不同行为随机化 scale 参数。
    scales = np.random.uniform(0.5, 5, 4)

    succ = np.random.exponential(scales[0], n)
    dis = np.random.exponential(scales[1], n)
    tran = np.random.exponential(scales[2], n)
    const = np.random.exponential(scales[3], n)

    # 归一化，确保 succ, dis, tran 对于每个节点的概率和为1
    succ_norm, dis_norm, tran_norm = _normalize_triplet(succ, dis, tran)
    # 将常数因子缩放到 [0, 1] 区间
    const_norm = _min_max_scale(const)

    return {
        'succ_distribution': succ_norm,
        'dis_distribution': dis_norm,
        'tran_distribution': tran_norm,
        'constantFactor_distribution': const_norm
    }


def _generate_gamma_distributions(n: int) -> dict:
    logging.info("===> Generating 'gamma' distributions...")
    # 伽马分布由形状参数 (k) 和尺度参数 (θ) 定义。
    # 为了增加多样性，我们随机化这两个参数。
    shapes = np.random.uniform(1, 5, 4)
    scales = np.random.uniform(0.5, 2, 4)

    succ = np.random.gamma(shapes[0], scales[0], n)
    dis = np.random.gamma(shapes[1], scales[1], n)
    tran = np.random.gamma(shapes[2], scales[2], n)
    const = np.random.gamma(shapes[3], scales[3], n)

    # 归一化，确保 succ, dis, tran 对于每个节点的概率和为1
    succ_norm, dis_norm, tran_norm = _normalize_triplet(succ, dis, tran)
    # 将常数因子缩放到 [0, 1] 区间
    const_norm = _min_max_scale(const)

    return {
        'succ_distribution': succ_norm,
        'dis_distribution': dis_norm,
        'tran_distribution': tran_norm,
        'constantFactor_distribution': const_norm
    }

def _normalize_triplet(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> tuple:
    total = a + b + c
    out_a = np.zeros_like(a, dtype=float)
    out_b = np.zeros_like(b, dtype=float)
    out_c = np.zeros_like(c, dtype=float)
    
    mask = total != 0
    
    out_a[mask] = a[mask] / total[mask]
    out_b[mask] = b[mask] / total[mask]
    out_c[mask] = c[mask] / total[mask]
    
    return out_a, out_b, out_c

def _min_max_scale(v: np.ndarray) -> np.ndarray:
    min_val = np.min(v)
    max_val = np.max(v)
    range_val = max_val - min_val
    
    if range_val == 0:
        return np.clip(v, 0, 1)
        
    return (v - min_val) / range_val

if __name__ == '__main__':
    distribution_type = "normal"
    get_distribution_test = "/home/wen/pythonspace/coup_spread/gzc-impl/get_distribution_test/test_" + distribution_type
    get_distribution(get_distribution_test, distribution_type, 5)
