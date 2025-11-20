import numpy as np
import logging
from typing import Dict, Any, List


def _generate_powerlaw_distributions_degree_aware(
        n: int,
        degrees: np.ndarray,
        config: Dict[str, Any],  # Using a Dict for a general example
) -> Dict[str, np.ndarray]:
    """
    生成与节点度数相关的、呈幂律分布的三种行为概率。
    1. 每个行为（接受、抛弃、转发）的“基础潜力”与节点的度数成幂函数关系。
       potential = (degree + 1)^gamma
       - gamma > 0: 度数越高的节点，潜力越大（正相关）。
       - gamma < 0: 度数越高的节点，潜力越小（负相关）。
    """
    logging.info("===> Generating 'Power-law Degree-Aware' distributions...")

    gamma_succ = config.get('degree_exponent_succ', -0.1)
    gamma_dis = config.get('degree_exponent_dis', -0.3)
    gamma_tran = config.get('degree_exponent_tran', 0.5)
    random_factor = config.get('randomness_factor', 0.2)

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

    return {
        'succ_distribution': succ_norm,
        'dis_distribution': dis_norm,
        'tran_distribution': tran_norm,
    }


# ==========================
# 主执行模块：演示如何调用此接口
# ==========================
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # 1. 准备输入数据
    NUM_NODES = 7
    # 创建一个度数差异巨大的数组来观察效果
    # 节点0: 度为0 (孤立点)
    # 节点1: 度为1 (普通点)
    # 节点2: 度为10 (重要节点)
    # 节点3: 度为100 (超级枢纽)
    node_degrees = np.array([0, 1, 3, 5, 7, 11, 17])

    # 定义配置
    config_example = {
        'degree_exponent_succ': -0.05,  # 度数越高，越不倾向于“接受”
        'degree_exponent_dis': -0.5,  # 度数越高，略微不倾向于“抛弃”
        'degree_exponent_tran': 0.2,  # 度数越高，越倾向于“转发”
        'randomness_factor': 0.1  # 10%的随机扰动
    }

    # 2. 调用函数
    distributions = _generate_powerlaw_distributions_degree_aware(
        n=NUM_NODES,
        degrees=node_degrees,
        config=config_example
    )

    # -0.5 -0.1 1.0
    # Node | Degree | P(succ) | P(dis) | P(tran) | Sum
    # -----------------------------------------------------------------
    # 0 | 0 | 0.3264 | 0.3379 | 0.3357 | 1.00
    # 1 | 1 | 0.1801 | 0.2559 | 0.5640 | 1.00
    # 2 | 10 | 0.0261 | 0.0712 | 0.9027 | 1.00
    # 3 | 100 | 0.0010 | 0.0064 | 0.9926 | 1.00
    # -----------------------------------------------------------------
    # 3. 打印并验证结果
    print("\n--- Generated Distributions ---")
    print(f"{'Node':<5} | {'Degree':<7} | {'P(succ)':<10} | {'P(dis)':<10} | {'P(tran)':<10} | {'Sum':<7}")
    print("-" * 65)

    for i in range(NUM_NODES):
        s = distributions['succ_distribution'][i]
        d = distributions['dis_distribution'][i]
        t = distributions['tran_distribution'][i]
        total = s + d + t
        print(f"{i:<5} | {node_degrees[i]:<7} | {s:<10.4f} | {d:<10.4f} | {t:<10.4f} | {total:<7.2f}")

    print("-" * 65)

    # 验证
    print("\n--- Verification ---")
    # 验证1: 检查转发概率是否随度数增加而显著增加
    tran_probs = distributions['tran_distribution']
    is_tran_increasing = np.all(np.diff(tran_probs) > 0)
    print(f"转发概率是否随度数增加而单调递增? {'是' if is_tran_increasing else '否 (由于随机性可能不严格单调)'}")
    assert tran_probs[3] > tran_probs[2] > tran_probs[1] > tran_probs[0]

    # 验证2: 检查接受概率是否随度数增加而显著减小
    succ_probs = distributions['succ_distribution']
    print(
        f"接受概率是否随度数增加而单调递减? {'是' if np.all(np.diff(succ_probs) < 0) else '否 (由于随机性可能不严格单调)'}")
    assert succ_probs[3] < succ_probs[2] < succ_probs[1] < succ_probs[0]

    # 验证3: 检查每个节点的概率和是否都接近1
    sums = distributions['succ_distribution'] + distributions['dis_distribution'] + distributions['tran_distribution']
    is_sum_one = np.allclose(sums, 1.0)
    print(f"所有节点的概率和是否都为1? {'是' if is_sum_one else '否'}")
    assert is_sum_one

    print("\n结果符合预期！")