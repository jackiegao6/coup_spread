import numpy as np
import logging
import pandas as pd
import os
from config import ExperimentConfig

def evaluate_seed_set(
    seed_list: list,
    simulation_function,  # 传入具体的单次模拟函数
    num_simulations: int,
    tran_matrix: np.ndarray,
    distributions: tuple
) -> tuple: # (mean, variance)

    # 创建一个列表来存储每一次模拟的结果
    all_simulation_results = []

    total_activated_users_by_seeds = 0
    succ_dist, dis_dist, _, const_factor_dist = distributions

    for _ in range(num_simulations):
        success_vector = simulation_function(
            tran_matrix,
            seed_list,
            succ_dist,
            dis_dist,
            const_factor_dist
        )
        single_activated_users  = np.sum(success_vector)
        all_simulation_results.append(single_activated_users)

    E_activated_users = np.mean(all_simulation_results)
    Var_activated_users = np.var(all_simulation_results)
    return E_activated_users, Var_activated_users


def simulation2(
    methods: list,
    seeds_list: list,
    init_tran_matrix: np.ndarray,
    usage_rate_file: str,
    distribution_list: tuple,
    simulation_times: list,
    single_sim_func, # 传入具体的单次模拟函数，如 monteCarlo_singleTime_improved
    seed_num: int,
    config: ExperimentConfig
):
    # 评估函数
    logging.info(f"评估函数开始")
    logging.info(f"当前模拟次数: {simulation_times}\n")

    num_methods = len(methods)
    logging.info(f"当前评估种子数: {seed_num} ---")

    # 中层循环：遍历不同的方法 控制评估的算法
    for i in range(num_methods):
        method = methods[i]
        # 获取当前方法对应的完整种子集
        current_seed_set = seeds_list[i]

        logging.info(f"  Evaluating method: '{method}' with {len(current_seed_set)} seeds.")
        # 内层循环： 控制评估的精度（模拟次数）
        for num_sims in simulation_times: #[50000, 100000] 注意这里是 in
            logging.info(f"    Running {num_sims} simulations...")

            E_activated_users, Var_activated_users = evaluate_seed_set(
                seed_list=current_seed_set,
                simulation_function=single_sim_func,
                num_simulations=num_sims,
                tran_matrix=init_tran_matrix,
                distributions=distribution_list
            )

            std_dev = np.sqrt(Var_activated_users)  # Standard Deviation, 标准差
            logging.info(f"    Result: E[Users] = {E_activated_users:.4f}, Var[Users] = {Var_activated_users:.4f}, Std_Dev = {std_dev:.4f}")

            file_exists = os.path.exists(usage_rate_file)
            if not file_exists: os.makedirs(os.path.dirname(usage_rate_file), exist_ok=True)

            df = pd.DataFrame({
                "method": [method],
                "seed_num": [seed_num],
                "num_sims": [num_sims],
                "E_activated_users": [E_activated_users],  # 保持为浮点数
                "variance": [Var_activated_users],
                "std_deviation": [std_dev],
                "tran_base_value": config.tran_base_value,
                "tran_degree_influence_factor": config.tran_degree_influence_factor,
                "succ_base_value": config.succ_base_value,
                "succ_degree_influence_factor": config.succ_degree_influence_factor,
                "dis_base_value": config.dis_base_value,
                "dis_degree_influence_factor": config.dis_degree_influence_factor
            })

            # to_csv 会自动处理浮点数格式
            df.to_csv(usage_rate_file, mode='a', header=not file_exists, index=False, encoding='utf-8-sig')