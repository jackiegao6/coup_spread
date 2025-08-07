import numpy as np
import logging
import pandas as pd
import os

def evaluate_seed_set(
    seed_list: list,
    simulation_function,  # 传入具体的单次模拟函数
    num_simulations: int,
    tran_matrix: np.ndarray,
    distributions: tuple
) -> float:

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
        single_activated_users_by_seeds = np.sum(success_vector)
        total_activated_users_by_seeds += single_activated_users_by_seeds
    
    E_activated_users_by_all_seeds = total_activated_users_by_seeds / num_simulations
    return E_activated_users_by_all_seeds


def simulation2(
    methods: list,
    seeds_list: list,
    init_tran_matrix: np.ndarray,
    usage_rate_file: str,
    distribution_list: tuple,
    simulation_times: list,
    single_sim_func, # 传入具体的单次模拟函数，如 monteCarlo_singleTime_improved
    seed_num: int
):
    logging.info(f"--- New Evaluation Run ---\n")
    logging.info(f"Simulation times for evaluation points: {simulation_times}\n")

    num_methods = len(methods)
    logging.info(f"--- Evaluating for seed number: {seed_num} ---")
    # 中层循环：遍历不同的方法 控制评估的算法
    for i in range(num_methods):
        method = methods[i]
        # 获取当前方法对应的完整种子集
        current_seed_set = seeds_list[i]

        logging.info(f"  Evaluating method: '{method}' with {len(current_seed_set)} seeds.")
        # 内层循环： 控制评估的精度（模拟次数）
        for num_sims in simulation_times: #[50000, 100000] 注意这里是 in
            logging.info(f"    Running {num_sims} simulations...")

            E_activated_users_by_all_seeds = evaluate_seed_set(
                seed_list=current_seed_set,
                simulation_function=single_sim_func,
                num_simulations=num_sims,
                tran_matrix=init_tran_matrix,
                distributions=distribution_list
            )

            logging.info(f"    Result: E_activated_users_by_all_seeds = {E_activated_users_by_all_seeds:.2f}")

            # 将这个方法在所有评估时间点的结果写入文件
            file_exists = os.path.exists(usage_rate_file)
            if not file_exists: os.makedirs(os.path.dirname(usage_rate_file), exist_ok=True)

            df = pd.DataFrame({"method": [method], "seed_num": [seed_num], "num_sims": [num_sims], "E_activated_users": [f"{E_activated_users_by_all_seeds:.4f}"]})
            df.to_csv(usage_rate_file, mode='a', header= not file_exists, index=False, encoding='utf-8-sig')


#deprecated
def simulation(
    methods: list,
    method_deliverers: list,
    init_tran_matrix: np.ndarray,
    usage_rate_file: str,
    distribution_list: tuple,
    seed_num_list: list, # deprecated
    simulation_times: list,
    single_sim_func # 传入具体的单次模拟函数，如 monteCarlo_singleTime_improved
):
    logging.info(f"--- New Evaluation Run ---\n")
    logging.info(f"Simulation times for evaluation points: {simulation_times}\n")

    num_methods = len(methods)

    # 外层循环：遍历不同的种子数量 控制评估的种子集大小
    for seed_num in seed_num_list:
        logging.info(f"--- Evaluating for seed number: {seed_num} ---")

        # 中层循环：遍历不同的方法 控制评估的算法
        for i in range(num_methods):
            method = methods[i]
            # 获取当前种子数量对应的完整种子集
            current_seed_set = method_deliverers[i][:seed_num]

            logging.info(f"  Evaluating method: '{method}' with {len(current_seed_set)} seeds.")

            usage_rates_at_times = []

            # 内层循环： 控制评估的精度（模拟次数）
            for num_sims in simulation_times:
                logging.info(f"    Running {num_sims} simulations...")

                avg_influence = evaluate_seed_set(
                    seed_set=current_seed_set,
                    simulation_function=single_sim_func,
                    num_simulations=num_sims,
                    tran_matrix=init_tran_matrix,
                    distributions=distribution_list
                )

                # 计算使用率 (平均影响力 / 种子数)
                usage_rate = avg_influence / seed_num if seed_num > 0 else 0

                logging.info(f"    Result: Avg. Influence = {avg_influence:.2f}, Usage Rate = {usage_rate:.4f}")

                # 将这个方法在所有评估时间点的结果写入文件
                file_exists = os.path.exists(usage_rate_file)
                if not file_exists:
                    os.makedirs(os.path.dirname(usage_rate_file), exist_ok=True)

                df = pd.DataFrame({"method": [method], "seed_num": [seed_num], "num_sims": [num_sims], "rate": [f"{usage_rate:.4f}"]})
                df.to_csv(usage_rate_file, mode='a', header= not file_exists, index=False, encoding='utf-8-sig')
