import numpy as np
import logging

def evaluate_seed_set(
    seed_set: list,
    simulation_function,  # 传入具体的单次模拟函数
    num_simulations: int,
    tran_matrix: np.ndarray,
    distributions: tuple
) -> float:
    """
    对一个给定的种子集运行多次模拟，并返回平均总影响力（激活人数）。

    Args:
        seed_set (list): 要评估的种子集。
        simulation_function: 用于单次模拟的纯函数。
        num_simulations (int): 运行模拟的总次数。
        tran_matrix (np.ndarray): 转发概率矩阵。
        distributions (tuple): 包含各种概率分布的元组。

    Returns:
        float: 在多次模拟中，平均的激活用户总数。
    """
    total_activated_count = 0
    succ_dist, dis_dist, _, const_factor_dist = distributions

    for _ in range(num_simulations):
        # 调用纯函数，它不修改任何输入，所以我们不需要 deepcopy
        success_vector = simulation_function(
            tran_matrix,
            seed_set,
            succ_dist,
            dis_dist,
            const_factor_dist
        )
        total_activated_count += np.sum(success_vector)
    
    return total_activated_count / num_simulations


def simulation(
    methods: list,
    method_deliverers: list,
    init_tran_matrix: np.ndarray,
    usage_rate_file: str,
    distribution_list: tuple,
    seed_num_list: list,
    simulation_times: list, # 替换原来的全局变量 `times`
    single_sim_func # 传入具体的单次模拟函数，如 monteCarlo_singleTime_improved
):
    """
    主评估流程，负责循环、聚合和保存结果。
    """
    # 确保文件是干净的，或者追加模式前有明确的分隔符
    with open(usage_rate_file, 'w') as f:
        f.write(f"--- New Evaluation Run ---\n")
        f.write(f"Simulation times for evaluation points: {simulation_times}\n")
    
    num_methods = len(methods)
    
    # 外层循环：遍历不同的种子数量 控制评估的种子集大小
    for seed_num in seed_num_list:
        logging.info(f"--- Evaluating for seed number: {seed_num} ---")
        
        with open(usage_rate_file, 'a') as f:
            f.write(f"\n# SeedNum:{seed_num}\n")
            
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
                
                # 调用清晰的评估器
                avg_influence = evaluate_seed_set(
                    seed_set=current_seed_set,
                    simulation_function=single_sim_func,
                    num_simulations=num_sims,
                    tran_matrix=init_tran_matrix,
                    distributions=distribution_list
                )
                
                # 计算使用率 (平均影响力 / 种子数)
                usage_rate = avg_influence / seed_num if seed_num > 0 else 0
                usage_rates_at_times.append(usage_rate)
                
                logging.info(f"    Result: Avg. Influence = {avg_influence:.2f}, Usage Rate = {usage_rate:.4f}")
            
            # 将这个方法在所有评估时间点的结果写入文件
            with open(usage_rate_file, 'a') as f:
                # 使用更易于解析的格式，如 CSV 或 JSON
                f.write(f"{method}:{','.join(f'{rate:.6f}' for rate in usage_rates_at_times)}\n")