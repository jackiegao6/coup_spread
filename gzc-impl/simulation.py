import numpy as np
import logging
import pandas as pd
import os
from config import ExperimentConfig
import multiprocessing

def evaluate_seed_set(
    seed_list: list,
    simulation_function,  # 传入具体的单次模拟函数
    num_simulations: int,
    tran_matrix: np.ndarray,
    distributions: tuple
) -> tuple: # (mean, variance)

    # 创建一个列表来存储每一次模拟的结果
    all_simulation_results = []


    succ_dist, dis_dist, _, const_factor_dist = distributions

    for i in range(num_simulations):
        print(f"\t\t当前模拟轮次: {i}")
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


def worker_evaluate_method(
        method_name: str,
        seed_set: list,
        simulation_times: list,  # 例如 [5, 10, 50]
        tran_matrix: np.ndarray,
        distribution_list: tuple,
        single_sim_func,
        seed_num: int,
        config_dict: dict  # 建议传 dict 而不是 Config 对象，防止 pickle 问题
):
    """
    单个进程执行的任务：评估某一个方法在不同模拟次数下的表现
    """
    results = []
    logging.info(f"[Process {os.getpid()}] 开始评估方法: {method_name}, 种子数: {len(seed_set)}")

    for num_sims in simulation_times:
        # 执行评估
        E_activated, Var_activated = evaluate_seed_set(
            seed_list=seed_set,
            simulation_function=single_sim_func,
            num_simulations=num_sims,
            tran_matrix=tran_matrix,
            distributions=distribution_list
        )

        std_dev = np.sqrt(Var_activated)

        # 将结果打包成字典返回，不要在进程里写文件！
        result_data = {
            "method": method_name,
            "seed_num": seed_num,
            "num_sims": num_sims,
            "E_activated_users": E_activated,
            "variance": Var_activated,
            "std_deviation": std_dev,
            "degree_exponent_succ": config_dict.get('succ_degree_influence_factor'),
            "degree_exponent_dis": config_dict.get('dis_degree_influence_factor'),
            "degree_exponent_tran": config_dict.get('tran_degree_influence_factor')
        }
        results.append(result_data)
        logging.info(f"[Process {os.getpid()}] {method_name} - {num_sims} sims 完成: E={E_activated:.2f}")

    return results

# --- 新增：Worker 函数 (独立于类之外，方便序列化) ---
def simulation2(
        methods: list,
        seeds_list: list,
        init_tran_matrix: np.ndarray,
        usage_rate_file: str,
        distribution_list: tuple,
        simulation_times: list,
        single_sim_func,
        seed_num: int,
        config
):
    logging.info(f"多进程评估开始，共有 {len(methods)} 个方法待评估...")

    # 1. 准备多进程参数
    # 将 config 转为 dict，确保安全序列化
    config_dict = {
        'succ_degree_influence_factor': config.succ_degree_influence_factor,
        'dis_degree_influence_factor': config.dis_degree_influence_factor,
        'tran_degree_influence_factor': config.tran_degree_influence_factor
    }

    tasks = []
    for i in range(len(methods)):
        # 打包每个任务需要的参数
        tasks.append((
            methods[i],
            seeds_list[i],
            simulation_times,
            init_tran_matrix,
            distribution_list,
            single_sim_func,
            seed_num,
            config_dict
        ))

    # 2. 启动进程池
    # processes数量：通常设为 min(任务数, CPU核心数 - 1)
    # 留一个核给系统和主进程写文件
    num_workers = len(methods)

    all_results_flat = []

    logging.info(f"启动进程池，Worker数量: {num_workers}")

    with multiprocessing.Pool(processes=num_workers) as pool:
        # 使用 starmap 因为 worker 接收多个参数
        # results_list 是一个列表的列表 [[method1_results], [method2_results], ...]
        results_list = pool.starmap(worker_evaluate_method, tasks)

    # 3. 结果处理与写入 (回到主进程，单线程安全写入)
    logging.info("所有进程执行完毕，开始写入数据...")

    # 展平结果列表
    for res in results_list:
        all_results_flat.extend(res)

    if not all_results_flat:
        logging.warning("没有任何结果生成！")
        return

    # 转换为 DataFrame
    df_new = pd.DataFrame(all_results_flat)

    # 写入文件 (处理表头逻辑)
    file_exists = os.path.exists(usage_rate_file)
    if not file_exists:
        os.makedirs(os.path.dirname(usage_rate_file), exist_ok=True)

    df_new.to_csv(usage_rate_file, mode='a+', header=not file_exists, index=False, encoding='utf-8-sig')

    logging.info(f"成功写入 {len(df_new)} 条记录至 {usage_rate_file}")


# --- 修改后的主函数 ---
def simulation2_without_parallel(
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
        for num_sims in simulation_times: #[5] 注意这里是 in
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
                # "succ_base_value": config.succ_base_value,
                "degree_exponent_succ": config.succ_degree_influence_factor,
                # "dis_base_value": config.dis_base_value,
                "degree_exponent_dis": config.dis_degree_influence_factor,
                # "tran_base_value": config.tran_base_value,
                "degree_exponent_tran": config.tran_degree_influence_factor
            })

            # to_csv 会自动处理浮点数格式
            df.to_csv(usage_rate_file, mode='a+', header=not file_exists, index=False, encoding='utf-8-sig')