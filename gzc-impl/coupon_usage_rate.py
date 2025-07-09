import os
import time
import pickle
import logging
from typing import List

from config import ExperimentConfig
import single_deliverer
import get_coupon_deliverers
import get_coupon_users
import coupon_usage_rate_get_distribution
import get_coupon_usage_rate_simulation



def load_experiment_data(config: ExperimentConfig):
    """加载所有实验所需的数据。"""
    logging.info(f"Loading data for dataset: {config.data_set}")
    with open(config.adj_file, 'rb') as f:
        adj = pickle.load(f)
    n = adj.shape[0]


    
    distribution_list = coupon_usage_rate_get_distribution.get_distribution(config.distribution_file, config.distribution, n)
    succ_dist, dis_dist, tran_dist, const_factor_dist = distribution_list
    
    init_tran_matrix, D = single_deliverer.getTranProMatrix(adj, tran_dist)
    
    return {
        "adj": adj,
        "n": n,
        "distributions": distribution_list,
        "init_tran_matrix": init_tran_matrix,
        "D": D
    }


def get_seed_sets(methods: list, config: ExperimentConfig, data: dict):
    """获取所有方法的种子集，优先从缓存加载，否则实时计算。"""
    if os.path.exists(config.deliverers_cache_file):
        logging.info(f"Loading seed sets from cache: {config.deliverers_cache_file}")

    
    logging.info("Cache not found. Calculating seed sets...")
    
    m = config.seed_num_list[-1]# todo gzc:why -1??

    selector_dict = {
        'deliverers_theroy': lambda: get_coupon_deliverers.deliverers_theroy(
            data["n"], m, data["init_tran_matrix"], *data["distributions"], config.personalization, data["D"]),
        'monterCarlo': lambda: get_coupon_deliverers.deliverers_monteCarlo(m=m,
                                                                           init_tranProMatrix=data["init_tran_matrix"],
                                                                           succ_distribution=data["distributions"][0],
                                                                           dis_distribution=data["distributions"][1],
                                                                           constantFactor_distribution=data["distributions"][3],
                                                                           L=config.monte_carlo_L,
                                                                           personalization=config.personalization),
        'random': lambda: get_coupon_deliverers.deliverers_random(data["n"], m),
        'degreeTopM': lambda: get_coupon_deliverers.deliverers_degreeTopM(data["adj"], m),
        # ... 其他方法也用lambda封装
    }

    method_to_seeds = {}
    method_to_runtime = {}
    for method in methods:
        if method not in selector_dict:
            logging.warning(f"Method '{method}' not found in registry. Skipping.")
            continue
        
        start_time = time.time()
        logging.info(f"Running method: {method}")
        
        # 调用注册表中对应的lambda函数
        seeds = selector_dict[method]()
        method_to_seeds[method] = seeds
        
        end_time = time.time()
        method_to_runtime[method] = end_time - start_time
        logging.info(f"Method {method} finished in {method_to_runtime[method]:.2f} seconds.")

    # 缓存结果
    # ... (此处省略写入缓存文件逻辑)
    with open(config.deliverers_cache_file, 'a+') as file:# '/root/autodl-tmp/data-processed/deliverers_cora_distrirandom_constantFactorrandom_monteCarloL5_seedNum10.txt'
                for key, value in method_to_seeds.items():
                    file.write(f'{key}:{value}\n')
                for key, value in method_to_runtime.items():
                    file.write(f'{key}:{value}\n')
    
    return method_to_seeds, method_to_runtime

def run_evaluation(method_to_seeds: dict, config: ExperimentConfig, data: dict):
    """根据个性化策略，运行评估模拟。"""
    logging.info(f"Starting evaluation with personalization: {config.personalization}")
    
    evaluation_dict = {
        'None': get_coupon_usage_rate_simulation.simulation,
        'firstUnused': get_coupon_usage_rate_simulation.simulation,
        'firstDiscard': get_coupon_usage_rate_simulation.simulation,
    }
    
    if config.personalization not in evaluation_dict:
        raise ValueError(f"Unknown personalization type: {config.personalization}")

    # 准备写入文件
    with open(config.usage_rate_file, 'a+') as f:
        f.write(f'times:{config.simulation_times}\n')

    # 调用正确的评估函数
    evaluation_func = evaluation_dict[config.personalization]
    methods = list(method_to_seeds.keys())
    deliverers = list(method_to_seeds.values())
    
    evaluation_func(methods=methods,
                    method_deliverers=deliverers,
                    init_tran_matrix=data["init_tran_matrix"],
                    usage_rate_file=config.usage_rate_file,
                    distribution_list=data["distributions"],
                    seed_num_list=config.seed_num_list,
                    simulation_times=config.simulation_times,
                    single_sim_func=get_coupon_users.monteCarlo_singleTime_improved)
    logging.info(f"Evaluation finished. Results saved to {config.usage_rate_file}")

def run_coupon_experiment(config: ExperimentConfig):

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # 1. 加载数据
    experiment_data = load_experiment_data(config)
    # 获得总节点数 n 
    n = experiment_data["n"]
    seedNumList = create_seed_num_list(total_nodes=n, num_steps=10, scale_factor=1000)
    my_config.seed_num_list = seedNumList
    
    # 2. 获取种子集
    # 这里可以添加对 'new' method_type 的处理逻辑，如果需要的话
    # methods = config.methods
    # method_to_seeds, _ = get_seed_sets(methods, config, experiment_data)
    
    # if not method_to_seeds:
    #     logging.error("No seed sets were generated. Aborting.")
    #     return

    # 3. 运行评估 [2465, 119, 2093, 997, 1613, 1560, 2089, 1101, 2501, 1312, 840, 2263, 1976, 523, 895, 453, 1958, 515, 72, 1601, 704, 1307, 545, 2475, 2243, 971, 702]
    method_to_seeds = {
    'monterCarlo': [2465, 119, 2093, 997, 1613, 1560, 2089, 1101, 2501,
                    1312, 840, 2263, 1976, 523, 895, 453, 1958, 515,
                    72, 1601, 704, 1307, 545, 2475, 2243, 971, 702]}

    run_evaluation(method_to_seeds, config, experiment_data)

def create_seed_num_list(
    total_nodes: int, 
    num_steps: int, 
    scale_factor: int = 1000
) -> List[int]:
    """
    根据总节点数和指定的步数，生成一个种子数量的列表。

    例如：total_nodes=10000, num_steps=10, scale_factor=1000
    会生成代表 0.1%, 0.2%, ..., 1.0% 节点数的列表 [10, 20, ..., 100]。

    Args:
        total_nodes (int): 网络中的总节点数 (n)。
        num_steps (int): 要生成的种子数量层级数 (例如，10个层级)。
        scale_factor (int): 用于计算比例的分母。默认为1000，表示千分比。

    Returns:
        list[int]: 一个包含不同种子数量的整数列表。
    """
    if total_nodes <= 0:
        return []
        
    seed_list = [round(total_nodes * i / scale_factor) for i in range(1, num_steps + 1)]
    
    # 去除可能因四舍五入产生的重复值，并确保列表非空
    unique_seeds = sorted(list(set(seed_list)))
    return [s for s in unique_seeds if s > 0]



if __name__ == '__main__':
    # 所有配置都集中在这里
    my_config = ExperimentConfig(
        data_set='cora',
        simulation_times=[10, 50], #[1000, 5000]
        methods=['monterCarlo'], # ['deliverers_theroy', 'monteCarlo', 'degreeTopM']
        seed_num_list=None,
        monte_carlo_L=5,
        distribution='random',
        constant_factor_distri='random',
        personalization='None',# firstUnused
        data_prefix='/root/autodl-tmp/data-processed',
        method_type='None' # new
    )
    run_coupon_experiment(my_config)