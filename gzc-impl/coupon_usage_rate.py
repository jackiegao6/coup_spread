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

def run_coupon_experiment(config: ExperimentConfig):
    '''主控制函数，负责按顺序调用其他模块 将种子集的计算耗时和最终的评估结果保存到文件中，以便后续分析和绘图'''

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # 1. 加载数据
    adj_and_n = load_experiment_data(config)
    n = adj_and_n["n"]

    seedNumList = create_seed_num_list(total_nodes=n, num_steps=config.num_steps, scale_factor=config.scale_factor)
    my_config.seed_num_list = seedNumList
    experiment_data = load_experiment_data_get_seedsnum(config=config, adj=adj_and_n["adj"], n=adj_and_n["n"])
    
    # 2. 获取种子集
    methods = config.methods
    method_to_seeds, _ = get_seed_sets(methods, config, experiment_data)
    
    if not method_to_seeds:
        logging.error("No seed sets were generated. Aborting.")
        return

    # 3. 运行评估 [2465, 119, 2093, 997, 1613, 1560, 2089, 1101, 2501, 1312, 840, 2263, 1976, 523, 895, 453, 1958, 515, 72, 1601, 704, 1307, 545, 2475, 2243, 971, 702]
    run_evaluation(method_to_seeds, config, experiment_data)


def load_experiment_data(config: ExperimentConfig):
    """从文件中加载实验所需的所有基础数据，包括社交网络图（邻接矩阵）、各种概率分布等"""
    logging.info(f"===> Loading data for dataset: {config.data_set}")
    with open(config.adj_file, 'rb') as f:
        adj = pickle.load(f)# 邻接矩阵
    n = adj.shape[0]

    return {
        "adj": adj,
        "n": n
    }

def load_experiment_data_get_seedsnum(config: ExperimentConfig, adj, n):
    distribution_list = coupon_usage_rate_get_distribution.get_distribution(config.distribution_file(m=config.seed_num_list[-1]), config.distribution_type, n)
    succ_dist, dis_dist, tran_dist, const_factor_dist = distribution_list
    
    init_tran_matrix, D = single_deliverer.getTranProMatrix(adj, tran_dist)
    
    return {
        "adj": adj,
        "n": n,
        "distributions": distribution_list,
        "init_tran_matrix": init_tran_matrix,
        "D": D
    }



def create_seed_num_list(
    total_nodes: int, 
    num_steps: int, 
    scale_factor: int = 1000
) -> List[int]:
    """
    根据总节点数和指定的步数，生成一系列不同大小的种子集规模

    根据网络的总节点数 total_nodes 和一个比例因子 scale_factor，生成从 1/scale_factor 到 num_steps/scale_factor 比例的节点数量
    total_nodes=10000, num_steps=10, scale_factor=1000
    会生成代表 0.1%, 0.2%, ..., 1.0% 节点数的列表 [10, 20, ..., 100]。

    Args:
        total_nodes (int): 网络中的总节点数 (n)。
        num_steps (int): 要生成的种子数量层级数 (例如，10个层级)。
        scale_factor (int): 用于计算比例的分母。默认为1000，表示千分比。

    Returns:
        list[int]: 一个包含不同种子数量的整数列表。
    """
        
    seed_list = [round(total_nodes * i / scale_factor) for i in range(1, num_steps + 1)]
    
    # 去除可能因四舍五入产生的重复值，并确保列表非空
    unique_seeds = sorted(list(set(seed_list)))
    return [s for s in unique_seeds]


def get_seed_sets(methods: list, config: ExperimentConfig, data: dict):
    """ 使用多种不同的算法（理论方法、蒙特卡洛、随机、度中心性）来计算出应该选择哪些用户作为种子集 todo RR"""
    
    m = config.seed_num_list[-1]# 避免重复计算

    selector_dict = { # 根据传入的 method 字符串，调用对应的种子选择函数
        'theroy': lambda: get_coupon_deliverers.deliverers_theroy(
            data["n"], m, data["init_tran_matrix"], *data["distributions"], config.personalization, data["D"]),
        'monterCarlo': lambda: get_coupon_deliverers.deliverers_monteCarlo(m=m,
                                                                           init_tranProMatrix=data["init_tran_matrix"],
                                                                           succ_distribution=data["distributions"][0],
                                                                           dis_distribution=data["distributions"][1],
                                                                           constantFactor_distribution=data["distributions"][3],
                                                                           L=config.monte_carlo_L,
                                                                           personalization=config.personalization),
        'random': lambda: get_coupon_deliverers.deliverers_random(data["n"], m), # 基线方法
        'degreeTopM': lambda: get_coupon_deliverers.deliverers_degreeTopM(data["adj"], m), # 基线方法
        'pageRank': lambda: get_coupon_deliverers.deliverers_pageRank(data["adj"], m),
        'succPro': lambda: get_coupon_deliverers.deliverers_succPro(succ_distribution=data['distributions'][0], m=m),
        '1_neighbor': lambda: get_coupon_deliverers.deliverers_1_neighbor(succ_distribution=data['distributions'][0], init_tranProMatrix=data['init_tran_matrix'],m=m),

    }

    method_to_seeds = {}
    method_to_runtime = {}
    for method in methods:
        if method not in selector_dict:
            logging.warning(f"===> Method '{method}' not found in registry. Skipping.")
            continue

        if os.path.exists(config.deliverers_cache_file(method=method, m=config.seed_num_list[-1])):
            logging.info(f"===> Loading seed sets from cache: {config.deliverers_cache_file(method=method,m=config.seed_num_list[-1])}")
        logging.info("===> Cache not found. Calculating seed sets...")
        
        
        start_time = time.time()
        logging.info(f"===> Running method: {method}")
        
        seeds = selector_dict[method]()
        method_to_seeds[method] = seeds
        
        end_time = time.time()
        method_to_runtime[method] = end_time - start_time
        logging.info(f"===> Method {method} finished in {method_to_runtime[method]:.2f} seconds.")

        deliverers_cache_file = config.deliverers_cache_file(   method=method,
                                                                m=config.seed_num_list[-1])
        
        os.makedirs(os.path.dirname(deliverers_cache_file), exist_ok=True)
        with open(deliverers_cache_file, 'a+') as file:
            for key, value in method_to_seeds.items():
                file.write(f'{key}:{value}\n')
  
    
    return method_to_seeds, method_to_runtime

def run_evaluation(method_to_seeds: dict, config: ExperimentConfig, data: dict):
    """对每种算法选出的种子集，通过大量的蒙特卡洛模拟来评估效果，即计算最终的“优惠券使用率”"""
    logging.info(f"Starting evaluation with personalization: {config.personalization}")
    
    evaluation_dict = {
        'None': get_coupon_usage_rate_simulation.simulation,
        'firstUnused': get_coupon_usage_rate_simulation.simulation,
        'firstDiscard': get_coupon_usage_rate_simulation.simulation,
    }
    
    if config.personalization not in evaluation_dict:
        raise ValueError(f"Unknown personalization type: {config.personalization}")

    usage_rate_file = config.usage_rate_file(m=config.seed_num_list[-1])
    os.makedirs(os.path.dirname(usage_rate_file), exist_ok=True)
    with open(usage_rate_file, 'a+') as f:
        f.write(f'times:{config.simulation_times}\n')

    evaluation_func = evaluation_dict[config.personalization]
    methods = list(method_to_seeds.keys())
    deliverers = list(method_to_seeds.values())
    
    evaluation_func(methods=methods,
                    method_deliverers=deliverers,
                    init_tran_matrix=data["init_tran_matrix"],
                    usage_rate_file=config.usage_rate_file(m=config.seed_num_list[-1]),
                    distribution_list=data["distributions"],
                    seed_num_list=config.seed_num_list,
                    simulation_times=config.simulation_times,
                    single_sim_func=get_coupon_users.monteCarlo_singleTime_improved)
    logging.info(f"Evaluation finished. Results saved to {config.usage_rate_file(m=config.seed_num_list[-1])}")


if __name__ == '__main__':
    my_config = ExperimentConfig(
        data_set='Twitter',
        simulation_times=[100, 200], #[1000, 5000]
        methods=['random','degreeTopM','pageRank','succPro','1_neighbor'], # ['theroy','monterCarlo','random','degreeTopM','pageRank','succPro','1_neighbor']
        seed_num_list=None,
        monte_carlo_L=5,
        distribution_type='random',
        constant_factor_distri='random',
        personalization='None',# firstUnused
        data_prefix='/root/autodl-tmp/processed-data',
        method_type='None', # new,

        num_steps=2,
        scale_factor=2000
    )
    run_coupon_experiment(my_config)