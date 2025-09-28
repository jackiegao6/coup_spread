from config import ExperimentConfig
import logging
import pickle
from typing import List, Dict, Any
import coupon_usage_rate_get_distribution_degree_aware as gd
import single_deliverer
import os
import get_coupon_deliverers
import time
import ast
import get_coupon_usage_rate_simulation
import get_coupon_users
from tools import generate_logger
from pathlib import Path
import numpy as np


def load_experiment_data(config: "ExperimentConfig") -> Dict[str, Any]:
    """
    :param config: 根据配置文件 单纯加载数据集
    :return: 返回邻接矩阵+度数
    """
    logging.info("加载数据集: %s", config.data_set)

    adj_path = Path(config.adj_file)
    if not adj_path.exists():
        raise FileNotFoundError(f"路径错误: {adj_path}")

    with adj_path.open("rb") as f:
        try:
            adj = pickle.load(f)
        except Exception as e:
            raise ValueError(f"文件加载错误 {adj_path}") from e

    return {"adj": adj, "n": adj.shape[0]}


def load_contribution_and_tran_matrix(config: "ExperimentConfig", adj, n: int) -> Dict[str, Any]:
    """
    根据每个节点的度生成不同的三个分布 比如通过幂律分布
    可以控制参数a
    通过 config中的 degree_influence_factor 联合度数 控制a大小

    :return: 转移分布、接收分布、拒绝分布、行为分布
    """
    distribution_res = gd.get_distribution_degree_aware(
        config.distribution_file(m=config.seeds_num),
        config.distribution_type,
        adj,
        config = config)

    succ_dist, dis_dist, tran_dist, const_factor_dist = distribution_res

    tran_matrix = single_deliverer.getTranProMatrix(adj, tran_dist)

    return {
        "adj": adj,
        "distributions": distribution_res,
        "init_tran_matrix": tran_matrix,
        "n": n
    }


def get_seed_sets(methods: list, config: ExperimentConfig, data: dict):
    """ 用多种算法（随机、PageRank、ris_coverage）来计算出应该选择哪些用户作为种子集"""

    m = config.seeds_num

    selector_dict = {
        # 'theroy': lambda: get_coupon_deliverers.deliverers_theroy(
        #     data["n"], m, data["init_tran_matrix"], *data["distributions"], config.personalization, data["D"]),
        'monterCarlo': lambda: get_coupon_deliverers.deliverers_monteCarlo(m=m,
                                                                           init_tranProMatrix=data["init_tran_matrix"],
                                                                           succ_distribution=data["distributions"][0],
                                                                           dis_distribution=data["distributions"][1],
                                                                           constantFactor_distribution=
                                                                           data["distributions"][3],
                                                                           L=config.monte_carlo_L,
                                                                           personalization=config.personalization),
        'random': lambda: get_coupon_deliverers.deliverers_random(data["n"], m),  # 基线方法
        'degreeTopM': lambda: get_coupon_deliverers.deliverers_degreeTopM(data["adj"], m),  # 基线方法
        'pageRank': lambda: get_coupon_deliverers.deliverers_pageRank(data["adj"], m),
        'succPro': lambda: get_coupon_deliverers.deliverers_succPro(succ_distribution=data['distributions'][0], m=m),
        '1_neighbor': lambda: get_coupon_deliverers.deliverers_1_neighbor(succ_distribution=data['distributions'][0],
                                                                          init_tranProMatrix=data['init_tran_matrix'],
                                                                          m=m),
        'ris_coverage': lambda: get_coupon_deliverers.deliverers_ris_coverage(
            adj=data["adj"],
            tranProMatrix=data["init_tran_matrix"],
            m=m,
            num_samples=config.num_samples  # 通过 config 对象来配置
        ),
    }

    methods_with_seeds = {}
    for method in methods:
        if method not in selector_dict:
            logging.warning(f"===> Method '{method}' not found in registry. Skipping.")
            continue

        deliverers_cache_file = config.deliverers_cache_file(method=method, m=config.seeds_num)

        if os.path.exists(deliverers_cache_file):
            logging.info(f"===> Loading seed sets from cache: {config.deliverers_cache_file(method=method, m=config.seeds_num)}")

            with open(deliverers_cache_file, 'r') as file:
                for line in file:
                    key, value = line.strip().split(':', 1)
                    methods_with_seeds[key] = ast.literal_eval(value)

            print(methods_with_seeds)
            continue
        logging.info("===> Cache not found. Calculating seed sets...")

        start_time = time.time()
        logging.info(f"===> Running method: {method}")

        seeds = selector_dict[method]()
        methods_with_seeds[method] = seeds

        end_time = time.time()
        cost_time = end_time - start_time
        logging.info(f"===> Method {method} finished in {cost_time:.2f} seconds.")

        os.makedirs(os.path.dirname(deliverers_cache_file), exist_ok=True)
        with open(deliverers_cache_file, 'a+') as file:
            for key, value in methods_with_seeds.items():
                file.write(f'{key}:{value}\n')

    return methods_with_seeds


def run_evaluation(methods_with_seeds: dict, config: ExperimentConfig, data: dict):
    logging.info(f"Starting evaluation with personalization: {config.personalization}")

    #deprecated
    evaluation_dict = {
        'None': get_coupon_usage_rate_simulation.simulation2,
        'firstUnused': get_coupon_usage_rate_simulation.simulation2,
        'firstDiscard': get_coupon_usage_rate_simulation.simulation2,
    }
    if config.personalization not in evaluation_dict: raise ValueError(f"Unknown personalization type: {config.personalization}")

    usage_rate_file = config.usage_rate_file(m=config.seeds_num)

    evaluation_func = evaluation_dict[config.personalization]
    methods = list(methods_with_seeds.keys())
    seeds = list(methods_with_seeds.values())

    evaluation_func(methods=methods,
                    seeds_list=seeds,
                    init_tran_matrix=data["init_tran_matrix"],
                    usage_rate_file=usage_rate_file,
                    distribution_list=data["distributions"],
                    simulation_times=config.simulation_times,
                    single_sim_func=get_coupon_users.monteCarlo_singleTime_improved2,
                    seed_num=config.seeds_num)
    logging.info(f"Evaluation finished. Results saved to {usage_rate_file}")


def run_coupon_experiment(config: ExperimentConfig):
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # 1. 加载数据
    adj_and_n = load_experiment_data(config)
    adj = adj_and_n["adj"]
    n = adj_and_n["n"]

    experiment_data = load_contribution_and_tran_matrix(config=config, adj=adj, n=n)

    # 2. 获取种子集
    get_seeds_methods = config.methods
    methods_with_seeds = get_seed_sets(get_seeds_methods, config, experiment_data)

    if not methods_with_seeds:
        logging.error("No seed sets were generated. Aborting.")
        return

    # 3. 评估种子集
    run_evaluation(methods_with_seeds, config, experiment_data)


if __name__ == '__main__':
    my_config = ExperimentConfig(
        data_set='Twitter',
        simulation_times=[5000], #[1000, 5000]
        methods=['random','pageRank','ris_coverage'], # ['theroy','monterCarlo','random','degreeTopM','pageRank','succPro','1_neighbor','ris_coverage']
        monte_carlo_L=15,
        distribution_type='powerlaw',# poisson gamma powerlaw random
        personalization='None',# firstUnused
        method_type='None', # new,

        num_samples = 600000,
        seeds_num = 16, # 32 64 128 256 512

        tran_degree_influence_factor = -10.0,
        succ_degree_influence_factor = 10.0,
        dis_degree_influence_factor = 10.0,

        rng= np.random.default_rng(1)
    )
    generate_logger.init_logger(log_file=my_config.log_file())
    run_coupon_experiment(my_config)