from config import ExperimentConfig
import logging
import pickle
from typing import List, Dict, Any
import distribution as gd
import SSR_method
import os
import get_seeds
import time
import ast
import simulation
import simulation_algo
from tools import generate_logger
from pathlib import Path
import numpy as np
import argparse
import get_trans_matrix
import scipy.sparse as sp
import networkx as nx

def load_experiment_data(config: "ExperimentConfig") -> Dict[str, Any]:
    logging.info("加载数据集: %s", config.data_set)

    adj_path = Path(config.adj_file)
    if not adj_path.exists():
        raise FileNotFoundError(f"路径错误 {adj_path}")

    with adj_path.open("rb") as f:
        try:
            adj = pickle.load(f)
        except Exception as e:
            raise ValueError(f"文件加载错误 {adj_path}") from e

    return {"adj": adj, "n": adj.shape[0]}


def load_contribution_and_tran_matrix(config: "ExperimentConfig", adj, n: int) -> Dict[str, Any]:
    distribution_res = gd.get_distribution_degree_aware(
        config.distribution_file(m=config.seeds_num),
        config.distribution_type,
        adj,
        config = config)

    succ_dist, dis_dist, tran_dist, const_factor_dist = distribution_res

    tran_matrix = get_trans_matrix.getTranProMatrix(adj)

    return {
        "adj": adj,
        "distributions": distribution_res,
        "init_tran_matrix": tran_matrix,
        "n": n
    }


def get_seed_sets(methods: list, config: ExperimentConfig, data: dict):
    m = config.seeds_num
    num_nodes = data["adj"].shape[0]
    
    succ_dis = data["distributions"][0] 
    dis_dis  = data["distributions"][1]
    p_dis    = data["distributions"][2] 

    alpha = {node: succ_dis[node] for node in range(len(succ_dis))}
    trans = {node: p_dis[node] for node in range(len(p_dis))}

    selector_dict = {
        'monterCarlo_CELF': lambda: get_seeds.deliverers_monteCarlo_CELF(
            n=data["n"], m=m, tranProMatrix=data["init_tran_matrix"],
            succ_distribution=succ_dis, dis_distribution=dis_dis,
            simulation_algo_func=simulation_algo.monteCarlo_singleTime_improved2,
            simulation_times=config.monte_carlo_L 
        ),
        'random': lambda: get_seeds.deliverers_random(data["n"], m),
        'degreeTopM': lambda: get_seeds.deliverers_degreeTopM(data["adj"], m),
        'pageRank': lambda: get_seeds.deliverers_pageRank2(adj=data["adj"], m=m, tranProMatrix=data["init_tran_matrix"]),
        'alpha_sort': lambda: get_seeds.deliverers_alpha_sort(adj=data["adj"], tranProMatrix=data["init_tran_matrix"], seeds_num=m, alpha=alpha),
        '1hop_sort': lambda: get_seeds.deliverers_teacher_alpha_1hop_sort(
            tranProMatrix=data["init_tran_matrix"],
            seeds_num=m,
            alpha_distribution=succ_dis
        ),      
        'ris_path_aware': lambda: SSR_method.deliverers_ris_path_aware(
            adj=data["adj"],
            tranProMatrix=data["init_tran_matrix"],
            seeds_num=m,
            num_samples=config.num_samples,
            alpha=succ_dis,
            beta=dis_dis,
            is_optimized=True
        ),
    }

    methods_with_seeds = {}
    for method in methods:
        if method not in selector_dict:
            logging.warning(f"'{method}' 暂未支持这种种子选择算法")
            continue

        deliverers_cache_file = config.deliverers_cache_file(method=method, m=config.seeds_num)

        if os.path.exists(deliverers_cache_file):
            logging.info(f"从当前位置 读取种子集 {config.deliverers_cache_file(method=method, m=config.seeds_num)}")

            with open(deliverers_cache_file, 'r') as file:
                for line in file:
                    key, value = line.strip().split(':', 1)
                    methods_with_seeds[key] = ast.literal_eval(value)

            print(methods_with_seeds)
            continue

        start_time = time.time()
        logging.info(f"执行当前算法开始: {method}")

        seeds = selector_dict[method]()
        methods_with_seeds[method] = seeds

        end_time = time.time()
        cost_time = end_time - start_time
        logging.info(f"当前用时 {cost_time:.2f} 秒")

        os.makedirs(os.path.dirname(deliverers_cache_file), exist_ok=True)
        with open(deliverers_cache_file, 'a+') as file:
            logging.info(f"将种子集 写入位置: {config.deliverers_cache_file(method=method, m=config.seeds_num)}")
            for key, value in methods_with_seeds.items():
                file.write(f'{key}:{value}\n')

    return methods_with_seeds

def run_evaluation(methods_with_seeds: dict, config: ExperimentConfig, data: dict):
    logging.info(f"评估种子集: {config.personalization}")

    evaluation_dict = {
        'None': simulation.simulation2,
    }

    if config.personalization not in evaluation_dict: raise ValueError(f"Unknown personalization type: {config.personalization}")

    usage_rate_file = config.usage_rate_file(m=config.seeds_num)

    evaluation_func = evaluation_dict[config.personalization]
    methods = list(methods_with_seeds.keys())
    seeds = list(methods_with_seeds.values())

    single_coupon_trans_process = {
        'AgainReJudge': simulation_algo.monteCarlo_singleTime_improved2,
    }

    evaluation_func(methods=methods,
                    seeds_list=seeds,
                    init_tran_matrix=data["init_tran_matrix"],
                    usage_rate_file=usage_rate_file,
                    distribution_list=data["distributions"],
                    simulation_times=config.simulation_times,
                    single_sim_func=single_coupon_trans_process[config.single_sim_func],
                    seed_num=config.seeds_num,
                    config=config)
    logging.info(f"保存评估文件至: {usage_rate_file}")

import get_trans_matrix

def run_coupon_experiment(config: ExperimentConfig):
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    adj_and_n = load_experiment_data(config)
    adj = adj_and_n["adj"]
    n = adj_and_n["n"]

    experiment_data = load_contribution_and_tran_matrix(config=config, adj=adj, n=n)

    get_seeds_methods = config.methods
    methods_with_seeds = get_seed_sets(get_seeds_methods, config, experiment_data)

    if not methods_with_seeds:
        logging.error("No seed sets were generated. Aborting.")
        return

    run_evaluation(methods_with_seeds, config, experiment_data)


#  python coupon_main.py --start 100 --end 301 --step 100
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run coupon experiment.")
    parser.add_argument('--start', type=int, default=100, help='起始 seeds_num')
    parser.add_argument('--end', type=int, default=101, help='结束 seeds_num')
    parser.add_argument('--step', type=int, default=100, help='步长')
    args = parser.parse_args()

    my_config = ExperimentConfig(
        data_set='network.netscience', 
        simulation_times=[600],  
        methods=['random', 'degreeTopM', 'pageRank', 'alpha_sort', 'ris_path_aware', '1hop_sort'],
        monte_carlo_L=100,
        distribution_type='log_continuous', 
        personalization='None',  
        method_type='None',  
        num_samples=100000,
        rng=np.random.default_rng(1),
        single_sim_func='AgainReJudge',  
        version='2026-4-20',
    )

    for num in range(args.start, args.end, args.step):
        my_config.seeds_num = num
        generate_logger.init_logger(log_file=my_config.log_file())
        run_coupon_experiment(my_config)
    print("done!!!!!!!!!!!!!!!!!\n")