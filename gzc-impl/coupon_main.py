from config import ExperimentConfig
import logging
import pickle
from typing import List
import coupon_usage_rate_get_distribution_degree_aware as gd
import single_deliverer
import os
import get_coupon_deliverers
import time
import ast



def load_experiment_data(config: ExperimentConfig):
    logging.info(f"===> 正在加载数据集: {config.data_set}")
    with open(config.adj_file, 'rb') as f:
        adj = pickle.load(f)
    n = adj.shape[0]
    return {"adj": adj, "n": n}


def prepare_seeds_model(config: ExperimentConfig, adj, n):
    distribution_res = gd.get_distribution_degree_aware(
        config.distribution_file(m=config.seeds_num),
        config.distribution_type,
        adj)

    succ_dist, dis_dist, tran_dist, const_factor_dist = distribution_res

    init_tran_matrix = single_deliverer.getTranProMatrix(adj, tran_dist)

    return {
        "adj": adj,
        "distributions": distribution_res,
        "init_tran_matrix": init_tran_matrix,
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

def run_coupon_experiment(config: ExperimentConfig):
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # 1. 加载数据
    adj_and_n = load_experiment_data(config)
    adj = adj_and_n["adj"]
    n = adj_and_n["n"]

    experiment_data = prepare_seeds_model(config=config, adj=adj, n=n)

    # 2. 获取种子集
    get_seeds_methods = config.methods
    methods_with_seeds = get_seed_sets(get_seeds_methods, config, experiment_data)

    if not methods_with_seeds:
        logging.error("No seed sets were generated. Aborting.")
        return

    # 3. 评估种子集
    # run_evaluation(methods_with_seeds, config, experiment_data)


if __name__ == '__main__':
    my_config = ExperimentConfig(
        data_set='BA',
        simulation_times=[5000], #[1000, 5000]
        methods=['random','pageRank','ris_coverage'], # ['theroy','monterCarlo','random','degreeTopM','pageRank','succPro','1_neighbor','ris_coverage']
        monte_carlo_L=15,
        distribution_type='random',# poisson gamma powerlaw random
        personalization='None',# firstUnused
        method_type='None', # new,

        num_samples = 600000,
        seeds_num = 16
    )


    run_coupon_experiment(my_config)