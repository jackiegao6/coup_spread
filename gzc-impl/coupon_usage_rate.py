# import os
# import time
# import pickle
# import logging
# from typing import List
#
# from config import ExperimentConfig
# import single_deliverer
# import get_coupon_deliverers
# import get_coupon_users
# import coupon_usage_rate_get_distribution
# import get_coupon_usage_rate_simulation
#
# def run_coupon_experiment(config: ExperimentConfig):
#
#     logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
#
#     # 1. 加载数据
#     adj_and_n = load_experiment_data(config)
#     n = adj_and_n["n"]
#
#     seedNumList = create_seed_num_list(total_nodes=n, num_steps=config.num_steps, scale_factor=config.scale_factor)
#     my_config.seed_num_list = seedNumList
#     experiment_data = load_experiment_data_get_seedsnum(config=config, adj=adj_and_n["adj"], n=adj_and_n["n"])
#
#     # 2. 获取种子集
#     methods = config.methods
#     method_to_seeds, _ = get_seed_sets(methods, config, experiment_data)
#
#     if not method_to_seeds:
#         logging.error("No seed sets were generated. Aborting.")
#         return
#
#     run_evaluation(method_to_seeds, config, experiment_data)
#
#
# def load_experiment_data(config: ExperimentConfig):
#     logging.info(f"===> Loading data for dataset: {config.data_set}")
#     with open(config.adj_file, 'rb') as f:
#         adj = pickle.load(f)# 邻接矩阵
#     n = adj.shape[0]
#
#     return {
#         "adj": adj,
#         "n": n
#     }
#
#
# def load_experiment_data_get_seedsnum(config: ExperimentConfig, adj, n):
#     distribution_list = coupon_usage_rate_get_distribution.get_distribution(config.distribution_file(m=config.seed_num_list[-1]), config.distribution_type, n)
#     succ_dist, dis_dist, tran_dist, const_factor_dist = distribution_list
#
#     init_tran_matrix, D = single_deliverer.getTranProMatrix(adj, tran_dist)
#
#     return {
#         "adj": adj,
#         "n": n,
#         "distributions": distribution_list,
#         "init_tran_matrix": init_tran_matrix,
#         "D": D
#     }
#
#
# def create_seed_num_list(
#     total_nodes: int,
#     num_steps: int,
#     scale_factor: int = 1000
# ) -> List[int]:
#
#     seed_list = [round(total_nodes * i / scale_factor) for i in range(1, num_steps + 1)]
#
#     # 去除可能因四舍五入产生的重复值，并确保列表非空
#     unique_seeds = sorted(list(set(seed_list)))
#     return [s for s in unique_seeds]
#
#
# def get_seed_sets(methods: list, config: ExperimentConfig, data: dict):
#     """ 使用多种不同的算法（理论方法、蒙特卡洛、随机、度中心性）来计算出应该选择哪些用户作为种子集 """
#
#     m = config.seed_num_list[-1]# 避免重复计算
#
#     selector_dict = {
#         'theroy': lambda: get_coupon_deliverers.deliverers_theroy(
#             data["n"], m, data["init_tran_matrix"], *data["distributions"], config.personalization, data["D"]),
#         'monterCarlo': lambda: get_coupon_deliverers.deliverers_monteCarlo(m=m,
#                                                                            init_tranProMatrix=data["init_tran_matrix"],
#                                                                            succ_distribution=data["distributions"][0],
#                                                                            dis_distribution=data["distributions"][1],
#                                                                            constantFactor_distribution=data["distributions"][3],
#                                                                            L=config.monte_carlo_L,
#                                                                            personalization=config.personalization),
#         'random': lambda: get_coupon_deliverers.deliverers_random(data["n"], m), # 基线方法
#         'degreeTopM': lambda: get_coupon_deliverers.deliverers_degreeTopM(data["adj"], m), # 基线方法
#         'pageRank': lambda: get_coupon_deliverers.deliverers_pageRank(data["adj"], m),
#         'succPro': lambda: get_coupon_deliverers.deliverers_succPro(succ_distribution=data['distributions'][0], m=m),
#         '1_neighbor': lambda: get_coupon_deliverers.deliverers_1_neighbor(succ_distribution=data['distributions'][0], init_tranProMatrix=data['init_tran_matrix'],m=m),
#         'ris_coverage': lambda: get_coupon_deliverers.deliverers_ris_coverage(
#                                                                             adj=data["adj"],
#                                                                             tranProMatrix=data["init_tran_matrix"],
#                                                                             m=m,
#                                                                             num_samples=config.num_samples # 通过 config 对象来配置
#                                                                         ),
#     }
#
#     method_to_seeds = {}
#     method_to_runtime = {}
#     for method in methods:
#         if method not in selector_dict:
#             logging.warning(f"===> Method '{method}' not found in registry. Skipping.")
#             continue
#
#         if os.path.exists(config.deliverers_cache_file(method=method, m=config.seed_num_list[-1])):
#             logging.info(f"===> Loading seed sets from cache: {config.deliverers_cache_file(method=method,m=config.seed_num_list[-1])}")
#         logging.info("===> Cache not found. Calculating seed sets...")
#
#
#         start_time = time.time()
#         logging.info(f"===> Running method: {method}")
#
#         seeds = selector_dict[method]()
#         method_to_seeds[method] = seeds
#
#         end_time = time.time()
#         method_to_runtime[method] = end_time - start_time
#         logging.info(f"===> Method {method} finished in {method_to_runtime[method]:.2f} seconds.")
#
#         deliverers_cache_file = config.deliverers_cache_file(   method=method,
#                                                                 m=config.seed_num_list[-1])
#
#         os.makedirs(os.path.dirname(deliverers_cache_file), exist_ok=True)
#         with open(deliverers_cache_file, 'a+') as file:
#             for key, value in method_to_seeds.items():
#                 file.write(f'{key}:{value}\n')
#
#     return method_to_seeds, method_to_runtime
#
# def run_evaluation(method_to_seeds: dict, config: ExperimentConfig, data: dict):
#     logging.info(f"Starting evaluation with personalization: {config.personalization}")
#
#     evaluation_dict = {
#         'None': get_coupon_usage_rate_simulation.simulation2,
#         'firstUnused': get_coupon_usage_rate_simulation.simulation2,
#         'firstDiscard': get_coupon_usage_rate_simulation.simulation2,
#     }
#
#     if config.personalization not in evaluation_dict:
#         raise ValueError(f"Unknown personalization type: {config.personalization}")
#
#     usage_rate_file = config.usage_rate_file(m=config.seed_num_list[-1])
#
#
#     evaluation_func = evaluation_dict[config.personalization]
#     methods = list(method_to_seeds.keys())
#     deliverers = list(method_to_seeds.values())
#
#     evaluation_func(methods=methods,
#                     method_deliverers=deliverers,
#                     init_tran_matrix=data["init_tran_matrix"],
#                     usage_rate_file=usage_rate_file,
#                     distribution_list=data["distributions"],
#                     seed_num_list=config.seed_num_list,
#                     simulation_times=config.simulation_times,
#                     single_sim_func=get_coupon_users.monteCarlo_singleTime_improved2)
#     logging.info(f"Evaluation finished. Results saved to {usage_rate_file}")
#
#
# if __name__ == '__main__':
#     my_config = ExperimentConfig(
#         data_set='Twitter',
#         simulation_times=[5000], #[1000, 5000]
#         methods=['random','degreeTopM','pageRank','succPro','1_neighbor','ris_coverage'], # ['theroy','monterCarlo','random','degreeTopM','pageRank','succPro','1_neighbor','ris_coverage']
#         seed_num_list=None,
#         monte_carlo_L=15,
#         distribution_type='random',
#         constant_factor_distri='random',
#         personalization='None',# firstUnused
#         data_prefix='/home/wen/pythonspace/processed-data',
#         method_type='None', # new,
#
#         num_steps=5,
#         scale_factor=5000,
#         num_samples = 600000
#     )
#
#
#     run_coupon_experiment(my_config)