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

def print_density(G):

    n = G.number_of_nodes()
    m = G.number_of_edges()

    density = nx.density(G)

    logging.info(f"节点数: {n}, 边数: {m}")
    logging.info(f"图密度: {density:.6f}") # 比如 0.005
    logging.info(f"稀疏度: {1 - density:.6f}") # 比如 0.995

    avg_degree = 2 * m / n  # 无向图
    logging.info(f"平均度数: {avg_degree:.2f}")

def check_sparsity_from_matrix(adj_matrix):
    # adj_matrix 是 sp.csr_matrix 或 sp.coo_matrix
    
    num_nodes = adj_matrix.shape[0]
    num_edges = adj_matrix.nnz  # nnz = Number of Non-Zero elements (非零元素个数，即边数)
    
    # 理论最大边数 (假设是有向的，就是 N*N；无向且对角线为0则是 N*(N-1))
    # 简单估算通常直接用 N*N
    max_possible_edges = num_nodes * num_nodes
    
    density = num_edges / max_possible_edges
    sparsity = 1.0 - density
    
    logging.info(f"--- 矩阵分析 ---")
    logging.info(f"节点数 (N): {num_nodes}")
    logging.info(f"边数 (E): {num_edges}")
    logging.info(f"密度 (Density): {density:.8f} (越小越稀疏)")
    logging.info(f"稀疏度 (Sparsity): {sparsity:.8f} (越接近1越稀疏)")
    logging.info(f"平均度数 (Avg Degree): {num_edges / num_nodes:.2f}")


def load_experiment_data(config: "ExperimentConfig") -> Dict[str, Any]:
    """
    :param config: 根据配置文件 单纯加载数据集
    :return: 返回邻接矩阵+度数
    """
    logging.info("加载数据集: %s", config.data_set)

    adj_path = Path(config.adj_file)
    if not adj_path.exists():
        raise FileNotFoundError(f"路径错误 {adj_path}")

    with adj_path.open("rb") as f:
        try:
            adj = pickle.load(f)
        except Exception as e:
            raise ValueError(f"文件加载错误 {adj_path}") from e

    check_sparsity_from_matrix(adj)
    return {"adj": adj, "n": adj.shape[0]}


def load_contribution_and_tran_matrix_watch(config: "ExperimentConfig", adj, n: int) -> Dict[str, Any]:
    """
    生成分布，并强制注入“看门人机制”以展示拓扑算法优势
    """
    # 1. 正常生成基础分布 (比如 Dirichlet=[5, 2, 8])
    # 这作为背景噪音，让大部分节点表现平庸
    distribution_res = gd.get_distribution_degree_aware(
        config.distribution_file(m=config.seeds_num),
        config.distribution_type,
        adj,
        config=config
    )

    # 解包分布 (确保是 numpy array)
    succ_dist, dis_dist, tran_dist, const_factor_dist = [np.array(d) for d in distribution_res]

    logging.info(">>> 正在注入‘社区效应/看门人机制’以区分算法能力...")

    # 1. 寻找“看门人”：度数最高的那个节点 (Facebook这种无标度网络，大V连接了非常多的人)
    # 注意：adj 是稀疏矩阵
    degrees = np.array(adj.sum(axis=1)).flatten()
    top_k_indices = np.argsort(degrees)[::-1][:50]
    
    # 2. 寻找“社区成员”：看门人的所有邻居
    # 对于 CSR 矩阵，高效获取邻居的方法：
    # 找到这些看门人覆盖的所有粉丝（去重）
    all_fans = set()
    for gx in top_k_indices:
        start = adj.indptr[gx]
        end = adj.indptr[gx+1]
        all_fans.update(adj.indices[start:end])
    community_nodes = list(all_fans)
    

    # 3. 【设置背景板】：让全网其他节点变“垃圾” (低成功、高丢弃)
    # 这样 Alpha_sort 就不会去选背景节点，而被逼去选我们设定的社区成员
    # 设定：P(succ)=0.05, P(dis)=0.8 (地狱难度)
    succ_dist[:] = 0.05
    dis_dist[:] = 0.8
    tran_dist[:] = 0.15

    # 4. 【设置社区成员】：Alpha_sort 的诱饵 (中等成功率，但也容易丢弃)
    # 设定：P(succ)=0.4。这在全网是最高的，Alpha_sort 肯定选他们。
    # 但单点期望只有 0.4。
    succ_dist[community_nodes] = 0.1
    dis_dist[community_nodes]  = 0.5
    tran_dist[community_nodes] = 0.4

    # 5. 【设置看门人】：RIS 的宝藏 (自己不用，但这辈子绝不丢弃，必转发)
    # 设定：P(succ)=0.0, P(dis)=0.0, P(tran)=1.0 (完美路由器)
    succ_dist[top_k_indices] = 0.0
    dis_dist[top_k_indices] = 0.0
    tran_dist[top_k_indices] = 1.0

    # 重新打包
    distribution_res = (succ_dist, dis_dist, tran_dist, const_factor_dist)

    tran_matrix = get_trans_matrix.getTranProMatrix(adj)

    return {
        "adj": adj,
        "distributions": distribution_res,
        "init_tran_matrix": tran_matrix,
        "n": n
    }

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
    """ 用多种算法（随机、PageRank、ris_coverage）来计算出应该选择哪些用户作为种子集"""

    m = config.seeds_num

    num_nodes = data["adj"].shape[0]
    succ_dis = data["distributions"][0] #ndarray:{23362} [0.33116881 0.40606414 ...]
    p_dis = data["distributions"][2]

    alpha = {node: succ_dis[node] for node in range(len(succ_dis))}
    trans = {node: p_dis[node] for node in range(len(p_dis))}

    selector_dict = {

        'monterCarlo_standard': lambda: get_seeds.deliverers_monteCarlo_greedy_standard(
            n=data["n"],
            m=m,
            tranProMatrix=data["init_tran_matrix"],
            succ_distribution=data["distributions"][0],
            dis_distribution=data["distributions"][1],
            simulation_algo_func=simulation_algo.monteCarlo_singleTime_improved2,
            simulation_times=config.monte_carlo_L 
        ),
        'monterCarlo_CELF': lambda: get_seeds.deliverers_monteCarlo_CELF(
            n=data["n"],
            m=m,
            tranProMatrix=data["init_tran_matrix"],
            succ_distribution=data["distributions"][0],
            dis_distribution=data["distributions"][1],
            simulation_algo_func=simulation_algo.monteCarlo_singleTime_improved2,
            simulation_times=config.monte_carlo_L 
        ),

        'random': lambda: get_seeds.deliverers_random(data["n"], m),  # 基线方法
        'degreeTopM': lambda: get_seeds.deliverers_degreeTopM(data["adj"], m),  # 基线方法
        'pageRank': lambda: get_seeds.deliverers_pageRank2(adj=data["adj"],
                                                          m=m,
                                                          tranProMatrix=data["init_tran_matrix"]), # 基线方法
        '1_neighbor': lambda: get_seeds.deliverers_1_neighbor(succ_distribution=data['distributions'][0],
                                                                          init_tranProMatrix=data['init_tran_matrix'],
                                                                          m=m),
        'ris_coverage': lambda: SSR_method.deliverers_ris_coverage(
            adj=data["adj"],
            tranProMatrix=data["init_tran_matrix"],
            seeds_num=m,
            num_samples=config.num_samples,  # 通过 config 对象来配置
            alpha=alpha
        ),
        'alpha_sort': lambda: get_seeds.deliverers_alpha_sort(
            adj=data["adj"],
            tranProMatrix=data["init_tran_matrix"],
            seeds_num=m,
            alpha=alpha
        ),
        'importance_sort': lambda: get_seeds.deliverers_importance_sort(
            adj=data["adj"],
            tranProMatrix=data["init_tran_matrix"],
            seeds_num=m,
            alpha=alpha,
            trans=trans
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

        logging.info(f"首次生成种子集 {method}")


        start_time = time.time()
        logging.info(f"执行当前算法开始: {method}")

        seeds = selector_dict[method]()
        methods_with_seeds[method] = seeds

        end_time = time.time()
        cost_time = end_time - start_time
        logging.info(f"当前用时 {cost_time:.2f} 秒")

        os.makedirs(os.path.dirname(deliverers_cache_file), exist_ok=True)
        with open(deliverers_cache_file, 'a+') as file:
            # todo 增加一下列数
            logging.info(f"将种子集 写入位置: {config.deliverers_cache_file(method=method, m=config.seeds_num)}")
            for key, value in methods_with_seeds.items():
                file.write(f'{key}:{value}\n')

    return methods_with_seeds



def run_evaluation(methods_with_seeds: dict, config: ExperimentConfig, data: dict):
    logging.info(f"评估种子集: {config.personalization}")

    evaluation_dict = {
        'None': simulation.simulation2,
        'firstUnused': simulation.simulation2,
        'firstDiscard': simulation.simulation2,
    }

    if config.personalization not in evaluation_dict: raise ValueError(f"Unknown personalization type: {config.personalization}")

    usage_rate_file = config.usage_rate_file(m=config.seeds_num)

    evaluation_func = evaluation_dict[config.personalization]
    methods = list(methods_with_seeds.keys())
    seeds = list(methods_with_seeds.values())

    single_coupon_trans_process = {
        'AgainReJudge': simulation_algo.monteCarlo_singleTime_improved2,
        'AgainContinue': simulation_algo.monteCarlo_singleTime_improved2_AgainContinue
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


def run_coupon_experiment(config: ExperimentConfig):
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # 1. 加载数据
    adj_and_n = load_experiment_data(config)
    adj = adj_and_n["adj"]
    n = adj_and_n["n"]

    experiment_data = load_contribution_and_tran_matrix(config=config, adj=adj, n=n)
    # experiment_data = load_contribution_and_tran_matrix_watch(config=config, adj=adj, n=n)


    # 2. 获取种子集
    get_seeds_methods = config.methods
    methods_with_seeds = get_seed_sets(get_seeds_methods, config, experiment_data)

    if not methods_with_seeds:
        logging.error("No seed sets were generated. Aborting.")
        return

    # 3. 评估种子集
    run_evaluation(methods_with_seeds, config, experiment_data)


#  python coupon_main.py --start 2300 --end 3000 --step 3000
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Run coupon experiment with range of seeds_num.")
    parser.add_argument('--start', type=int, default=3, help='起始 seeds_num')
    parser.add_argument('--end', type=int, default=10, help='结束 seeds_num（不包含）')
    parser.add_argument('--step', type=int, default=500, help='步长')
    args = parser.parse_args()

    my_config = ExperimentConfig(
        # Twitter facebook Amherst Pepperdine Wellesley Mich Rochester Oberlin students 
        # network.netfacebookego network.netDog network.Amazon network.douban11core network.netYeast network.doubanrandom network.netactorcollaboration network.netfacebookego
        data_set='network.netDog', 

        simulation_times=[1000],  # [1000, 5000]

        # methods=['random', 'degreeTopM', 'pageRank','alpha_sort', 'importance_sort', 'ris_coverage', 'monterCarlo_CELF'],
        methods=['random', 'degreeTopM', 'pageRank','alpha_sort', 'importance_sort', 'ris_coverage','monterCarlo_CELF'],

        monte_carlo_L=300,

        distribution_type='random',  # powerlaw powerlaw-old random poisson gamma
        personalization='None',  # firstUnused
        method_type='None',  # new,

        num_samples=100000,
        # seeds_num=num,  # 32 64 128 256 512

        succ_degree_influence_factor = 0.0, # 稍微降低大V自用率即可
        dis_degree_influence_factor = 0.0,  # 降低丢弃，但别降到0
        tran_degree_influence_factor = 0.0,  # 提高转发，但别提太高

        rng=np.random.default_rng(1),

        single_sim_func='AgainReJudge',  # AgainReJudge(接受过的用户可以再次接受) 、 AgainContinue(采用)(吸收态用户接收到券的使用概率为0)(目的：不是让券的使用率最大，而是让券的尽可能地覆盖)
        version='2026-1-7-3-[500,500,500]',
        random_dirichlet=[50,50,900]
    )

    for num in range(args.start, args.end, args.step):
        my_config.seeds_num = num
        generate_logger.init_logger(log_file=my_config.log_file())
        run_coupon_experiment(my_config)
    print("done!!!!!!!!!!!!!!!!!\ndone!!!!!!!!!!!!!!!!!!!!!!!!\ndone!!!!!!!!!!!!!!!!!!!!!!\n")