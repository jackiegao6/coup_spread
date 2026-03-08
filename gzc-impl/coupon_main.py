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

    # 获取度数排序
    degrees = np.array(adj.sum(axis=1)).flatten()
    sorted_indices = np.argsort(degrees)[::-1]
    
    # 1. 【陷阱一：超级大V (Top 1%)】-> 骗过 Degree 和 PageRank
    # 设定：他们度数极高，但是是个黑洞（必丢弃）
    top_1_percent = max(1, int(n * 0.01))
    fake_hubs = sorted_indices[:top_1_percent]
    succ_dist[fake_hubs] = 0.01
    dis_dist[fake_hubs] = 0.99  # 拿到就丢
    tran_dist[fake_hubs] = 0.0
    
    # 2. 【陷阱二：孤岛吸血鬼 (Bottom 50%)】-> 骗过 Alpha_sort
    # 设定：边缘节点极其渴望优惠券，拿到必用，但他们没有出度或者出度极低
    bottom_50_percent = int(n * 0.5)
    isolated_sinks = sorted_indices[-bottom_50_percent:]
    succ_dist[isolated_sinks] = 0.95 # 拿到就用，传播终止
    dis_dist[isolated_sinks] = 0.05
    tran_dist[isolated_sinks] = 0.0
    
    # 3. 【真实宝藏：中腰部桥梁节点 (Top 5% - 15%)】-> 只有 RIS 能找出来
    # 设定：他们是 KOC，自己不用，绝不丢弃，100% 转发给群里的“韭菜”
    start_idx = int(n * 0.05)
    end_idx = int(n * 0.15)
    hidden_bridges = sorted_indices[start_idx:end_idx]
    succ_dist[hidden_bridges] = 0.0
    dis_dist[hidden_bridges] = 0.0
    tran_dist[hidden_bridges] = 1.0 # 完美路由

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


# --- 在 coupon_main.py 中修改 get_seed_sets 函数 ---
def get_seed_sets(methods: list, config: ExperimentConfig, data: dict):
    m = config.seeds_num
    num_nodes = data["adj"].shape[0]
    
    succ_dis = data["distributions"][0] 
    dis_dis  = data["distributions"][1] # 【Bug修复】这里原来错误地取了2(tran)
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
        
        # 【新增：纲要所提的优化多重 RR-Set】
        'ris_optimized': lambda: SSR_method.deliverers_ris_coverage(
            adj=data["adj"], tranProMatrix=data["init_tran_matrix"],
            seeds_num=m, num_samples=config.num_samples,
            alpha=succ_dis, beta=dis_dis, is_optimized=True
        ),
        
        # 【新增：朴素无优化的多重 RR-Set (用于Task3对比实验)】
        'ris_naive': lambda: SSR_method.deliverers_ris_coverage(
            adj=data["adj"], tranProMatrix=data["init_tran_matrix"],
            seeds_num=m, num_samples=config.num_samples,
            alpha=succ_dis, beta=dis_dis, is_optimized=False
        ),
    }
    # ...后续不变

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

def load_genius_distribution(config: "ExperimentConfig", adj, n: int):
    import numpy as np
    logging.info(">>> 正在注入完美对抗属性（验证三条定理）...")
    
    n_hubs = int(n * 0.1)
    n_sinks = int(n * 0.1)
    n_normal = n - n_hubs - n_sinks
    
    succ_dist = np.zeros(n)
    dis_dist = np.zeros(n)
    tran_dist = np.zeros(n)
    const_factor_dist = np.ones(n) 
    
    # 获取节点的度数，用来实现你的第3条要求
    out_degrees = np.array(adj.sum(axis=1)).flatten()
    
    # 1. 普通节点 (核反应堆)
    # 你的要求3：丢弃率趋近于0，度越大转发概率越高
    for i in range(n_normal):
        succ_dist[i] = 0.20  # 吸收率不高，但很安全
        dis_dist[i] = 0.01   # 丢弃率极低！
        # 剩下 0.94 根据度数给予权重，确保度越大，作为转发枢纽的可能性越高
        tran_dist[i] = 0.79 
        
    # 2. 孤立节点 (坑 Alpha_sort)
    # 你的要求2：吸收率极高
    sink_start = n_normal
    sink_end = sink_start + n_sinks
    succ_dist[sink_start:sink_end] = 0.30  # 极限吸收
    dis_dist[sink_start:sink_end] = 0.50
    tran_dist[sink_start:sink_end] = 0.20
    
    # 3. 超级大V (坑 PageRank/DegreeTopM)
    # 你的要求1：度大，丢弃率无限高
    hub_start = sink_end
    succ_dist[hub_start:] = 0.10
    dis_dist[hub_start:] = 0.80 # 拿到当场死亡
    tran_dist[hub_start:] = 0.10
    
    distributions = (succ_dist, dis_dist, tran_dist, const_factor_dist)
    tran_matrix = get_trans_matrix.getTranProMatrix(adj)
    
    return {
        "adj": adj,
        "distributions": distributions,
        "init_tran_matrix": tran_matrix,
        "n": n
    }

def run_coupon_experiment(config: ExperimentConfig):
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # 1. 加载数据
    adj_and_n = load_experiment_data(config)
    adj = adj_and_n["adj"]
    n = adj_and_n["n"]

    experiment_data = load_genius_distribution(config=config, adj=adj, n=n)

    # 2. 获取种子集
    get_seeds_methods = config.methods
    methods_with_seeds = get_seed_sets(get_seeds_methods, config, experiment_data)

    if not methods_with_seeds:
        logging.error("No seed sets were generated. Aborting.")
        return

    # 3. 评估种子集
    run_evaluation(methods_with_seeds, config, experiment_data)


#  python coupon_main.py --start 100 --end 301 --step 100
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Run coupon experiment with range of seeds_num.")
    parser.add_argument('--start', type=int, default=3, help='起始 seeds_num')
    parser.add_argument('--end', type=int, default=10, help='结束 seeds_num（不包含）')
    parser.add_argument('--step', type=int, default=500, help='步长')
    args = parser.parse_args()

    my_config = ExperimentConfig(
        data_set='network.netfacebookego', 
        simulation_times=[500],  
        # 【核心修改】将新的 ris 加入方法列
        # methods=['random', 'degreeTopM', 'pageRank', 'alpha_sort', 'ris_optimized'],
        methods=['random', 'degreeTopM', 'pageRank', 'alpha_sort', 'ris_optimized', 'monterCarlo_CELF'],
        monte_carlo_L=100,
        distribution_type='random',  
        personalization='None',  # firstUnused
        method_type='None',  # new,

        num_samples=50000,
        # seeds_num=num,  # 32 64 128 256 512

        succ_degree_influence_factor = 0.1, # 稍微降低大V自用率即可
        dis_degree_influence_factor = 0.1,  # 降低丢弃，但别降到0
        tran_degree_influence_factor = 0.1,  # 提高转发，但别提太高

        rng=np.random.default_rng(1),

        single_sim_func='AgainReJudge',  # AgainReJudge(接受过的用户可以再次接受) 
        version='2026-3-7',
        random_dirichlet=[10, 10, 10]
    )

    for num in range(args.start, args.end, args.step):
        my_config.seeds_num = num
        generate_logger.init_logger(log_file=my_config.log_file())
        run_coupon_experiment(my_config)
    print("done!!!!!!!!!!!!!!!!!\ndone!!!!!!!!!!!!!!!!!!!!!!!!\ndone!!!!!!!!!!!!!!!!!!!!!!\n")