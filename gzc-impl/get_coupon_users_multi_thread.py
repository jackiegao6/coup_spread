import numpy as np
import logging
import single_deliverer
import pickle
import networkx as nx
import random

from multiprocessing import Pool, cpu_count
import functools


def _calculate_marginal_gain_for_candidate(
    candidate: int,
    base_influence: float,
    current_deliverers: list,
    # 其他 _run_full_simulation 需要的固定参数
    L: int,
    tranProMatrix: np.ndarray,
    succ_distribution: np.ndarray,
    dis_distribution: np.ndarray,
    constantFactor_distribution: np.ndarray,
    personalization: str
) -> tuple:
    """
    这是一个为单个候选节点计算边际增益的工作函数，专门用于并行处理。

    Args:
        candidate (int): 要评估的候选节点ID。
        base_influence (float): 基础影响力值。
        current_deliverers (list): 当前的种子集。
        ... (其他参数)

    Returns:
        tuple: 一个包含 (候选节点ID, 边际增益) 的元组。
    """
    # 构造临时投放集合进行测试
    test_deliverer_set = current_deliverers + [candidate]
    
    # 计算加入候选节点后的新影响力
    new_influence = _run_full_simulation(
        L, test_deliverer_set, tranProMatrix, succ_distribution,
        dis_distribution, constantFactor_distribution, personalization
    )
    
    marginal_gain = new_influence - base_influence
    
    # 打印日志可以帮助追踪进度，但在大量并行任务中可能会使输出混乱。
    # 在生产环境中可以考虑移除或使用更高级的日志管理。
    # print(f"  (Worker) Evaluated candidate {candidate}: Marginal gain = {marginal_gain:.4f}")

    return (candidate, marginal_gain)

def find_next_best_deliverer_parallel(
    current_deliverers: list,
    tranProMatrix: np.ndarray,
    L: int,
    succ_distribution: np.ndarray,
    dis_distribution: np.ndarray,
    constantFactor_distribution: np.ndarray,
    personalization: str,
    num_workers: int = None
) -> int:
    """
    通过并行化的蒙特卡洛模拟计算边际增益，找到下一个最优的投放者。

    Args:
        ... (原始参数)
        num_workers (int, optional): 使用的工作进程数。如果为None，则使用所有可用的CPU核心。

    Returns:
        int: 下一个最优投放者的节点ID。
    """
    n = tranProMatrix.shape[0]
    candidate_nodes = [node for node in range(n) if node not in current_deliverers]

    if not candidate_nodes:
        logging.warning("没有候选节点了，无法选择。")
        return None

    # 1. 计算当前集合的基础影响力 (这步仍然在主进程中执行)
    base_influence = _run_full_simulation(
        L, current_deliverers, tranProMatrix, succ_distribution, 
        dis_distribution, constantFactor_distribution, personalization
    )
    print(f"当前投放者 {current_deliverers} 的基础影响力: {base_influence:.4f}")

    # 准备并行计算
    if num_workers is None:
        num_workers = cpu_count() / 4
    print(f"使用 {num_workers} 个工作进程进行并行计算...")

    # 2. 创建任务列表
    # 使用 functools.partial 来“冻结”那些在所有任务中都相同的参数
    # 这样可以简化传递给 starmap 的参数
    worker_func_with_args = functools.partial(
        _calculate_marginal_gain_for_candidate,
        base_influence=base_influence,
        current_deliverers=current_deliverers,
        L=L,
        tranProMatrix=tranProMatrix,
        succ_distribution=succ_distribution,
        dis_distribution=dis_distribution,
        constantFactor_distribution=constantFactor_distribution,
        personalization=personalization
    )
    
    # 3. 使用进程池执行任务
    # Pool() 上下文管理器会在结束时自动关闭和加入进程
    with Pool(processes=num_workers) as pool:
        # pool.map 只接受一个可迭代的参数，所以我们只迭代 candidate_nodes
        results = pool.map(worker_func_with_args, candidate_nodes)
        
    # `results` 将会是一个列表，形如 [(candidate1, gain1), (candidate2, gain2), ...]
    if not results:
        logging.error("并行计算没有返回任何结果。")
        return None

    # 4. 在主进程中处理结果，找到最佳候选人
    best_candidate, max_marginal_gain = max(results, key=lambda item: item[1])
    
    # 为了处理增益相同的情况，找到所有具有最大增益的候选者
    best_candidates = [
        candidate for candidate, gain in results if gain >= max_marginal_gain
    ]
    
    # 从最佳候选中随机选择一个
    best_next_deliverer = random.choice(best_candidates)

    print(f"选择的最优新投放者: {best_next_deliverer} (最大边际增益: {max_marginal_gain:.4f})")
    
    return best_next_deliverer


def find_next_best_deliverer(
    current_deliverers: list,
    tranProMatrix: np.ndarray,
    L: int,
    succ_distribution: np.ndarray,
    dis_distribution: np.ndarray,
    constantFactor_distribution: np.ndarray,
    personalization: str
) -> int:
    
    best_next_deliverer = find_next_best_deliverer_parallel(current_deliverers=current_deliverers,
                                      tranProMatrix=tranProMatrix,
                                      L=L,
                                      succ_distribution=succ_distribution,
                                      dis_distribution=dis_distribution,
                                      constantFactor_distribution=constantFactor_distribution,
                                      personalization=personalization
    )
    return best_next_deliverer


def _run_full_simulation(
    L: int,
    deliverer_set: list,
    tranProMatrix: np.ndarray,
    succ_distribution: np.ndarray,
    dis_distribution: np.ndarray,
    constantFactor_distribution: np.ndarray,
    personalization: str
) -> float:
    """
    对给定的种子集，重复运行 L 次单次模拟，并计算平均总影响力
    """
    n = tranProMatrix.shape[0]
    total_influence_accumulator = 0.0

    # 根据个性化策略选择正确的模拟函数
    if personalization == 'firstUnused':
        single_simulation_func = monteCarlo_singleTime_firstUnused_improved
    elif personalization == 'firstDiscard':
        single_simulation_func = monteCarlo_singleTime_firstDiscard_improved
    else: # 默认或None
        single_simulation_func = monteCarlo_singleTime_improved

    for _ in range(L):
        # 它应该返回一个(1, n)或(n,)的数组，代表此轮模拟中各节点的成功状态(0或1)
        success_vector = single_simulation_func(
            tranProMatrix,
            deliverer_set,
            succ_distribution,
            dis_distribution,
            constantFactor_distribution
        )
        # 累加本轮模拟的总成功人数
        total_influence_accumulator += np.sum(success_vector)

    # 返回平均总影响力
    return total_influence_accumulator / L


def monteCarlo_singleTime_improved(
    tranProMatrix: np.ndarray,
    initial_deliverers: list,
    succ_distribution: np.ndarray,
    dis_distribution: np.ndarray,
    constantFactor_distribution: np.ndarray
) -> np.ndarray:
    """
    执行单次蒙特卡洛模拟，模拟从初始投放者开始的多路随机游走。

    Args:
        tranProMatrix (np.ndarray): 描述转发概率的矩阵 (只读)。
        initial_deliverers (list): 初始投放者（种子节点）的列表。
        ... (其他概率分布)

    Returns:
        np.ndarray: 一个n维的0/1向量，1表示该节点在本轮模拟中成功使用了优惠券。
    """
    n = tranProMatrix.shape[0]
    users_useAndDis = set()
    
    # 为每个初始投放者启动一个独立的随机游走
    for start_user in initial_deliverers:
        current_user = start_user
        
        # 模拟单张优惠券的随机游走过程
        while True:
            rand_pro = np.random.rand()
            # 检查当前节点是否已经做出过决定
            if current_user in users_useAndDis:
                # 做出过决定 再次接触优惠券的逻辑
                # 免疫/饱和效应：节点再次收到后更容易使用或丢弃
                if rand_pro < (dis_distribution[current_user] + 
                              constantFactor_distribution[current_user] * succ_distribution[current_user]):
                    break # 游走在此中断
            else:
                # 首次接触优惠券的逻辑
                if rand_pro < succ_distribution[current_user]:
                    # 决定“使用”
                    users_useAndDis.add(current_user)
                    # 游走在此中断，因为优惠券被使用了
                    break 
                elif rand_pro < (succ_distribution[current_user] + dis_distribution[current_user]):
                    # 决定“丢弃”
                    # todo gzc: 丢弃也算成功使用 吸收态
                    users_useAndDis.add(current_user) # 即使丢弃，也算“处理过”
                    break # 游走在此中断
            
            # 如果没有中断，则意味着节点决定“转发”
            next_node = _select_next_neighbor(current_user, tranProMatrix)
            
            if next_node is None:
                # 没有邻居可转发，游走中断
                break
            else:
                # 更新当前节点，继续游走
                current_user = next_node

    # 将最终成功使用的节点集合转换为0/1向量
    success_vector = np.zeros(n, dtype=int)
    if users_useAndDis: # 如果集合不为空
        activated_list = list(users_useAndDis)
        success_vector[activated_list] = 1
        
    return success_vector


def _select_next_neighbor(current_user: int, 
                          tranProMatrix: np.ndarray
                          ) -> int:
    """
    从当前节点的邻居中，根据转发概率矩阵选择下一个节点。
    """
    # 找到邻居及其对应的转发概率
    neighbors = np.nonzero(tranProMatrix[:, current_user])[0]
    
    if len(neighbors) == 0:
        return None 
        
    probabilities = tranProMatrix[neighbors, current_user]
    prob_sum = np.sum(probabilities)
    
    if prob_sum <= 0:
        # 如果概率和为0（或负数，异常情况），则均匀选择一个邻居
        return np.random.choice(neighbors)
    
    # 归一化概率并选择
    normalized_probs = probabilities / prob_sum
    return np.random.choice(neighbors, p=normalized_probs)

def monteCarlo_singleTime_firstDiscard_improved(
    tranProMatrix: np.ndarray,
    initial_deliverers: list,
    succ_distribution: np.ndarray,
    dis_distribution: np.ndarray,
    constantFactor_distribution: np.ndarray
) -> np.ndarray:
    """
    执行单次蒙特卡洛模拟，采用"首次丢弃即改变"的个性化策略。
    此模型下，成功用户被定义为所有“首次使用者”和“首次丢弃者”的集合。
    此函数是“纯”的，不会修改任何输入参数。

    Args:
        tranProMatrix (np.ndarray): 描述转发概率的矩阵 (只读)。
        initial_deliverers (list): 初始投放者（种子节点）的列表。
        ... (其他概率分布)

    Returns:
        np.ndarray: 一个n维的0/1向量，1表示该节点在本轮模拟中被视为成功。
    """
    n = tranProMatrix.shape[0]
    # 使用集合(set)来高效地管理不同状态的节点
    successful_users = set()  # 记录首次“使用”的节点
    first_discarders = set()  # 记录首次“丢弃”的节点
    
    # 为每个初始投放者启动一个独立的随机游走
    for start_node in initial_deliverers:
        current_node = start_node
        
        # 模拟单张优惠券的随机游走过程
        while True:
            # 检查节点是否已经被处理过（无论是使用还是首次丢弃）
            if current_node in successful_users or current_node in first_discarders:
                # 再次接触的逻辑
                rand_pro = np.random.rand()
                # 再次接触后决定丢弃
                if rand_pro < (dis_distribution[current_node] + 
                              constantFactor_distribution[current_node] * succ_distribution[current_node]):
                    # 原始代码在这里有特殊逻辑：如果再次接触后丢弃，并且之前不是首次丢弃者，
                    # 把它加入firstdiscard列表。我们在这里也模拟这个行为。
                    if current_node not in first_discarders:
                        first_discarders.add(current_node)
                    break # 游走中断
            else:
                # 首次接触的逻辑
                rand_pro = np.random.rand()
                if rand_pro < succ_distribution[current_node]:
                    # 决定“使用”
                    successful_users.add(current_node)
                    break 
                elif rand_pro < (succ_distribution[current_node] + dis_distribution[current_node]):
                    # 决定“丢弃”
                    first_discarders.add(current_node)
                    break
                # 否则，节点决定“转发”，游走继续
            
            # 转发逻辑
            next_node = _select_next_neighbor(current_node, tranProMatrix)
            
            if next_node is None:
                # 无处可去，游走中断
                break
            else:
                # 更新当前节点，继续游走
                current_node = next_node

    # 根据模型假设，成功用户是首次使用者和首次丢弃者的并集
    final_activated_nodes = successful_users.union(first_discarders)
    
    # 将最终成功用户的集合转换为0/1向量
    success_vector = np.zeros(n, dtype=int)
    if final_activated_nodes:
        activated_list = list(final_activated_nodes)
        success_vector[activated_list] = 1
        
    return success_vector

def monteCarlo_singleTime_firstUnused_improved(
    tranProMatrix: np.ndarray,
    initial_deliverers: list,
    succ_distribution: np.ndarray,
    dis_distribution: np.ndarray,
    constantFactor_distribution: np.ndarray
) -> np.ndarray:
    """
    执行单次蒙特卡洛模拟，采用"首次接触即改变"的个性化策略。
    在此模型中，我们假设所有接触过优惠券的节点最终都会被视为“成功”。

    Args:
        tranProMatrix (np.ndarray): 描述转发概率的矩阵 (只读)。
        initial_deliverers (list): 初始投放者（种子节点）的列表。
        ... (其他概率分布)

    Returns:
        np.ndarray: 一个n维的0/1向量，1表示该节点在本轮模拟中接触过优惠券。
    """
    n = tranProMatrix.shape[0]
    # 使用集合(set)来高效存储所有接触过优惠券的节点
    contacted_nodes = set()
    
    # 为每个初始投放者启动一个独立的随机游走
    for start_node in initial_deliverers:
        current_node = start_node
        
        # 模拟单张优惠券的随机游走过程
        while True:
            # 检查当前节点是否已经接触过优惠券
            if current_node in contacted_nodes:
                # 再次接触的逻辑
                rand_pro = np.random.rand()
                if rand_pro < (dis_distribution[current_node] + 
                              constantFactor_distribution[current_node] * succ_distribution[current_node]):
                    break # 游走在此中断
            else:
                # 首次接触的逻辑
                # 无论做出何种决定，该节点都算“被接触过”
                contacted_nodes.add(current_node)
                
                rand_pro = np.random.rand()
                if rand_pro < succ_distribution[current_node]:
                    # 决定“使用”，游走中断
                    break 
                elif rand_pro < (succ_distribution[current_node] + dis_distribution[current_node]):
                    # 决定“丢弃”，游走中断
                    break 
                # 否则，节点决定“转发”，游走继续
            
            # 转发逻辑
            next_node = _select_next_neighbor(current_node, tranProMatrix)
            
            if next_node is None:
                # 无处可去，游走中断
                break
            else:
                # 更新当前节点，继续游走
                current_node = next_node

    # 根据模型假设，所有接触过的节点都算成功，将其转换为0/1向量
    success_vector = np.zeros(n, dtype=int)
    if contacted_nodes:
        contacted_list = list(contacted_nodes)
        success_vector[contacted_list] = 1
        
    return success_vector


def monteCarloSimulation(tranProMatrix,indexes,L,succ_distribution,dis_distribution,constantFactor_distribution,personalization):
    best_next_deliverer = find_next_best_deliverer(current_deliverers=indexes, tranProMatrix=tranProMatrix,L=L,
                             succ_distribution=succ_distribution, dis_distribution=dis_distribution, constantFactor_distribution=constantFactor_distribution,
                             personalization=personalization)
    return best_next_deliverer


if __name__ == '__main__':

    G = nx.Graph()
    G.add_nodes_from([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]) 
    G.add_edges_from([(0, 2), (0, 3), (1, 2), (1, 4), (2, 4), (4,5), (4,6), (4,7),(4,8),(4,9), (5,6),(5,9)])
    adj = nx.adjacency_matrix(G)

    n = adj.shape[0]
    use_pro = [0.01,0.2,0.3,0.4,0.3,
               0.1,0.3,0.5,0.2,0.2]
    dis_pro = [0.04,0.2,0.2,0.1,0.1,
               0.4,0.3,0.2,0.1,0.3]
    L = 5
    constantFactor = [1,1,1,1,1,
                      1,1,1,1,1]
    users = []
    initial_tran_distribution = np.array([0.001, 0.5, 0.9, 0.2, 0.7, 0.8, 0.5, 0.9, 0.2, 0.7])

    tranProMatrix, neighbor_having = single_deliverer.getTranProMatrix(adj,initial_tran_distribution)
    bestSingleDeliverer = single_deliverer.getBestSingleDeliverer(tranProMatrix,use_pro,neighbor_having)
    indexes = [bestSingleDeliverer]
    best_next_deliverer = monteCarloSimulation(tranProMatrix,indexes,L,
                                               succ_distribution=use_pro,
                                               dis_distribution=dis_pro,
                                               constantFactor_distribution=constantFactor,
                                               personalization=users)
    print("best_next_deliverer is: ",best_next_deliverer)