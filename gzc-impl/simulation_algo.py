from typing import Optional
import numpy as np

def _select_next_neighbor_old(
    current_user: int,
    tranProMatrix: np.ndarray,
    rand_pro
) -> Optional[int]:
    """
    从当前节点的邻居中，根据转发概率矩阵 和 rand_pro参数 选择邻居
    Args:
        current_user: 当前节点编号
        tranProMatrix: 转移概率矩阵 (n x n)，tranProMatrix[i, j] 表示 j -> i 的转发概率
    Returns:
        邻居节点编号
    """
    # 找到邻居及其对应的转发概率
    neighbors = np.flatnonzero(tranProMatrix[:, current_user])
    if neighbors.size == 0:
        return None

    probabilities = tranProMatrix[neighbors, current_user]
    prob_sum = np.sum(probabilities)

    if prob_sum == 0:
        return None

    # 归一化概率
    normalized_probs = probabilities / prob_sum
    # 计算累积概率分布
    cumulative_probs = np.cumsum(normalized_probs)
    selected_index = np.searchsorted(cumulative_probs, rand_pro)
    if selected_index >= len(neighbors):
        selected_index = len(neighbors) - 1
    return neighbors[selected_index]


# AgainReJudge
def monteCarlo_singleTime_improved2(
    tranProMatrix: np.ndarray,
    initial_deliverers: list,
    succ_distribution: np.ndarray,
    dis_distribution: np.ndarray,
    constantFactor_distribution: np.ndarray
) -> np.ndarray:
    """
    如果一个用户已经使用过优惠券 activatedUsers, 当他再次收到券时，他依然会按照概率决定是“再次使用”、“丢弃”还是“转发
    由于用户已经是“激活”状态，即使他决定“使用”，总激活人数也不会增加
    在这种模式下，已激活用户变成了优惠券的“汇点”（Sink）。券一旦被他们再次决定使用或丢弃，游走就终止，券就从网络中消失了。这模拟了现实中“用户对重复优惠券感到厌烦或直接核销但不产生额外收益”的场景
    """

    n = tranProMatrix.shape[0]
    activatedUsers = set()

    for start_user in initial_deliverers: # 为每个初始投放者启动一个独立的随机游走

        current_user = start_user

        # 模拟单张优惠券的随机游走过程 一旦使用 || 丢弃 循环将就break
        while True:
            rand_pro = np.random.rand()
            p_succ = succ_distribution[current_user]
            p_dis = dis_distribution[current_user]
            threshold = p_succ + p_dis # 动作发生的总概率（使用+丢弃）

            if current_user in activatedUsers:
                # 当一个节点被激活过了 再次接触优惠券 可能会消耗掉券但不增加新激活人数
                if rand_pro < succ_distribution[current_user]:
                    # 继续用
                    break
                elif rand_pro < (succ_distribution[current_user] + dis_distribution[current_user]):
                    # 继续丢弃
                    break

            else:
                # 没有被激活过的节点 接触优惠券的逻辑
                if rand_pro < succ_distribution[current_user]:
                    # 决定“使用”
                    activatedUsers.add(current_user)
                    # 游走在此中断，因为优惠券被使用了
                    break
                elif rand_pro < (succ_distribution[current_user] + dis_distribution[current_user]):
                    # 决定“丢弃”
                    break # 游走在此中断

            # 如果没有中断，则意味着节点决定“转发”
            remaining_prob = 1.0 - threshold

            # 为了在转发给不同邻居时依然能公平地利用这个随机数，代码将其线性映射回了 [0, 1] 区间 在逻辑上保证了单次决策中概率空间的完整性
            rescaled_rand_pro = (rand_pro - threshold) / remaining_prob

            next_node = _select_next_neighbor_old(current_user, tranProMatrix, rescaled_rand_pro)

            if next_node is None:
                # 没有邻居可转发，游走中断
                break
            else:
                # 更新当前节点，继续游走
                current_user = next_node

    # 将最终成功使用的节点集合转换为0/1向量
    success_vector = np.zeros(n, dtype=int)
    success_vector[list(activatedUsers)] = 1
    return success_vector

# deprecated 如若使用 记得缩放概率
def monteCarlo_singleTime_improved2_AgainContinue(
    tranProMatrix: np.ndarray,
    initial_deliverers: list,
    succ_distribution: np.ndarray,
    dis_distribution: np.ndarray,
    constantFactor_distribution: np.ndarray
) -> np.ndarray:
    '''
    如果用户已经激活，他收到新券后直接跳过“使用/丢弃”判定，必定尝试转发给邻居
    已激活用户不再拦截券，而是充当了纯粹的“中继站（Relay）”
    这种模式下，优惠券在网络中的寿命会更长，最终的总激活人数通常会显著高于第一种模式。这模拟了“自动转发”或“用户即便用过了也会顺手发给别人”的情景
    '''

    n = tranProMatrix.shape[0]
    activatedUsers = set()
    activated_list = []

    # 为每个初始投放者启动一个独立的随机游走
    for start_user in initial_deliverers:
        # logging.info(f"\t\t\t当前模拟的起始节点: {start_user}")
        current_user = start_user

        # 模拟单张优惠券的随机游走过程
        while True:
            rand_pro = np.random.rand()

            # 检查当前节点是否已经做出过决定
            if current_user not in activatedUsers:
                # 首次接触优惠券的逻辑
                if rand_pro < succ_distribution[current_user]:
                    # 决定“使用”
                    activatedUsers.add(current_user)
                    activated_list.append(current_user)
                    # 游走在此中断，因为优惠券被使用了
                    break
                elif rand_pro < (succ_distribution[current_user] + dis_distribution[current_user]):
                    # 决定“丢弃”
                    break # 游走在此中断

            # 做出过决定 再次接触优惠券的逻辑 直接转发 || 如果没有中断，则意味着节点决定“转发”
            next_node = _select_next_neighbor_old(current_user, tranProMatrix, rand_pro)

            if next_node is None:
                # 没有邻居可转发，游走中断
                break
            else:
                # 更新当前节点，继续游走
                current_user = next_node

    # 将最终成功使用的节点集合转换为0/1向量
    success_vector = np.zeros(n, dtype=int)

    success_vector[activated_list] = 1

    return success_vector
