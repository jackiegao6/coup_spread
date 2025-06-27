



def create_seed_num_list(
    total_nodes: int, 
    num_steps: int, 
    scale_factor: int = 1000
) -> list[int]:
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