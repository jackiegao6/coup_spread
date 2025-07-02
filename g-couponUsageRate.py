import networkx as nx
import pickle
import logging


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


def preprocess_graph_from_edges(edge_file_path: str, output_adj_path: str) -> bool:
    """
    从边列表文件加载图，计算其邻接矩阵，并保存为 pickle 文件。

    Args:
        edge_file_path (str): 输入的 .edges 文件的路径。
        output_adj_path (str): 输出的邻接矩阵 .pkl 文件的路径。

    Returns:
        bool: 如果处理成功则返回 True，否则返回 False。
    """
    logging.info(f"Attempting to load edges from: {edge_file_path}")
    
    try:
        with open(edge_file_path, "r") as f:
            network = nx.Graph()
            for i, line in enumerate(f):
                try:
                    parts = line.strip().split()
                    if len(parts) < 2:
                        logging.warning(f"Skipping malformed line {i+1} in {edge_file_path}: not enough parts.")
                        continue
                    node_from = int(parts[0])
                    node_to = int(parts[1])
                    network.add_edge(node_from, node_to)
                except (ValueError, IndexError) as e:
                    logging.warning(f"Skipping malformed line {i+1} in {edge_file_path}: {e}")
                    continue
    except FileNotFoundError:
        logging.error(f"Edge file not found at: {edge_file_path}")
        return False

    if not network.nodes:
        logging.warning("Graph is empty after loading edges. No adjacency matrix will be saved.")
        return False
        
    logging.info(f"Graph loaded successfully: {network.number_of_nodes()} nodes, {network.number_of_edges()} edges.")
    
    logging.info("Calculating adjacency matrix...")
    # NetworkX 默认会创建 scipy.sparse 矩阵，这对于大图是高效的
    adj = nx.adjacency_matrix(network)
    
    logging.info(f"Saving adjacency matrix to: {output_adj_path}")
    try:
        with open(output_adj_path, "wb") as f:
            pickle.dump(adj, f)
    except IOError as e:
        logging.error(f"Failed to write to output file {output_adj_path}: {e}")
        return False
        
    logging.info("Processing complete.")
    return True

if __name__ == '__main__':

    # In your main orchestrator script
    # ...
    experiment_data = load_experiment_data(config)
    n = experiment_data["n"]

    # 现在调用纯粹的计算函数
    seedNumList = create_seed_num_list(total_nodes=n, num_steps=10, scale_factor=1000)
    # ...
    