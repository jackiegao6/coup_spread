import pickle
import numpy as np
import scipy.sparse as sp

def analyze_graph_from_pkl(pkl_path):
    # 1. 读取 .pkl 文件
    with open(pkl_path, "rb") as f:
        obj = pickle.load(f)

    print("读取到的对象类型:", type(obj))

    # 如果是 CSR 稀疏矩阵
    if sp.issparse(obj):
        print("检测到 scipy CSR 稀疏矩阵")
        matrix = obj.tocsr()     # 保证是 csr 格式
        dense = matrix.toarray() # 转为普通矩阵
    else:
        print("❌ 文件不是稀疏矩阵，也不是常规矩阵")
        return

    # ==========================
    #     图结构分析
    # ==========================
    num_nodes = dense.shape[0]

    # 是否对称（无向图）
    is_symmetric = np.array_equal(dense, dense.T)

    # 边数量
    if is_symmetric:
        # 对于无向图，只统计上三角
        num_edges = np.sum(np.triu(dense, 1) != 0)
    else:
        # 有向图，直接数所有非 0 元素
        num_edges = np.sum(dense != 0)

    print("\n分析结果：")
    print(f"节点数量: {num_nodes}")
    print(f"是否对称: {is_symmetric}")
    print(f"边数量: {num_edges}")

    return {
        "nodes": num_nodes,
        "edges": num_edges,
        "symmetric": is_symmetric
    }


# 调用示例
# result = analyze_graph_from_pkl("facebook-adj.pkl")

# 调用示例
if __name__ == '__main__':

    '''
    分析结果：
        节点数量: 4039
        是否对称: True
        边数量: 88234
    '''
    # result = analyze_graph_from_pkl("/home/wen/pythonspace/data-test/datasets/facebook-adj.pkl")

    '''
    分析结果：
        节点数量: 23362
        是否对称: True
        边数量: 32774
    '''
    # result = analyze_graph_from_pkl("/home/wen/pythonspace/data-test/datasets/Twitter-adj.pkl")

    '''
    分析结果：
        节点数量: 3748
        是否对称: True
        边数量: 81903
    '''
    result = analyze_graph_from_pkl("/home/wen/pythonspace/data-test/datasets/Mich-adj.pkl")

