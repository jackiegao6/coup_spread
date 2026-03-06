import numpy as np
import scipy.sparse as sp
import networkx as nx
import pickle
import os

def create_ultimate_adversarial_graph(n=2000, out_dir="/home/wen/work/coup_spread/dataset/"):
    print("正在生成终极对抗图 (The Ultimate Adversarial Graph)...")
    
    # 精确划分阵营
    n_comm = int(n * 0.70)      # 1400 节点: 社区
    n_sinks = int(n * 0.20)     # 400 节点: Alpha孤岛
    n_traps = int(n * 0.05)     # 100 节点: PageRank死亡陷阱
    n_bridges = int(n * 0.05)   # 100 节点: RIS专属桥梁

    G = nx.DiGraph()
    G.add_nodes_from(range(n))

    # 1. 社区 (0 到 1399)
    # 生成一个随机规则图 (每个节点连10个人)，确保内部连通，但不连向外面
    comm_sub = nx.random_regular_graph(10, n_comm)
    for u, v in comm_sub.edges():
        G.add_edge(u, v)
        G.add_edge(v, u) # 双向边

    # 2. Alpha孤岛 (1400 到 1799)
    # 孤立节点，无任何出边，天然的断头路

    # 3. 死亡陷阱大V群 (1800 到 1899)
    # 建立一个“全连接”的克里格(Clique)！每个人都连向另外99个人。
    # 这里的度数和 PageRank 将会是全网绝对最高！吸走 Baseline 的所有注意力！
    trap_start = n_comm + n_sinks
    for i in range(trap_start, trap_start + n_traps):
        for j in range(trap_start, trap_start + n_traps):
            if i != j:
                G.add_edge(i, j)

    # 4. 隐形桥梁 (1900 到 1999)
    # 度数极低（只随机连向社区的3个人），PageRank和Degree极低
    bridge_start = trap_start + n_traps
    for i in range(bridge_start, bridge_start + n_bridges):
        targets = np.random.choice(n_comm, 3, replace=False)
        for t in targets:
            G.add_edge(i, t) # 单向进入社区

    # 转换为稀疏矩阵并保存
    adj = nx.to_scipy_sparse_array(G, format='csr')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"ultimate_{n}-adj.pkl")
    
    with open(out_path, 'wb') as f:
        pickle.dump(adj, f, protocol=pickle.HIGHEST_PROTOCOL)
        
    print(f"✅ 终极图生成完毕！保存在 {out_path}")
    print(f"   Trap节点最大度数: {n_traps - 1} (诱饵)")
    print(f"   Bridge节点度数: 3 (隐藏桥梁)")

if __name__ == "__main__":
    create_ultimate_adversarial_graph(2000)
    '''
    正在生成终极对抗图 (The Ultimate Adversarial Graph)...
✅ 终极图生成完毕！保存在 /home/wen/work/coup_spread/dataset/ultimate_2000-adj.pkl
   Trap节点最大度数: 99 (诱饵)
   Bridge节点度数: 3 (隐藏桥梁)
   '''