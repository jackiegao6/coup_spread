import urllib.request
import networkx as nx
import scipy.sparse as sp
import pickle
import os
import gzip

def download_and_convert(url, output_path, is_directed=False, is_snap=False):
    print(f"正在处理网络，保存至: {output_path}")
    
    # 1. 下载数据
    temp_file = "temp_edgelist.txt"
    if url.endswith('.gz'):
        temp_file = "temp_edgelist.txt.gz"
        
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    with urllib.request.urlopen(req) as response, open(temp_file, 'wb') as out_file:
        out_file.write(response.read())

    # 2. 读取为 NetworkX 图
    G = nx.DiGraph() if is_directed else nx.Graph()
    
    open_func = gzip.open if temp_file.endswith('.gz') else open
    with open_func(temp_file, 'rt') as f:
        for line in f:
            line = line.strip()
            # 忽略注释
            if line.startswith('#') or line.startswith('%') or not line:
                continue
            parts = line.split()
            if len(parts) >= 2:
                u, v = int(parts[0]), int(parts[1])
                G.add_edge(u, v)
                if not is_directed:
                    G.add_edge(v, u)

    # 3. 提取最大连通子图 (剔除离群孤岛，让实验更紧凑)
    if is_directed:
        largest_cc = max(nx.weakly_connected_components(G), key=len)
    else:
        largest_cc = max(nx.connected_components(G), key=len)
    
    G_core = G.subgraph(largest_cc).copy()
    
    # 重映射节点 ID 为连续的 0 ~ N-1
    mapping = {old_id: new_id for new_id, old_id in enumerate(G_core.nodes())}
    G_core = nx.relabel_nodes(G_core, mapping)

    print(f"  -> 最大连通子图提取完毕: 节点数={G_core.number_of_nodes()}, 边数={G_core.number_of_edges()}")

    # 4. 转换为 CSR 稀疏矩阵并保存
    adj_sparse = nx.to_scipy_sparse_array(G_core, format='csr')
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'wb') as f:
        pickle.dump(adj_sparse, f, protocol=pickle.HIGHEST_PROTOCOL)
        
    # 清理临时文件
    os.remove(temp_file)
    print(f"✅ 转换完成！\n")


if __name__ == "__main__":
    out_dir = "/home/wen/work/coup_spread/dataset/downloaded_real_networks/"
    
    # 1. Email-Eu-core (约 1000 节点) - SNAP 斯坦福数据集
    download_and_convert(
        url="https://snap.stanford.edu/data/email-Eu-core.txt.gz",
        output_path=os.path.join(out_dir, "real_email_1000-adj.pkl"),
        is_directed=True
    )

    # 2. C. elegans 神经网络 (约 300 节点)
    download_and_convert(
        url="http://konect.cc/files/download.tsv.celegansneural.tar.bz2", 
        # Konect 的下载有时不稳定，这里用 GitHub 上备份的经典小数据集 URL
        # 考虑到服务器连通性，我用一个公共的 raw url 代替
        # url="https://raw.githubusercontent.com/networkrepository/networkrepository.github.io/master/network/soc-jazz.edges",
        output_path=os.path.join(out_dir, "real_jazz_198-adj.pkl"),
        is_directed=False
    )
    
    print("全部真实数据集下载与转换完毕！")