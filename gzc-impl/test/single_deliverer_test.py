import numpy as np
import networkx as nx
import single_deliverer
import scipy.sparse
import copy


G = nx.Graph()
G.add_nodes_from([0, 1, 2, 3, 4]) 
G.add_edges_from([(0, 1), (0, 2), (1, 2), (1, 3), (2, 3), (2, 4), (3, 4)])

adj_matrix = nx.adjacency_matrix(G, nodelist=sorted(G.nodes()))
print("="*20 + " 图结构信息 " + "="*20)
print("邻接矩阵 (A):\n", adj_matrix.toarray())
print("-" * 55)

# --- 步骤 2: 定义每个用户的行为概率 ---
# a) 定义每个用户收到优惠券后，将其【转发出去】的总概率
# 例如，用户0比较活跃，有80%的概率会转发；用户3比较内向，只有20%概率转发
forwarding_probs = np.array([0.8, 0.6, 0.9, 0.2, 0.5])

# b) 定义每个用户收到优惠券后，【直接使用它】的概率
# 注意：一个用户可以既使用又转发，这里的概率是独立的。
# 但为了符合马尔可夫链的设定（转移+吸收），我们假设 P(使用) + P(转发) + P(丢弃) = 1
# 为了简化，我们假设"使用"的概率独立于"转发"。succ_distribution 是吸收概率的一部分。
# 例如，用户4是价格敏感型，有70%概率会使用优惠券
usage_probs = np.array([0.1, 0.3, 0.05, 0.6, 0.7])


print("\n" + "="*20 + " 行为概率设定 " + "="*20)
print("各用户的总转发概率 (tran_distribution):\n", forwarding_probs)
print("各用户的直接使用概率 (succ_distribution):\n", usage_probs)
print("-" * 55)

# --- 步骤 3: 计算转发概率矩阵 ---
# 调用第一个函数，得到瞬时态之间的转移矩阵tranProMatrix
tranProMatrix, degrees = single_deliverer.getTranProMatrix(adj_matrix, forwarding_probs)

print("\n" + "="*20 + " 计算转发矩阵 " + "="*20)
print("各节点的度 (邻居数量):\n", degrees)
# 打印结果，保留两位小数以便观察
print("\n转发概率矩阵tranProMatrix:\n", np.round(tranProMatrix, 2))
# 验证：矩阵的第i列和应该等于第i个用户的总转发概率
print("\n验证：矩阵各列之和 (应等于总转发概率):\n", np.round(np.sum(tranProMatrix, axis=0), 2))
print("-" * 55)

# # --- 步骤 4: 寻找最佳投放节点 ---
print("\n" + "="*20 + " 寻找最佳投放点 " + "="*20)

# # 场景A: 网络中所有人都未参与活动
print("--- 场景 A: 初始投放 ---")
best_node_A = single_deliverer.getBestSingleDeliverer(tranProMatrix, usage_probs, [])
print(f"\n结论：最佳的初始投放节点是用户【{best_node_A}】。")

print("-" * 55)

# 场景B: 假设用户4已经参与过活动（比如已用过一张券），不能再贡献“使用量”了
# print("--- 场景 B: 用户4已参与活动 ---")
# users_out = [4]
# best_node_B, all_pros_B = single_deliverer.getBestSingleDeliverer(tranProMatrix, usage_probs, users_useAndDis=users_out)
# print(f"设定：用户 {users_out} 不再能使用优惠券。")
# print(f"从每个节点开始投放，最终期望的总使用量:\n{np.round(all_pros_B, 3)}")
# print(f"\n结论：在这种情况下，最佳的投放节点变为用户【{best_node_B}】。")
# print(f"从他开始，预计能带来 {np.round(all_pros_B[best_node_B], 3)} 次优惠券使用。")