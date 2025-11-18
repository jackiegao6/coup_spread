import numpy as np
import networkx as nx
import get_coupon_deliverers
import single_deliverer

if __name__ == "__main__":
    print("\n" + "="*20 + " 测试开始 " + "="*20)

    # --- a. 设置测试场景 ---
    # 假设有5个用户 (0, 1, 2, 3, 4)
    # 假设一个邻接关系 (为了模拟find_next_best_deliverer中的重叠效应)
    # 0-1, 1-2, 2-3, 3-4, 4-0
    G = nx.Graph()
    G.add_nodes_from([0, 1, 2, 3, 4]) 
    # 定义他们之间的好友关系
    G.add_edges_from([(0, 1), (0, 2), (1, 2), (1, 3), (2, 3), (2, 4), (3, 4)])

    # 获取图的邻接矩阵（SciPy稀疏矩阵格式）
    adj_matrix = nx.adjacency_matrix(G, nodelist=sorted(G.nodes()))
    print("="*20 + " 图结构信息 " + "="*20)
    print("节点列表:", sorted(G.nodes()))
    print("边列表:", G.edges())
    print("邻接矩阵 (A):\n", adj_matrix.toarray())
    print("-" * 55)
    
    # 假设每个用户的行为概率
    # succ_distribution: 用户4的使用意愿最高，其次是用户2
    succ_distribution = np.array([0.1, 0.2, 0.6, 0.3, 0.8])
    dis_distribution = np.array([0.8, 0.7, 0.2, 0.1, 0.1])
    forwarding_probs = np.array([0.1, 0.1, 0.2, 0.6, 0.1])

    constantFactor_distribution = np.zeros(5) # 简化，不使用

    # --- b. 定义测试参数 ---
    num_to_select = 3  # 我们想选出3个最佳投放者
    simulation_rounds = 1000 # 蒙特卡洛模拟次数 (在我们的mock中未使用，但需传入)
    personalization_params = {} # 简化，不使用

    # --- c. 执行被测试的函数 ---
    print(f"\n目标：从5个用户中选择 {num_to_select} 个最佳投放者。")
    print("用户使用意愿分布:", succ_distribution)
    print("-" * 55)

    tranProMatrix, degrees = single_deliverer.getTranProMatrix(adj_matrix, forwarding_probs)


    selected_nodes = get_coupon_deliverers.select_deliverers_improved(
        m=num_to_select,
        init_tranProMatrix=tranProMatrix, 
        succ_distribution=succ_distribution,
        dis_distribution=dis_distribution,
        constantFactor_distribution=constantFactor_distribution,
        L=simulation_rounds,
        personalization=personalization_params
    )
    
    print("-" * 55)
    print(f"函数返回的最终投放者列表: {selected_nodes}")

    # --- d. 验证结果 ---
    # 根据我们的模拟逻辑，我们期望：
    # 1. 第一个选出的应该是 succ_distribution 最高的节点，即节点 4。
    # 2. 第二个选出时，节点4的邻居（2和3）的边际收益会打折。
    #    - 节点0收益: 0.1
    #    - 节点1收益: 0.2
    #    - 节点2收益: 0.6 * 0.5 = 0.3
    #    - 节点3收益: 0.3 * 0.5 = 0.15
    #    因此，第二个选出的应该是节点 2。
    # 3. 第三个选出时，已选集合为 {4, 2}。
    #    - 节点0 (邻居2): 0.1 * 0.5 = 0.05
    #    - 节点1 (邻居2): 0.2 * 0.5 = 0.1
    #    - 节点3 (邻居2,4): 0.3 * 0.5 = 0.15
    #    因此，第三个选出的应该是节点 3。
    # 期望的最终结果是 [4, 2, 3]
    expected_result = [4, 2, 3]
    
    print(f"期望的结果列表: {expected_result}")

    assert selected_nodes == expected_result, f"测试失败！期望得到 {expected_result}，但实际得到 {selected_nodes}"

    print("\n测试成功！函数行为符合预期。")
    print("="*20 + " 测试结束 " + "="*20)