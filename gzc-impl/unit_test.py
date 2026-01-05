import numpy as np
import scipy.sparse as sp
import logging
import sys

# 假设你的算法都在 coupon_usage_rate_get_distribution_degree_aware.py 或者其他地方
# 这里我们需要用到你的 CELF 和 RIS 函数，以及辅助函数
import get_trans_matrix  # 用于生成转移矩阵
from SSR_method import CouponInfluenceMaximizer # 你的RIS类
# 引入你的 CELF 函数（假设在某个文件里，如果没有，请把CELF函数贴在下面）
# from your_file import deliverers_monteCarlo_CELF 

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(message)s')

def run_sanity_check():
    print("="*50)
    print("开始进行【星型网络】自查测试")
    print("目标：验证 RIS/CELF 是否能识别出'传播枢纽'")
    print("="*50)

    # 1. 构造数据：10个节点，0指向1-9
    n = 10
    # 构造邻接矩阵 (0->1, 0->2 ... 0->9)
    # adj[i, j] = 1 表示 i->j
    adj = np.zeros((n, n))
    for i in range(1, 10):
        adj[0, i] = 1 
    
    adj_sparse = sp.csr_matrix(adj)
    
    # 2. 生成转移矩阵 (列归一化)
    # 你的 getTranProMatrix 逻辑是 adj.T 然后归一化
    # 0->i, 所以在 adj.T 中是 [i, 0] = 1. 0 是源，在列上。
    # 2. 生成转移矩阵
    tran_matrix = get_trans_matrix.getTranProMatrix(adj_sparse)
    
    # ==========================================
    # 【关键修复】：强制将所有边的权重设为 1.0
    # 这样 RIS 反向传播时，0 必定能到达 1-9
    # ==========================================
    if sp.issparse(tran_matrix):
        tran_matrix.data[:] = 1.0  # 将所有非零元素设为 1
    else:
        tran_matrix[tran_matrix > 0] = 1.0
    
    # 3. 构造极端的概率分布
    # succ: 0号节点极低，其他人极高
    succ_dist = np.array([0.01] + [0.99]*9) 
    # dis: 全为 0
    dis_dist = np.zeros(n)
    # tran: 全为 1
    tran_dist = np.ones(n)
    
    # 构造 alpha 字典供 RIS/Alpha_sort 使用
    alpha_dict = {i: succ_dist[i] for i in range(n)}
    
    print(f"设定：节点 0 (P_succ=0.01) -> 连接 -> 节点 1-9 (P_succ=0.99)")
    print(f"设定：转发率 P_tran = 1.0, 丢弃率 P_dis = 0.0")
    print("-" * 30)

    # ------------------------------------------------
    # 测试 A: Alpha_Sort
    # ------------------------------------------------
    print("\n[测试 1] Alpha_Sort (预期：不选 0，选 1-9)")
    # 模拟简单的 alpha sort
    sorted_idx = np.argsort(succ_dist)[::-1]
    alpha_seed = sorted_idx[0]
    print(f"-> Alpha_Sort 选择了: {alpha_seed} (P_succ={succ_dist[alpha_seed]})")
    
    # ------------------------------------------------
    # 测试 B: CELF (模拟验证)
    # ------------------------------------------------
    print("\n[测试 2] CELF / Monte Carlo (预期：必须选 0)")
    
    # 简易版 CELF/MC 评估函数
    def simulate_spread(seed_node):
        # 简单模拟：AgainContinue
        total_activated = 0
        for _ in range(100): # 模拟100次
            activated = set()
            # 种子先尝试激活
            if np.random.rand() < succ_dist[seed_node]:
                activated.add(seed_node)
                continue # 用了就没了
            
            # 没用，尝试转发
            # 0 只有出边到 1-9
            targets = np.nonzero(tran_matrix[:, seed_node])[0] # 找出邻居
            if len(targets) > 0:
                # 随机传给一个 (在这个星型图中，0传给谁概率都是均等的 1/9)
                next_node = np.random.choice(targets)
                # 邻居尝试激活
                if np.random.rand() < succ_dist[next_node]:
                    activated.add(next_node)
            total_activated += len(activated)
        return total_activated / 100

    # 评估节点 0
    score_0 = simulate_spread(0)
    # 评估节点 1 (孤家寡人，也没人传给它，它只能靠自己)
    score_1 = simulate_spread(1)
    
    print(f"-> 节点 0 (枢纽) 的模拟期望得分: {score_0:.2f}")
    print(f"-> 节点 1 (叶子) 的模拟期望得分: {score_1:.2f}")
    
    if score_0 > score_1:
        print("★ 结论: MC 逻辑正常 (枢纽 > 个体)")
    else:
        print("❌ 结论: MC 逻辑异常 (Bug!)")

    # ------------------------------------------------
    # 测试 C: RIS (代码验证)
    # ------------------------------------------------
    print("\n[测试 3] RIS (预期：必须选 0)")
    
    # 调用你的 RIS 类
    try:
        # 注意：这里的 tranProMatrix 需要是你的代码接受的格式 (通常是 numpy array)
        if hasattr(tran_matrix, 'toarray'):
            tran_matrix_dense = tran_matrix.toarray()
        else:
            tran_matrix_dense = tran_matrix

        maximizer = CouponInfluenceMaximizer(
            adj=adj_sparse,
            tranProMatrix=tran_matrix_dense,
            alpha=alpha_dict,
            k=1 # 选1个种子
        )
        
        # 运行采样
        maximizer.generate_rr_sets_parallel(N=10, workers=1) # 单进程跑
        selected_seeds, est_inf = maximizer.select_seeds()
        
        print(f"-> RIS 选择了: {selected_seeds[0]}")
        
        if selected_seeds[0] == 0:
            print("★ 结论: RIS 逻辑正常 (成功识别枢纽)")
        else:
            print(f"❌ 结论: RIS 逻辑异常 (选择了 {selected_seeds[0]})")
            print("   分析: 可能是反向遍历没跑通，或者 RR-Set 生成有问题。")

    except Exception as e:
        print(f"❌ RIS 运行报错: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    run_sanity_check()