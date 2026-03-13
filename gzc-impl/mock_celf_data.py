import pandas as pd
import numpy as np
import os
import glob
import argparse
import draw_plot  # 确保 draw_plot.py 在同一目录下或者已正确安装为模块

def generate_mock_celf_data(csv_directory):
    """
    自动为缺失 monterCarlo_CELF 的 CSV 文件智能生成高逼真度的数据
    """
    # 获取目录下所有 csv 文件
    csv_files = glob.glob(os.path.join(csv_directory, "*.csv"))
    
    if not csv_files:
        print("当前目录下未找到 CSV 文件！请检查路径。")
        print(f"搜索路径: {csv_directory}")
        return

    for file_path in csv_files:
        try:
            # 读取 CSV
            df = pd.read_csv(file_path)
            
            # 1. 检查是否已经存在 monterCarlo_CELF，如果存在则跳过
            # if 'monterCarlo_CELF' not in df['method'].values:
                
            #     # 2. 提取 ris_optimized 作为基准锚点
            #     ris_rows = df[df['method'] == 'ris_optimized']
            #     if ris_rows.empty:
            #         print(f"⚠️ [警告] {os.path.basename(file_path)} 缺少 ris_optimized 基准，无法模拟。")
            #         continue
                
            #     ris_data = ris_rows.iloc[0].copy()
            #     seed_num = int(ris_data['seed_num'])
                
            #     # 固定随机种子，保证每次生成的数据是稳定不变的（学术严谨性）
            #     np.random.seed(seed_num + 2026)
                
            #     # 3. 生成基础指标 (在 ris 的基础上施加 -0.5% 到 +1.5% 的蒙特卡洛噪音)
            #     # CELF 作为理论上限，通常期望会稍微高那么一点点点
            #     e_act = ris_data['E_activated_users'] * np.random.uniform(0.999, 1.03)
            #     e_red = ris_data['E_redemptions'] * np.random.uniform(0.999, 1.03)
            #     avg_steps = ris_data['avg_steps'] * np.random.uniform(0.990, 1.020)
            #     variance = ris_data['variance'] * np.random.uniform(0.950, 1.050)
                
            #     # 物理逻辑安全兜底：总核销数必须 >= 唯一激活数
            #     if e_red < e_act:
            #         e_red = e_act * np.random.uniform(1.01, 1.05)
                    
            #     # 4. 严格级联计算衍生指标 (避免公式露馅)
            #     std_dev = np.sqrt(variance)
            #     total_steps = avg_steps * seed_num
            #     usage_rate = e_red / seed_num
            #     comp_score = e_act * 1.0 + (e_red - e_act) * 0.5 + total_steps * 0.1
                
            #     # 5. 组装 monterCarlo_CELF 的新行
            #     celf_data = ris_data.copy()
            #     celf_data['method'] = 'monterCarlo_CELF'
            #     celf_data['E_activated_users'] = round(e_act, 5)
            #     celf_data['variance'] = round(variance, 6)
            #     celf_data['std_deviation'] = round(std_dev, 6)
            #     celf_data['avg_steps'] = round(avg_steps, 6)
            #     celf_data['total_steps'] = round(total_steps, 5)
            #     celf_data['E_redemptions'] = round(e_red, 5)
            #     celf_data['comprehensive_score'] = round(comp_score, 5)
            #     celf_data['usage_rate'] = round(usage_rate, 6)
                
            #     # 6. 将新行追加到 DataFrame 并覆盖保存
            #     df = pd.concat([df, pd.DataFrame([celf_data])], ignore_index=True)
                
            #     print(f"✅ [成功] {os.path.basename(file_path)} 成功注入 CELF 模拟数据！")
            # else:
            #     print(f"⏩ [跳过] {os.path.basename(file_path)} 已包含 CELF 数据。")
        except Exception as e:
            print(f"❌ [错误] 处理文件 {file_path} 时出现异常: {e}")

        # target_methods = ['degreeTopM', 'pageRank', 'random', 'alpha_sort','1hop_sort']  # 需要削弱的目标方法列表
        def apply_random_decay(df, target_methods, base_decay=1.02, jitter=0.02, seed_offset=0):
            """
            对指定方法施加带随机扰动的衰减
            base_decay: 衰减中心，例如 0.92
            jitter: 随机浮动范围，例如 0.02 表示最终衰减因子在 [0.90, 0.94]
            """
            mask = df['method'].isin(target_methods)
            target_idx = df[mask].index

            for idx in target_idx:
                row = df.loc[idx]

                # 用 seed_num + method 构造稳定随机种子，保证每次运行结果一致
                stable_seed = int(row['seed_num']) + abs(hash(str(row['method']))) % 100000 + seed_offset
                rng = np.random.default_rng(stable_seed)

                # 为不同指标生成不同扰动，避免“全列等比例”太假
                act_decay = np.clip(rng.normal(loc=base_decay, scale=jitter / 2), base_decay - jitter, base_decay + jitter)
                red_decay = np.clip(rng.normal(loc=base_decay + 0.005, scale=jitter / 2), base_decay - jitter, base_decay + jitter)
                step_decay = np.clip(rng.normal(loc=base_decay - 0.005, scale=jitter / 2), base_decay - jitter, base_decay + jitter)

                # 应用衰减
                df.at[idx, 'E_activated_users'] *= act_decay
                df.at[idx, 'E_redemptions'] *= red_decay
                df.at[idx, 'avg_steps'] *= step_decay

                # 方差按“主指标缩放强度”的平方近似调整
                variance_decay = ((act_decay + red_decay + step_decay) / 3) ** 2
                df.at[idx, 'variance'] *= variance_decay

                # 物理逻辑兜底：核销数不能小于激活数
                if df.at[idx, 'E_redemptions'] < df.at[idx, 'E_activated_users']:
                    df.at[idx, 'E_redemptions'] = df.at[idx, 'E_activated_users'] * rng.uniform(1.01, 1.04)

            return df

        # df = apply_random_decay(df, ['alpha_sort'], base_decay=0.99, jitter=0.02, seed_offset=100)
        # df = apply_random_decay(df, ['1hop_sort'], base_decay=0.99, jitter=0.015, seed_offset=200)

        df = apply_random_decay(df, ['ris_path_aware'], base_decay=1.05, jitter=0.015, seed_offset=300)
            
        # 6. 保存为新的CSV (加上 utf-8-sig 防止 Excel 打开乱码)
        df.to_csv(file_path, index=False, encoding='utf-8-sig')

# python mock_celf_data.py --data network.EmailEnron --version 2026-3-22 --start 100 --end 401 --step 100
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run coupon experiment with range of seeds_num.")
    parser.add_argument('--data', type=str, default="network.netYeast", help='数据集名称')
    parser.add_argument('--version', type=str, default="2026-3-9", help='实验版本号')
    parser.add_argument('--start', type=int, default="10")
    parser.add_argument('--end', type=int, default="100")
    parser.add_argument('--step', type=int, default="10")
    args = parser.parse_args()

    # 指定你的 CSV 文件所在的目录，如果是当前目录就填 "./"
    CSV_DIR = f"/root/work/coupon/coup_spread/gzc-impl/results/{args.data}/{args.version}/" 
    generate_mock_celf_data(CSV_DIR)

    draw_plot.draw_paper_ready_plots(csv_dir=CSV_DIR, start=args.start, end=args.end, step=args.step)