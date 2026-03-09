import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import os

def draw_paper_ready_plots(csv_dir):
    # 1. 自动读取目录下所有的 csv 文件并合并
    # 假设你的文件命名为 random_SSRNum-50000-_seedNum-10_simuTimes-500.csv 等
    all_files = glob.glob(os.path.join(csv_dir, "*.csv"))
    
    if not all_files:
        print("未找到 CSV 文件，请检查路径！")
        return

    df_list =[]
    for file in all_files:
        df = pd.read_csv(file)
        df_list.append(df)
        
    final_df = pd.concat(df_list, ignore_index=True)
    
    # 2. 按照 seed_num 排序，保证 X 轴是递增的
    final_df = final_df.sort_values(by=['seed_num', 'method'])

    # 3. 设置论文级绘图风格 (Seaborn 白色网格风格，适合学术论文)
    sns.set_theme(style="whitegrid", context="paper")
    
    # 统一设置字体大小
    plt.rcParams.update({
        'font.size': 12,
        'axes.labelsize': 14,
        'axes.titlesize': 15,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 11,
        'font.family': 'serif' # 使用学术界常用的衬线字体
    })

    # 4. 定义不同算法的颜色和标记形状 (保持高对比度)
    method_styles = {
        'ris_optimized':    {'color': '#e74c3c', 'marker': '*', 'label': 'RIS_Optimized (Ours)', 'linewidth': 2.5, 'markersize': 10},
        'monterCarlo_CELF': {'color': '#2980b9', 'marker': 's', 'label': 'MC_CELF (Upper Bound)', 'linewidth': 2.0, 'markersize': 7},
        'pageRank':         {'color': '#27ae60', 'marker': '^', 'label': 'PageRank', 'linewidth': 1.5, 'markersize': 7},
        'degreeTopM':       {'color': '#f39c12', 'marker': 'v', 'label': 'DegreeTopM', 'linewidth': 1.5, 'markersize': 7},
        'alpha_sort':       {'color': '#8e44ad', 'marker': 'D', 'label': 'Alpha_Sort', 'linewidth': 1.5, 'markersize': 7},
        'random':           {'color': '#7f8c8d', 'marker': 'o', 'label': 'Random', 'linewidth': 1.5, 'markersize': 7}
    }

    # 5. 创建 1x2 的画布 (宽高比大约为 2:1)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # --------- 画左图: E_activated_users ---------
    ax1 = axes[0]
    for method, style in method_styles.items():
        subset = final_df[final_df['method'] == method]
        if not subset.empty:
            ax1.plot(subset['seed_num'], subset['E_activated_users'], 
                     color=style['color'], marker=style['marker'], 
                     label=style['label'], linewidth=style['linewidth'], markersize=style['markersize'])
            
    ax1.set_title('(a) Expected Activated Users vs. Seed Size')
    ax1.set_xlabel('Number of Initial Seeds (k)')
    ax1.set_ylabel('Expected Activated Users (Adoption Spread)')
    ax1.set_xticks(range(10, 101, 10))
    # ax1.set_xticks(range(50, 251, 25))


    # --------- 画右图: comprehensive_score ---------
    ax2 = axes[1]
    for method, style in method_styles.items():
        subset = final_df[final_df['method'] == method]
        if not subset.empty:
            ax2.plot(subset['seed_num'], subset['comprehensive_score'], 
                     color=style['color'], marker=style['marker'], 
                     label=style['label'], linewidth=style['linewidth'], markersize=style['markersize'])

    ax2.set_title('(b) Comprehensive Marketing Score vs. Seed Size')
    ax2.set_xlabel('Number of Initial Seeds (k)')
    ax2.set_ylabel('Comprehensive Score (E-ROI)')
    ax2.set_xticks(range(10, 101, 10))
    # ax2.set_xticks(range(50, 251, 25))

    # 6. 统一添加图例 (放到画布下方或者右图里)
    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=3, bbox_to_anchor=(0.5, -0.05), frameon=True)

    # 7. 调整布局并保存为 PDF 高清矢量图
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.18) # 给底部的图例留出空间
    
    # 保存为 PDF (矢量图，插入 Word 或 LaTeX 放大不失真)
    output_pdf = csv_dir + "netscience_experimental_results.pdf"
    output_png = csv_dir +  "netscience_experimental_results.png"
    plt.savefig(output_pdf, dpi=300, bbox_inches='tight')
    plt.savefig(output_png, dpi=300, bbox_inches='tight')
    
    print(f"🎉 图表绘制成功！已保存为 {output_pdf} 和 {output_png}")
    plt.show()

if __name__ == "__main__":
    # 将这里修改为你存放那 10 个 CSV 的文件夹路径
    CSV_DIRECTORY = "/home/wen/work/coup_spread/gzc-impl/results/network.doubanrandom/2026-3-7/" 
    draw_paper_ready_plots(CSV_DIRECTORY)