import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import os
import numpy as np
from matplotlib.ticker import AutoMinorLocator

def draw_elegant_ablation_plot(csv_dir, target_method="ris_path_aware"):
    # =========================
    # 1. 读取并合并数据
    # =========================
    all_files = glob.glob(os.path.join(csv_dir, "*.csv"))
    if not all_files:
        print("未找到 CSV 文件！")
        return

    df_list = [pd.read_csv(file) for file in all_files]
    final_df = pd.concat(df_list, ignore_index=True)

    df_plot = final_df[final_df["method"] == target_method].copy()
    if df_plot.empty:
        print(f"未找到方法: {target_method}")
        return

    # 提取唯一的种子数(k)和 h 值，并排序
    k_values = sorted(df_plot["seed_num"].unique())
    h_values = sorted(df_plot["degree_power_h"].unique())

    # 如果 h 的取值太多，为了美观，我们只挑 3-4 个最具代表性的（比如极小、中间、极大）
    # 这里假设我们挑 3 个：最小值，接近1的值，最大值
    if len(h_values) > 4:
        h_values = [h_values[0], 1.0, h_values[-1]] # 你可以手动修改这里的列表，比如 [-1.0, 1.0, 2.0]
        df_plot = df_plot[df_plot["degree_power_h"].isin(h_values)]

    # =========================
    # 2. 顶会风格全局配置
    # =========================
    sns.set_theme(style="ticks", context="paper")
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 12,
        "axes.labelsize": 13,
        "axes.titlesize": 14,
        "legend.fontsize": 11,
        "axes.linewidth": 1.0,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    # 定义一套高级学术配色 (红, 蓝, 绿, 紫)
    colors = ["#E45756", "#4C78A8", "#59A14F", "#B279A2"]
    markers = ["o", "s", "^", "D"]

    # 创建 1x2 的双子图画布，比例适合论文双栏
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # =========================
    # 3. 左图 (a): 折线图 (激活人数)
    # =========================
    for idx, h_val in enumerate(h_values):
        subset = df_plot[df_plot["degree_power_h"] == h_val].sort_values("seed_num")
        
        ax1.plot(
            subset["seed_num"], 
            subset["E_activated_users"], 
            color=colors[idx % len(colors)], 
            marker=markers[idx % len(markers)], 
            linewidth=2.5, 
            markersize=7, 
            alpha=0.9,
            markeredgecolor="white",
            markeredgewidth=1.0,
            label=f"$h = {h_val:g}$"
        )

    ax1.set_title("(a) Expected Activated Users vs. Seed Budget", pad=10)
    ax1.set_xlabel(r"Seed Budget $k$")
    ax1.set_ylabel(r"Expected Activated Users")
    ax1.grid(axis="y", linestyle="--", alpha=0.3)
    ax1.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax1.yaxis.set_minor_locator(AutoMinorLocator(2))

    # =========================
    # 4. 右图 (b): 分组柱状图 (总步数)
    # =========================
    x_indexes = np.arange(len(k_values)) # X 轴基准坐标
    bar_width = 0.8 / len(h_values)      # 动态计算柱子宽度，确保不拥挤

    for idx, h_val in enumerate(h_values):
        subset = df_plot[df_plot["degree_power_h"] == h_val].sort_values("seed_num")
        
        # 确保数据对齐（防止某些 k 缺失导致错位）
        y_bar = []
        for k in k_values:
            val = subset[subset["seed_num"] == k]["total_steps"]
            y_bar.append(val.values[0] if not val.empty else 0)
        
        # 计算每组柱子的偏移量
        offset = (idx - len(h_values)/2 + 0.5) * bar_width
        
        ax2.bar(
            x_indexes + offset, 
            y_bar, 
            width=bar_width * 0.9, # 乘以0.9留一点缝隙
            color=colors[idx % len(colors)], 
            alpha=0.85, 
            edgecolor="black", 
            linewidth=0.7,
            label=f"$h = {h_val:g}$"
        )

    ax2.set_title("(b) Total Survival Steps vs. Seed Budget", pad=10)
    ax2.set_xlabel(r"Seed Budget $k$")
    ax2.set_ylabel(r"Total Survival Steps")
    
    # 替换 X 轴刻度为真实的 k 值
    ax2.set_xticks(x_indexes)
    ax2.set_xticklabels(k_values)
    ax2.grid(axis="y", linestyle="--", alpha=0.3)

    # =========================
    # 5. 细节修饰与统一图例
    # =========================
    sns.despine(fig=fig) # 去除顶部和右侧边框

    # 提取左图的图例句柄（因为左右图的颜色和标签是一一对应的，取一个就行）
    handles, labels = ax1.get_legend_handles_labels()
    
    # 将图例放在整个画布的底部居中
    fig.legend(
        handles, labels, 
        loc="lower center", 
        ncol=len(h_values), 
        bbox_to_anchor=(0.5, -0.05), # 负数表示放在图表下方
        frameon=True, 
        edgecolor="#CCCCCC",
        handlelength=2.0
    )

    # 调整子图间距，给底部图例留出空间
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.18, wspace=0.25)

    # =========================
    # 6. 保存与展示
    # =========================
    output_pdf = os.path.join(csv_dir, "ablation_h_elegant.pdf")
    output_png = os.path.join(csv_dir, "ablation_h_elegant.png")
    
    plt.savefig(output_pdf, dpi=300, bbox_inches="tight")
    plt.savefig(output_png, dpi=300, bbox_inches="tight")
    
    print(f"🎉 顶级学术插图已生成！\n  - {output_pdf}\n  - {output_png}")
    plt.show()

if __name__ == "__main__":
    # ⚠️ 替换为你的 CSV 目录
    CSV_DIRECTORY = "/root/work/coupon/coup_spread/gzc-impl/results/network.netscience/paper-netscience-h-test/" 
    draw_elegant_ablation_plot(CSV_DIRECTORY, target_method="ris_path_aware")