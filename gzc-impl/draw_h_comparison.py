import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import os
from matplotlib.ticker import AutoMinorLocator

def draw_h_ablation_plot(csv_dir, target_method="ris_path_aware"):
    # 1. 读取并合并 CSV
    all_files = glob.glob(os.path.join(csv_dir, "*.csv"))
    if not all_files:
        print("未找到 CSV 文件！")
        return

    df_list = [pd.read_csv(file) for file in all_files]
    final_df = pd.concat(df_list, ignore_index=True)

    # 2. 过滤出目标算法（比如你的核心算法 ris_path_aware）
    df_plot = final_df[final_df["method"] == target_method].copy()
    
    if df_plot.empty:
        print(f"CSV中没有找到方法: {target_method}")
        return

    # 确保按 seed_num 排序
    df_plot = df_plot.sort_values(by=["degree_power_h", "seed_num"])

    # 3. 顶会风格设置
    sns.set_theme(style="ticks", context="paper")
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman"],
        "font.size": 12,
        "axes.labelsize": 14,
        "axes.titlesize": 15,
        "legend.fontsize": 12,
        "axes.linewidth": 1.0,
        "grid.alpha": 0.2,
    })

    fig, ax = plt.subplots(figsize=(7, 5.5))

    # 4. 为不同的 h 定义样式
    h_styles = {
        -0.5: {"color": "#E45756", "marker": "o", "label": r"$h = -0.5$ (Inverse)"},
        1.0:  {"color": "#4C78A8", "marker": "s", "label": r"$h = 1.0$ (Linear)"},
        1.5:  {"color": "#59A14F", "marker": "^", "label": r"$h = 1.5$ (Convex)"}
    }

    # 5. 画线
    h_values_in_data = df_plot["degree_power_h"].unique()
    for h_val in sorted(h_values_in_data):
        subset = df_plot[df_plot["degree_power_h"] == h_val]
        
        # 如果有预设样式就用预设，没有就动态生成
        style = h_styles.get(h_val, {"color": None, "marker": "d", "label": f"$h = {h_val}$"})
        
        ax.plot(
            subset["seed_num"],
            subset["E_activated_users"],
            color=style["color"],
            marker=style["marker"],
            label=style["label"],
            linewidth=2.5,
            markersize=8,
            alpha=0.9,
            markeredgecolor="white",
            markeredgewidth=1.0
        )

    # 6. 细节修饰
    ax.set_title(f"Impact of Degree Mapping Exponent ($h$) on {target_method}", pad=12)
    ax.set_xlabel(r"Seed Budget $k$")
    ax.set_ylabel(r"Expected Activated Users")
    
    ax.grid(axis="y", linestyle="--")
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    sns.despine(fig=fig)

    # 图例放在左上角
    ax.legend(loc="upper left", frameon=True, edgecolor="#CCCCCC")

    plt.tight_layout()
    
    # 保存图片
    output_pdf = os.path.join(csv_dir, "h_ablation_study.pdf")
    plt.savefig(output_pdf, dpi=300, bbox_inches="tight")
    print(f"图表已保存至: {output_pdf}")
    plt.show()

if __name__ == "__main__":
    # 替换为你的 CSV 所在目录
    CSV_DIRECTORY = "/root/work/coupon/coup_spread/gzc-impl/results/network.netscience/2026-4-20-ablation_h/" 
    draw_h_ablation_plot(CSV_DIRECTORY, target_method="ris_path_aware")