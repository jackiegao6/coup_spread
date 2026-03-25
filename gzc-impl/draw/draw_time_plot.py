import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from matplotlib.ticker import AutoMinorLocator

def draw_time_consumption_plot(time_csv_path):
    if not os.path.exists(time_csv_path):
        print(f"找不到时间数据文件: {time_csv_path}")
        return

    # 1. 读取数据
    df = pd.read_csv(time_csv_path)

    # 2. 过滤出我们要对比的两个方法
    target_methods = ["monterCarlo_CELF", "ris_path_aware", "pageRank", "degreeTopM", "random","alpha_sort","1hop_sort"]
    df_plot = df[df["method"].isin(target_methods)].copy()
    
    if df_plot.empty:
        print("CSV中没有找到指定的算法数据！")
        return

    # 排序保证连线正常
    df_plot = df_plot.sort_values(by=["method", "seed_num"])

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

    # 4. 样式定义
    styles = {
        "monterCarlo_CELF": {
            "color": "#4C78A8", "marker": "s", "label": "MC-CELF (Upper Bound)",
            "linewidth": 2.0, "markersize": 6.2, "zorder": 3, "alpha": 0.95
        },
        "ris_path_aware": {
            "color": "#E45756", "marker": "*", "label": "RIS-Optimized (Ours)",
            "linewidth": 2.8, "markersize": 10.5, "zorder": 6, "alpha": 0.98
        },
        "pageRank": {
            "color": "#59A14F", "marker": "^", "label": "PageRank",
            "linewidth": 1.8, "markersize": 6.0, "zorder": 3, "alpha": 0.92
        },
        "degreeTopM": {
            "color": "#F28E2B", "marker": "v", "label": "DegreeTopM",
            "linewidth": 1.8, "markersize": 6.0, "zorder": 3, "alpha": 0.92
        },
        "alpha_sort": {
            "color": "#B279A2", "marker": "D", "label": "Alpha-Sort",
            "linewidth": 1.8, "markersize": 5.8, "zorder": 3, "alpha": 0.92
        },
        "random": {
            "color": "#9D9DA1", "marker": "o", "label": "Random",
            "linewidth": 1.7, "markersize": 5.8, "zorder": 2, "alpha": 0.90
        },
        "1hop_sort": {
            "color": "#9C755F", "marker": "P", "label": "1Hop-Sort",
            "linewidth": 1.8, "markersize": 6.0, "zorder": 3, "alpha": 0.92
        },
    }

    # 5. 画线
    for method in target_methods:
        subset = df_plot[df_plot["method"] == method]
        if subset.empty: continue
        
        style = styles[method]
        ax.plot(
            subset["seed_num"],
            subset["time_ms"],
            color=style["color"],
            marker=style["marker"],
            label=style["label"],
            linewidth=2.5,
            markersize=style.get("markersize", 8),
            alpha=0.9,
            markeredgecolor="white",
            markeredgewidth=1.0
        )

    # 6. 细节修饰
    ax.set_title("Expected Time Consumption vs. Seed Size", pad=12)
    ax.set_xlabel(r"Seed Budget $k$")
    ax.set_ylabel(r"Time Consumption (milliseconds)")
    
    # 【最关键的一步】：Y轴设置为对数坐标 (Log Scale)
    ax.set_yscale("log")
    
    ax.grid(axis="y", which="major", linestyle="--")
    ax.grid(axis="y", which="minor", linestyle=":", alpha=0.1)
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    sns.despine(fig=fig)

    # 图例
    ax.legend(loc="upper left", frameon=True, edgecolor="#CCCCCC")

    plt.tight_layout()
    
    # 保存图片
    output_pdf = os.path.join(os.path.dirname(time_csv_path), "time_consumption_vs_k.pdf")
    plt.savefig(output_pdf, dpi=300, bbox_inches="tight")
    print(f"🎉 时间对比图已保存至: {output_pdf}")
    plt.show()

if __name__ == "__main__":
    # 替换为你实际生成的 TimeCost CSV 文件路径
    # /root/work/coupon/coup_spread/gzc-impl/results/network.netscience/paper-netscience-h-1.0-timecost/TimeCost_log_continuous_SSRNum-100000.csv
    TIME_CSV_FILE = "/root/work/coupon/coup_spread/gzc-impl/results/network.netscience/paper-netscience-h-1.0-timecost/TimeCost_log_continuous_SSRNum-100000.csv" 
    draw_time_consumption_plot(TIME_CSV_FILE)