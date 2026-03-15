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
    target_methods = ["monterCarlo_CELF", "ris_path_aware"]
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
        "monterCarlo_CELF": {"color": "#E45756", "marker": "s", "label": "MC-CELF (Baseline)"},
        "ris_path_aware":   {"color": "#4C78A8", "marker": "*", "label": "RIS-Optimized (Ours)", "markersize": 12}
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
    TIME_CSV_FILE = "/root/work/coupon/coup_spread/gzc-impl/results/network.netscience/2026-4-20-time_test/TimeCost_log_continuous_SSRNum-100000.csv" 
    draw_time_consumption_plot(TIME_CSV_FILE)