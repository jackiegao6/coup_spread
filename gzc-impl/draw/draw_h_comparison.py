import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import os
from matplotlib.ticker import AutoMinorLocator

def draw_h_ablation_plot(csv_dir, target_method="ris_path_aware"):
    # 1. 读取并合并 CSV（不变）
    all_files = glob.glob(os.path.join(csv_dir, "*.csv"))
    if not all_files:
        print("未找到 CSV 文件！")
        return

    df_list = [pd.read_csv(file) for file in all_files]
    final_df = pd.concat(df_list, ignore_index=True)

    df_plot = final_df[final_df["method"] == target_method].copy()
    if df_plot.empty:
        print(f"CSV中没有找到方法: {target_method}")
        return

    df_plot = df_plot.sort_values(by=["degree_power_h", "seed_num"])

    # ====================== 新增：动态样式生成 ======================
    h_values = sorted(df_plot["degree_power_h"].unique())
    
    # 颜色：husl 调色板（支持任意数量，颜色鲜艳且区分度高）
    colors = sns.color_palette("husl", n_colors=len(h_values))
    
    # Marker 循环列表（足够 20+ 个不重复）
    marker_list = ['o', 's', '^', 'D', 'v', 'p', '*', 'h', 'x', '+', 
                   '<', '>', '1', '2', '3', '4', '8', 'P', 'H']
    markers = [marker_list[i % len(marker_list)] for i in range(len(h_values))]

    # 动态构建 h_styles
    h_styles = {}
    for i, h_val in enumerate(h_values):
        if h_val < 0:
            desc = "(Inverse)"
        elif abs(h_val - 1.0) < 1e-6:
            desc = "(Linear)"
        elif h_val > 1:
            desc = "(Convex)"
        else:
            desc = ""
        
        label = rf"$h = {h_val:g}$ {desc}".strip()
        
        h_styles[h_val] = {
            "color": colors[i],
            "marker": markers[i],
            "label": label
        }
    # ============================================================

    # 顶会风格设置（不变）
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

    # 画线（现在完全使用动态样式）
    for h_val in h_values:
        subset = df_plot[df_plot["degree_power_h"] == h_val]
        style = h_styles[h_val]
        
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

    # 细节修饰（不变）
    ax.set_title(f"Impact of Degree Mapping Exponent ($h$) on {target_method}", pad=12)
    ax.set_xlabel(r"Seed Budget $k$")
    ax.set_ylabel(r"Expected Activated Users")
    
    ax.grid(axis="y", linestyle="--")
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    sns.despine(fig=fig)

    ax.legend(loc="upper left", frameon=True, edgecolor="#CCCCCC")

    plt.tight_layout()
    
    output_pdf = os.path.join(csv_dir, "h_ablation_study.pdf")
    plt.savefig(output_pdf, dpi=300, bbox_inches="tight")
    print(f"图表已保存至: {output_pdf}")
    plt.show()

def draw_h_ablation_plot2(csv_dir, target_method="ris_path_aware"):
    # 1. 读取并合并 CSV
    all_files = glob.glob(os.path.join(csv_dir, "*.csv"))
    if not all_files:
        print("未找到 CSV 文件！")
        return

    df_list = [pd.read_csv(file) for file in all_files]
    final_df = pd.concat(df_list, ignore_index=True)

    df_plot = final_df[final_df["method"] == target_method].copy()
    if df_plot.empty:
        print(f"CSV中没有找到方法: {target_method}")
        return

    df_plot = df_plot.sort_values(by=["degree_power_h", "seed_num"])

    # ====================== 新样式：完全模仿你给的图片 ======================
    h_values = sorted(df_plot["degree_power_h"].unique())
    
    # 颜色：bright（粉蓝青绿黄，和示例图片一模一样）
    colors = sns.color_palette("bright", n_colors=len(h_values))
    
    # Marker：大量三角形循环（和图片里各种▲▼完全一致）
    marker_list = ['o', 's', '^', 'D', 'v', 'p', '*', 'h', 'x', '+', 
                   '<', '>', '1', '2', '3', '4', '8', 'P', 'H']
    markers = [marker_list[i % len(marker_list)] for i in range(len(h_values))]

    h_styles = {}
    for i, h_val in enumerate(h_values):
        if h_val < 0:
            desc = "(Inverse)"
        elif abs(h_val - 1.0) < 1e-6:
            desc = "(Linear)"
        elif h_val > 1:
            desc = "(Convex)"
        else:
            desc = ""
        label = rf"$h = {h_val:g}$ {desc}".strip()
        h_styles[h_val] = {
            "color": colors[i],
            "marker": markers[i],
            "label": label
        }
    # ====================================================================

    sns.set_theme(style="ticks", context="paper")
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman"],
        "font.size": 12,
        "axes.labelsize": 14,
        "axes.titlesize": 15,
        "legend.fontsize": 11,
        "axes.linewidth": 1.0,
    })

    fig, ax = plt.subplots(figsize=(7.2, 4.8))  # 比例接近示例图片

    for h_val in h_values:
        subset = df_plot[df_plot["degree_power_h"] == h_val]
        style = h_styles[h_val]
        ax.plot(
            subset["seed_num"],
            subset["E_activated_users"],
            color=style["color"],
            marker=style["marker"],
            label=style["label"],
            linewidth=2.2,
            markersize=9.5,
            alpha=0.92,
            markeredgecolor="white",
            markeredgewidth=0.8
        )

    ax.set_title(f"Impact of Degree Mapping Exponent ($h$) on {target_method}", pad=10)
    ax.set_xlabel(r"Seed Budget $k$")
    ax.set_ylabel(r"Expected Activated Users")
    
    # === 关键：完全无网格 + 无 minor ticks（和示例图片一样干净）===
    sns.despine(fig=fig)

    ax.legend(loc="upper left", frameon=True, edgecolor="#CCCCCC", fontsize=10.5)

    plt.tight_layout()
    
    output_pdf = os.path.join(csv_dir, "h_ablation_study.pdf")
    plt.savefig(output_pdf, dpi=300, bbox_inches="tight")
    print(f"图表已保存至: {output_pdf}")
    plt.show()

if __name__ == "__main__":
    # 替换为你的 CSV 所在目录
    CSV_DIRECTORY = "/root/work/coupon/coup_spread/gzc-impl/results/network.netscience/paper-netscience-h-test/" 
    draw_h_ablation_plot2(CSV_DIRECTORY, target_method="ris_path_aware")