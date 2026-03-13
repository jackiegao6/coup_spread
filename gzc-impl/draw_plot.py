import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import os

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import os

from matplotlib.ticker import AutoMinorLocator, FuncFormatter

def draw_paper_ready_plots(csv_dir, start=10, end=100, step=10):
    # =========================
    # 1. 自动读取目录下所有 csv 并合并
    # =========================
    all_files = glob.glob(os.path.join(csv_dir, "*.csv"))
    if not all_files:
        print("未找到 CSV 文件，请检查路径！")
        return

    df_list = []
    for file in all_files:
        df = pd.read_csv(file)
        df_list.append(df)

    final_df = pd.concat(df_list, ignore_index=True)

    # =========================
    # 2. 排序，保证 X 轴递增
    # =========================
    final_df = final_df.sort_values(by=["seed_num", "method"])

    # =========================
    # 3. 顶会论文风格配置
    # =========================
    sns.set_theme(style="ticks", context="paper")

    plt.rcParams.update({
        # 字体
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Nimbus Roman", "DejaVu Serif", "STIXGeneral"],
        "mathtext.fontset": "stix",
        "axes.unicode_minus": False,

        # 字号
        "font.size": 11,
        "axes.labelsize": 13,
        "axes.titlesize": 13.5,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 10.5,

        # 线条与边框
        "axes.linewidth": 0.9,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "xtick.minor.width": 0.6,
        "ytick.minor.width": 0.6,
        "xtick.major.size": 4.0,
        "ytick.major.size": 4.0,
        "xtick.minor.size": 2.5,
        "ytick.minor.size": 2.5,

        # 网格
        "grid.linewidth": 0.6,
        "grid.alpha": 0.18,

        # 输出
        "savefig.dpi": 300,
        "pdf.fonttype": 42,   # TrueType，Word/Adobe 兼容更好
        "ps.fonttype": 42,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "lines.antialiased": True,
    })

    # =========================
    # 4. 颜色与样式：更接近论文图
    #    采用色盲友好 + 克制风格
    # =========================
    method_styles = {
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

    # =========================
    # 5. 创建双子图
    #    适合论文双栏插图：宽而不高
    # =========================
    fig, axes = plt.subplots(1, 2, figsize=(13.8, 5.4))

    # 千分位格式
    def thousands_formatter(x, pos):
        if abs(x) >= 1000:
            return f"{int(x):,}"
        return f"{x:g}"

    # X 轴刻度
    xticks = list(range(start, end + 1, step*2))

    # =========================
    # 6. 左图：Expected Activated Users
    # =========================
    ax1 = axes[0]
    for method, style in method_styles.items():
        subset = final_df[final_df["method"] == method]
        if not subset.empty:
            ax1.plot(
                subset["seed_num"],
                subset["E_activated_users"],
                color=style["color"],
                marker=style["marker"],
                label=style["label"],
                linewidth=style["linewidth"],
                markersize=style["markersize"],
                alpha=style["alpha"],
                zorder=style["zorder"],
                markeredgecolor="white",
                markeredgewidth=0.9,
                solid_capstyle="round",
                solid_joinstyle="round",
            )

    ax1.set_title(r"(a) Expected Activated Users vs. Seed Size", pad=8)
    ax1.set_xlabel(r"Seed Budget $k$")
    ax1.set_ylabel(r"Expected Activated Users, $E[N_{\mathrm{act}}(k)]$")

    ax1.set_xticks(xticks)
    ax1.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax1.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax1.yaxis.set_major_formatter(FuncFormatter(thousands_formatter))

    # 更顶会风的网格：只保留 y 主网格，x 方向弱化
    ax1.grid(axis="y", which="major", linestyle="--", alpha=0.22)
    ax1.grid(axis="y", which="minor", linestyle=":", alpha=0.10)
    ax1.grid(axis="x", which="major", linestyle=":", alpha=0.06)

    # =========================
    # 7. 右图：Comprehensive Score
    # =========================
    ax2 = axes[1]
    for method, style in method_styles.items():
        subset = final_df[final_df["method"] == method]
        if not subset.empty:
            ax2.plot(
                subset["seed_num"],
                subset["comprehensive_score"],
                color=style["color"],
                marker=style["marker"],
                label=style["label"],
                linewidth=style["linewidth"],
                markersize=style["markersize"],
                alpha=style["alpha"],
                zorder=style["zorder"],
                markeredgecolor="white",
                markeredgewidth=0.9,
                solid_capstyle="round",
                solid_joinstyle="round",
            )

    ax2.set_title(r"(b) Comprehensive Marketing Score vs. Seed Size", pad=8)
    ax2.set_xlabel(r"Seed Budget $k$")
    ax2.set_ylabel(r"Comprehensive Score, $\text{E-ROI}(k)$")

    ax2.set_xticks(xticks)
    ax2.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax2.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax2.yaxis.set_major_formatter(FuncFormatter(thousands_formatter))

    ax2.grid(axis="y", which="major", linestyle="--", alpha=0.22)
    ax2.grid(axis="y", which="minor", linestyle=":", alpha=0.10)
    ax2.grid(axis="x", which="major", linestyle=":", alpha=0.06)

    # =========================
    # 8. 坐标轴细节
    # =========================
    for ax in axes:
        ax.tick_params(axis="both", which="major", direction="in")
        ax.tick_params(axis="both", which="minor", direction="in")
        ax.margins(x=0.02)

    # 去掉顶部/右侧边框，常见论文风格
    sns.despine(fig=fig, top=True, right=True, trim=False)
    for ax in axes:
        ax.tick_params(axis='x', which='major', direction='in', pad=4)

    # =========================
    # 9. 统一图例：底部居中
    # =========================
    handles, labels = ax1.get_legend_handles_labels()
    legend = fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=3,
        bbox_to_anchor=(0.5, -0.03),
        frameon=True,
        fancybox=False,
        framealpha=1.0,
        borderpad=0.55,
        columnspacing=1.4,
        handlelength=2.2,
        handletextpad=0.55,
    )
    legend.get_frame().set_edgecolor("#C8C8C8")
    legend.get_frame().set_linewidth(0.8)
    legend.get_frame().set_facecolor("white")

    # =========================
    # 10. 布局与保存
    # =========================
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.22, wspace=0.22)

    output_pdf = os.path.join(csv_dir, "topconf_style.pdf")
    output_png = os.path.join(csv_dir, "topconf_style.png")

    plt.savefig(output_pdf, bbox_inches="tight")
    plt.savefig(output_png, bbox_inches="tight")

    print(f"🎉 顶会风格图表已保存为:\n{output_pdf}\n{output_png}")
    plt.show()

if __name__ == "__main__":
    CSV_DIRECTORY = "/home/wen/work/coup_spread/gzc-impl/results/network.netscience/2026-4-3/" 
    draw_paper_ready_plots(CSV_DIRECTORY)