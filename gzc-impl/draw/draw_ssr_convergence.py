import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import os
import re
from matplotlib.ticker import FuncFormatter, AutoMinorLocator

def draw_ssr_convergence_plot(csv_dir, target_method="ris_path_aware", target_seed_num=100):
    # 1. 获取目录下所有相关的 CSV 文件
    all_files = glob.glob(os.path.join(csv_dir, "*SSRNum-*.csv"))
    if not all_files:
        print("未找到包含 SSRNum 的 CSV 文件，请检查路径！")
        return

    df_list = []
    for file in all_files:
        # 2. 使用正则表达式从文件名中提取 SSRNum
        # 匹配规则：寻找 "SSRNum-" 后面跟着的一串数字
        match = re.search(r'SSRNum-(\d+)', os.path.basename(file))
        if match:
            ssr_num = int(match.group(1))
            df = pd.read_csv(file)
            
            # 只保留目标算法和目标种子数的数据
            df = df[(df["method"] == target_method) & (df["seed_num"] == target_seed_num)].copy()
            
            if not df.empty:
                # 将提取到的 SSRNum 作为新列加入 DataFrame
                df["SSRNum"] = ssr_num
                df_list.append(df)

    if not df_list:
        print(f"未找到 method={target_method} 且 seed_num={target_seed_num} 的有效数据！")
        return

    # 3. 合并数据并按 SSRNum 排序
    final_df = pd.concat(df_list, ignore_index=True)
    # 如果同一个 SSRNum 有多条记录（比如跑了多次），取平均值
    # 修改后的代码
    final_df = final_df.groupby("SSRNum", as_index=False).mean(numeric_only=True)
    final_df = final_df.sort_values(by="SSRNum")

    print("提取到的收敛数据：")
    print(final_df[["SSRNum", "E_activated_users"]])

    # 4. 顶会风格设置
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

    # 5. 画折线图
    ax.plot(
        final_df["SSRNum"],
        final_df["E_activated_users"],
        color="#D62728",      # 经典的学术红
        marker="o",           # 圆点标记
        label=f"{target_method} (k={target_seed_num})",
        linewidth=2.5,
        markersize=8,
        alpha=0.9,
        markeredgecolor="white",
        markeredgewidth=1.0
    )

    # 6. 细节修饰
    ax.set_title("Convergence of Expected Activated Users vs. SSR Samples", pad=12)
    ax.set_xlabel(r"Number of SSR Samples ($N$)")
    ax.set_ylabel(r"Expected Activated Users")

    # 格式化 X 轴：将 100000 显示为 100K，让图表更专业
    def format_k(x, pos):
        if x >= 1000:
            return f'{int(x/1000)}K'
        return str(int(x))
    ax.xaxis.set_major_formatter(FuncFormatter(format_k))

    # 添加网格线
    ax.grid(axis="y", which="major", linestyle="--")
    ax.grid(axis="x", which="major", linestyle=":", alpha=0.15)
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    sns.despine(fig=fig)

    # 7. 添加“收敛辅助线” (可选，增加学术感)
    # 找到 100k 时的值作为收敛基准线
    if 100000 in final_df["SSRNum"].values:
        converge_val = final_df[final_df["SSRNum"] == 100000]["E_activated_users"].values[0]
        ax.axhline(y=converge_val, color="gray", linestyle="--", alpha=0.5, zorder=0)
        ax.text(final_df["SSRNum"].max() * 0.7, converge_val + 0.2, "Convergence Plateau", color="gray", fontsize=11)

    # 图例
    ax.legend(loc="lower right", frameon=True, edgecolor="#CCCCCC")

    plt.tight_layout()
    
    # 8. 保存图片
    output_pdf = os.path.join(csv_dir, "ssr_convergence_plot.pdf")
    plt.savefig(output_pdf, dpi=300, bbox_inches="tight")
    print(f"🎉 收敛图已保存至: {output_pdf}")
    plt.show()

if __name__ == "__main__":
    # 替换为你存放这些 CSV 文件的实际目录
    CSV_DIRECTORY = "/root/work/coupon/coup_spread/gzc-impl/results/network.netscience/paper-netscience-num_samples/" 
    draw_ssr_convergence_plot(CSV_DIRECTORY)