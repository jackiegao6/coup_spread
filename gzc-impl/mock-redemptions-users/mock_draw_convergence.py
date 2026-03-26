import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import FuncFormatter, AutoMinorLocator

def generate_mock_convergence_data():
    # 模拟不同的 SSR 采样数量 N
    n_values = [1000, 10000, 50000, 100000, 200000, 300000, 400000, 500000]
    
    # 模拟激活人数的收敛过程 (指数逼近上限)
    max_activated = 285.5  # 假设的理论上限
    data = []
    
    for n in n_values:
        # 当 N 较小时，误差较大；当 N 很大时，逼近 max_activated
        error_margin = 150.0 / (n ** 0.4) 
        activated = max_activated - error_margin + np.random.normal(0, 0.5)
        data.append({"SSRNum": n, "activated": activated})
        
    return pd.DataFrame(data)

def draw_convergence():
    df = generate_mock_convergence_data()
    
    sns.set_theme(style="ticks", context="paper")
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["Times New Roman"],
        "font.size": 12, "axes.labelsize": 14, "axes.titlesize": 15,
        "legend.fontsize": 12, "axes.linewidth": 1.0, "grid.alpha": 0.2,
    })

    fig, ax = plt.subplots(figsize=(7, 5.5))

    ax.plot(df["SSRNum"], df["activated"],
            color="#D62728", marker="o", label="RIS-Optimized (k=50)",
            linewidth=2.5, markersize=8, alpha=0.9)

    ax.set_title("Convergence of Expected Activated Users vs. SSR Samples", pad=12)
    ax.set_xlabel(r"Number of SSR Samples ($N$)")
    ax.set_ylabel(r"Expected Activated Users, $\mathbb{E}[N_{act}]$")

    # X轴格式化为 K (千)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: f'{int(x/1000)}K' if x >= 1000 else str(int(x))))
    
    # 添加收敛基准线
    converge_val = df["activated"].iloc[-1]
    ax.axhline(y=converge_val, color="gray", linestyle="--", alpha=0.6, zorder=0)
    ax.text(df["SSRNum"].max() * 0.55, converge_val + 1.5, "Convergence Plateau", color="gray", fontsize=12)

    ax.grid(axis="y", linestyle="--")
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    sns.despine(fig=fig)
    ax.legend(loc="lower right", frameon=True)

    plt.tight_layout()
    plt.savefig("mock_ssr_convergence.pdf", dpi=300, bbox_inches="tight")
    print("✅ 收敛图已生成: mock_ssr_convergence.pdf")
    plt.show()

if __name__ == "__main__":
    draw_convergence()