import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import AutoMinorLocator

def generate_mock_time_data():
    k_values = np.arange(10, 101, 10)
    data = []
    
    for k in k_values:
        # MC-CELF: 耗时随 k 线性/多项式增长，基数极大 (模拟 EmailEnron 级别网络)
        mc_time = 50000 * (k ** 1.2) + np.random.normal(0, 5000)
        
        # RIS: 耗时极低，几乎不随 k 增长 (因为 SSR 已经提前生成好了)
        ris_time = 150 + k * 0.5 + np.random.normal(0, 10)
        
        data.append({"method": "MC-CELF (Baseline)", "seed_num": k, "time_ms": mc_time})
        data.append({"method": "RIS-Optimized (Ours)", "seed_num": k, "time_ms": ris_time})
        
    return pd.DataFrame(data)

def draw_time():
    df = generate_mock_time_data()
    
    sns.set_theme(style="ticks", context="paper")
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["Times New Roman"],
        "font.size": 12, "axes.labelsize": 14, "axes.titlesize": 15,
        "legend.fontsize": 12, "axes.linewidth": 1.0, "grid.alpha": 0.2,
    })

    fig, ax = plt.subplots(figsize=(7, 5.5))

    styles = {
        "MC-CELF (Baseline)": {"color": "#E45756", "marker": "s"},
        "RIS-Optimized (Ours)": {"color": "#4C78A8", "marker": "*", "markersize": 12}
    }

    for method, style in styles.items():
        subset = df[df["method"] == method]
        ax.plot(subset["seed_num"], subset["time_ms"], label=method,
                color=style["color"], marker=style["marker"], 
                linewidth=2.5, markersize=style.get("markersize", 8), alpha=0.9)

    ax.set_title("Expected Time Consumption vs. Seed Size", pad=12)
    ax.set_xlabel(r"Seed Budget $k$")
    ax.set_ylabel(r"Time Consumption (milliseconds)")
    
    # 核心：对数坐标轴
    ax.set_yscale("log")
    
    ax.grid(axis="y", which="major", linestyle="--")
    ax.grid(axis="y", which="minor", linestyle=":", alpha=0.1)
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    sns.despine(fig=fig)
    ax.legend(loc="center right", frameon=True)

    plt.tight_layout()
    plt.savefig("mock_time_consumption.pdf", dpi=300, bbox_inches="tight")
    print("✅ 时间开销图已生成: mock_time_consumption.pdf")
    plt.show()

if __name__ == "__main__":
    draw_time()