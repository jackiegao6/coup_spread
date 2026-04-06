import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator, LogLocator

# 1. 定义数据集及其规模因子 (用于模拟不同网络规模下的时间开销)
DATASETS = [
    {"name": "netscience",     "nodes": 379,   "edges": 1828,   "scale": 1.0},    
    {"name": "netfacebookego", "nodes": 2888,  "edges": 5962,   "scale": 10.0},   
    {"name": "doubanrandom",   "nodes": 4723,  "edges": 11774,  "scale": 20.0},   
    {"name": "EmailEnron",     "nodes": 33696, "edges": 361622, "scale": 200.0}   
]

def generate_total_time_mock_data():
    k_values = np.arange(10, 201, 10)
    
    methods = [
        "MC-CELF (Upper Bound)", "RIS-Optimized (Ours)", "1Hop-Sort", 
        "Alpha-Sort", "PageRank", "DegreeTopM"
    ]
    
    data = []
    for ds in DATASETS:
        ds_name = ds["name"]
        scale = ds["scale"]
        
        # 模拟生成 100,000 个 SSR 集合的固定预处理时间 (毫秒)
        # 预处理时间随网络规模 (scale) 线性增长
        ssr_preprocessing_time = 1500 * scale + np.random.normal(0, 100 * scale)
        
        for method in methods:
            for k in k_values:
                if method == "MC-CELF (Upper Bound)":
                    # MC-CELF: 无预处理，选种时间随 k 剧烈增长
                    # 模拟公式: a * k^1.5
                    base_time = 20 * scale * (k ** 1.9)
                    noise = np.random.normal(0, base_time * 0.05)
                    time_ms = max(10, base_time + noise)
                    
                elif method == "RIS-Optimized (Ours)":
                    # 【核心修正】：RIS 总时间 = 固定的预处理时间 + 极低的选种时间
                    # 选种时间随 k 缓慢线性增长
                    selection_time = 7 * scale * (k ** 1.4)
                    noise = np.random.normal(0, selection_time * 0.1)
                    time_ms = ssr_preprocessing_time + max(1, selection_time + noise)
                    
                elif method == "PageRank":
                    # PageRank: 矩阵迭代时间 (固定) + 排序时间
                    base_time = 800 * scale
                    noise = np.random.normal(0, base_time * 0.05)
                    time_ms = max(5, base_time + noise)
                    
                elif method in ["DegreeTopM", "Alpha-Sort", "1Hop-Sort"]:
                    # 启发式: 仅需遍历节点或一跳邻居，时间极低
                    base_time = 50 * np.sqrt(scale)
                    noise = np.random.normal(0, base_time * 0.1)
                    time_ms = max(2, base_time + noise)
                    
                elif method == "Random":
                    # Random: 瞬间完成
                    base_time = 2.0
                    noise = np.random.normal(0, 0.5)
                    time_ms = max(0.5, base_time + noise)
                
                data.append({
                    "dataset": ds_name,
                    "method": method, 
                    "seed_num": k, 
                    "time_ms": time_ms
                })
            
    return pd.DataFrame(data)

def draw_total_time_1x4():
    df = generate_total_time_mock_data()
    
    # 2. 纯正的顶会 Matplotlib 风格设置
    plt.rcParams.update({
        "font.family": "serif", 
        "font.serif": ["Times New Roman"],
        "font.size": 12, 
        "axes.labelsize": 14, 
        "legend.fontsize": 11, 
        "axes.linewidth": 1.2, 
        "xtick.direction": "in",  
        "ytick.direction": "in",
        "xtick.top": True,        
        "ytick.right": True       
    })

    # 3. 严格复刻参考图的颜色和标记
    method_styles = {
        "RIS-Optimized (Ours)":  {"color": "red", "marker": ">", "label": "RIS-Optimized(ours)", "markersize": 6, "linewidth": 2.0, "zorder": 10},
        "1Hop-Sort":             {"color": "#FF00FF", "marker": "<", "label": "1Hop-Sort"},
        "Alpha-Sort":            {"color": "#808000",     "marker": "o", "label": "Alpha-Sort"},
        "PageRank":              {"color": "blue",    "marker": "^", "label": "PageRank"},

    }

    # 4. 创建 1行4列 的大图
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    letters = ['a', 'b', 'c', 'd']
    
    for col_idx, ds in enumerate(DATASETS):
        ds_name = ds["name"]
        subset_ds = df[df["dataset"] == ds_name]
        ax = axes[col_idx]
        
        # --- 绘制总时间开销 ---
        for method, style in method_styles.items():
            subset = subset_ds[subset_ds["method"] == method]
            y_values = np.maximum(subset["time_ms"], 0.1)
            
            ax.plot(subset["seed_num"], y_values, label=style["label"],
                    color=style["color"], marker=style["marker"], 
                    linewidth=style.get("linewidth", 1.2), 
                    markersize=style.get("markersize", 5), 
                    alpha=0.9, zorder=style.get("zorder", 1))

        # --- 设置坐标轴和内嵌文本 ---
        ax.set_xlabel("k", labelpad=4)
        ax.set_xlim(0, 205)
        ax.set_xticks([0, 50, 100, 150, 200])
        
        # Y 轴设置为对数坐标 (Log Scale)
        ax.set_yscale("log")
        
        # Y 轴标签只在最左侧显示
        if col_idx == 0:
            ax.set_ylabel(r"Total time consumption (ms)", labelpad=4)
            
        # 设置对数坐标的刻度显示格式
        locmin = LogLocator(base=10.0, subs=(0.2,0.4,0.6,0.8), numticks=12)
        ax.yaxis.set_minor_locator(locmin)
        ax.yaxis.set_minor_formatter(plt.NullFormatter())
        
        # 将 (a) dataset_name 写在图表内部左上角
        letter = letters[col_idx]
        ax.text(0.04, 0.94, f"({letter}) {ds_name}", 
                transform=ax.transAxes, fontsize=12, verticalalignment='top',
                bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=1.5))

    # 5. 提取图例并放置在整个大图的最顶部
    handles, labels = axes[0].get_legend_handles_labels()
    legend = fig.legend(handles, labels, loc="upper center", ncol=7, 
                        bbox_to_anchor=(0.5, 1.12), frameon=True, 
                        edgecolor='black', fancybox=False, handletextpad=0.4, columnspacing=1.2)
    legend.get_frame().set_linewidth(1.0)

    # 6. 调整子图间距
    plt.subplots_adjust(top=0.85, bottom=0.15, left=0.06, right=0.98, wspace=0.25)
    
    # 保存高清 PDF
    output_filename = "neurips_style_total_time_1x4.pdf"
    plt.savefig(output_filename, dpi=300, bbox_inches="tight")
    print(f"✅ 包含预处理时间的 1x4 总时间开销图已生成: {output_filename}")
    plt.show()

if __name__ == "__main__":
    draw_total_time_1x4()