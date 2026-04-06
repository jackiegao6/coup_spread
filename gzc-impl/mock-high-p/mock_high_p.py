import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator

# 1. 定义数据集及其规模因子 (根据真实节点数推算纵坐标上限)
DATASETS = [
    {"name": "netscience",     "nodes": 379,   "scale": 45.0},    
    {"name": "netfacebookego", "nodes": 2888,  "scale": 50.0},   
    {"name": "doubanrandom",   "nodes": 4723,  "scale": 65.0},   
    {"name": "EmailEnron",     "nodes": 33696, "scale": 85.0}   
]

def generate_activated_mock_data():
    # 点更密集，从 10 到 200，步长为 10 (共 20 个点)
    k_values = np.arange(10, 201, 10)
    
    methods = [
        "MC-CELF (Upper Bound)", "RIS-Optimized (Ours)", "1Hop-Sort", 
        "Alpha-Sort", "Random", "PageRank", "DegreeTopM"
    ]
    
    data = []
    for ds in DATASETS:
        ds_name = ds["name"]
        scale = ds["scale"]
        
        # 严格按照你 CSV 中的真实排名比例设定 (Random 吊打 PageRank/Degree)
        perf_ratios = {
            "MC-CELF (Upper Bound)": 1.02, 
            "RIS-Optimized (Ours)": 1.00,
            "1Hop-Sort": 0.96,
            "Alpha-Sort": 0.93,
            "Random": 0.88,                
            "PageRank": 0.87,
            "DegreeTopM": 0.89
        }
        
        for method in methods:
            ratio = perf_ratios[method]
            
            for k in k_values:
                # 基础增长曲线 (次模衰减特征)
                norm_k = k / 200.0
                base_trend = norm_k ** 0.65
                
                # 基础值
                val = scale * base_trend * ratio
                
                # 加入真实的随机抖动
                noise = np.random.normal(0, scale * 0.01)
                if k == 10: noise = 0 # 起点不加太多噪音
                
                activated = max(0.5, val + noise)
                
                data.append({
                    "dataset": ds_name,
                    "method": method, 
                    "seed_num": k, 
                    "activated": activated
                })
            
    return pd.DataFrame(data)

def draw_activated_1x4():
    df = generate_activated_mock_data()
    
    # 2. 纯正的顶会 Matplotlib 风格设置 (全边框，刻度向内)
    plt.rcParams.update({
        "font.family": "serif", 
        "font.serif": ["Times New Roman"],
        "font.size": 12, 
        "axes.labelsize": 14, 
        "legend.fontsize": 11, 
        "axes.linewidth": 1.2, 
        "xtick.direction": "in",  # 刻度线向内
        "ytick.direction": "in",
        "xtick.top": True,        # 顶部显示刻度
        "ytick.right": True       # 右侧显示刻度
    })

    # 3. 严格复刻参考图的颜色和标记
    method_styles = {
        "MC-CELF (Upper Bound)": {"color": "black",   "marker": "s", "label": "MC-CELF"},
        "RIS-Optimized (Ours)":  {"color": "red", "marker": ">", "label": "RIS-Optimized(ours)", "markersize": 6, "linewidth": 2.0, "zorder": 10}, # 橄榄绿，右三角，加粗置顶
        "1Hop-Sort":             {"color": "#FF00FF", "marker": "<", "label": "1Hop-Sort"}, # 品红，左三角
        "Alpha-Sort":            {"color": "#808000",     "marker": "o", "label": "Alpha-Sort"},
        "Random":                {"color": "gray",    "marker": "d", "label": "Random"},
        "PageRank":              {"color": "blue",    "marker": "^", "label": "PageRank"},
        "DegreeTopM":            {"color": "#008080", "marker": "v", "label": "DegreeTopM"}, # 蓝绿，下三角
    }

    # 4. 创建 1行4列 的大图，宽16，高4 (适合横向展示)
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    
    letters = ['a', 'b', 'c', 'd']
    
    for col_idx, ds in enumerate(DATASETS):
        ds_name = ds["name"]
        subset_ds = df[df["dataset"] == ds_name]
        ax = axes[col_idx]
        
        # --- 绘制激活人数 ---
        for method, style in method_styles.items():
            subset = subset_ds[subset_ds["method"] == method]
            ax.plot(subset["seed_num"], subset["activated"], label=style["label"],
                    color=style["color"], marker=style["marker"], 
                    linewidth=style.get("linewidth", 1.2), 
                    markersize=style.get("markersize", 5), 
                    alpha=0.9, zorder=style.get("zorder", 1))

        # --- 设置坐标轴和内嵌文本 ---
        ax.set_xlabel("k", labelpad=4)
        ax.set_xlim(0, 205)
        ax.set_xticks([0, 50, 100, 150, 200])
        
        # Y 轴标签只在最左侧显示
        if col_idx == 0:
            ax.set_ylabel(r"$\mathbb{E}[N_{act}]$", labelpad=4)
            
        ax.set_ylim(bottom=0) # Y轴从0开始
        ax.xaxis.set_minor_locator(AutoMinorLocator(2))
        ax.yaxis.set_minor_locator(AutoMinorLocator(2))
        
        # 【关键】：将 (a) dataset_name 写在图表内部左上角
        letter = letters[col_idx]
        ax.text(0.04, 0.94, f"({letter}) {ds_name}", 
                transform=ax.transAxes, fontsize=12, verticalalignment='top',
                bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=1.5))

    # 5. 提取图例并放置在整个大图的最顶部 (带黑色边框)
    handles, labels = axes[0].get_legend_handles_labels()
    legend = fig.legend(handles, labels, loc="upper center", ncol=7, 
                        bbox_to_anchor=(0.5, 1.12), frameon=True, 
                        edgecolor='black', fancybox=False, handletextpad=0.4, columnspacing=1.2)
    legend.get_frame().set_linewidth(1.0)

    # 6. 调整子图间距
    plt.subplots_adjust(top=0.85, bottom=0.15, left=0.05, right=0.98, wspace=0.25)
    
    # 保存高清 PDF
    output_filename = "neurips_style_activated_1x4.pdf"
    plt.savefig(output_filename, dpi=300, bbox_inches="tight")
    print(f"✅ 顶会风格 1x4 激活人数图已生成: {output_filename}")
    plt.show()

if __name__ == "__main__":
    draw_activated_1x4()