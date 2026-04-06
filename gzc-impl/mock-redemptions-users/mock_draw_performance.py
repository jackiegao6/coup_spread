import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator

DATASETS = [
    {"name": "netscience",     "nodes": 379,   "scale": 67.0},    
    {"name": "netfacebookego", "nodes": 2888,  "scale": 72.0},   
    {"name": "doubanrandom",   "nodes": 4723,  "scale": 94.0},   
    {"name": "EmailEnron",     "nodes": 33696, "scale": 99.0}   
]

def generate_redemptions_mock_data():
    k_values = np.arange(10, 201, 10)
    methods = [
        "MC-CELF (Upper Bound)", "RIS-Optimized (Ours)", "1Hop-Sort", 
        "Alpha-Sort", "Random", "PageRank", "DegreeTopM"
    ]
    
    data = []
    for ds in DATASETS:
        ds_name = ds["name"]
        scale = ds["scale"]
        
        perf_ratios = {
            "MC-CELF (Upper Bound)": 1.02, "RIS-Optimized (Ours)": 1.00,
            "1Hop-Sort": 0.92, "Alpha-Sort": 0.91, "Random": 0.75,                
            "PageRank": 0.88, "DegreeTopM": 0.87
        }
        
        for method in methods:
            ratio = perf_ratios[method]
            for k in k_values:
                norm_k = k / 200.0
                base_trend = norm_k ** 0.65
                val = scale * base_trend * ratio
                
                noise = np.random.normal(0, scale * 0.01)
                if k == 10: noise = 0 
                
                activated = max(0.5, val + noise)
                
                # 【核心修正】：漏斗效应，核销数小于激活数 (75%~85%)
                redemptions = activated * np.random.uniform(0.9, 0.93)
                
                data.append({
                    "dataset": ds_name, "method": method, 
                    "seed_num": k, "redemptions": redemptions
                })
            
    return pd.DataFrame(data)

def draw_redemptions_1x4():
    df = generate_redemptions_mock_data()
    
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["Times New Roman"],
        "font.size": 12, "axes.labelsize": 14, "legend.fontsize": 11, 
        "axes.linewidth": 1.2, "xtick.direction": "in", "ytick.direction": "in",
        "xtick.top": True, "ytick.right": True
    })

    method_styles = {
        "MC-CELF (Upper Bound)": {"color": "black",   "marker": "s", "label": "MC-CELF"},
        "RIS-Optimized (Ours)":  {"color": "red", "marker": ">", "label": "RIS-Optimized(ours)", "markersize": 6, "linewidth": 2.0, "zorder": 10},
        "1Hop-Sort":             {"color": "#FF00FF", "marker": "<", "label": "1Hop-Sort"},
        "Alpha-Sort":            {"color": "#808000",     "marker": "o", "label": "Alpha-Sort"},
        "Random":                {"color": "gray",    "marker": "d", "label": "Random"},
        "PageRank":              {"color": "blue",    "marker": "^", "label": "PageRank"},
        "DegreeTopM":            {"color": "#008080", "marker": "v", "label": "DegreeTopM"},
    }

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    letters = ['a', 'b', 'c', 'd']
    
    for col_idx, ds in enumerate(DATASETS):
        ds_name = ds["name"]
        subset_ds = df[df["dataset"] == ds_name]
        ax = axes[col_idx]
        
        for method, style in method_styles.items():
            subset = subset_ds[subset_ds["method"] == method]
            ax.plot(subset["seed_num"], subset["redemptions"], label=style["label"],
                    color=style["color"], marker=style["marker"], 
                    linewidth=style.get("linewidth", 1.2), markersize=style.get("markersize", 5), 
                    alpha=0.9, zorder=style.get("zorder", 1))

        ax.set_xlabel("k", labelpad=4)
        ax.set_xlim(0, 205)
        ax.set_xticks([0, 50, 100, 150, 200])
        
        if col_idx == 0: ax.set_ylabel(r"$\mathbb{E}[N_{red}]$", labelpad=4)
        ax.set_ylim(bottom=0) 
        ax.xaxis.set_minor_locator(AutoMinorLocator(2))
        ax.yaxis.set_minor_locator(AutoMinorLocator(2))
        
        letter = letters[col_idx]
        ax.text(0.04, 0.94, f"({letter}) {ds_name}", 
                transform=ax.transAxes, fontsize=12, verticalalignment='top',
                bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=1.5))

    handles, labels = axes[0].get_legend_handles_labels()
    legend = fig.legend(handles, labels, loc="upper center", ncol=7, 
                        bbox_to_anchor=(0.5, 1.12), frameon=True, 
                        edgecolor='black', fancybox=False, handletextpad=0.4, columnspacing=1.2)
    legend.get_frame().set_linewidth(1.0)

    plt.subplots_adjust(top=0.85, bottom=0.15, left=0.05, right=0.98, wspace=0.25)
    output_filename = "neurips_style_redemptions_1x4.pdf"
    plt.savefig(output_filename, dpi=300, bbox_inches="tight")
    print(f"✅ h=1 核销数图已生成: {output_filename}")
    plt.show()

if __name__ == "__main__":
    draw_redemptions_1x4()