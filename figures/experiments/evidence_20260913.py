"""Render Chinese-draft evidence separately from archived English figures."""
from pathlib import Path
import argparse
import csv
import statistics
import matplotlib.pyplot as plt
from _plot_style import apply_style, save_outputs, METHOD_COLORS, METHOD_MARKERS
import validated_study_figures as study

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "experiments/results/evidence-20260913"
OUTPUT = ROOT / "figures/experiments/evidence-20260913"


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert rows and all(row.get('status') == 'REAL_EXPERIMENT' for row in rows)
    return rows


def evidence_read(path):
    if path.name == 'validated_summary.csv':
        path = RESULTS / 'ci/validated_summary.csv'
    elif path.name == 'validated_sampler_ablation.csv':
        path = RESULTS / 'sampler/summary.csv'
    return read_csv(path)


def evidence_save(fig, script_path, description=None):
    save_outputs(fig, str(OUTPUT / Path(script_path).name), description)


def plot_large():
    rows = read_csv(RESULTS / 'large-graph/summary.csv')
    raw = read_csv(RESULTS / 'large-graph/raw.csv')
    apply_style()
    fig, (ax, gain_ax) = plt.subplots(1, 2, figsize=(7.1, 3.2))
    for method in ('CIM-RIS', 'IC-RIS', '1Hop-Sort', 'Alpha-Sort', 'DegreeTopM', 'PageRank', 'Random'):
        selected = sorted((r for r in rows if r['method'] == method), key=lambda r: int(r['k']))
        ax.errorbar([int(r['k']) for r in selected], [float(r['mean_adopters']) for r in selected], yerr=[float(r['ci95_across_selection_runs']) for r in selected], color=METHOD_COLORS[method], marker=METHOD_MARKERS[method], label=method, capsize=2, linewidth=1.3)
    ax.set_xlabel('Coupon budget, $k$')
    ax.set_ylabel('Distinct adopters')
    ax.set_title('(a) Independent forward quality', loc='left')
    ax.set_xticks([10, 25, 50, 100, 150, 200])
    budgets, gains, intervals = [], [], []
    for k in (10, 25, 50, 100, 150, 200):
        means = {r['method']: float(r['mean_adopters']) for r in rows if int(r['k']) == k}
        best = max((method for method in means if method != 'CIM-RIS'), key=means.get)
        selected = {(r['method'], r['selection_seed']): float(r['mean_adopters']) for r in raw if int(r['k']) == k}
        paired = [100 * (selected[('CIM-RIS', str(seed))] / selected[(best, str(seed))] - 1) for seed in range(20260715, 20260720)]
        budgets.append(k)
        gains.append(statistics.mean(paired))
        intervals.append(2.7764451051977987 * statistics.stdev(paired) / 5**.5)
    gain_ax.errorbar(budgets, gains, yerr=intervals, color=METHOD_COLORS['CIM-RIS'], marker='o', capsize=3)
    gain_ax.axhline(0, color='#777777', linestyle='--', linewidth=.8)
    gain_ax.set_xlabel('Coupon budget, $k$')
    gain_ax.set_ylabel('Paired relative difference (%)')
    gain_ax.set_title('(b) CIM-RIS vs. strongest baseline', loc='left')
    gain_ax.set_xticks([10, 25, 50, 100, 150, 200])
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=4, fontsize=7, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, .83))
    evidence_save(fig, 'large_graph_quality.py', 'Full Douban independent quality evaluation; Student-t 95% intervals over five selection runs.')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--part', choices=['quality', 'sampler', 'large', 'all'], default='all')
    args = parser.parse_args()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    study.read_real_csv = evidence_read
    study.save_outputs = evidence_save
    if args.part in ('quality', 'all'):
        study.plot_quality_combined()
    if args.part in ('sampler', 'all'):
        study.plot_sampler()
    if args.part in ('large', 'all'):
        plot_large()


if __name__ == '__main__':
    main()
