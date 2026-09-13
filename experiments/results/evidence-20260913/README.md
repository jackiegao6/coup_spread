# 中文稿补充证据（2026-09-13）

本目录保留原 `validated-v2` 之外的新证据。受控扩散模拟不等于实地投放，历史 Linux 计时与本机 Windows 计时不能混合比较。

- `ci/`：沿用全部原始主实验任务，重算 df=4 的双侧 95% Student-t 区间，均值不变。
- `sampler/summary_batches.csv`：两个图、三个预算、两种采样器各 30 批，共 360 批；每批 20,000 样本。保存批次种子、估计值、零贡献比例和计时。
- `sampler/summary.csv`：逐批汇总；`mean_batch_seconds` 包含根权重/根列表生成和采样循环，`mean_loop_seconds` 保留原先的循环计时范围。
- `sampler/summary.metadata.json`：输入、源码、结果指纹及运行环境。采样均值与标准差复现历史汇总，新时间来自本轮实际运行。
- `large-graph/`：完整 Douban、均衡场景、六预算、五随机种子、七方法的独立前向质量评估。每个任务 JSON 保存分配和随机流，不用样本内估计替代质量。`training_estimate_not_quality` 仅作诊断，可以超过真实采纳规模的物理上限。
- `provenance.json`：五个 pickle 图与本地 MAT 的矩阵值和 CSR 索引对应关系，以及仍缺失的上游来源字段。

运行前固定的范围见 `plan/task-packets/2026-09-13-experiment-evidence-repair.md`。源码、原始结果、摘要和图均不应根据收益正负筛选。

在仓库根目录复核：

```powershell
py -3.12 -m unittest discover -s experiments -p 'test_*.py'
py -3.12 experiments/verify_evidence_repair.py
py -3.12 experiments/audit_dataset_provenance.py
```

生成过程：

```powershell
py -3.12 experiments/aggregate_validated_study.py --output-dir experiments/results/evidence-20260913/ci
py -3.12 experiments/run_real_sampler_ablation.py --validated-job-dir experiments/results/validated-v2/jobs --datasets Netscience,EmailEnron --budgets 10,50,200 --batch-size 20000 --repeats 30 --protocol-version evidence-20260913-sampler --output experiments/results/evidence-20260913/sampler/summary.csv
py -3.12 experiments/run_large_graph_quality.py
py -3.12 figures/experiments/evidence_20260913.py
```

消融脚本拒绝覆盖已有结果；复跑时用新的 `--output` 路径。大图脚本只在参数、代码、数据和环境指纹一致时复用已完成任务。以上哈希为运行时字节指纹，跨平台检出时应保留文件原有换行；旧存档的 CRLF/LF 差异不代表数值被修改。

原始发布链接、MAT 生成前的清洗抽样过程及节点 ID 映射仍待提供。不能把已验证的本地 MAT 对应关系扩大为对原始发布来源的认证。
