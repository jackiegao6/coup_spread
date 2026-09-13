# 数据来源核查

## 已验证

`experiments/audit_dataset_provenance.py` 对五个主实验图执行了独立 SciPy 读取比较：

| 实验名 | network.mat 变量 | 节点数 | 有向边数 |
|---|---|---:|---:|
| Netscience | netscience | 379 | 1,828 |
| NetFacebookEgo | netfacebookego | 2,888 | 5,962 |
| DoubanRandom | doubanrandom | 4,723 | 11,774 |
| EmailEnron | EmailEnron | 33,696 | 361,622 |
| network.douban | douban | 154,907 | 654,206 |

五个 CSR 文件均与对应 MAT 变量具有相同矩阵值、列索引和行指针；所有存储边互反，没有自环和非单位权重。完整哈希和统计保存在 `experiments/results/evidence-20260913/provenance.json`。

`gzc-impl/convert_mat_to_pickle.py` 展示了从 MAT 变量转为 CSR 再 pickle 的过程，其历史示例入口为 EmailEnron。文件相等支持这一转换关系，但不能单独证明其他数据的历史转换命令曾被执行。

## 尚未建立的上游链条

- 原始发布者及每个数据集的下载地址、版本、许可证。
- 下载文件到 network.mat 之间的清洗、抽样、连通分量选择规则及随机种子。
- 原始用户 ID 到矩阵行号的映射。
- DoubanRandom 的具体抽样方法及是否为完整 Douban 的诱导子图。

已向用户询问原始来源；没有得到资料前不补写推测性来源。中文稿已将“来自 KONECT”的无条件断言改为可核实的本地存档描述，并披露上游来源尚待补全。本地 MAT 一致性不能作为原始数据真实性的独立认证。
