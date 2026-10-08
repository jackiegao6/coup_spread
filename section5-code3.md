# 每个节点对应 k 个随机数的 RR-set 生成

## 1. 从单券缓存改为“节点 × 券编号”缓存

对于每个节点 v，独立生成 k 个随机数

$
r_{v,1},r_{v,2},\ldots,r_{v,k}\overset{\mathrm{i.i.d.}}{\sim}\operatorname{Uniform}[0,1).
$

所有不同的“节点—券编号”对使用独立的抽样。第 j 张券的可能世界为

$
W_j=\{\phi_j(v):v\in V\},\qquad
\phi_j(v)=\operatorname{Action}(v,r_{v,j}).
$

也就是说：

- 同一节点、同一张券：使用同一个 r 和同一个完整行为，后续检查不重新抽样。
- 同一节点、不同券：分别使用各自的 r，行为可以不同，也可能恰好相同。
- 生成一批新的独立样本时：重新生成整张随机数表，不能一直沿用上一批的表。

代码保存两张表：

| 表 | 内容 |
| --- | --- |
| `r_values[v][j]` | 节点 v 在第 j 张券世界中的随机数 |
| `choices[v][j]` | 该随机数确定的完整行为：消费、丢弃或唯一转发邻居 |

数学中的券编号为 1,…,k；Python 使用 0,…,k−1，代码下标 j 对应数学上的第 j+1 张券。

## 2. 一个 r 如何确定完整行为

沿用现有实验“选择转发后均匀选择出邻居”的分布。记

$
p_v^t=1-p_v^a-p_v^d.
$

当 r 小于消费概率时消费，落在随后长度为丢弃概率的区间时丢弃。剩余长度为 \($p_v^t$\) 的区间，按固定的出邻居顺序等分为 \($|N^+(v)|$\) 段，每一段对应一个转发邻居。因此

$
p(v,w)=\frac{p_v^t}{|N^+(v)|}.
$

**同一个 r 同时确定“是否转发”和“转发给谁”。** 

## 3. Python 代码

为直接展示“每个节点有 k 个 r”，这里一次性生成全部随机数和行为，再按券编号分别执行反向搜索。输入假设为合法的图和概率：节点编号非负，所有节点都有字典条目，入邻居表与出邻居表一致；无出邻居的节点满足消费与丢弃概率之和为 1。邻接表顺序在生成本批样本期间保持不变。

```python
import random
from collections import deque


CONSUME = -2
DROP = -1


def sample_coupon_worlds(
    out_neighbors: dict[int, list[int]],
    p_a: dict[int, float],
    p_d: dict[int, float],
    k: int,
    rng: random.Random,
) -> tuple[dict[int, list[float]], dict[int, list[int]]]:
    """生成 k 个独立世界：每个节点存储 k 个 r 和 k 个完整行为。"""
    if k < 1:
        raise ValueError("k must be positive")

    r_values = {}
    choices = {}

    for v, neighbors in out_neighbors.items():
        r_values[v] = [rng.random() for _ in range(k)]
        choices[v] = []

        for r in r_values[v]:
            if r < p_a[v]:
                action = CONSUME
            elif r < p_a[v] + p_d[v]:
                action = DROP
            else:
                # 此分支下，合法输入保证有出邻居且转发概率大于 0。
                p_transfer = 1.0 - (p_a[v] + p_d[v])
                position = (r - (p_a[v] + p_d[v])) / p_transfer
                # 将转发区间等分；min 仅防止浮点舍入越界。
                index = min(int(position * len(neighbors)), len(neighbors) - 1)
                action = neighbors[index]

            choices[v].append(action)

    return r_values, choices


def generate_rr_set_for_coupon(
    root: int,
    coupon_id: int,
    in_neighbors: dict[int, list[int]],
    choices: dict[int, list[int]],
) -> set[int]:
    """在已固定的第 coupon_id 个世界中，生成一个完整 RR 集合。"""
    if not 0 <= coupon_id < len(choices[root]):
        raise ValueError("coupon_id out of range")

    # 根节点也读取同一列；不能另抽一个 r 来决定消费。
    if choices[root][coupon_id] != CONSUME:
        return set()

    rr_set = {root}
    queue = deque([root])

    while queue:
        v = queue.popleft()
        for w in in_neighbors[v]:
            # 每次都读取 w 在当前券世界中的固定行为。
            action = choices[w][coupon_id]
            if action == v and w not in rr_set:
                rr_set.add(w)
                queue.append(w)

    return rr_set
```

`sample_coupon_worlds` 负责抽样，`generate_rr_set_for_coupon` 只负责搜索。后一个函数完全不生成随机数，也不会修改已固定的行为表。

### 调用示例

下面对同一个根节点生成 k 个单券 RR set。`rr_sets[j]` 保存第 j 张券对应的集合，不将不同券的集合合并，也不做覆盖选择。

```python
in_neighbors = {
    0: [],
    1: [0],
    2: [0, 1],
}
out_neighbors = {
    0: [1, 2],
    1: [2],
    2: [],
}
p_a = {0: 0.2, 1: 0.3, 2: 0.5}
p_d = {0: 0.1, 1: 0.2, 2: 0.5}
root, k = 2, 3

r_values, choices = sample_coupon_worlds(
    out_neighbors, p_a, p_d, k, random.Random(42)
)
rr_sets = [
    generate_rr_set_for_coupon(root, j, in_neighbors, choices)
    for j in range(k)
]

print("节点 0 的 k 个随机数：", r_values[0])
print("节点 0 对各张券的行为：", choices[0])
print("各张券的 RR 集合：", [sorted(rr) for rr in rr_sets])
```

在该例中，一次初始化会生成 3×3=9 个随机数；后续三次反向搜索均不会增加抽样次数。重复使用同一张表查询同一根节点、同一券编号，得到的是同一个样本，而不是新的独立样本。

## 4. 对应伪代码

### 算法 A：生成每个节点的 k 个随机数与行为

```text
算法：SampleCouponWorlds(G, k, pᵃ, pᵈ)
输入：有向图 G，券数量 k，消费与丢弃概率
输出：随机数表 r[v,j]，行为表 choices[v,j]

 1. 对每个节点 v ∈ V：
 2.     对每张券 j = 1,…,k：
 3.         独立生成 r[v,j] ∈ [0,1)
 4.         若 r[v,j] < pᵃ_v：
 5.             choices[v,j] ← CONSUME
 6.         否则，若 r[v,j] < pᵃ_v + pᵈ_v：
 7.             choices[v,j] ← DROP
 8.         否则：
 9.             将剩余转发区间按出邻居顺序等分
10.             确定 r[v,j] 落入的区间对应的邻居 x
11.             choices[v,j] ← x

12. 返回 r 和 choices
```

### 算法 B：在第 j 张券的世界中生成 RR set

```text
算法：GenerateRRSetForCoupon(G, u, j, choices)
输入：有向图 G，根节点 u，券编号 j，已生成的行为表
输出：第 j 张券的完整反向集合 R_j(u)

 1. 若 choices[u,j] ≠ CONSUME：返回空集
 2. R_j(u) ← {u}
 3. Q ← 只包含 u 的队列

 4. 当 Q 非空时：
 5.     从 Q 取出节点 v
 6.     对每个入邻居 w ∈ N⁻(v)：
 7.         读取已保存的 choices[w,j]，不重新抽样
 8.         若 choices[w,j] = v 且 w ∉ R_j(u)：
 9.             将 w 加入 R_j(u)
10.             将 w 加入 Q

11. 返回 R_j(u)
```

### 算法 C：对 k 张券分别调用

```text
1. (r, choices) ← SampleCouponWorlds(G, k, pᵃ, pᵈ)
2. 对 j = 1,…,k：
3.     R_j(u) ← GenerateRRSetForCoupon(G, u, j, choices)
4. 返回 (R_1(u),…,R_k(u))
```

根节点和普通节点使用完全相同的列索引 j。R_j 中的成员标记和队列也按单券分别初始化，不跨券共用。

