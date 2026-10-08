#  RR-set 生成

## 1. 重访时不重新采样

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

| 表               | 内容                                             |
| ---------------- | ------------------------------------------------ |
| `r_values[v][j]` | 节点 v 在第 j 张券世界中的随机数                 |
| `choices[v][j]`  | 该随机数确定的完整行为：消费、丢弃或唯一转发邻居 |





## 3. Python 代码

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



## 4. 伪代码

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

### 调用链：对 k 张券分别调用

```text
1. (r, choices) ← SampleCouponWorlds(G, k, pᵃ, pᵈ)
2. 对 j = 1,…,k：
3.     R_j(u) ← GenerateRRSetForCoupon(G, u, j, choices)
4. 返回 (R_1(u),…,R_k(u))
```





## 5. 正向传播与反向采样的概率等价性

**结论：节点 v 对第 j 张券始终使用固定的 r[v,j] 和对应行为，那么正向消费概率与反向集合包含概率一致。每个节点拥有 k 个随机数并不破坏这一关系，而是为 k 张券分别建立了独立的可能世界。**

### 5.1 事件与概率的定义

固定券编号 j。第 j 列随机数决定可能世界

$
W_j=\{\phi_j(v):v\in V\}.
$

> 记 \($\phi_j(v)$\) 为节点 \(v\) 在第 \(j\) 张优惠券的可能世界 \($W_j$\) 中，由随机数 \($r_{v,j}$\) 确定的行为。



正向传播从源节点 s 出发，每次到达 v 时读取 $\phi_j(v)$：消费则在 v 终止，丢弃则无消费终止，选择邻居 w 则沿 (v,w) 转发。若进入固定转发环，则该券不产生消费；实现中可以在检测到重访时结束模拟。

记 $A_j(u,s;W_j)$ 为“第 j 张券从 s 出发，在世界 $W_j$ 下最终被 u 消费”的事件。这里讨论的概率为

$
M_{u,s}=\Pr_{W_j}[A_j(u,s;W_j)].
\tag{1}
$

由于所有券使用相同的行为概率分布，该边际概率不依赖 j。

记 $R_j(u;W_j)$ 为代码 `generate_rr_set_for_coupon` 输出的完整集合：若该列中 u 不消费，则集合为空；否则从 u 开始沿该列确定的转发边反向搜索。

### 5.2 单张券的概率等价性定理

**定理 1。** 对任意券编号 j、投放源 s 和目标节点 u，均有

$
\Pr_{W_j}[s\in R_j(u;W_j)]=M_{u,s}.
\tag{2}
$

**证明。** 固定任意一个世界 $W_j$。此时所有节点对第 j 张券的行为已经确定，正向传播与反向搜索均不再抽样。我们先证明逐个世界中的**事件等价**：

$
A_j(u,s;W_j)
\quad\Longleftrightarrow\quad
s\in R_j(u;W_j).
\tag{3}
$

**情况一：根节点 u 的固定行为不是消费。**

正向过程中，即使券到达 u，也不会在 u 消费；重复到达也不会改变其固定行为。因此左侧事件不成立。反向代码此时直接返回空集，右侧事件也不成立。

**情况二：根节点 u 的固定行为是消费。** 分别证明两个方向。

（1）正向在 u 消费，必然有 s 属于反向集合。

假设第 j 张券从 s 出发最终在 u 消费，其轨迹为

$
s=v_0\to v_1\to\cdots\to v_\ell=u.
$

轨迹上每个中间节点的固定行为均满足

$
\phi_j(v_i)=v_{i+1},\qquad 0\le i<\ell,
$

且 $\phi_j(u)=\mathrm{CONSUME}$。成功的轨迹不会包含重复节点；否则固定行为会使其陷入转发环，无法最终到达消费状态。

反向搜索首先将 u 加入集合。处理 u 时，$v_{\ell-1}$ 的固定行为指向 u，因此它会被加入；处理 $v_{\ell-1}$ 时，其前驱又会被加入。沿轨迹逐步向前，最终必然发现 s。

若某个节点此前已从其他分支加入，它仍属于集合，且会被队列处理，因此去重不会遗漏该轨迹。s=u 时，根节点直接属于集合，结论也成立。

（2）s 属于反向集合，必然正向在 u 消费。

假设 s 被反向搜索加入。若 s=u，结论直接成立。否则，s 首次被加入时，一定存在某个已经加入的节点 v，使得 $\phi_j(s)=v$。

沿每个节点“首次加入时指向的已加入节点”继续追溯，每一步都指向更早加入的节点，因此不可能成环。这条有限的链最终到达初始化时唯一的节点 u。

于是得到一条正向路径

$
s=v_0\to v_1\to\cdots\to v_\ell=u,
\qquad \phi_j(v_i)=v_{i+1}.
$

在固定世界 $W_j$ 中，一张券到达每个中间节点时都执行这个已确定的转发行为，不会改为消费、丢弃或转发给其他邻居。因此从 s 投放的第 j 张券必然沿该路径到达 u，并由 u 消费。

综上，式（3）对每一个 $W_j$ 都成立，所以相应指示变量相等：

$
\mathbf 1\{A_j(u,s;W_j)\}
=\mathbf 1\{s\in R_j(u;W_j)\}.
$

对 $W_j$ 取期望即得

$
\begin{aligned}
M_{u,s}
&=\mathbb E_{W_j}[\mathbf 1\{A_j(u,s;W_j)\}]\\
&=\mathbb E_{W_j}[\mathbf 1\{s\in R_j(u;W_j)\}]\\
&=\Pr_{W_j}[s\in R_j(u;W_j)].
\end{aligned}
$

证毕。

这个证明是在同一个世界中建立事件对应，并不要求实际运行时正向与反向必须使用同一个随机种子。分别运行时，只要它们生成相同分布的世界，事件概率仍然相等。

### 5.3 k 张券为何仍然成立

第 j 张券只查询第 j 列；新增其他券的列不会改变 $W_j$ 的分布，也不会改变 $R_j(u;W_j)$。因此，定理 1 对每个 j 分别成立。

进一步，固定根节点 u 和预先给定的投放方案 $(s_1,\ldots,s_k)$，其中第 j 张券投放给 $s_j$。对任意一组固定世界 $(W_1,\ldots,W_k)$，由式（3）逐券取并集，有

$
\{\text{用户 }u\text{ 至少消费一张券}\}
\quad\Longleftrightarrow\quad
\{\exists j\in\{1,\ldots,k\}:s_j\in R_j(u;W_j)\}.
\tag{4}
$

由于各列随机数相互独立，各券的消费事件在给定 u 和投放方案后相互独立。因此

$
\begin{aligned}
\Pr[\text{用户 }u\text{ 至少消费一张券}]
&=1-\prod_{j=1}^{k}(1-M_{u,s_j})\\
&=\Pr[\exists j:s_j\in R_j(u;W_j)].
\end{aligned}
\tag{5}
$

所以，不仅每张券的正反向概率一致，k 张券中“至少一张被 u 消费”的概率，也与对应集合中“至少一个编号匹配的源节点命中”的概率一致。一个用户即使命中多张券，也只计一次激活。

这里的独立性是在根节点 u 固定后讨论的；如果根节点也随机抽取，不能忽略各个集合共享根节点带来的依赖。上述推导也没有加入非空组条件采样。

