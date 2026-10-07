可以。下面这版我先按**“中文方法稿”**来写，目标不是追求英文论文腔，而是把老师要求的核心逻辑完整落下来：先把 RR-set 怎么生成讲清楚，再证明正向传播与反向可达概率一致。这里严格按照你刚才确认的代码机制：

\[
\text{同一 visit state 只采样一次并复用，}
\]

\[
\text{同一物理节点再次访问时产生新的 visit id，并重新独立采样。}
\]

---

## 5.1 优惠券传播模型下的反向可达集构造

经典影响力最大化中的反向可达集（Reverse Reachable Set, RR set）通过反向搜索，将“从某个种子节点出发能否激活目标节点”的正向传播事件转化为“该种子节点是否属于目标节点的 RR set”的集合包含事件。对于本文的优惠券传播模型，一张 coupon 在每次访问用户时只能执行消费、丢弃或向一个邻居转发中的一种动作，并且只有最终消费 coupon 的用户才被视为激活。因此，经典 IC 模型中的 RR-set 生成过程不能直接应用，需要根据 coupon 的单路径传播和重复访问机制重新构造反向可达集。

### Visit-state representation

由于一张 coupon 的传播轨迹可能形成环并重复访问同一用户，仅使用物理节点不足以描述一次完整的随机传播过程。为此，我们进一步引入 **visit state**。

对于节点 \(v\)，记

\[
(v,h)
\]

为 coupon 对节点 \(v\) 的第 \(h\) 个访问状态，其中 \(h\) 为该状态对应的 visit id。同一物理节点的不同访问状态，例如

\[
(v,1),\quad (v,2),
\]

对应相互独立的随机行为。

对于每个 visit state \((v,h)\)，随机采样

\[
r_{v,h}\sim {\rm Uniform}[0,1),
\]

并根据节点 \(v\) 的行为概率确定唯一动作。记该动作为

\[
\phi(v,h)
\in
\{\text{consume},\text{drop}\}\cup N^+(v).
\]

若

\[
r_{v,h}<p_v^a,
\]

则 \(\phi(v,h)=\text{consume}\)；若

\[
p_v^a\le r_{v,h}<p_v^a+p_v^d,
\]

则 \(\phi(v,h)=\text{drop}\)；否则根据边概率区间唯一确定某个 \(x\in N^+(v)\)，并令

\[
\phi(v,h)=x,
\]

表示 coupon 在该访问状态下沿边 \((v,x)\) 转发。

算法实现中，每个节点 \(v\) 维护一个以 visit id 为键的 Hashtable \(H_v\)，其中

\[
H_v[h]=\phi(v,h).
\]

当算法第一次需要揭示状态 \((v,h)\) 时，独立生成 \(r_{v,h}\) 并将对应动作保存至 \(H_v[h]\)。之后若再次检查同一个状态 \((v,h)\)，直接复用已经存储的动作，而不重新采样。若传播过程中再次访问物理节点 \(v\)，则创建新的 visit id，例如 \((v,h+1)\)，并独立采样新的动作。

因此，本模型遵循如下原则：

\[
\boxed{
\text{same node + same visit id}
\Longrightarrow
\text{reuse the sampled action},
}
\]

而

\[
\boxed{
\text{same node + different visit ids}
\Longrightarrow
\text{independent actions}.
}
\]

这一状态表示与 Section~3 中“coupon 重访节点时重新采样动作”的传播规则一致。

### State-level reverse reachable set

给定一个目标用户 \(u\)，考虑一张 coupon 的一次反向样本。外层采样过程首先判断作为传播终点的 root state 是否发生消费。如果 root 对应的消费事件不成立，则本次 coupon 不可能通过该 root 产生激活，因此直接返回空 RR set。

若 root 的消费事件成立，则固定该状态为本次 coupon 的最终消费状态，并从 root 开始反向搜索。

为区分重复访问，我们首先在状态空间上定义反向可达集，记为

\[
\widetilde R(u).
\]

\(\widetilde R(u)\) 中保存的是 visit states，而不是单纯的物理节点。

初始化时将 root consumption state 加入集合和搜索队列。随后，每次从队列中取出当前状态 \((v,h_v)\)，检查所有满足

\[
(w,v)\in E
\]

的物理入邻居 \(w\)。

对于每个可能的 predecessor visit state \((w,h_w)\)，算法查询节点 \(w\) 的 Hashtable：

- 如果 \(H_w[h_w]\) 尚未生成，则采样一个新的 \(r_{w,h_w}\)，确定唯一动作 \(\phi(w,h_w)\)，并保存；
- 如果 \(H_w[h_w]\) 已经存在，则直接读取已有动作；
- 只有当
  \[
  \phi(w,h_w)=v
  \]
  时，状态 \((w,h_w)\) 才能够在正向传播中将 coupon 转发至当前状态对应的节点 \(v\)，因此将其加入 \(\widetilde R(u)\)，并继续反向扩展。

如果 \(\phi(w,h_w)\) 为 consume、drop，或者选择了其他出邻居，则此次反向检查失败。

需要注意的是，反向搜索会检查当前状态的所有可能 predecessor，因此反向结构可以出现多个分支。这里的分支并不表示一张 coupon 在正向传播中被复制，而表示**存在多个不同的候选初始位置，它们分别可能形成一条最终到达同一消费节点 \(u\) 的单 coupon trajectory**。

由于 seed \(s\) 初始获得 coupon 时对应的是其第一次访问状态 \((s,1)\)，我们最终定义节点级 RR set 为

\[
R(u)
=
\{s\in V:(s,1)\in\widetilde R(u)\}.
\tag{15}
\]

也就是说，只有当 seed 的初始状态 \((s,1)\) 能够在本次反向随机实现中到达 root consumption state 时，\(s\) 才属于 \(R(u)\)。

---

### Algorithm: Generate one coupon RR set

下面给出对应的抽象伪代码。实际实现中，visit id 的创建与状态对应关系由每个节点的 Hashtable 维护。

```latex
\begin{algorithm}[t]
\caption{\textsc{GenerateCouponRR}$(u)$}
\label{alg:coupon-rr}
\begin{algorithmic}[1]
\Require Directed graph $G=(V,E)$, root $u$, behavior probabilities
\Ensure A coupon reverse reachable set $R(u)$

\State Sample the action of the root consumption state
\If{the root does not consume the coupon}
    \State \Return $\emptyset$
\EndIf

\State Initialize the state-level set $\widetilde R(u)$ with the root state
\State Initialize a queue $Q$ with the root state
\State Initialize the visit-state hash tables $\{H_v\}_{v\in V}$

\While{$Q$ is not empty}
    \State Remove a visit state $(v,h_v)$ from $Q$
    \For{each physical in-neighbor $w\in N^-(v)$}
        \State Determine the corresponding predecessor visit state $(w,h_w)$
        \If{$H_w[h_w]$ is unrevealed}
            \State Sample $r_{w,h_w}\sim {\rm Uniform}[0,1)$
            \State Determine the unique action $\phi(w,h_w)$
            \State Store $H_w[h_w]\gets\phi(w,h_w)$
        \EndIf
        \If{$\phi(w,h_w)=v$ and $(w,h_w)\notin\widetilde R(u)$}
            \State Add $(w,h_w)$ to $\widetilde R(u)$
            \State Insert $(w,h_w)$ into $Q$
        \EndIf
    \EndFor
\EndWhile

\State $R(u)\gets\{s\in V:(s,1)\in\widetilde R(u)\}$
\State \Return $R(u)$
\end{algorithmic}
\end{algorithm}
```

其中，“determine the corresponding predecessor visit state” 对应程序中由节点 Hashtable 和 visit-id bookkeeping 完成的状态管理。

---

## 5.2 正向传播与反向可达的概率等价性

上面的 RR-set 构造只有在其与原始正向 coupon propagation 具有相同的概率语义时，才能用于估计 influence spread。因此，本节证明本文方法最核心的性质：

\[
\boxed{
\Pr[\text{coupon seeded at }s\text{ is eventually consumed by }u]
=
\Pr[s\in R(u)].
}
\tag{16}
\]

### Theorem 1. Forward--reverse probability equivalence

对于任意节点 \(s,u\in V\)，设 \(A(u,s)\) 表示一张初始投放于 \(s\) 的 coupon 最终被用户 \(u\) 消费的事件。按照 Algorithm~\ref{alg:coupon-rr} 生成以 \(u\) 为根的随机 RR set \(R(u)\)，则有

\[
\Pr[A(u,s)]
=
\Pr[s\in R(u)].
\tag{17}
\]

### Proof

考虑一条从 seed \(s\) 出发并最终在 \(u\) 被消费的具体 visit-state trajectory：

\[
\pi=
\bigl(
(v_0,h_0),
(v_1,h_1),
\ldots,
(v_\ell,h_\ell)
\bigr),
\]

其中

\[
v_0=s,\qquad h_0=1,\qquad v_\ell=u.
\]

这里允许同一个物理节点多次出现，但不同出现位置具有不同的 visit id。

例如，物理轨迹

\[
s\rightarrow w\rightarrow x\rightarrow w\rightarrow u
\]

在状态空间中表示为

\[
(s,1)
\rightarrow
(w,1)
\rightarrow
(x,1)
\rightarrow
(w,2)
\rightarrow
(u,1).
\]

因此，第一次和第二次访问 \(w\) 使用的是两个独立的随机状态。

对于轨迹 \(\pi\)，其正向传播成立需要：

\[
(v_i,h_i)
\]

在每个中间位置选择将 coupon 转发至 \(v_{i+1}\)，且最终状态 \((u,h_\ell)\) 选择消费。

根据模型定义，该具体状态轨迹出现的概率为

\[
\Pr_F(\pi)
=
\left[
\prod_{i=0}^{\ell-1}
p(v_i,v_{i+1})
\right]
p_u^a.
\tag{18}
\]

即使某个物理节点在轨迹中重复出现，由于不同 visit id 对应独立采样，因此各次转发事件仍按照 Eq.~(18) 相乘。

现在考虑从 \(u\) 开始的反向 RR-set 构造。

为了使反向搜索沿与 \(\pi\) 相反的方向依次发现

\[
(v_{\ell-1},h_{\ell-1}),
(v_{\ell-2},h_{\ell-2}),
\ldots,
(v_0,h_0),
\]

首先要求 root state 的动作是消费，该事件概率为

\[
p_u^a.
\]

随后，对于每个 \(i=0,\ldots,\ell-1\)，反向搜索要求状态 \((v_i,h_i)\) 的唯一动作恰好为

\[
\phi(v_i,h_i)=v_{i+1}.
\]

该事件的概率为

\[
p(v_i,v_{i+1}).
\]

由于不同 visit states 的动作彼此独立，而同一个 visit state 的动作只生成一次并在后续检查中复用，因此反向搜索生成该完整状态序列的概率为

\[
\Pr_R(\pi)
=
p_u^a
\prod_{i=0}^{\ell-1}
p(v_i,v_{i+1}).
\tag{19}
\]

比较 Eq.~(18) 和 Eq.~(19)，有

\[
\Pr_F(\pi)=\Pr_R(\pi).
\tag{20}
\]

也就是说，对于任意一条合法的有限 visit-state trajectory，正向传播产生该 trajectory 并最终在 \(u\) 消费的概率，与反向 RR 构造沿反方向揭示相同状态序列的概率完全相同。

又因为在假设

\[
p_v^a+p_v^d>0,\qquad \forall v\in V
\]

下，一张 coupon 以概率 1 最终消费或丢弃，因此所有能够使 \(s\) 最终在 \(u\) 消费的有限 terminal trajectories 构成完整的互斥事件集合。

对所有这类 trajectories 求和，可得

\[
\Pr[A(u,s)]
=
\Pr[(s,1)\in\widetilde R(u)].
\tag{21}
\]

由节点级 RR set 的定义

\[
s\in R(u)
\iff
(s,1)\in\widetilde R(u),
\]

最终得到

\[
\boxed{
\Pr[A(u,s)]
=
\Pr[s\in R(u)].
}
\tag{22}
\]

证毕。

---

## 多优惠券情形的直接推论

对于 \(k\) 张相互独立传播的 coupons，给定固定 root \(u\)，分别在 \(k\) 个独立 coupon worlds 中生成

\[
R_1(u),R_2(u),\ldots,R_k(u).
\]

设第 \(j\) 张 coupon 分配给 seed \(s_j\)，则根据 Theorem 1，

\[
s_j\in R_j(u)
\]

与“coupon \(j\) 最终被 \(u\) 消费”具有相同的概率语义。

因此，用户 \(u\) 被至少一张 coupon 激活的事件可以等价表示为

\[
A(u;S)
\iff
\bigcup_{j=1}^{k}
\{s_j\in R_j(u)\},
\]

从而

\[
\boxed{
\Pr[A(u;S)]
=
\Pr
\left[
\exists j\in[k]:
s_j\in R_j(u)
\right].
}
\tag{23}
\]

Eq.~(23) 是后续 RR sampling 的核心桥梁：它将原本的 coupon 正向传播事件转换为了 seed 与反向可达集之间的覆盖事件。

下一步即可进一步利用

\[
\sigma(S)
=
\sum_{u\in V}\Pr[A(u;S)]
\]

将其转换为**均匀随机选择 root \(U\)** 后的期望形式，并说明为什么一次样本需要围绕同一个 root 生成 \(k\) 个 RR sets。这个正好就是后面的 **5.3 RR Sampling and Coverage Reformulation**。