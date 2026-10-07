$ \text{同一 visit state 只采样一次并复用，} $$ \text{同一物理节点再次访问时产生新的 visit id，并重新独立采样。} $

------

## 5.1 优惠券传播模型下的反向可达集构造

经典影响力最大化中的反向可达集（Reverse Reachable Set, RR set）通过反向搜索，将“从某个种子节点出发能否激活目标节点”的正向传播事件转化为“该种子节点是否属于目标节点的 RR set”的集合包含事件。

对于本文的优惠券传播模型，一张 coupon 在每次访问用户时只能执行消费、丢弃或向一个邻居转发中的一种动作，并且只有最终消费 coupon 的用户才被视为激活。

因此，经典 IC 模型中的 RR-set 生成过程不能直接应用，需要根据 coupon 的单路径传播和重复访问机制重新构造反向可达集。

### Visit-state representation

由于一张 coupon 的传播轨迹可能形成环并重复访问同一用户，仅使用物理节点不足以描述一次完整的随机传播过程。为此，我们进一步引入 **visit state**。

对于节点 \(v\)，记

$ (v,h) $

为 coupon 对节点 \(v\) 的第 \(h\) 个访问状态，其中 \(h\) 为该状态对应的 visit id。同一物理节点的不同访问状态，例如

$ (v,1),\quad (v,2), $

对应相互独立的随机行为。

对于每个 visit state \((v,h)\)，随机采样

$ r_{v,h}\sim {\rm Uniform}[0,1), $

并根据节点 \(v\) 的行为概率确定唯一动作。记该动作为

$ \phi(v,h) \in \{\text{consume},\text{drop}\}\cup N^+(v). $

若

$ r_{v,h}<p_v^a, $

则 \(\phi(v,h)=\text{consume}\)；若

$ p_v^a\le r_{v,h}<p_v^a+p_v^d, $

则 \(\phi(v,h)=\text{drop}\)；否则根据边概率区间唯一确定某个 \(x\in N^+(v)\)，并令

$ \phi(v,h)=x, $

表示 coupon 在该访问状态下沿边 \((v,x)\) 转发。

算法实现中，每个节点 \(v\) 维护一个以 visit id 为键的 Hashtable \(H_v\)，其中

$ H_v[h]=\phi(v,h). $

当算法第一次需要揭示状态 \((v,h)\) 时，独立生成 \(r_{v,h}\) 并将对应动作保存至 \(H_v[h]\)。之后若再次检查同一个状态 \((v,h)\)，直接复用已经存储的动作，而不重新采样。若传播过程中再次访问物理节点 \(v\)，则创建新的 visit id，例如 \((v,h+1)\)，并独立采样新的动作。

因此，本模型遵循如下原则：

$ \boxed{ \text{same node + same visit id} \Longrightarrow \text{reuse the sampled action}, } $

而

$ \boxed{ \text{same node + different visit ids} \Longrightarrow \text{independent actions}. } $

这一状态表示与 Section~3 中“coupon 重访节点时重新采样动作”的传播规则一致。

### State-level reverse reachable set

给定一个目标用户 \(u\)，考虑一张 coupon 的一次反向样本。外层采样过程首先判断作为传播终点的 root state 是否发生消费。如果 root 对应的消费事件不成立，则本次 coupon 不可能通过该 root 产生激活，因此直接返回空 RR set。

若 root 的消费事件成立，则固定该状态为本次 coupon 的最终消费状态，并从 root 开始反向搜索。

为区分重复访问，我们首先在状态空间上定义反向可达集，记为

$ \widetilde R(u). $

\(\widetilde R(u)\) 中保存的是 visit states，而不是单纯的物理节点。

初始化时将 root consumption state 加入集合和搜索队列。随后，每次从队列中取出当前状态 \((v,h_v)\)，检查所有满足

$ (w,v)\in E $

的物理入邻居 \(w\)。

对于每个可能的 predecessor visit state \((w,h_w)\)，算法查询节点 \(w\) 的 Hashtable：

- 如果 \(H_w[h_w]\) 尚未生成，则采样一个新的 \(r_{w,h_w}\)，确定唯一动作 \(\phi(w,h_w)\)，并保存；
- 如果 \(H_w[h_w]\) 已经存在，则直接读取已有动作；
- 只有当$ \phi(w,h_w)=v $时，状态 \((w,h_w)\) 才能够在正向传播中将 coupon 转发至当前状态对应的节点 \(v\)，因此将其加入 \(\widetilde R(u)\)，并继续反向扩展。

如果 \(\phi(w,h_w)\) 为 consume、drop，或者选择了其他出邻居，则此次反向检查失败。

需要注意的是，反向搜索会检查当前状态的所有可能 predecessor，因此反向结构可以出现多个分支。这里的分支并不表示一张 coupon 在正向传播中被复制，而表示**存在多个不同的候选初始位置，它们分别可能形成一条最终到达同一消费节点 \(u\) 的单 coupon trajectory**。

由于 seed \(s\) 初始获得 coupon 时对应的是其第一次访问状态 \((s,1)\)，我们最终定义节点级 RR set 为

$ R(u) = \{s\in V:(s,1)\in\widetilde R(u)\}. \tag{15} $

也就是说，只有当 seed 的初始状态 \((s,1)\) 能够在本次反向随机实现中到达 root consumption state 时，\(s\) 才属于 \(R(u)\)。

------

### Algorithm: Generate one coupon RR set

下面给出对应的抽象伪代码。实际实现中，visit id 的创建与状态对应关系由每个节点的 Hashtable 维护。

```
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

------

## 5.2 正向传播与反向可达的概率等价性

上面的 RR-set 构造只有在其与原始正向 coupon propagation 具有相同的概率语义时，才能用于估计 influence spread。因此，本节证明本文方法最核心的性质：

$ \boxed{ \Pr[\text{coupon seeded at }s\text{ is eventually consumed by }u] = \Pr[s\in R(u)]. } \tag{16} $

### Theorem 1. Forward--reverse probability equivalence

对于任意节点 \(s,u\in V\)，设 \(A(u,s)\) 表示一张初始投放于 \(s\) 的 coupon 最终被用户 \(u\) 消费的事件。按照 Algorithm~\ref{alg:coupon-rr} 生成以 \(u\) 为根的随机 RR set \(R(u)\)，则有

$ \Pr[A(u,s)] = \Pr[s\in R(u)]. \tag{17} $

### Proof

考虑一条从 seed \(s\) 出发并最终在 \(u\) 被消费的具体 visit-state trajectory：

$ \pi= \bigl( (v_0,h_0), (v_1,h_1), \ldots, (v_\ell,h_\ell) \bigr), $

其中

$ v_0=s,\qquad h_0=1,\qquad v_\ell=u. $

这里允许同一个物理节点多次出现，但不同出现位置具有不同的 visit id。

例如，物理轨迹

$ s\rightarrow w\rightarrow x\rightarrow w\rightarrow u $

在状态空间中表示为

$ (s,1) \rightarrow (w,1) \rightarrow (x,1) \rightarrow (w,2) \rightarrow (u,1). $

因此，第一次和第二次访问 \(w\) 使用的是两个独立的随机状态。

对于轨迹 \(\pi\)，其正向传播成立需要：

$ (v_i,h_i) $

在每个中间位置选择将 coupon 转发至 \(v_{i+1}\)，且最终状态 \((u,h_\ell)\) 选择消费。

根据模型定义，该具体状态轨迹出现的概率为

$ \Pr_F(\pi) = \left[ \prod_{i=0}^{\ell-1} p(v_i,v_{i+1}) \right] p_u^a. \tag{18} $

即使某个物理节点在轨迹中重复出现，由于不同 visit id 对应独立采样，因此各次转发事件仍按照 Eq.~(18) 相乘。

现在考虑从 \(u\) 开始的反向 RR-set 构造。

为了使反向搜索沿与 \(\pi\) 相反的方向依次发现

$ (v_{\ell-1},h_{\ell-1}), (v_{\ell-2},h_{\ell-2}), \ldots, (v_0,h_0), $

首先要求 root state 的动作是消费，该事件概率为

$ p_u^a. $

随后，对于每个 \(i=0,\ldots,\ell-1\)，反向搜索要求状态 \((v_i,h_i)\) 的唯一动作恰好为

$ \phi(v_i,h_i)=v_{i+1}. $

该事件的概率为

$ p(v_i,v_{i+1}). $

由于不同 visit states 的动作彼此独立，而同一个 visit state 的动作只生成一次并在后续检查中复用，因此反向搜索生成该完整状态序列的概率为

$ \Pr_R(\pi) = p_u^a \prod_{i=0}^{\ell-1} p(v_i,v_{i+1}). \tag{19} $

比较 Eq.~(18) 和 Eq.~(19)，有

$ \Pr_F(\pi)=\Pr_R(\pi). \tag{20} $

也就是说，对于任意一条合法的有限 visit-state trajectory，正向传播产生该 trajectory 并最终在 \(u\) 消费的概率，与反向 RR 构造沿反方向揭示相同状态序列的概率完全相同。

又因为在假设

$ p_v^a+p_v^d>0,\qquad \forall v\in V $

下，一张 coupon 以概率 1 最终消费或丢弃，因此所有能够使 \(s\) 最终在 \(u\) 消费的有限 terminal trajectories 构成完整的互斥事件集合。

对所有这类 trajectories 求和，可得

$ \Pr[A(u,s)] = \Pr[(s,1)\in\widetilde R(u)]. \tag{21} $

由节点级 RR set 的定义

$ s\in R(u) \iff (s,1)\in\widetilde R(u), $

最终得到

$ \boxed{ \Pr[A(u,s)] = \Pr[s\in R(u)]. } \tag{22} $

证毕。

------

## 多优惠券情形的直接推论

对于 \(k\) 张相互独立传播的 coupons，给定固定 root \(u\)，分别在 \(k\) 个独立 coupon worlds 中生成

$ R_1(u),R_2(u),\ldots,R_k(u). $

设第 \(j\) 张 coupon 分配给 seed \(s_j\)，则根据 Theorem 1，

$ s_j\in R_j(u) $

与“coupon \(j\) 最终被 \(u\) 消费”具有相同的概率语义。

因此，用户 \(u\) 被至少一张 coupon 激活的事件可以等价表示为

$ A(u;S) \iff \bigcup_{j=1}^{k} \{s_j\in R_j(u)\}, $

从而

$ \boxed{ \Pr[A(u;S)] = \Pr \left[ \exists j\in[k]: s_j\in R_j(u) \right]. } \tag{23} $

Eq.~(23) 是后续 RR sampling 的核心桥梁：它将原本的 coupon 正向传播事件转换为了 seed 与反向可达集之间的覆盖事件。

下一步即可进一步利用

$ \sigma(S) = \sum_{u\in V}\Pr[A(u;S)] $

将其转换为**均匀随机选择 root \(U\)** 后的期望形式，并说明为什么一次样本需要围绕同一个 root 生成 \(k\) 个 RR sets。这个正好就是后面的 **5.3 RR Sampling and Coverage Reformulation**。





------

## 5.3 基于反向可达集的 SSR 构造与传播度估计

由上一节可知，对于任意目标节点 \(u\)、候选种子节点 \(s\)，从 \(s\) 正向传播的 coupon 最终被 \(u\) 消费的概率，等于 \(s\) 被包含在以 \(u\) 为根生成的 coupon RR set 中的概率，即

$ \Pr[A(u,s)] = \Pr[s\in R(u)]. \tag{23} $

然而，CIM 中同时存在 \(k\) 张相互独立的 coupons。节点 \(u\) 最终被激活，当且仅当至少有一张 coupon 最终被 \(u\) 消费。因此，一次用于估计 spread 的反向样本不能只包含一个 RR set，而需要同时描述 \(k\) 张 coupon 对同一目标用户 \(u\) 的独立传播结果。

为此，我们将围绕同一个 root 生成的 \(k\) 个 coupon RR sets 组织为一组 SSR。

设第 \(t\) 次采样首先从节点集合 \(V\) 中均匀随机选择一个 root

$ U_t\sim \operatorname{Uniform}(V). $

由于本文的 spread 对每个用户赋予相同权重，因此采用均匀 root sampling。若后续考虑带权用户价值，则可以相应地改变 root sampling distribution。

固定采样得到的 root \(U_t=u\)。对于每张 coupon \(j\in[k]\)，独立生成一个对应的随机传播世界，并按照 Section 5.1 的方法构造以 \(u\) 为终点的反向可达集

$ R_j^{(t)}(u). $

具体地，首先独立判断 root \(u\) 在第 \(j\) 个 coupon world 中是否消费该 coupon。若消费事件未发生，则令

$ R_j^{(t)}(u)=\emptyset. $

若消费事件发生，则固定 \(u\) 为该 coupon 的消费终点，并调用前述 RR-set generation procedure，在 visit-state 空间中反向搜索得到

$ R_j^{(t)}(u). $

不同 coupon 对应的 possible worlds 相互独立，因此：

$ R_1^{(t)}(u), R_2^{(t)}(u), \ldots, R_k^{(t)}(u) $

也是在相同 root \(u\) 下独立生成的 \(k\) 个反向集合。

我们将第 \(t\) 次采样得到的整体对象记为

$ \mathcal R^{(t)} = \left( R_1^{(t)}(U_t), R_2^{(t)}(U_t), \ldots, R_k^{(t)}(U_t) \right), \tag{24} $

并称其为一组 SSR。

这里必须强调，一组 SSR 对应的是**一个随机目标用户和 \(k\) 张 coupons**，而不是 \(k\) 个彼此无关的普通 RR samples。后续 spread estimation 和 seed selection 都以整个 SSR group 为基本单位。

------

### SSR 与激活事件的关系

设 \(k\) 张 coupons 的投放位置分别为

$ \mathbf S = (s_1,s_2,\ldots,s_k), $

其中 \(s_j\) 表示第 \(j\) 张 coupon 的 seed。

对于固定 root \(u\)，由正向—反向概率等价性，第 \(j\) 张 coupon 最终被 \(u\) 消费的事件，与

$ s_j\in R_j(u) $

具有相同的概率。

因此，节点 \(u\) 至少消费一张 coupon，即被激活，当且仅当至少存在一个 \(j\in[k]\) 满足

$ s_j\in R_j(u). $

于是有

$ \Pr[A(u;\mathbf S)] = \Pr \left[ \bigcup_{j=1}^{k} \{s_j\in R_j(u)\} \right]. \tag{25} $

对第 \(t\) 个 SSR group 定义覆盖指示变量

$ X_t(\mathbf S) = \mathbb I \left[ \exists j\in[k]: s_j\in R_j^{(t)}(U_t) \right]. \tag{26} $

如果 \(X_t(\mathbf S)=1\)，表示在该 SSR sample 中，至少一张 coupon 对应的 seed 命中了自己的反向可达集，因此 root \(U_t\) 被该投放方案覆盖。

注意，即使有多张 coupon 同时满足

$ s_j\in R_j^{(t)}(U_t), $

该 SSR group 仍然只贡献一次。这与 CIM 的定义完全一致：同一个用户即使消费多张 coupons，也只被计为一个 activated user。

------

### 由期望得到 spread estimator

根据 CIM 的定义，

$ \sigma(\mathbf S) = \sum_{u\in V} \Pr[A(u;\mathbf S)]. \tag{27} $

因为 \(U_t\) 在 \(V\) 上均匀采样，

$ \Pr[U_t=u] = \frac{1}{n}. $

于是

$ \begin{aligned} \mathbb E[X_t(\mathbf S)] &= \sum_{u\in V} \Pr[U_t=u]\, \Pr[A(u;\mathbf S)] \\ &= \frac{1}{n} \sum_{u\in V} \Pr[A(u;\mathbf S)] \\ &= \frac{\sigma(\mathbf S)}{n}. \end{aligned} \tag{28} $

因此，

$ \boxed{ \sigma(\mathbf S) = n\,\mathbb E[X_t(\mathbf S)]. } \tag{29} $

Eq. (29) 就是整个 SSR sampling 方法的理论基础。

它说明，我们不需要对每个节点分别执行大量正向 Monte Carlo simulations，而只需要重复执行：

$ \text{uniformly sample a root} \rightarrow \text{generate one SSR group} \rightarrow \text{check whether the group is covered}. $

给定 \(\theta\) 个独立生成的 SSR groups

$ \mathcal R^{(1)},\ldots,\mathcal R^{(\theta)}, $

定义传播度估计量

$ \widehat{\sigma}_{\theta}(\mathbf S) = \frac{n}{\theta} \sum_{t=1}^{\theta} X_t(\mathbf S). \tag{30} $

由 Eq. (28) 立即得到

$ \mathbb E[ \widehat{\sigma}_{\theta}(\mathbf S) ] = \sigma(\mathbf S). \tag{31} $

因此 \(\widehat{\sigma}_{\theta}\) 是 coupon influence spread 的无偏估计。

当采样数量 \(\theta\) 增大时，根据大数定律，

$ \frac1\theta \sum_{t=1}^{\theta}X_t(\mathbf S) \longrightarrow \mathbb E[X_t(\mathbf S)] $

从而

$ \widehat{\sigma}_{\theta}(\mathbf S) \longrightarrow \sigma(\mathbf S). $

后续可以进一步利用集中不等式确定达到给定误差概率所需的 SSR 数量。

------

### SSR generation 的伪代码

这部分最好也给算法，因为老师明确说“算法要有 input/output/return”。

```
\begin{algorithm}[t]
\caption{\textsc{GenerateSSR}}
\label{alg:ssr-generation}
\begin{algorithmic}[1]
\Require Directed graph $G=(V,E)$; coupon budget $k$;
         behavior and transfer probabilities
\Ensure One SSR group
        $\mathcal R=(R_1,\ldots,R_k)$

\State Uniformly sample a root $u$ from $V$

\For{$j=1,\ldots,k$}
    \State Independently sample the root consumption event
           for coupon $j$
    \If{root $u$ does not consume coupon $j$}
        \State $R_j\gets\emptyset$
    \Else
        \State $R_j\gets\Call{GenerateCouponRR}{u}$
    \EndIf
\EndFor

\State \Return $\mathcal R=(R_1,\ldots,R_k)$
\end{algorithmic}
\end{algorithm}
```

这里需要注意一个实现细节：

$ R_1,\ldots,R_k $

必须使用各自独立的 visit-state Hashtable / possible world。

不能让第 1 张 coupon 生成过的

$ H_v[h] $

直接被第 2 张 coupon 复用，因为模型假设不同 coupons 独立传播。

------

# 5.4 基于 SSR 组覆盖的种子选择

有了 \(\theta\) 个 SSR groups 后，CIM 的 seed selection 可以转化为一个带 coupon 位置约束的覆盖优化问题。

第 \(t\) 个 SSR sample 为

$ \mathcal R^{(t)} = \left( R_1^{(t)}, R_2^{(t)}, \ldots, R_k^{(t)} \right). $

对于 seed allocation

$ \mathbf S=(s_1,\ldots,s_k), $

只要存在某个 \(j\) 满足

$ s_j\in R_j^{(t)}, $

第 \(t\) 个 SSR group 就被覆盖。

因此 empirical objective 等价于最大化：

$ F_\theta(\mathbf S) = \sum_{t=1}^{\theta} \mathbb I \left[ \exists j\in[k]: s_j\in R_j^{(t)} \right]. \tag{32} $

并且

$ \widehat{\sigma}_\theta(\mathbf S) = \frac{n}{\theta}F_\theta(\mathbf S). \tag{33} $

------

## 倒排索引

直接反复扫描所有 SSR groups 的代价较高。因此，实验实现针对每个 coupon index \(j\) 和每个节点 \(v\) 建立倒排索引：

$ \operatorname{Cov}[j][v] = \left\{ t: v\in R_j^{(t)} \right\}. \tag{34} $

例如：

$ \operatorname{Cov}[2][v] = \{2,5,8\} $

表示节点 \(v\) 出现在第 2、5、8 个 SSR groups 的第二个 RR set 中。

这样在计算节点 \(v\) 作为第 \(j\) 张 coupon seed 的收益时，无需扫描全部 \(\theta\) 个 samples，只需要访问

$ \operatorname{Cov}[j][v]. $

------

## SSR group 的覆盖状态

为每一个 SSR group \(t\) 维护布尔变量

$ C_t\in\{0,1\}, $

初始时

$ C_t=0. $

若此前已经有某一张 coupon 的 seed 命中该组，即存在 \(l<j\) 满足

$ s_l\in R_l^{(t)}, $

则设

$ C_t=1. $

由于一个 root 用户无论消费多少张 coupon 都只贡献一次 activation，因此一旦某个 SSR group 已经被覆盖，之后其他 coupons 再次命中该组不会产生额外收益。

这就是实验代码中 `covered[t]` 的含义。

------

## 第 \(j\) 轮的边际覆盖收益

按照实验实现，种子按照 coupon index

$ j=1,2,\ldots,k $

依次选择。

假设前 \(j-1\) 张 coupons 已经选择了

$ s_1,\ldots,s_{j-1}. $

对于一个尚未被选择的候选节点 \(v\)，定义其在第 \(j\) 张 coupon 位置上的新增覆盖收益为

$ g_j(v) = \left| \left\{ t: v\in R_j^{(t)} \ \land\ C_t=0 \right\} \right|. \tag{35} $

等价地，

$ g_j(v) = \left| \operatorname{Cov}[j][v] \setminus \mathcal C \right|, $

其中

$ \mathcal C=\{t:C_t=1\} $

为当前已经覆盖的 SSR groups。

第 \(j\) 轮选择

$ s_j \in \arg\max_{v\notin\{s_1,\ldots,s_{j-1}\}} g_j(v). \tag{36} $

选择 \(s_j\) 后，对所有

$ t\in\operatorname{Cov}[j][s_j] $

更新

$ C_t\gets1. $

然后进入下一张 coupon 的 seed selection。

------

你给的那个两张券、三组 SSR 的例子非常值得放正文，因为一下就能解释为什么算法比较的是“新增覆盖”，而不是出现次数。

假设：

| SSR group | \(R_1^{(t)}\) | \(R_2^{(t)}\) |
| --------- | ------------- | ------------- |
| 1         | \(\{a\}\)     | \(\{c\}\)     |
| 2         | \(\{a\}\)     | \(\{c\}\)     |
| 3         | \(\{b\}\)     | \(\{d\}\)     |

第一轮：

$ g_1(a)=2,\qquad g_1(b)=1, $

因此选择

$ s_1=a. $

第 1、2 组被覆盖。

第二轮虽然节点 \(c\) 出现在两个 \(R_2\) 中，但是它们对应的第 1、2 组已经被前一张 coupon 覆盖，因此：

$ g_2(c)=0. $

而

$ d\in R_2^{(3)} $

且第 3 组尚未覆盖，因此：

$ g_2(d)=1. $

于是第二轮选择

$ s_2=d. $

这个例子非常直观地体现出：

$ \boxed{ \text{算法优化的是新增 SSR-group coverage， 而不是节点在 RR sets 中的总出现次数。} } $

------

对应的选种伪代码可以写成：

```
\begin{algorithm}[t]
\caption{\textsc{SelectSeedsBySSR}}
\label{alg:ssr-greedy}
\begin{algorithmic}[1]
\Require $\theta$ SSR groups
         $\{\mathcal R^{(t)}\}_{t=1}^{\theta}$;
         coupon budget $k$
\Ensure Seed allocation $(s_1,\ldots,s_k)$

\State Build the inverted index
       $\operatorname{Cov}[j][v]
       =\{t:v\in R_j^{(t)}\}$
\State $C_t\gets\mathrm{false}$ for all $t$
\State $S\gets\emptyset$

\For{$j=1,\ldots,k$}
    \For{each $v\in V\setminus S$}
        \State $g_j(v)\gets
        |\{t\in\operatorname{Cov}[j][v]:C_t=\mathrm{false}\}|$
    \EndFor

    \State $s_j\gets
    \arg\max_{v\in V\setminus S}g_j(v)$
    \State $S\gets S\cup\{s_j\}$

    \For{each $t\in\operatorname{Cov}[j][s_j]$}
        \State $C_t\gets\mathrm{true}$
    \EndFor
\EndFor

\State \Return $(s_1,\ldots,s_k)$
\end{algorithmic}
\end{algorithm}
```

------

还有一个理论问题我建议我们现在就标出来，不要等老师再指出。

**这套代码的 greedy 和我们上一版 Section 5 里那个标准 maximum-coverage greedy 不是同一个算法。**

上一版是从所有剩余节点里，每轮找对整个 set function 最大的 marginal gain，所以能直接套：

$ 1-\frac1e. $

而现在真实代码是：

$ j=1,\ldots,k $

固定 coupon position，第 \(j\) 轮只优化第 \(j\) 个 RR-set column：

$ R_j^{(1)},\ldots,R_j^{(\theta)}. $

也就是说，决策实际上是：

$ (s_1,\ldots,s_k) $

这种 **coupon-indexed assignment**，而不是普通无序集合 \(S\)。

所以这一版方法和代码终于对上之后，**原先那个 \(1-1/e-\varepsilon\) theorem 不能原封不动搬回来**。下一步要单独分析：

$ F_\theta(s_1,\ldots,s_k) = \sum_t \mathbf1[ \exists j:s_j\in R_j^{(t)} ] $

在这种“每个 coupon position 选一个 seed + seed 不重复”的约束下，顺序 greedy 到底能保证什么 approximation ratio。

这个问题应该放到后面的理论保证里重新证明，而不是先假定还是 \(1-1/e\)。

但从**老师现在要求的方法逻辑**来看，这版结构已经对了：

$ \boxed{ \text{单个 RR 怎么生成} \rightarrow \text{正反向概率一致} \rightarrow \text{同一 root 下生成 }k\text{ 个 RR} \rightarrow \text{形成一个 SSR group} \rightarrow \text{期望 = spread}/n \rightarrow \text{SSR group coverage} \rightarrow \text{按券编号逐轮选种} } $

这条线比我们之前那版真正贴合你们的算法和实验代码。

















