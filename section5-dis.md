# 5.1 优惠券反向可达集的构造

为了从目标用户的角度描述优惠券传播，我们引入优惠券反向可达集。正向传播从投放节点出发，追踪优惠券最终被哪个用户使用；

反向构造则固定一个目标用户，沿入边寻找能够使该用户使用优惠券的投放节点。

本小节先定义单券反向可达集，再说明反向检查的概率区间、随机数抽取与扩展操作，最后将多张券的反向集合组织为一个 SSR 。



**单券反向可达集生成过程**

固定目标用户 $u$ 和第 $j$ 张券的可能世界 $W_j$。由第三节的消费事件 $A_j(u,s;W_j)$，定义
$$
R_j(u;W_j)=\{s\in V:A_j(u,s;W_j)\text{ 发生}\}.
$$

在该可能世界中，从$R_j$内任一节点投放第 $j$ 张券，最终都会由 $u$ 使用。

要求券最终在目标节点处使用，仅仅经过目标用户不构成成功结果。

一个$R_j$可以包含多个候选投放节点，每个节点分别对应一种投放选择，并不表示一张券可以复制。

下文采用每次检查独立重抽随机数的简化方案，其与上述目标集合定义的关系见文末备注。



**反向图与概率区间**

令 $G^{\mathsf T}$ 为将 $G$ 中各条有向边反向后得到的图。

原图中的边 $(w,v)$ 在反向图中表示为 $(v,w)$，其权值仍为 $p(w,v)$。

因此，在反向图中从 $v$ 检查邻居 $w$ 时，实际判断的是：券在正向到达 $w$ 后，是否会转发给 $v$。

为便于判断，为每个节点预先建立行为概率区间。设 $w$ 的出邻居按固定次序排列为 $v_1,\ldots,v_d$，将区间 $[0,1)$ 依次分配给使用、废弃以及各个转发行为。对于出边 $(w,v_i)$，记录其区间的左右端点

$$
s_{w,v_i}=p_w^a+p_w^d+\sum_{h<i}p(w,v_h),
\qquad
e_{w,v_i}=s_{w,v_i}+p(w,v_i).
$$

其中 $s_{w,v_i}$ 是区间起点，与种子节点编号 $s_j$ 区分。边区间长度满足 $e_{w,v_i}-s_{w,v_i}=p(w,v_i)$。每次反向检查均独立生成一个新的随机数 $r\sim\mathrm{Uniform}[0,1)$。各区间在正向模型中的行为含义如下。

| 随机数所在区间 | 节点 $w$ 的行为 |
|---|---|
| $[0,p_w^a)$ | 使用该券，传播终止 |
| $[p_w^a,p_w^a+p_w^d)$ | 废弃该券，传播终止 |
| $[s_{w,v_i},e_{w,v_i})$ | 将该券转发给唯一邻居 $v_i$ |

由于各区间互不重叠，一次抽出的随机数至多对应一条转发边；不同检查分别抽取随机数，其结果可以对应不同的边。所有转发区间的长度之和为 $p_w^t$，与使用、废弃区间一起覆盖 $[0,1)$；没有出邻居的节点仅包含使用和废弃区间。

例如，设 $p_w^a=0.2$、$p_w^d=0.1$，且 $w$ 向 $v_1$、$v_2$ 转发的概率分别为 $0.4$、$0.3$，则对应区间为

$$
\underbrace{[0,0.2)}_{\text{使用}}\quad
\underbrace{[0.2,0.3)}_{\text{废弃}}\quad
\underbrace{[0.3,0.7)}_{w\to v_1}\quad
\underbrace{[0.7,1)}_{w\to v_2}.
$$

若检查边 $(w,v_1)$ 时抽得 $r=0.56$，则本次检查成功。之后再次检查该边或检查 $(w,v_2)$ 时，都重新生成随机数，不沿用 $0.56$。



**每次检查独立抽样**

每次反向检查边 $(w,v)$ 时，均独立生成新的随机数 $r\sim\mathrm{Uniform}[0,1)$，并判断是否满足 $s_{w,v}\le r<e_{w,v}$。满足时，该次检查支持沿边继续反向扩展；否则，该次检查失败，继续处理其他待检查边。

首次遇到节点、不同分支再次遇到同一节点，以及再次检查同一条边，都重新抽取随机数。不保存随机数供后续判断复用，也不查询之前的抽样结果。各节点的行为概率和边区间保持不变，重新生成的是随机数 $r$。



**SSR 的组织方式** 对同一个目标用户 $u$，分别为 $k$ 张券生成反向集合，并按券编号组成一个样本组，称为 SSR：
$$
\operatorname{SSR}(u)=\bigl(R_1(u),R_2(u),\ldots,R_k(u)\bigr).
$$

各集合共享目标用户，给定该目标后，其随机世界 $W_1,\ldots,W_k$ 相互独立。满足上述集合定义的单券生成过程确定后，SSR 按以下步骤组织。

1. 从节点集合 $V$ 中均匀抽取一个目标用户 $u$。
2. 对 $j=1,\ldots,k$，使用独立的随机性生成 $R_j(u)$，分别保存其券编号。
3. 将目标用户 $u$ 与这 $k$ 个集合一起保存为一个 SSR。空集保留原位置，整组为空的样本也保留。
4. 独立重复上述过程，得到xxx采样规模个 SSR 样本。一组包含 $k$ 个单券集合。

给定固定编号的投放方案 $S=\{s_1,\ldots,s_k\}$，第 $j$ 张券从 $s_j$ 投放。该方案命中一个 SSR 的条件为

$$
\exists j\in\{1,\ldots,k\}:\ s_j\in R_j(u).
$$

即只比较种子与对应券编号的集合。一组中即使有多张券命中，也只记录一次命中，对应目标用户只计一次激活。











---

## 构造衔接备注

上文给出了目标集合定义、概率区间、每次检查独立抽样的简化规则和 SSR 的组织方式。以下问题仍需处理，暂不将简化规则视为满足上述定义的完整单券生成算法。

- **简化抽样与目标集合的区别。** 原模型中，同一次持券行为只能选择一个后继；每次边检查独立重抽后，不同分支对该行为的检查可能分别选择不同后继。这会改变分支间的依赖关系，在一般情况下不保持原模型的反向成员概率。因此，简化过程的输出不能直接认定为上述 $R_j(u;W_j)$ 的正确样本，也不能据此宣称与第三节概率等价。
- **目标使用事件如何初始化。** 参考文档中的部分流程先判断根节点是否立即使用，失败便不展开该张券的反向集合。当前第三节还允许到达目标后先转发、返回后才使用，因此不能直接把一次使用判断作为整个样本的开关。需要明确这些使用事件如何组织为初始反向记录。
- **如何完成探索。** 每次检查重新抽样，只确定了局部随机数的使用方式；仍需确定哪些检查加入待处理队列、如何识别初始投放位置，以及何时完成本次集合的生成。队列为空能否作为返回条件，取决于初始化和扩展规则。

参考文档将样本组称为 SRR（super RR set）；本稿沿用当前讨论中的 SSR 记号。这部分备注用于继续讨论，不作为上述正文的算法结论。



---

## 5.1 优惠券传播模型下的反向可达集

传统IM中的反向可达集（Reverse Reachable Set, RR set）建立在正向传播和反向可达之间的概率等价关系上。具体而言，对于一个给定目标节点u，若能够构造一个随机集合 \(R(u)\)，使得任意节点 \(s\) 属于 \(R(u)\) 的概率恰好等于从 \(s\) 出发的传播最终激活u 的概率，则对种子集的影响力估计可以转化为对随机 RR sets 的覆盖统计。







我把这段录音和你们前面的版本对起来看了。老师这次的意思其实非常明确：**我们现在第五节的整体路线写偏了。**之前那版“从所有候选 seed 正向模拟 trajectory，再按 terminal consumer 建 inverted index”的东西虽然数学上能自洽，但它不是老师要你写的核心算法。老师要的是你们原来真正讨论和实现的那套：

$ \boxed{\text{先定义并构造 coupon-specific RR set}} $$ \boxed{\text{证明正向激活概率 = seed 与 RR set 相交的概率}} $$ \boxed{\text{再由 }\sigma(S)=\sum_u\Pr[u\text{ activated}]\text{ 推出随机根节点采样}} $$ \boxed{\text{多 coupon 时，一次 sample 是以同一个 root 为中心的 }k\text{ 个 RR sets}} $$ \boxed{\text{最后才进入覆盖和 greedy}} $

这和你之前开题材料里写的原始方案是一致的：先均匀选根节点，再针对 \(k\) 个 coupon world 构造 \(k\) 个 RR sets；根节点的消费事件决定对应 RR set 是否有效。figure5_method_overview

而且老师特别强调了：**“反向可达集怎么生成”本身就是你们工作的重要创新点，不能只写一个 \(R(u)\) 的定义，然后跳到概率等价。**

所以我建议你现在先不要继续修英文第五节。先按老师要求把中文方法文档写出来，而且先只把前两块写清楚。

------



我会重构成下面这样。

### 5.1 Coupon Reverse Reachable Set

这一节只解决两个问题：

1. **RR set 到底是什么？**
2. **RR set 到底怎么生成？**

也就是说要从正向传播规则出发，明确给一个输入 rootu，如何反向得到一个随机集合 \(R_u\)。

然后证明最核心的式子：

$ \boxed{ \Pr[s\in R_u] = \Pr[\text{coupon seeded at }s\text{ is eventually consumed by }u] } $

多 coupon 后进一步得到：

$ \boxed{ \Pr[\exists j:\ s_j\in R_{u,j}] = \Pr[u\text{ is activated by }S]. } $

这才是后面所有 RIS/RR 方法能用的理论基础。

------

### 5.2 RR Sampling and Coverage Reformulation

有了上面的 probability equivalence，才开始处理目标函数：

$ \sigma(S) = \sum_{u\in V} \Pr[u\text{ activated}]. $

因为每个用户在 spread 中权重都是 1，所以可以把上式写成：

$ \sigma(S) = n\cdot \mathbb E_{u\sim \mathrm{Unif}(V)} [ \Pr[u\text{ activated}] ]. $

再用 5.1 的 RR equivalence 替换：

$ \sigma(S) = n\cdot \Pr[ \text{seed assignment covers a random RR sample} ]. $

这一步才解释了：

> 为什么我们可以随机采一个 root，然后生成 RR set，而不是对所有节点都计算激活概率。

老师录音里一直说的“根据期望”“均匀采样”“大数定理”其实就是这一段。

然后再从期望走到 empirical estimation：

$ \hat\sigma(S) = \frac{n}{\theta} \sum_{i=1}^{\theta}X_i(S), $

由大数定律/集中不等式说明，当 \(\theta\) 足够大时，

$ \hat\sigma(S)\approx\sigma(S). $

------













## 5.1 优惠券传播模型下的反向可达集

经典影响力最大化算法中的反向可达集（Reverse Reachable Set, RR set）建立在正向传播和反向可达之间的概率等价关系上。具体而言，对于一个给定目标节点 u，若能够构造一个随机集合 R(u)，使得任意节点 s 属于 R(u) 的概率恰好等于从 s 出发的传播最终激活 u的概率，则对种子集的影响力估计可以转化为对随机 RR sets 的覆盖统计。

优惠券传播与经典 IC 模型存在明显区别。一张 coupon 在任意时刻只能沿一条边继续传播，并且仅有最终消费 coupon 的用户才被激活。因此，我们需要针对该传播机制重新定义反向可达过程。

给定目标节点 u，我们称以 u 为根的一次随机反向传播所得到的节点集合记作$ R(u). $

其含义是：在本次随机实现下，对于任意节点 $s\in R(u)$，若将一张 coupon 初始分配给 s，则该 coupon 按照对应的正向传播随机实现能够最终到达u，并由u 消费。

因此，反向 RR-set 的构造不能仅仅判断网络拓扑上的可达性，还必须同时满足两类随机事件：

1. 中间节点只负责转发，因此其对应随机状态必须允许 coupon 沿当前反向搜索方向继续传播；
2. 根节点u 则必须发生消费事件。

### RR-set 的生成

对一个指定的根节点u，一次 RR-set 生成过程包括两个部分。

首先判断节点u 在本次可能世界中是否发生 consumption。若未发生，则即使券能够传播至u，也不能激活u，因此该 realization 对u 的激活贡献为 0。

若u 发生 consumption，则从u 开始执行反向搜索。对于当前节点 v，依次考虑其入邻居 w。根据正向模型中的转发规则，随机判断一张位于w的 coupon 是否会选择边$ (w,v) $

进行转发。若该 transfer event 成立，则将 \(w\) 加入当前 RR set，并继续从 \(w\) 向其入邻居扩展；否则跳过 \(w\)。

这一反向过程持续进行，直至不存在新的可扩展节点，最终得到以u 为根的一次随机 RR set。







------

这里我要特意停一下：

**上面这段只是老师录音已经明确要求的“逻辑骨架”。**

你们新版模型现在规定“revisit 后重新采样”，所以**具体到“反向搜索访问 \(w\) 时到底如何采随机数、重复节点如何处理、状态如何保存”这一段，我现在不应该替你编。**

老师也明确说了，这部分是你们自己讨论很久才确定的、程序已经实现的核心创新。

我们必须把**你们现有代码/老师手写的 reverse generation rule**拿出来，逐行翻成论文描述。

你之前开题材料里的旧版本确实给过一种 BFS 形式：从 root 出发检查入邻居 \(w\)，随机判断 \((w,v)\) 是否为对应 possible world 中的有效转移，然后把 \(w\) 加入 RR-set。figure5_method_overview 但是那个材料对应的是你们较早的 possible-world 版本，**不能直接假设它就是老师现在 revisit-resampling 模型的最终算法**。

这一点这次不能再猜。

---

### 引理：正向传播概率和反向激活概率一致

对于任意根节点u 和候选 seed \(s\)，有

$ \boxed{ \Pr[s\in R(u)] = \Pr[A(u,s)] } $

其中 A(u,s) 表示从 s 发出的一张 coupon 最终由u 消费的事件。

应该按传播序列来。

假设正向存在一条具体 forwarding trajectory：

$ \pi: s=v_0\rightarrow v_1\rightarrow \cdots\rightarrow v_\ell=u. $

该 trajectory 成立且最终在u consumption 的概率是：

$ P_F(\pi) = \left[ \prod_{i=0}^{\ell-1} p(v_i,v_{i+1}) \right] p_u^a. $

然后证明按照你们定义的反向 RR generation procedure，从u 出发依次得到：

$ u=v_\ell \leftarrow v_{\ell-1} \leftarrow\cdots\leftarrow v_0=s $

这一反向 realization 的概率同样是：

$ P_R(\pi) = p_u^a \prod_{i=0}^{\ell-1} p(v_i,v_{i+1}). $

因此：

$ P_F(\pi)=P_R(\pi). $

再对所有能够从 \(s\) 到u 且最终在u consumption 的 admissible trajectories 求和，就得到：

$ \Pr[A(u,s)] = \Pr[s\in R(u)]. $

---

但是同样，因为现在允许 revisit，具体求和对象到底是 path 还是 walk，以及反向过程中重复访问怎么对应，必须跟你们真正的 RR generator 对上。

------

## 5.2：从 RR 等价走到采样

这一部分我认为可以先完整写出来，因为逻辑已经很明确。

设$ S=(s_1,\ldots,s_k) $ 为 k张 coupon 对应的 seed allocation。

对于一个固定节点u，独立为 k 张 coupon 生成：

$ R_{u,1},R_{u,2},\ldots,R_{u,k}. $

第 j 张 coupon 能够使u activated，当且仅当：

$ s_j\in R_{u,j}. $

因此：

$ A(u;S) \iff \bigcup_{j=1}^{k} \{s_j\in R_{u,j}\}. $

由上一节的正反向概率等价：

$ \Pr[A(u;S)] = \Pr \left[ \exists j\in[k]: s_j\in R_{u,j} \right]. \tag{1} $

这一式就是**从传播模型进入 RR sampling 的桥梁**。

------

接下来考虑 spread：

$ \sigma(S) = \sum_{u\in V} \Pr[A(u;S)]. $

令随机变量u 在 V 上均匀采样：

$ \Pr[U=u]=\frac1n. $

于是：

$ \begin{aligned} \sigma(S) &= n\sum_{u\in V} \frac1n \Pr[A(u;S)] \\ &= n\, \mathbb E_U [ \Pr[A(U;S)] ]. \end{aligned} $

代入式 (1)：

$ \boxed{ \sigma(S) = n\, \Pr \left[ \exists j\in[k]: s_j\in R_{U,j} \right]. } \tag{2} $

这就是老师录音里说的：

> “随机选择一个目标节点……种子集合能够激活u 的概率和反向可达集交集概率一致。”

也就是说，一次 sampling 的正确方式应该是：

$ \boxed{ \text{uniformly sample one root }U } $

然后围绕同一个 root 生成：

$ \boxed{ (R_{U,1},\ldots,R_{U,k}) } $

作为**一组 sample**。

你们开题材料里原来其实就是这么设计的：一个 root 对应 \(k\) 个 possible worlds / RR sets。figure5_method_overview

------

## k 个 RR-set 为一组SSR

这是老师录音里很重要的一句话，我们之前英文第五节基本丢掉了。

一次 sample 不是一个普通 RR set：

$ R. $

而应该是：

$ \boxed{ \mathcal R(U) = (R_{U,1},R_{U,2},\ldots,R_{U,k}). } $

因为你们不是传播一个可复制的信息，而是有 k 张**相互独立的 coupons**。

对于这个 rootu，只要存在一个 coupon：

$ s_j\in R_{U,j}, $

那么u 就已经被激活。

后面再有其他 coupon：

$ s_l\in R_{U,l}, $

不会产生第二个 activated user。

因此一个 sample 的 coverage indicator 应该是：

$ X(S,\mathcal R) = \mathbb I \left[ \bigvee_{j=1}^k (s_j\in R_j) \right]. $

这个就已经和经典 IM 的

$ \mathbb I[S\cap R\neq\emptyset] $

不一样了。

**这里正是老师说“我们这个覆盖和 IM 的覆盖还有点不一样”的地方。**

然后采样 \(\theta\) 组：

$ \mathcal R^{(1)},\ldots,\mathcal R^{(\theta)}, $

定义：

$ \hat\sigma(S) = \frac{n}{\theta} \sum_{i=1}^{\theta} X(S,\mathcal R^{(i)}). $

由：

$ \mathbb E[X] = \frac{\sigma(S)}n, $

有：

$ \mathbb E[\hat\sigma(S)] = \sigma(S). $

当 \(\theta\) 足够大时，由大数定律：

$ \hat\sigma(S)\rightarrow\sigma(S). $

后面如果要严格给 approximation guarantee，再上 Chernoff/Bernstein 等 concentration bound。





------

# 六、所以老师这次真正否掉的是哪部分

我们应该把这个认识统一。

他**不是**说：

> NP-hard/submodular/greedy 的思路错了。

他说的是：

> **从传播模型到 RR-set，再从 RR-set 到 sampling/coverage 的逻辑链条没有写出来。**

我们之前的稿子直接跳成了：

$ \text{simulate trajectories} \rightarrow \text{group terminal endpoints} \rightarrow \text{coverage}. $

而老师要的是：

$ \boxed{ \text{forward coupon diffusion} } $$ \Downarrow $$ \boxed{ \text{construct coupon RR set} } $$ \Downarrow $$ \boxed{ \Pr[\text{forward activation}] = \Pr[\text{RR intersection}] } $$ \Downarrow $$ \boxed{ \sigma(S) = n\Pr[\text{random RR group covered}] } $$ \Downarrow $$ \boxed{ \text{RR sampling} } $$ \Downarrow $$ \boxed{ \text{special coverage/greedy for }k\text{ coupons}. } $

这个就是我们现在必须重写的中文方法主线。

------

下一步最重要的不是让我继续生成英文，而是**把你们程序里“单个 RR set 到底怎样从 root 反向生成”的算法拿出来**。你把那段代码、伪代码、老师以前画的草稿，三者任意一个给我。我会严格按它写出：

> **5.1 反向可达集定义 → Algorithm 1 输入/输出 → 逐步生成过程 → 正反向概率等价 Lemma + proof**

然后再接我上面已经整理好的 **5.2 期望—随机根—\(k\) 个 RR-set 一组—覆盖估计**。这两节整理完，才真正符合老师今天要求的“两页中文方案”。







---







这个就非常关键了。现在“实验代码里的 RR-set 到底怎么生成”已经清楚了，我们终于可以不猜算法，而是**按真实实现来写第五节**。

不过我先把结论说在前面：**这套代码的算法流程很好写，也很直观；但在写老师要求的“正向传播概率 = 反向可达概率”证明之前，有一个随机性细节必须核对。** 我下面先严格按你给的代码把中文方法写出来，再说明这个问题。

## 5.1 优惠券传播模型下的反向可达集构造

经典影响力最大化中的反向可达集（Reverse Reachable Set, RR set）通过反向搜索，将“一个种子节点是否能够激活目标节点”的正向传播事件转化为“种子节点是否属于目标节点的 RR set”的集合包含事件。对于优惠券传播模型，由于一张优惠券只能被逐节点转发，并且仅当优惠券最终被用户消费时该用户才被激活，因此需要根据 coupon 的消费—丢弃—转发机制重新构造 RR set。

给定目标节点 \(u\)，我们首先确定 \(u\) 对当前 coupon 的消费事件。该判断由外层采样过程完成。如果本次随机实现中 \(u\) 不消费该 coupon，则即使优惠券能够到达 \(u\)，也不会产生激活，因此该 coupon 对应的反向可达集为空。若 \(u\) 的消费事件发生，则固定 \(u\) 为本次传播的最终消费节点，并从 \(u\) 出发执行反向搜索。

记本次生成的反向可达集为 \(R(u)\)。初始化

\[ R(u)=\{u\}, \]

并将 \(u\) 加入待扩展队列 \(Q\)。将 \(u\) 本身加入反向可达集是因为，在已经固定 \(u\) 发生消费的条件下，若 coupon 直接投放给 \(u\)，则 coupon 可以立即被 \(u\) 消费。

随后，每次从队列 \(Q\) 中取出一个当前节点 \(v\)，并检查其所有入邻居

\[ N^-(v)=\{w:(w,v)\in E\}. \]

对于每一个入邻居 \(w\)，反向搜索需要判断：

\[ \text{如果 coupon 当前位于 }w,\text{ 是否会被转发给 }v? \]

因此，该判断使用节点 \(w\) 的行为概率。按照正向传播模型，对 \(w\) 抽取一个随机数

\[ r\sim\mathrm{Uniform}[0,1), \]

并依据 \(p_w^a,p_w^d\) 以及 \(\{p(w,x)\}_{x\in N^+(w)}\) 判定其动作。若此次动作恰好为沿边

\[ (w,v) \]

转发，则称本次反向检查成功。若此时 \(w\notin R(u)\)，则将 \(w\) 加入 \(R(u)\)，并同时加入队列 \(Q\)，使反向搜索继续向 \(w\) 的入邻居扩展。若 \(w\) 选择消费、丢弃或转发给其他出邻居，则此次检查失败，不将 \(w\) 加入集合。

这里需要强调，处理当前节点 \(v\) 时，需要检查 **所有** 入邻居，而不是随机选择一个入邻居。因此，一个 RR set 在反向生成过程中可以形成多个分支。这种分支并不表示一张 coupon 在正向传播时被复制成多张 coupon，而是表示在本次反向样本中，存在多个不同的候选初始投放节点，它们可能分别使 coupon 最终到达同一个消费节点 \(u\)。

每个节点只在第一次进入 \(R(u)\) 时加入队列，因此每个被发现节点最多被扩展一次。当队列为空时，反向搜索结束并返回 \(R(u)\)。

可以把伪代码先写成：

```
\begin{algorithm}[t]
\caption{\textsc{GenerateRRSet}$(u)$}
\label{alg:rr-generation}
\begin{algorithmic}[1]
\Require Graph $G=(V,E)$; root $u$; node and transfer probabilities
\Ensure A reverse reachable set $R(u)$
\If{the consumption event of $u$ does not occur}
    \State \Return $\emptyset$
\EndIf
\State $R(u)\gets\{u\}$
\State initialize a queue $Q$ with $u$
\While{$Q$ is not empty}
    \State remove a node $v$ from $Q$
    \For{each $w\in N^-(v)$}
        \State sample a fresh $r\sim\mathrm{Uniform}[0,1)$
        \State determine the action of $w$ according to $r$
        \If{$w$ transfers the coupon to $v$ and $w\notin R(u)$}
            \State $R(u)\gets R(u)\cup\{w\}$
            \State insert $w$ into $Q$
        \EndIf
    \EndFor
\EndWhile
\State \Return $R(u)$
\end{algorithmic}
\end{algorithm}
```

你给的例子也非常适合紧跟算法之后。

对于

\[ c\rightarrow a\rightarrow u,\qquad d\rightarrow a,\qquad d\rightarrow b\rightarrow u, \]

若本次样本中 \(u\) 的消费事件成立，并且反向检查得到：

\[ a\rightarrow u,\qquad c\rightarrow a \]

成功，而 \(b\rightarrow u\) 和 \(d\rightarrow a\) 失败，则生成过程为

\[ \{u\} \rightarrow \{u,a\} \rightarrow \{u,a,c\}, \]

最终：

\[ R(u)=\{u,a,c\}. \]

其直观含义是：**在该次反向样本中，从 \(u,a,c\) 中任意一个节点投放 coupon，都被该样本判定为能够最终由 \(u\) 消费。**

这已经比我们之前第五节里的 `inverted coverage set` 好解释得多，也完全符合老师说的“首先得告诉读者反向可达集到底怎么生成”。

------

## 但现在有一个必须解决的概率问题

你特别说了一句：

> **每次检查都重新抽 \(r\)**。

如果这里确实是指：**同一个节点 \(w\) 在检查不同出边时，也重新独立抽一次 \(r\)**，那么按照目前 Section 3 的正向模型，老师要求的

\[ \Pr[s\in R(u)] = \Pr[\text{coupon from }s\text{ is consumed by }u] \]

一般情况下**不能直接成立**。

一个很小的例子就能看出来。

假设当前反向搜索中 \(a,b\) 都已经能够到达 root \(u\)，而节点 \(w\) 有：

\[ w\rightarrow a,\qquad w\rightarrow b, \]

并且

\[ p(w,a)=0.4,\qquad p(w,b)=0.4,\qquad p_w^d=0.2. \]

在正向传播中，coupon 第一次到达 \(w\) 时只做**一次**决策，所以它成功进入能够到达 \(u\) 的后继节点的概率为：

\[ 0.4+0.4=0.8. \]

但是按你描述的旧反向代码，如果检查 \((w,a)\) 时独立抽一次，检查 \((w,b)\) 时又独立抽一次，则

\[ \Pr[w\in R(u)] = 1-(1-0.4)^2 = 0.64. \]

于是：

\[ 0.64\neq0.8. \]

也就是说，**算法流程现在完全清楚了，但这个“每条入边独立重新抽 \(r\)”的细节会直接影响老师最需要的正反向概率等价证明。**

这不是文字怎么写的问题，而是模型和代码必须先对齐。

如果旧代码实际上是：

> 第一次遇到 \(w\) 时只生成一个 \(r_w\)，以后检查 \(w\) 时复用这个 \(r_w\)，从而 \(w\) 在该 possible world 中只选择一个固定的 outgoing edge，

那正反向证明就很好写，而且正是你们以前手稿里的 live-edge 方案。

但如果实验代码**确实是每检查一次 \((w,v)\) 就重新采样**，那我们下一步必须根据代码重新证明它对应的究竟是什么概率；不能在论文里直接写“显然与正向传播等价”。

所以现在我建议先把这一个问题钉死：

\[ \boxed{ \text{同一个 }w\text{ 在一次 RR-set 生成过程中，如果先后检查 }(w,v_1),(w,v_2), \text{代码到底是共用一个随机数，还是各自重新采样？} } \]

你刚才的文字说的是后者。如果你能把这段**真实代码**贴给我，我可以直接按代码逐行判断。这个一旦确认，我们马上就能继续写老师要求的核心内容：

\[ \text{RR-set construction} \rightarrow \text{forward--reverse probability equivalence proof} \rightarrow \text{uniform root sampling} \rightarrow k\text{ 个 RR sets 为一组}. \]

这次最重要的是先保证这条链在数学上真的和程序一致。









































