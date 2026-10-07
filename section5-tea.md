

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

