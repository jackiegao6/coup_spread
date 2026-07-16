# Independent Proof Audit

## Scope and verdict

This audit independently re-derives the model identities, NP-hardness
reduction, DR-submodularity result, conditioned RR estimators, approximation
ratio, and expected running time in paper-v2 copy.tex. The audit found no
fatal mathematical error. The manuscript changes made during the audit close
four presentation gaps: the repeated-source exchange loss in the reduction,
the residual-gain inequality for greedy, the recurrence expansion in the
approximation proof, and the applicability of all-$k$ root conditioning to
greedy prefixes.

## Model identities

**Result: pass.** For one coupon seeded at $v$, $q_{vu}$ is the probability
that $u$ is its unique redeemer. Independent coupon realizations therefore
give

\[
\sigma(\boldsymbol{x})
=\sum_u\left(1-\prod_v(1-q_{vu})^{x_v}\right).
\]

The fixed-action realization semantics are essential: a repeated visit follows
the same transfer action and hence identifies a zero-adoption cycle. Ordinary
absorbing-Markov equations that resample a node action after a revisit would
describe a different process and must not be used as an exact evaluator.

## NP-hardness

**Result: pass after clarification.** The reduction is from vertex cover on
cubic graphs. For a source holding $r\ge 2$ coupons, removing one coupon
reduces its private adopter's probability by

\[
\alpha(1-\alpha)^{r-1}\le \alpha(1-\alpha)
\]

and reduces the three incident edge-adopter contributions by at most $3p$.
Moving that coupon to an unused source gains $\alpha$ at the new private
adopter. With $\alpha=1/2$ and $p=1/24$, the net gain is at least
$\alpha^2-3p=1/8>0$, so every optimum uses distinct sources.

For a distinct size-$K$ selection, if $n_i$ counts graph edges with $i$
selected endpoints, cubicity gives $n_1+2n_2=3K$ and
$n_0+n_1+n_2=|E_H|$. Hence

\[
\sigma=K\alpha+3Kp-(3K-|E_H|)p^2-n_0p^2.
\]

The threshold is reached exactly when $n_0=0$, i.e., exactly when the
selected vertices form a size-$K$ vertex cover. The construction is
polynomial and its probabilities are valid.

**Failure checks performed:** repeated allocations; edge nodes with two
selected endpoints; padding a cover of size at most $K$ to exactly $K$; and
the probability-mass constraint $\alpha+3p\le 1$.

## Monotonicity, DR-submodularity, and exact greedy

**Result: pass.** The marginal of one coupon at $v$ is

\[
\Delta_v(\boldsymbol{x})
=\sum_u q_{vu}\prod_w(1-q_{wu})^{x_w}.
\]

It is nonnegative and decreases componentwise with the current allocation.
Replacing a seed of capacity $c_v$ by $c_v$ identical copies converts the
problem to cardinality-constrained monotone submodular maximization, so exact
greedy has the standard $1-(1-1/k)^k\ge 1-1/e$ guarantee.

For the later approximate-greedy proof, the residual bound is justified using
$\boldsymbol{x}_i\vee\boldsymbol{x}^*$. Every positive missing-optimum term is
feasible, and there are at most $k$ such copies. This establishes

\[
\max_v\Delta_v(\boldsymbol{x}_i)
\ge (\mathrm{OPT}-\sigma(\boldsymbol{x}_i))/k.
\]

## RR coverage and conditioned sampling

**Result: pass.** A live edge records the unique transfer selected by a node
for one coupon realization. Conditional on the root adopting coupon $j$,
reverse traversal therefore returns exactly the seeds from which that coupon
would be redeemed at the root. Keeping coupon indices prevents invalid
cross-realization matches.

For root $v$, let $C_v=\bigvee_{j=1}^k A_{v,j}$,
$w_v=1-(1-p_v^a)^k$, and $W=\sum_v w_v$. Sampling $v$ with probability
$w_v/W$, then sampling its gates conditional on $C_v$, gives importance
weight $W$. Any prefix-coverage event and any greedy marginal event implies
$C_v$, even though $C_v$ uses all $k$ gates. Therefore division by $w_v$ is
valid for every prefix and marginal, yielding unbiased estimators.

**Failure checks performed:** prefixes shorter than $k$; gates that fire only
after the current prefix; zero-adoption roots; empty samples; repeated visits;
and coupon-index swaps.

## Approximation guarantee

**Result: pass after clarification.** There are at most
$\sum_{i=1}^k |V_s|^i\le k|V_s|^k$ deterministic prefix-candidate pairs.
For each pair, the per-sample variable lies in $[0,W]$, has the true marginal
as its expectation, and has variance at most $W$, because one coupon creates
at most one new adopter. With $\tau=\epsilon\mathrm{LB}/(2k)$, the stated
sample count and Bernstein's inequality support the union bound over all
adaptive executions.

On the resulting event, each chosen marginal is within
$\epsilon\mathrm{LB}/k$ of the best true feasible marginal. The recurrence

\[
g_{i+1}\le(1-1/k)g_i+\epsilon\mathrm{LB}/k
\]

expands to $g_k\le e^{-1}\mathrm{OPT}+\epsilon\mathrm{LB}$, proving the
claimed $(1-1/e-\epsilon)$ ratio because
$\mathrm{LB}\le\mathrm{OPT}$.

## Expected running time

**Result: pass.** For one conditioned joint sample,

\[
\mathbb{E}[\text{memberships}]
\le kn/W,\qquad
\mathbb{E}[\text{scanned in-edges}]
\le km/W,
\]

because one coupon seeded at $w$ is redeemed by at most one user, so
$\sum_v q_{wv}\le1$. Thus one sample costs $O(k(m+n)/W)$ in expectation.
Multiplication by the stated $T$ cancels $W$. The facts
$\mathrm{LB}\le\mathrm{OPT}\le\min\{k,W\}$ ensure that the main term also
dominates the $O(m+n)$ alias-table preprocessing for root and node-action
distributions and the extra sample introduced by the ceiling. After this
preprocessing, each lazily generated action and live-edge test takes constant
time. Inverted membership lists account for sample processing, and per-round
gain initialization contributes $O(k|V_s|)$.

## Residual theoretical risks

1. The theorem requires a valid positive lower bound. Fixed empirical sample
   budgets used in the experiments do not receive the theorem's per-run
   $(\epsilon,\delta)$ certificate unless they satisfy the displayed bound.
2. The guarantee is for the fixed-action, zero-adoption-on-transfer-cycle
   model. A process that resamples actions on revisits is a different model.
3. The running time is near-linear in graph size only with $k$, $\epsilon$,
   and the spread lower bound treated as fixed, as stated in the manuscript.
