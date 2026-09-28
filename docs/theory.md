# Theory notes for "Three Objects, One Image": what an ideal predictor returns, and what the read-out gives for it

Derivations and numerical checks behind Section 2.2 of the paper ("What an ideal predictor returns") and the
values the excess tear of Section 2.3 takes on each loss-optimal answer. The scripts are in `theory/`; run them from
that folder (`theory/README.md` lists them, their run order and their runtimes). Every number below comes from one of
them, on the exact depths of the three objects in `theory/pilot_compat/stim/texture_a90.npz` (76,197 foreground
pixels, 15 cubes, n = 6). Statements we could not check are marked as not verified.

These notes began (Thu-Fri 24-25/09/2026) as working notes for an earlier framing of the project, which asked
whether a one-step predictor averages the three objects and a multi-step sampler picks one. The paper no longer
argues that. It asks what each loss-optimal answer pays at the joints and compares that with what the networks pay
(the excess tear, Section 7). The derivations still hold, and Section 0 lists the ones the paper uses. Sections 4c, 5 and 6, the table in Section 0.2 and the passages marked as pilot material in Sections 1 and 4a are the record of the earlier framing and are not used
by the paper.

## 0. What the paper uses

| Paper (Sec. 2.2-2.3) | Here | Checks |
|---|---|---|
| The image is exactly three rigid objects, $d_k=d_1-T\thinspace\mathbf 1[c<c_k]$ | Lemma 1 (§1) | check1 |
| Prop. 1: a squared error with targets that do not depend on the prediction (fixed, or each target normalised on its own as in Lotus and Marigold) returns $\sum_kp_kd_k$, which jumps by $p_kT$ at joint $k$ | Props 1-2 (§2.1-2.2) | check1, check2 (a, b) |
| Prop. 2: a per-pixel absolute error returns $d_k$ if $p_k>\frac12$; otherwise the pointwise middle of the three values, which is their mean (a jump of $T/3$ at every joint) once each target is centred on its own median and the prediction is not renormalised | Prop. 5, Lemma 6 (§3.1) | check1 (e), check3 (a) |
| If the prediction is median/MAD normalised as well (Depth Anything, MiDaS), the middle map is infeasible and, when the object fills the frame, the optimum is a $p$-dependent patchwork; the middle map comes back when a shared far background dominates the normalisation | Theorem 7 (§3.2), §3.5 | check3 (b), check3c, check8, check6 |
| Least-squares scale and shift alignment: a squared error returns the posterior's principal component, which can flatten the bars (none of the ten networks uses this loss); an absolute error (E2E-FT) returns a single object, or the middle map when a shared far background dominates the normalisation | Theorem 3 (§2.3), §3.4 | check2 (c), check3b, check6b |
| Depth Anything V2's gradient term, and trimming its absolute-error term, leave the single-object optimum in place; trimming the gradient term as well lets a two-object mixture win at a two-way tie | §3.3 | check3b, check7, check7b |
| MiDaS's trimmed loss stitches two objects together near a two-way tie and prefers a mean-like map at uniform $p$ | §3.4 | check3b |
| The one-step output of a diffusion model is the posterior mean given its starting noise, and given the image when the schedule reaches zero signal (Marigold v1.1) | Prop. 8 (§4a) | check4a, check4b |
| What the excess tear $e_j$ reads on each of these optima | §7 | check14 |

### 0.1 The corollary

In ground-truth units (the references $r_j$, $g_j$ of the excess tear taken from the exact objects; §7):

* The mean $\sum_kp_kd_k$, each single object and the middle map of the median-centred objects are built from the
  three objects with weights that sum to one. They share the objects' slope on every bar and have $\sum_je_j=1$
  ($e=p$, one-hot, and $\frac13$ each), exactly for the mean and the objects and to $10^{-3}$ for the middle map.
* A posterior mean over the objects **and their depth reversals** (weight $w$ upright, $1-w$ reversed) has
  $\sum_je_j=P(\text{up})-P(\text{rev})=2w-1$, with one common slope $(2w-1)$ times the objects' on all three bars.
  So a total near 0 does not by itself rule out a mean, if that mean includes reversed readings.
* The least-squares-aligned squared error returns a principal component. Near uniform $p$ and at two-way ties it has
  $\sum_kw_k\approx0$: flat bars, joint excesses of opposite signs and $\sum_je_j\approx0$ at every scale and
  either sign. None of the ten networks uses this loss.
* At the two-way tie $(0.45,0.45,0.1)$ the MiDaS trimmed loss and the Depth Anything loss with its gradient term
  also trimmed are best served by maps made of objects 1 and 2 (a patchwork for MiDaS, a mixture for Depth
  Anything), and at uniform $p$ the MiDaS loss by a mean-like map. Their bars all recede, the two-object maps
  unevenly, and their joints pay more than the whole mismatch ($\sum_je_j=1.95$-3.08 at the objects' MAD).
* The scale-invariant losses (median/MAD, least-squares alignment) fix the output only up to a positive scale
  (least squares: any nonzero scale), so $\sum_je_j=1$ is a statement at the objects' own scale. A positive scale
  keeps only the sign of $\sum_je_j$ and the ratios of the bar slopes. At the objects' MAD the mean and the middle
  map read $\sum_je_j\approx3$ at uniform $p$.
* $e_j$ is unchanged by a shift of each output and by a scale shared by an image and its references. It is not
  unchanged by a scale applied to the image alone.
* In a network's own units the reference steps $r_j$ of ordinary joints are not zero, so $e=0$ means "the network
  makes its ordinary step", not "continuous", and the value $-1$ for a reversed mean does not carry over (§7.4).

### 0.2 Working hypotheses of the earlier framing, and what became of them

Regime A = the loss normalises over the object alone (tight crop). Regime B = the object sits on a shared far background that dominates the normalisation (the pilot's 29%-foreground renders are closer to B).

| Working-hypothesis claim | Verdict | § |
|---|---|---|
| Each valid object is a ramp with one jump $T$ at its corner; the per-cube read-out recovers it | **Holds exactly.** $d_k=d_1-T\mathbf 1[c<c_k]$ to $4\times10^{-15}$; read-out matrix $E=T(I+O(10^{-4}))$; the read-out is exactly linear on the span of the objects | 1 |
| L2 / posterior mean gives tear $p_kT$ at corner $k$ | **Holds** for plain L2 and for per-target normalisation (Lotus, Marigold targets): split $=p$ to $5\times10^{-4}$ | 2.1-2.2 |
| ... also under scale-and-shift-invariant L2 | **Wrong in regime A.** With least-squares alignment the optimum is the *top principal component*: split (1.45, -0.39, -0.06) for $p=(0.6,0.3,0.1)$, and undefined at uniform $p$; the mean is the *worst* candidate. $p$ can still be recovered as $w\oslash Gw$. **Right in regime B** (split within 0.04 of $p$ with equal object medians, 0.06 in the renderer gauge) | 2.3, 3.5 |
| L1 / median commits to $k$ iff $p_k>\frac12$ | **Holds exactly** (certified by a lower bound) for per-pixel L1 and for median/MAD SSI-L1, in both regimes | 3.1-3.2 |
| All $p_k<\frac12$: a fixed map equal to the equal-weight mean | **Holds** for per-pixel L1 with equal-median gauge, and in regime B for SSI-MAE (certified) and for the DAv2, MiDaS and LS-L1 losses (best found). **Gauge-dependent:** in the renderer's stored gauge the middle map is object 2. **Wrong for SSI-L1 in regime A:** a $p$-dependent patchwork with a full tear at the argmax corner and steps inside bars; closed form in the continuous model (the argmax object's region grows continuously to the whole loop as $p_{(1)}\to\frac12$) | 3.1, 3.2, 3.5 |
| DAv2 ($L_{ssi}+2L_{gm}$) is a median, so it commits iff $p_k>\frac12$ | **Regime A: commits to argmax $p$ for every $p$ tested** (GM punishes anything that is not a full-amplitude object), also with the 10% loss trimming DAv2 applies to pseudo-labelled images when only the ssi term is trimmed. If the GM term is trimmed too (an implementation guess), a two-object mixture beats the objects at the two-way tie $(0.45,0.45,0.1)$. **Regime B: commit iff $p_k>\frac12$, else $\frac13$ tears** (with and without trimming). Best-of-search, not certified. The loss analysed is the teacher's; the released students are distilled from it | 3.3, 3.5 |
| First diffusion step = posterior mean (Tweedie) | **True but given the noise:** $\mathbb E[z\mid x_T,c]$. It equals $\mathbb E[z\mid c]$ only when $\kappa\ll1$. For Marigold $\kappa$ is 1.4-2.7 at the flatness the pilot outputs actually have (object = 7-27% of the depth range), 3.5-5.5 only if the object filled 55-106% of the range, and 15-45% lower again if the network is the Bayes denoiser for its multi-resolution *training* noise rather than for i.i.d. noise (1.1-3.1 overall). $\kappa>1$ in every case, so a *perfect* model's one-step output already votes: seed-to-seed sd 0.2-0.4, and 82-98% of seeds keep their first-step choice. "One step averages whatever the seed" is **false** for SD's schedule (this includes the Bayes-optimal Lotus-G); it holds for Lotus-D (no noise input) | 4a |
| Latent mean, then decode: tears still read $p$ | **Approximately.** Decoded convex combinations of encoded objects keep the split within 0.02-0.06 of the weights (object 12%, 55% and 106% of the range) | 4b |
| A correct sampler picks $k$ with frequency $p_k$ | **Only from the true noisy marginal $p_T$.** From $\mathcal N(0,I)$ with SD's $\bar\alpha_T=0.0047$, $q$ drifts from $p$ by an amount that depends on $\kappa$, on the noise law and on small differences between the atoms' norms. For the encoded ground-truth geometries $q$ matches the *seed-averaged one-step weights* within 0.001-0.026 under both noise laws, but this is **not a theorem**: for random atoms whose norms differ by <1%, $\max\lvert q-s\rvert$ reaches 0.39. The within-model test therefore compares measured $q$ with $q$ *simulated* for the model's own atoms (P6) | 4c |
| The span diagnostic (split, ramp share, residual) separates mean / principal component / patchwork / object | **Only when the model's control residual is $\lesssim0.1$.** At the DAv2 control level (0.2-0.4) a perturbed mean is labelled "patchwork" in 87-100% of draws and an object "principal component" in 28-33% | 1 |

---

## 1. Setup: the valid set and its loop structure

**Notation.** Foreground pixels $\Omega$, $M=|\Omega|$. Cube index $c(x)\in\lbrace 0,\dots,N-1\rbrace$ along the loop, $N=3(n-1)=15$.
Corner $j$ (joint $j$ in the paper) is the first cube of bar $j$: $c_1=0,\ c_2=n-1=5,\ c_3=2(n-1)=10$; bar $j$ = cubes $[c_j, c_{j+1})$.
Object $k$ (gap at corner $k$) has depth map $d_k$ (larger = farther). Posterior weights $p=(p_1,p_2,p_3)$ on the simplex $\Delta$.

**Lemma 1 (the three maps).** With $T=(n-1)\sqrt3$ and $r:=d_1$,
$$d_k(x) = r(x) - T\thinspace\mathbf 1[c(x)<c_k],\qquad k=1,2,3 .$$
*Proof.* Object $k$ moves the cubes before corner $k$ by $(n-1)(1,1,1)$, i.e. by $(n-1)\sqrt3$ towards the camera along $v=(1,1,1)/\sqrt3$; the shown face and its image footprint are unchanged (`pdepth/render.py`). $\square$
Check: $\max_\Omega |(d_k-d_1)+T\thinspace\mathbf 1[c<c_k]| \le 3.6\times10^{-15}$ (check1).

**Lemma 2 (one ramp, one jump).** Split $r(x)=P(c(x))+\rho(x)$ into the per-cube mean $P$ and a within-cube residual $\rho$ (face-plane geometry, zero mean on each cube). Then

1. $\rho$ is the same for all three objects (they differ by constants on cubes), so the object-dependent part lives entirely in the per-cube profile $P_k(i)=P(i)-T\thinspace\mathbf 1[i<c_k]$.
2. $P(i+1)-P(i)\approx-\delta$ with $\delta=1/\sqrt3$ (each cube step moves one unit along an axis, which is $1/\sqrt3$ along $v$). The loop closes in the image because the chain's total displacement $(n-1)(1,1,1)$ is parallel to $v$, so the depth lost around the loop is $N\delta=(n-1)\sqrt3=T$. **The jump equals the accumulated ramp: $T=N\delta$.**
3. Read cyclically from corner $k$, $P_k$ is a monotone ramp (step $-\delta$) through all $N$ cubes and jumps up by $T-\delta$ on the step that enters corner $k$. So each valid object is "one ramp with one jump of size $T$ at its own corner".

Measured per-cube steps of object 1: $-0.5765\ldots-0.5795$ on the 14 bar/corner-free steps and $+8.0828=T-\delta$ entering corner 1 (check1).

*Pilot material of the earlier framing, not used by the paper: the rest of this section, except the cohomological reading. The paper's read-out is the one of Section 7.*

**The pilot's tear read-out (`theory/pilot_compat/tears.py`; the paper uses a different one, Section 7).** Steps $s_i=P(i+1)-P(i)$ (cyclic), ramp $\hat\rho=\operatorname{median}$ of the 12 steps that do not enter a corner, tear $e_j=s_{c_j-1}-\hat\rho$, split $=e/\sum_j e_j$.

**Lemma 3 (the read-out is exactly linear on the span of the objects).** For any $w\in\mathbb R^3$, $b\in\mathbb R$:
$$e\Big(\textstyle\sum_k w_k d_k + b\Big)=w^\top E,\qquad E_{kj}:=e_j(d_k).$$
*Proof.* $d_k-d_1$ is constant on every bar (Lemma 1: the shifted set is a union of whole bars), so the 12 in-bar steps of $\sum_k w_kd_k+b$ are $(\sum_k w_k)$ times those of $r$; the median is homogeneous of degree 1 for any real factor (12 values, midpoint median), and corner steps are linear. $\square$

Measured: $E/T = I + O(10^{-4})$ (largest off-diagonal $8\times10^{-5}$; row sums $1.00001$). So for maps in the span, $\text{split}=w/\textstyle\sum w + O(10^{-4})$, invariant to any affine map $m\mapsto am+b$, $a\neq0$. 200 random mixtures: $\max|\text{split}-p| = 7.8\times10^{-5}$; the residual is exactly $p^\top E/(p^\top E\mathbf1)$ to $6\times10^{-16}$ (check1).

**Cohomological reading.** The steps of any depth map around the loop sum to zero. The rigid bars demand the step $-\delta$ everywhere, which sums to $-T\neq 0$: the demand is not the coboundary of any depth map, and the obstruction is exactly $T$. Every depth map whose bars are straight with the rigid slope $-\delta$ therefore pays total tear $\sum_j e_j = T$. More generally, a map with straight bars of common step $-\hat\delta$ pays $N\hat\delta$. "Where it tears" is the choice of representative. A valid object puts all of $T$ on one corner. This is Penrose's own view of the tribar: R. Penrose, "On the cohomology of impossible figures", *Leonardo* 25(3/4):245-247, 1992.

**The span diagnostic.** The span of $\lbrace P_1,P_2,P_3,\mathbf 1\rbrace$ is exactly "every bar a straight ramp with a common slope, any bar offsets". Fitting a profile $y\approx\sum_k w_kP_k+c$ gives (i) the split $w/\sum w$, (ii) the *ramp share* $\sum w/\sum|w|$, equal to 1 for every convex mixture and $<1$ when the bars are flatter than the tears require, and (iii) the relative residual $\lVert y-\text{fit}\rVert/\lVert y-\bar y\rVert$, 0 for any mixture.

*On noise-free theory maps* the three numbers separate the cases (check9): objects give a one-hot split, ramp 1, residual 0; posterior means give split $=p$, ramp 1, residual 0; SSI-L2 principal components give ramp 0.12-0.53, residual 0. SSI-MAE patchworks are the weak case. Their profile has ramp **+1.00**, a split *inside* the simplex, e.g. (0.53, 0.27, 0.21) at $p=(0.4,0.35,0.25)$ and (0.34, 0.33, 0.33) at uniform $p$, and a residual of only 0.19-0.29 (0.05 at $p=(0.5,0.3,0.2)$, where the patchwork is almost object 1). So a patchwork looks like a posterior mean plus a moderate residual.

*At the error level of real outputs they do not separate* (check9). DAv2 S/B/L already have residual 0.21-0.30 and ramp share 0.17-0.65 on the **one-answer control**. A nearest-template classifier on (split, ramp, residual) was applied to every theory map plus a perturbation. With i.i.d. per-cube noise at residual level $\rho$, it labels means correctly in 100% / 81% / 16% of draws at $\rho=0.05/0.1/0.2$. The other 84% at $\rho=0.2$ are called "patchwork". Next, the control's own error (control output affinely aligned to object 1, minus object 1) was added to each theory map. Then means are called "patchwork" in 87-100% of draws, objects are called "principal component" in 28-33%, and principal components are called "object" in 50-98%. **So ramp share $<1$ is not a principal-component signature for these models (their one-answer controls show it), and a residual of 0.2-0.3 does not tell a patchwork from a noisy mean.**

*A statistic aimed at the patchwork.* The patchwork's defining feature is steps inside bars. Take the 12 in-bar steps $s_i$ and $z_i=\lvert s_i-\operatorname{median}s\rvert/\operatorname{std}(y)$. Clean patchworks have $\max z$ of 0.5-1.1 (0.23 at $p=(0.5,0.3,0.2)$, where the patchwork is almost object 1), clean mixtures 0. DAv2's controls have $\max z$ of 0.39-0.69. The threshold "1.25 × the model's own control value" flags 36-87% of perturbed patchworks, but also 13-40% of perturbed means. That is useful but weak.

**Rule used from now on.** Every diagnostic number is read against the same model's control. The split/ramp/residual triple is interpreted only if the control residual is $\le0.1$ (there the classifier is $\ge80$% correct). Above that, only coarse statements are made: one-hot vs not, split inside vs outside the simplex, and more or fewer in-bar steps than the control. The predictions in Section 5 are stated with these thresholds.

---

## 2. L2 predictors

A deterministic network trained to convergence on a loss $\ell$ approximates the population optimum $f^\star=\arg\min_f \sum_k p_k\thinspace\ell(f,d_k)$ (all statements below are about this optimum; a finite network only approximates it).

### 2.1 Fixed target gauge: the mean (holds)

**Proposition 1.** If $\ell(f,d)=\lVert f-d\rVert^2/M$ with targets that do not depend on $f$, then $f^\star=\sum_kp_kd_k$ and its tears are $T\thinspace p$ (to $E$'s $10^{-4}$).
If each object also comes with an arbitrary distribution of global offsets $b$ (depth along $v$ is not identifiable in an orthographic image), $f^\star=\sum_kp_kd_k+\mathbb E[b]$ and the tears are unchanged. *Proof:* the conditional mean; then Lemma 3. $\square$

### 2.2 Per-target normalisation (Marigold, Lotus): still the mean, reweighted by the normalisation scales

Marigold normalises each ground-truth map by its own 2%/98% percentiles to $[-1,1]$, and Lotus follows the same affine-invariant recipe (the exact percentiles for Lotus are not verified). Neither aligns the prediction, so the target is $a_kd_k+b_k$ with $(a_k,b_k)$ a function of $d_k$ only.

**Proposition 2.** $f^\star=\sum_kp_k(a_kd_k+b_k)$, so $\text{split}_k = p_ka_k/\sum_jp_ja_j$ (exactly $p$ iff the $a_k$ are equal).
Measured on the foreground: $a_k/\bar a = (1.00009,\ 0.99950,\ 1.00041)$ for the 2/98 percentile normalisation and $(1.0005, 0.9991, 1.0004)$ for median/MAD; the optimum reproduces $p$ to 3 decimals for $p=(0.6,0.3,0.1),(0.4,0.35,0.25),$ uniform (check2 (b)).
The scales are equal because the three objects have the same depth histogram (each is a cyclic shift of one sawtooth plus a constant). If the loss also covers a background whose depth relative to each object differs, the $a_k$ can differ; the correction is then this explicit reweighting.

### 2.3 Scale-and-shift-invariant L2 with least-squares alignment (MiDaS "ssimse"): NOT the mean

MiDaS (Ranftl et al., TPAMI 2022, arXiv:1907.01341) aligns the prediction to each target by least squares, $\hat d=s f+t$ with $(s,t)=\arg\min\sum_i(sf_i+t-d_i)^2$, then takes the MSE. None of the ten networks in the paper is trained with this loss: MiDaS DPT-L uses the trimmed absolute error of Section 3.4, and E2E-FT aligns by least squares but then takes an absolute error (Section 3.4).

**Theorem 3 (the SSI-L2 optimum is a principal component).** Let $\tilde d_k=d_k-\bar d_k$ (centred on $\Omega$), $G_{jk}=\tilde d_j^\top\tilde d_k/M$, $P=\operatorname{diag}(p)$. The population loss is
$$L(f)=\sum_kp_k\min_{s,t}\tfrac1M\lVert sf+t-d_k\rVert^2=\sum_kp_k\tfrac{\lVert\tilde d_k\rVert^2}{M}-\frac{\tilde f^\top C\tilde f}{M\thinspace\lVert\tilde f\rVert^2},\qquad C=\sum_kp_k\tilde d_k\tilde d_k^\top ,$$
so the minimisers are $f^\star=\sum_kw_kd_k+b$ (any nonzero scale, either sign), with $w$ the top eigenvector of $PG$, unique up to scale when $\lambda_{\max}$ is simple:

$$p_k\thinspace(Gw)_k=\lambda_{\max}\thinspace w_k\quad(k=1,2,3).$$

*Proof.* For fixed $f$, $\min_{s,t}\lVert sf+t-d\rVert^2=\lVert\tilde d\rVert^2-(\tilde f^\top\tilde d)^2/\lVert\tilde f\rVert^2$ (regression of $d$ on $f$ and the constant $\mathbf 1$). Summing gives the Rayleigh quotient of $C$, maximised by its top eigenvector, which lies in $\operatorname{span}\lbrace\tilde d_k\rbrace$. Writing $\tilde f=\sum_jw_j\tilde d_j$: $C\tilde f=\sum_k\tilde d_k\thinspace p_k(Gw)_k$, and linear independence of the $\tilde d_k$ gives the eigen-equation. Tears follow from Lemma 3. $\square$

**Corollary 3a (inversion).** $p\propto w\oslash(Gw)$. Since $G$ is computable from the ground truth, $p$ can still be recovered from an SSI-L2 output; it is just not the split. (Verified: recovered $p$ equals the true $p$ to 3 decimals for all 7 test cases.)

**Corollary 3b (closed form, circulant $G$).** The three centred objects are pairwise anti-correlated: measured $\cos(\tilde d_j,\tilde d_k)=-0.336$ ($-1/3$ for three sawtooths a third of a period apart), $G\approx cI+\gamma J$ with $c=8.33$, $\gamma=-2.09$ per pixel. Then

$$\text{split}_j=\frac{\gamma\thinspace p_j}{\lambda-c\thinspace p_j},\qquad \sum_j\frac{p_j}{\lambda-cp_j}=\frac1\gamma,\quad \lambda\in(c\thinspace p_{(2)},\thinspace c\thinspace p_{(1)})\ \text{if}\ \gamma<0,\quad \lambda>c\thinspace p_{(1)}\ \text{if}\ \gamma>0.$$

Because $\gamma<0$, the most probable corner gets a tear **larger than $T$** and the other two get **negative** tears; the bars are flattened (ramp share $<1$). At uniform $p$ the top eigenspace is $\lbrace w:\sum w=0\rbrace$: flat bars and tears of both signs that sum to zero (the optimum found in check2 (c) has $w\propto(-0.174,+0.500,-0.326)$: one up-step and two down-steps), split undefined.
$\gamma<0$ holds whenever the object-independent part of the depth variance is small ($\gamma\approx\alpha-\beta/3$ with within-cube variance $\alpha=0.056$ and loop variance $\beta\approx6.18$). A large *shared* component (e.g. a far background inside the loss) makes $\gamma>0$; then all split entries are positive and the split tends to $p$ as $\gamma/c\to\infty$ (Section 3.5).

Direct optimisation of the population loss over all 76,197 pixel values (L-BFGS, 4 starts, float64; check2 (c)):

| $p$ | split, eigen-theory | split, direct optimum | ramp share $\sum w/\sum\lvert w\rvert$ | loss: optimum / best single object / mean |
|---|---|---|---|---|
| (0.6, 0.3, 0.1) | (1.449, -0.386, -0.063) | (1.449, -0.386, -0.063) | 0.53 | 2.108 / 2.217 / 3.209 |
| (0.8, 0.15, 0.05) | (1.103, -0.081, -0.022) | (1.103, -0.081, -0.022) | 0.83 | 1.093 / 1.108 / 1.317 |
| (0.5, 0.3, 0.2) | (1.799, -0.582, -0.217) | (1.799, -0.582, -0.217) | 0.39 | 2.657 / 2.770 / 4.158 |
| (0.4, 0.35, 0.25) | (4.645, -3.164, -0.481) | (4.645, -3.163, -0.481) | 0.12 | 3.088 / 3.324 / 5.149 |
| (0.36, 0.33, 0.31) | (7.67, -4.68, -1.99) | (7.67, -4.68, -1.99) | 0.07 | 3.339 / 3.545 / 5.502 |
| uniform | undefined ($\sum w=0$) | $\sum w/\sum\lvert w\rvert = 0.000$ | 0.00 | 3.458 / 3.687 / 5.556 |

The fit residual of every optimum to the span is $<10^{-8}$, and the circulant closed form matches to 2 decimals away from ties (e.g. (1.448, -0.385, -0.063)); near ties it is up to 0.04 off at (0.4, 0.35, 0.25) and 0.11 off at (0.36, 0.33, 0.31). **Under SSI-L2 the mean is the worst of the three candidates**: it has small variance and correlates weakly with each object ($\cos^2\approx0.11$), so after per-target rescaling it explains little. Committing to one object always beats the mean here.

**Variant (median/MAD normalisation of both maps, then MSE).** Not used by the models we test. Direct optimisation (Adam) gives split (0.566, 0.359, 0.076) for $p=(0.6,0.3,0.1)$ and (1/3, 1/3, 1/3) for uniform $p$, but the output leaves the span (residual 0.26-0.34): the Lagrangian of the MAD constraint is $g=\bar n+\tfrac\lambda2\operatorname{sign}(g)$, the mean pushed outward, with a step inside each bar where $\bar n$ crosses zero.

**Correction to the working hypothesis.** "L2 averages" is true only when the target normalisation does not depend on the prediction (Props 1-2). With least-squares scale-and-shift alignment the L2 optimum is the top principal component of the posterior, which overshoots at the likely corner and reverses the others.

**What the paper's read-out gives for it (Section 7).** The principal component is $\sum_kw_kd_k$ with $\sum_kw_k<\sum_k\lvert w_k\rvert$, so its bars recede at $\sum_kw_k$ times the objects' rate and $\sum_je_j=\sum_kw_k$ at the scale where the $w_k$ are the weights. Near uniform $p$ and at two-way ties $\sum_kw_k\approx0$: the bars are flat, the joints get excesses of opposite sign, and $\sum_je_j\approx0$ at every scale and either sign. Among the optima for a belief over the three upright objects it is the only one with flat bars and $\sum_je_j\approx0$; none of the ten networks uses its loss.

---

## 3. L1 predictors

### 3.1 Per-pixel median with fixed targets

**Lemma 4 (weighted median of three).** At one pixel let the three target values sort as $v_{(1)}\le v_{(2)}\le v_{(3)}$ with weights $q_{(1)},q_{(2)},q_{(3)}$. The minimisers of $\sum_iq_{(i)}|v-v_{(i)}|$ are the $v$ with $W(<v)\le\frac12$ and $W(>v)\le\frac12$. Hence: if some $q_{(i)}>\frac12$ the minimiser is $v_{(i)}$; if all $q_{(i)}<\frac12$ it is $v_{(2)}$ (the middle value) whatever the weights; if $q_{(1)}=\frac12$ (or $q_{(3)}=\frac12$) it is the whole interval $[v_{(1)},v_{(2)}]$ (or $[v_{(2)},v_{(3)}]$). Ties between values merge their weights. $\square$

**Proposition 5 (fixed targets $t_k$, loss $\sum_kp_k\lVert f-t_k\rVert_1$).**
(a) $p_k>\frac12\Rightarrow f^\star=t_k$ exactly: **commit**, for any gauge.
(b) all $p_k<\frac12\Rightarrow f^\star=\operatorname{mid}(t_1,t_2,t_3)$ pointwise, **a fixed map independent of $p$**.
(c) $p_k=\frac12$: every map lying pointwise between $t_k$ and the middle map is optimal. Check: $g=(1-\tau)n_1+\tau\thinspace\text{mid}$ has the same loss 0.890667 for $\tau\in\lbrace 0,.25,.5,.75,1\rbrace$ at $p=(0.5,0.3,0.2)$, while its split slides from (1,0,0) to (1/3,1/3,1/3) (check3 (a)).

**What the fixed map is depends on the gauge.** For a non-invariant L1 loss the absolute depth of each object matters, and it is not determined by an orthographic image.

* *Renderer gauge* (pilot npz as stored): on every bar two of the three objects coincide, so the middle map is **object 2** exactly (max difference 0, split (0, 1.000, 0)) for every $p$ with all $p_k<\frac12$. This is a gauge artefact, not a property of the image.
* *Equal-median gauge* (each target centred on its own median, the prediction not renormalised): the middle map is the equal-weight mean, as the working hypothesis guessed. A loss that also median/MAD-normalises the prediction (Depth Anything, MiDaS) does not give this: there the middle map is infeasible (Section 3.2) unless a shared background dominates the normalisation (Section 3.5).

**Lemma 6 (middle = mean in the equal-median gauge).** Parametrise the loop by $\theta\in[0,1)$ with corner $k$ at $\theta_k=(k-1)/3$. Centred objects are $d_k(\theta)=T(\tfrac12-\lbrace\theta-\theta_k\rbrace)$. On bar $j$, $\lbrace\theta-\theta_k\rbrace=(\theta-\theta_j)+m/3$ with $m=(j-k)\bmod 3$, so the three values are $T(\tfrac12-(\theta-\theta_j)-m/3)$, $m=0,1,2$: an arithmetic progression with gap $T/3$. The middle term is the mean. Its tears are $T/3$ at every corner. $\square$
(On bar $j$ the three objects put that bar first, second or last in their chain; the median picks "second", which is also the average position.)
Measured: on every bar the ordering of the three normalised maps is the same at all ~25,400 pixels, gaps 1.334-1.339 ($=T/3$ in MAD units; sd within a bar $\le5.3\times10^{-4}$); $\max|\text{mid}-\text{mean}|=0.0013$ against an rms of 0.38; split(mid) $=(0.3338,0.3329,0.3333)$ (check1 (e)).

So the tentative claim is **correct for a pure per-pixel L1 with per-target normalisation**: exact commitment when $p_k>\frac12$, a fixed map equal to the equal-weight mean otherwise, and an interval of optima on the boundary. It is **not** what the scale-invariant L1 losses of real networks give (next).

### 3.2 SSI-MAE with median/MAD normalisation of prediction and target (MiDaS ssimae; Depth Anything's $L_{ssi}$)

$N(x)=(x-\operatorname{med}x)/\operatorname{mean}|x-\operatorname{med}x|$, $n_k=N(d_k)$, $L(f)=\sum_kp_k\operatorname{mean}|N(f)-n_k|$. Since $N$ maps every non-constant $f$ onto $S=\lbrace g:\operatorname{med}g=0,\operatorname{mean}|g|=1\rbrace$ and fixes $S$, the problem is $\min_{g\in S}C(g)$ with $C(g)=\sum_kp_k\operatorname{mean}|g-n_k|$.

**Theorem 7.**
(a) If $p_k>\frac12$, $f^\star=d_k$ up to a positive affine map, uniquely. *Proof:* dropping the constraint gives the lower bound $\min_gC=C(n_k)$ (Prop. 5a), and $n_k\in S$. $\square$ Certificate: relaxed bound = dual bound = loss of $d_1$ = 0.7126 at (0.6,0.3,0.1) and 0.3563 at (0.8,0.15,0.05) (check3 (b)).
(b) If all $p_k<\frac12$, the unconstrained optimum (the middle map) is **infeasible**: its MAD is $1/3$ of the objects' (measured 0.327). Normalising multiplies it by 3, and the relaxed bound is not reached (at $p=(0.4,0.35,0.25)$ the rescaled middle map costs 1.1193 against the relaxed bound 0.8908). The constraint binds. Relaxing $\operatorname{med}g=0$ and dualising $\operatorname{mean}|g|=1$,

$$D(\lambda)=\lambda+\operatorname{mean}_x\min_{v}\big[c_x(v)-\lambda|v|\big]\ \le\ \min_{g\in S}C(g),\qquad c_x(v)=\sum_kp_k|v-n_k(x)|,\ \lambda<1,$$

and since $c_x-\lambda|\cdot|$ is piecewise linear, the inner minimum is at $v\in\lbrace n_1(x),n_2(x),n_3(x),0\rbrace$. **The optimum is a patchwork: every pixel takes one of the three objects' normalised values.**
(c) Closed form in the symmetric continuous model (check8). Centred objects $n_k(\theta)=T(\frac12-\lbrace\theta-\theta_k\rbrace)$, $h=T/3$, $a_k:=1-2p_k>0$. On every bar the middle value is the mean, and it crosses zero at the bar's midpoint. Cut each bar at its midpoint. Each of the six half-bars is then an ordered pair $(c,c')$: $c$ is the corner it touches and $c'$ the corner at the far end of the same bar. All six ordered pairs of distinct corners occur exactly once. Let $s\in[0,1)$ be the distance from the bar midpoint in units of a half-bar ($s=1$ at corner $c$). A pixel has three useful moves:
* stay at the middle value;
* take the value of object $c$, which lies on the **same side** of zero: MAD gain $h$, cost $a_c h$;
* take the value of object $c'$, which lies on the **other side** of zero: MAD gain $(1-s)h$, cost $a_{c'}h$.

The Lagrangian picks per pixel $\max\lbrace 0,\ \lambda-a_c,\ \lambda(1-s)-a_{c'}\rbrace$ (in units of $h$), and the dual is
$$D(\lambda)=\tfrac23h+\tfrac{\lambda h}2-\tfrac h6\sum_{c\ne c'}\mathbb E_{s\sim U[0,1)}\max\lbrace 0,\lambda-a_c,\lambda(1-s)-a_{c'}\rbrace.$$
$\lambda^\star$ is fixed by the MAD budget: the total gain over the six half-bars must be 3 (in units of $h/6$). If $\lambda^\star$ sits at a jump $\lambda=a_k$, the budget is met by filling part of object $k$'s same-side pixels, which are all equally priced. **Object $k$'s region** is its two adjacent half-bars (when $\lambda^\star\ge a_k$) plus a band of relative width $1-a_k/\lambda^\star$ next to the midpoint on each of the two far half-bars, unless a more probable object outbids it there. As $p_k\uparrow\frac12$ we have $a_k\to0$, and the band grows to the whole far half-bar. Object $k$ then covers both bars next to its corner. On the third bar $n_k$ *is* the middle value, so the map becomes $n_k$ exactly. This proves the continuity claim and meets (a).

Check (check8, $T=4$, 120,000 pixels): the closed-form dual equals the brute-force dual over $\lbrace n_1,n_2,n_3,0\rbrace$. It also equals a dual that prices the median constraint. The Lagrangian map (partial fill, then median/MAD-normalised) is feasible and within $5\times10^{-5}$ of the dual, i.e. optimal. The exception is the degenerate uniform $p$, where $\lambda^\star=1/3$ falls between the grid points: there the brute-force dual is $3\times10^{-4}$ lower and the gap is $1.1\times10^{-4}$.

| $p$ | $\lambda^\star$ | optimum | old construction (same-side only) | object 1 | share of loop overwritten by objects 1 / 2 |
|---|---|---|---|---|---|
| (0.49, 0.26, 0.25) | 0.48 | 0.9065 | 1.0045 (+11%) | 0.9067 | 0.65 / 0 |
| (0.48, 0.30, 0.22) | 0.40 | 0.9236 | 0.9956 (+8%) | 0.9244 | 0.63 / 0.00 |
| (0.40, 0.35, 0.25) | 0.30 | 1.0370 | 1.0445 (+0.7%) | 1.0667 | 0.44 / 0.07 |
| (0.36, 0.33, 0.31) | 0.34 | 1.0865 | 1.0889 (+0.2%) | 1.1378 | 0.39 / 0.11 |
| (0.45, 0.45, 0.10) | 0.10 | 0.9556 | 0.9556 | 0.9778 | degenerate tie |
| uniform | 1/3 | 1.1111 $=5T/18$ | 1.1111 | 1.1852 | degenerate |

**Result.** Below $\frac12$ the optimum is the mean map overwritten by the most probable object. That object takes its two half-bars and bands that reach past the midpoints of the adjacent bars, and the second object fills the rest of the budget. The argmax corner gets a full tear $T$, the other corners keep partial tears, and the loop closes through steps inside the two bars next to the argmax corner. Near $p_{(1)}=\frac12$ the optimum is almost object 1 (cost within 0.02%). This matches the discrete patchwork of check3c, where object 1 takes 3 of the 5 cubes of each adjacent bar at $p=(0.4,0.35,0.25)$, more than a half-bar. The old same-side-only construction is exact only at two-way ties and uniform $p$. It is 0.2-11% too expensive otherwise, and could never approach object 1. At uniform $p$ the optimum is degenerate and includes the (rescaled) mean map (both cost $5T/18$).

Numbers (check3 (b); patchwork structure in check3c):

| $p$ | relaxed bound (pointwise median) | dual lower bound | best patchwork (primal) | single object (argmax) | posterior mean $\sum_kp_kd_k$ | middle map |
|---|---|---|---|---|---|---|
| (0.6, 0.3, 0.1) | 0.7126 | 0.7126 | 0.7126 (= object 1) | 0.7126 | 0.8927 | 1.1188 |
| (0.5, 0.3, 0.2) | 0.8907 | 0.8907 | 0.8989 | 0.8907 | 1.0303 | 1.1190 |
| (0.45, 0.45, 0.1) | 0.8909 | 0.9581 | 0.9621 | 0.9798 | 1.0304 | 1.1192 |
| (0.4, 0.35, 0.25) | 0.8908 | 1.0402 | 1.0433 | 1.0688 | 1.1008 | 1.1193 |
| (0.36, 0.33, 0.31) | 0.8909 | 1.0903 | 1.0903 | 1.1400 | 1.1174 | 1.1193 |
| uniform | 0.8909 | 1.1150 | 1.1151 | 1.1875 | 1.1194 | 1.1194 |

Duality gaps are at most 0.008, so the patchwork is optimal to within 1%. At $p=(0.4,0.35,0.25)$ the optimum takes object 1's values on cubes 0-2 and 12-14 (both sides of corner 1), and mostly the middle map elsewhere, with some object-2 values on cubes 3-6 and 11. Its per-cube steps minus the in-bar ramp (MAD units, full tear $\approx4.0$) are $4.0$ into corner 1, $2.0$ into corner 2, $1.34$ ($=T/3$) into corner 3, and $-1.2,-0.4,-0.6,-0.7$ inside bars 1 and 3.

**Correction.** For the SSI-L1 loss that DAv1/DAv2 actually use, "commit iff some $p_k>\frac12$" is exact on the commit side only. Below $\frac12$ the output is not a fixed map. It is a $p$-dependent patchwork with a full tear at the argmax corner and steps inside bars. (With a shared background the fixed-map result comes back, Section 3.5.)

### 3.3 Depth Anything V2 adds a gradient-matching term

**What DAv2 actually trains with.** From arXiv:2406.09414 (HTML, re-read 25/09): "We use two loss terms for optimization on labeled images: a scale- and shift-invariant loss $\mathcal L_{ssi}$ and a gradient matching loss $\mathcal L_{gm}$" (Sec. 5.2), with weight ratio 1:2 (Sec. 7.1). $L_{ssi}$ is the median/MAD MAE of Depth Anything V1 (arXiv:2401.10891). MiDaS's $L_{reg}=\frac1M\sum_{s=0}^{3}\sum_i(|\nabla_xR^s_i|+|\nabla_yR^s_i|)$, with $R=N(f)-N(d)$ subsampled by $2^s$. We use $K=4$ scales as in MiDaS; DAv2 does not state its $K$ (not verified). Sec. 5.2 says that "for each pseudo-labeled sample, we ignore its top-n-largest-loss regions during training, where n is set as 10%". On pseudo-labelled images the authors also "add an additional feature alignment loss" to frozen DINOv2 features, following V1. The released S/B/L checkpoints are students trained only on pseudo-labelled real images. The labels come from a teacher trained on synthetic images. Whether $L_{gm}$ is also applied to the pseudo-labelled images, and whether "regions" means pixels or patches, is not stated (not verified).

**Which loss decides, then.** We argue in two steps.
1. The *teacher* is trained on synthetic images with $L_{ssi}+2L_{gm}$ and no trimming (trimming is stated only for pseudo-labelled samples). An image that is ambiguous for the teacher is therefore governed by the analysis below.
2. The *students* are trained on the teacher's deterministic labels. For any loss that is minimised when the prediction equals the label up to an affine map, trimmed or not, the student's population optimum on those images is the teacher's output. Trimming and feature alignment change how well a finite student matches the teacher, not what it is matched to.

So the released models inherit the teacher's response to ambiguity to the extent that they imitate it. That is an inductive-bias statement, not a theorem, and the feature-alignment term (which acts on encoder features, not on the output) cannot be analysed at the population level. As a robustness check we still evaluate the student-style trimmed loss on the three-object posterior (check7, check7b, below).

**Mechanism.** GM compares gradients. Two objects differ in gradient only across bar boundaries, because $d_j-d_k$ is constant on each bar (Lemma 1), so $\mathrm{GM}(n_j-n_k)$ is small. A map that is not a full-amplitude object pays GM on many edges. In regime A (below) the normalisation stretches the mean map by 3, so its bars are 3 times too steep against *every* target at *every* in-bar edge. A patchwork pays at each step it puts inside a bar.

**Numbers, object-only loss** (check3b: exact loss of all $3^6=729$ maps that take one object per half-bar, then Adam for 400 steps from the best of them and from the mean). Loss values:

| $p$ | object 1 | object 2 | object 3 | best non-object half-bar map | mean map | middle map | Adam from mean ends at |
|---|---|---|---|---|---|---|---|
| (0.6, 0.3, 0.1) | **0.8993** | 1.5789 | 1.9805 | 1.2761 | 1.5115 | 2.2124 | 1.0818 |
| (0.45, 0.45, 0.1) | **1.2390** | **1.2392** | 1.9809 | 1.3072 | 1.7573 | 2.2135 | 1.3638 |
| (0.4, 0.35, 0.25) | **1.3425** | 1.4563 | 1.6507 | 1.4796 | 2.1522 | 2.2205 | 1.5893 |
| uniform | 1.4881 | 1.4888 | **1.4673** | 1.5753 | 2.2248 | 2.2247 | 1.6793 |

The best map found is always a single object, the most probable one. At (0.45, 0.45, 0.1) objects 1 and 2 tie (1.2390 vs 1.2392). At uniform $p$ object 3 wins by 1.4% because of small asymmetries in the render. Adam started from the mean stalls in worse local minima (e.g. 1.589 at (0.4, 0.35, 0.25)). This is the best of a structured search, **not a certified optimum**. The only lower bound we have is the SSI-MAE dual bound (0.71-1.12 at the four $p$ of the table), which is loose.

**With the students' 10% trimming (check7).** `dav2trim` = mean of the 90% smallest $\lvert N(f)-n_k\rvert$ + 2 GM. `dav2trimgm` additionally zeroes the trimmed pixels in the GM residual; whether DAv2 does this is unknown, so it is an implementation guess. The candidates are the objects, the mean, the middle map, all 729 half-bar maps, and the two maps with which the trimmed MiDaS loss beat every object (the two-object patchwork found at (0.45, 0.45, 0.1) and the mean-like map found at uniform $p$). Adam (400 steps) was run from the argmax object, the mean and those two maps.

| $p$ | `dav2trim`: best object | best non-object found (which) | `dav2trimgm`: best object | best non-object found (which) |
|---|---|---|---|---|
| (0.6, 0.3, 0.1) | **0.8597** | 0.9910 (Adam from mean; split (0.69, 0.29, 0.02)) | **0.8949** | 0.9580 (Adam from mean) |
| (0.45, 0.45, 0.1) | **1.1845** | 1.2482 (Adam from mean; split (0.49, 0.49, 0.03)) | 1.2323 | **1.1962** (Adam from mean; split (0.47, 0.50, 0.03), span residual 0.29) |
| (0.4, 0.35, 0.25) | **1.2830** | 1.4174 (half-bar map) | **1.3377** | 1.3920 (Adam from mean) |
| uniform | **1.4013** | 1.5090 (half-bar map) | **1.4716** | 1.5308 (Adam from the MiDaS mean-like map) |

When only the ssi term is trimmed, a single object wins at every $p$, by 5-15%. The MiDaS-style patchwork and mean-like maps, which win under 20% trimming with weight-0.5 GM (Sec. 3.4), lose clearly here (1.35-1.59, and 1.32-1.55 after Adam, against 0.86-1.40), because the weight-2 GM term still punishes their bent bars. If the GM residual is trimmed too, the conclusion fails at the two-way tie: a two-object mixture beats both objects by 3%. So near ties the commit prediction depends on an implementation detail we cannot check. (Under `dav2trimgm`, Adam started *from an object* is unstable and climbs to 1.4-2.2, so the objects' own values are the reference.) The Adam end values depend slightly on the number of torch threads. The logs use NT=3; with 1 or 2 threads the mixture ends at 1.2108 instead of 1.1962 and still beats both objects. As before, this is a structured search, not a certificate.

### 3.4 Other scale-invariant losses

**MiDaS ssitrim + 0.5 GM** (median/MAD, drop the 20% largest residuals, $\frac1{2M}\sum$ of the rest; arXiv:1907.01341). Object-only loss (check3b):

| $p$ | best single object | best map found | its split, span residual | mean-start ends at |
|---|---|---|---|---|
| (0.6, 0.3, 0.1) | 0.2960 (obj 1) | object 1 (0.2971 after Adam) | one-hot, 0.01 | 0.3147 |
| (0.4, 0.35, 0.25) | 0.4424 (obj 1) | object 1 (0.4416) | (0.97, 0.03, 0), 0.04 | 0.4443, split (0.44, 0.36, 0.20) |
| (0.45, 0.45, 0.1) | 0.4077 (obj 1) | **patchwork of objects 1 and 2** (0.3824) | (0.34, 0.49, 0.17), 0.52 | 0.3901 |
| uniform | 0.4856 (obj 3) | **mean-like map** (0.4834) | (0.333, 0.329, 0.339), 0.27 | same |

Trimming changes the picture. Throwing away the worst 20% of each target's residuals rewards maps that match each object *exactly* on a large part of the frame. So near a two-way tie the optimum stitches the two leading objects together, and at uniform $p$ a mean-like map beats every object. With a clear leader it still commits. There is no clean threshold rule, and the mean-start local minimum at (0.4, 0.35, 0.25) is only 0.6% worse than committing, so a real network could land on either.

**E2E-FT depth loss: least-squares scale and shift, then L1** (Martin Garcia et al., arXiv:2409.11355: "$`\mathcal L_D=\frac1{HW}\sum|d^*-\hat d|`$" after LS alignment). Object-only: a single object wins at every $p$ tested, and Adam started from the mean converges to a single object at (0.6, 0.3, 0.1), (0.4, 0.35, 0.25) and (0.45, 0.45, 0.1): to the winner at the first two, and at the tie to object 2 (1.1028, against object 1's 1.1018). The winning objects' losses are 0.8015, 1.2028 and 1.1018; the mean scores 1.44-1.97 and the best non-object half-bar map 1.18-1.54. At uniform $p$ the three objects tie (1.3350-1.3367), and Adam started from the mean stays at the mean (1.9245), a much worse local minimum.

**Shared far background (regime B, check6b, gap $T$):** both losses return the pointwise median. For $p=(0.6,0.3,0.1)$ that is object 1 (MiDaS 0.0425 vs mean 0.0459; LS-L1 0.4913 vs 0.5595). For $(0.4,0.35,0.25)$ and uniform $p$ it is the middle map with split (0.334, 0.333, 0.333) (MiDaS 0.0385 vs object 1 0.0632; LS-L1 0.5595 vs 0.7485), and Adam from the mean converges there, except for MiDaS at (0.4, 0.35, 0.25), where 150 Adam steps stop at 0.0419 with split (0.38, 0.35, 0.27).

**Scale-invariant log loss (ZoeDepth).** ZoeDepth's metric fine-tuning uses $\alpha\sqrt{Q}$, $Q=\operatorname{var}(g)+0.15\operatorname{mean}(g)^2$, $g=\log f-\log d$, $\alpha=10$ (isl-org/ZoeDepth, `zoedepth/trainers/loss.py`), that is $Q=\operatorname{mean}(g^2)-\lambda\operatorname{mean}(g)^2$ with $\lambda=0.85$ up to the $n/(n-1)$ of the sample variance. For $\lambda<1$, $\sqrt{Q}$ is a norm of $g$ (an unsquared norm, so like the L1 losses it commits under a majority), so for fixed positive targets the loss $\sum_kp_k\thinspace\alpha\sqrt{Q_k}$ is a weighted sum of distances from $\log f$ to the $\log d_k$. If one object has $p_k\ge 1/2$, the triangle inequality puts the minimum at that object. Away from the objects the loss is smooth, and averaging the pixel condition $\sum_kw_k(g_k-\lambda\bar g_k)=0$, with $w_k\propto p_k/\sqrt{Q_k}$ and $\bar g_k=\operatorname{mean}(g_k)$, over the pixels gives $(1-\lambda)\sum_kw_k\bar g_k=0$, so $\log f=\sum_kw_k\log d_k$. Either way the optimum is a weighted geometric mean of the objects, and its bars are close to the objects' slope when the depth range is small against the camera distance.

### 3.5 The normalisation regime matters as much as the loss

Everything in 3.2-3.4 so far normalises over the object alone (**regime A**). Real networks are trained on whole frames. Our stimulus has 29% foreground. If the model puts the background at one depth shared by all three hypotheses, the frame median lies in the background and every object pixel lies on one side of it (**regime B**). The MAD is then a *linear* functional of the object pixels, the pointwise median stays normalised, and Proposition 5 comes back exactly.

check6/check6b place each object with the same median depth in front of a wall at (max object depth + gap) and use the full 512×512 frame:

* **LS-SSI-L2** becomes the mean. The shared wall makes $\gamma$ large and positive ($\gamma=+35$ vs $c=2.4$ at gap $T$), and by Corollary 3b the split tends to $p$: (0.636, 0.279, 0.086), (0.606, 0.297, 0.098), (0.601, 0.299, 0.100) for $p=(0.6,0.3,0.1)$ at gap $0.1T,\ T,\ 3T$.
* **SSI-MAE**: the pointwise median has median 0.0000 and MAD 1.0000, so it is the certified optimum (loss equals the relaxed bound to $10^{-5}$) at every gap: object 1 for $p=(0.6,0.3,0.1)$, and the middle map with split (0.334, 0.333, 0.333) for every $p$ with no majority.
* **DAv2 loss (gap $T$)**: the structured search (objects, mean, pointwise median, 729 half-bar maps) picks the pointwise median every time. For $p=(0.6,0.3,0.1)$ that is object 1 (0.2320, vs mean 0.2825). For $(0.4,0.35,0.25)$ it is the middle map at 0.2998, ahead of the mean (0.3080) and object 1 (0.3458). For $(0.45,0.45,0.1)$: 0.3014, vs 0.3198 for object 1. For uniform $p$: 0.2993, vs 0.380-0.387 for the objects. So in regime B, **DAv2 averages when no object has a majority**. It is the GM term plus object-only normalisation that makes it commit. *With 10% trimming of the ssi term over the full frame* (check7b; that is 34% of the object's pixel count), the result is the same. For $p=(0.6,0.3,0.1)$ the pointwise median is object 1 (0.1791, vs mean 0.2186). For $(0.4,0.35,0.25)$ it is the middle map (0.2323, vs mean 0.2385 and object 1 0.2666), for $(0.45,0.45,0.1)$ 0.2339 (vs 0.2367 and 0.2471), and for uniform $p$ 0.2318 (vs 0.2320 and 0.2915-0.2986). Adam from the mean and from the pointwise median does not improve on it.
* **Gauge caveat.** When the three objects sit at different depths relative to the wall (the renderer gauge), the MADs differ (3.84, 4.68, 5.52) and the no-majority middle map has split (0.309, 0.333, 0.358). The LS-SSI-L2 split becomes (0.545, 0.328, 0.128). The commit case $p_k>\frac12$ is gauge-free. Nothing else is.

**Summary of Section 3 (and 2).**

| loss (model) | regime A: object fills the frame | regime B: object on a shared far background |
|---|---|---|
| L2, per-target normalisation (Lotus, Marigold targets) | mean: split $=p$ | mean: split $=p$ |
| L2, least-squares scale+shift (MiDaS ssimse) | top principal component: tear $>T$ at the likely corner, reversed tears elsewhere, flattened bars | $\approx$ mean (split within 0.04 of $p$) |
| L1 per pixel, fixed targets | commit if $p_k>\frac12$, else the fixed middle map (= mean under equal medians) | same |
| SSI-MAE, median/MAD (DAv1 $L_{ssi}$) | commit if $p_k>\frac12$ (exact); else a $p$-dependent patchwork with a full tear at the argmax corner | commit if $p_k>\frac12$, else middle map with $T/3$ tears (exact) |
| SSI-MAE + 2 GM (DAv2 teacher; students distilled from it) | commit to argmax $p$ for every $p$ tested, also with 10% ssi trimming; with GM trimmed too, a two-object mixture at a two-way tie | commit if $p_k>\frac12$, else middle map (also with 10% trimming) |
| MiDaS ssitrim + 0.5 GM | commit for a clear leader; a two-object patchwork near a two-way tie; a mean-like map at uniform $p$ | commit if $p_k>\frac12$, else middle map |
| LS alignment + L1 (Marigold E2E-FT) | commit to argmax $p$ (every $p$ tested) | commit if $p_k>\frac12$, else middle map |
| scale-invariant log loss (ZoeDepth) | the object if $p_k\ge\frac12$, else a weighted geometric mean of the objects | same |

---

## 4. Diffusion depth models

### 4a. The first step is a posterior mean given the noise, not given the image

**Proposition 8 (MMSE at every noise level).** Let $x_t=a_tz_0+\sigma_t\varepsilon$ with $a_t=\sqrt{\bar\alpha_t}$, $\sigma_t^2=1-\bar\alpha_t$, $\varepsilon\sim\mathcal N(0,I)$ independent of $(z_0,c)$, $c$ the image. If the network is trained by MSE on $\varepsilon$, on $z_0$ or on $v=a_t\varepsilon-\sigma_tz_0$, the population optimum satisfies
$$\hat z_0(x_t,t,c)=\mathbb E[z_0\mid x_t,c]=\frac{x_t+\sigma_t^2\nabla\log p_t(x_t\mid c)}{a_t}\quad\text{(Tweedie)}.$$
*Proof.* $z_0=(x_t-\sigma_t\varepsilon)/a_t$ and $z_0=a_tx_t-\sigma_tv$ (because $a_t^2+\sigma_t^2=1$) are affine in the regression target given $x_t$, so the MSE minimiser of each target maps to $\mathbb E[z_0\mid x_t,c]$. $\square$
This needs only MSE training, not Gaussian data. But it holds for the noise law used *in training*. Marigold trains with annealed multi-resolution noise, which is at full strength at $t=T$ (Ke et al., arXiv:2312.02145, App. A.2), and then samples with i.i.d. Gaussian noise. So its $t=T$ output is the posterior mean under a noise law it was not trained on, and Proposition 8 applies to it only approximately. We checked the law in the released training code (prs-eth/Marigold, `src/util/multi_res_noise.py`, config `strength: 0.9, annealed: true, downscale_strategy: original`, trainer `strength * t/1000`). On a 64×64 latent it adds, per channel, a second full-resolution $\mathcal N(0,I)$, a 16-31 px level (variance $s^2\approx0.81$ at $t=T$), a 1-7 px level ($s^4\approx0.65$) and a 1×1 level ($s^6\approx0.53$), and then divides by the global std. Its covariance $\Sigma_t$ has eigenvalues 0.62-820 at $t=T$; the Gaussian model matches the real function's variance along our directions to 1-6% (Monte Carlo, check11). The Bayes denoiser *for that law*, fed i.i.d. noise as diffusers does, has log-odds
$$\log\frac{\pi_k}{\pi_j}=\log\frac{p_k}{p_j}+\frac{a_T}{\sigma_T^2}\thinspace x_T^\top\Sigma_T^{-1}(z_k-z_j)-\frac{a_T^2}{2\sigma_T^2}\big(z_k^\top\Sigma_T^{-1}z_k-z_j^\top\Sigma_T^{-1}z_j\big),$$
with noise sd $\kappa^{\rm MR}_{jk}=\frac{a_T}{\sigma_T^2}\lVert\Sigma_T^{-1}(z_j-z_k)\rVert$. The two laws bracket what a real Marigold does on i.i.d. input. Which one it follows is not verified, and it is probably neither exactly.

**For the three-object posterior** ($z_0=z_k$ with probability $p_k$, where $z_k$ is object $k$ in whatever space the model denoises):
$$\mathbb E[z_0\mid x_T,c]=\sum_k\pi_k(x_T)\thinspace z_k,\qquad \log\frac{\pi_k}{\pi_j}=\log\frac{p_k}{p_j}+\frac{a_T}{\sigma_T^2}\langle x_T,z_k-z_j\rangle-\frac{a_T^2}{2\sigma_T^2}\big(\lVert z_k\rVert^2-\lVert z_j\rVert^2\big).$$
With $x_T\sim\mathcal N(0,I)$ the middle term is Gaussian with standard deviation
$$\kappa_{jk}=\frac{\sqrt{\bar\alpha_T}}{1-\bar\alpha_T}\thinspace\lVert z_j-z_k\rVert .$$
So the first-step prediction is a mixture of the three objects whose weights are $p$ **perturbed in log-odds by noise of size $\kappa$**. To first order, $\operatorname{sd}(\pi_k)\approx p_k\frac{a_T}{\sigma_T^2}\lVert z_k-\sum_jp_jz_j\rVert$. Simulation check: predicted 0.075, simulated 0.072 at $\kappa=0.34$.

* Stable Diffusion's scaled-linear schedule (Marigold v1-0: $\beta\in[0.00085,0.012]$, 1000 steps, read from the cached scheduler config) gives $\bar\alpha_{999}=0.00466$, $\sqrt{\bar\alpha_{999}}=0.0683$, $\mathrm{SNR}_T=0.0047$ (check4b). So $\kappa=1$ at $\lVert z_j-z_k\rVert=14.6$ latent units.
* The first step is $\approx\mathbb E[z_0\mid c]$ and seed-independent only if $\kappa\ll1$. At $\kappa\approx1$ it is already a noisy vote. In the exact simulations of Section 4c the final object equals the first-step argmax in 78-91% of seeds at $\kappa=1.0$ and in 94-98% at $\kappa=2.7$. **So the pilot's "the seed decides at step 1" is what an exact model does when $\kappa\gtrsim1$. On its own it is not evidence of a defect.**
* Tower property: $\mathbb E_{x_T\sim p_T}[\pi(x_T)]=p$ exactly, where $p_T$ is the true noisy marginal. Averaging the one-step output over seeds therefore estimates the mean. With $x_T\sim\mathcal N(0,I)$ this holds only approximately (Section 4c).

**Measured for Marigold v1-0 (check4a, check10, check11).** We encoded the three ground-truth maps the way Marigold normalises training depth: 2/98 percentiles over the whole frame to $[-1,1]$, replicated to 3 channels, the cached SD2 VAE, latent mean times 0.18215. A back wall is placed behind the object at a variable gap, and the gap sets how much of the depth range the object gets. **The pilot's own Marigold outputs are much flatter than an object that fills 55% or 106% of the range**: over all 32 per-seed maps the object spans 7-27% of the frame's 2/98 range (median 15%), and the per-cube profile only 3-15%. So $\kappa$ has to be quoted as a function of that share, and under both noise laws. All rows below use $p=(0.6,0.3,0.1)$, the exact denoiser, 20,000 seeds and trailing DDIM.

| object's share of $[-1,1]$ | $\lVert z_j-z_k\rVert$ | $\kappa$ i.i.d. law | $\kappa$ multi-res law | sd of 1-step $\pi_1$ (iid / MR) | P(final = 1-step argmax) (iid / MR) | valid at 1 step (iid / MR) |
|---|---|---|---|---|---|---|
| 6.7% (wall $15T$) | 20-21 | 1.39-1.47 | 1.12-1.23 | 0.25 / 0.22 | 0.85 / 0.82 | 0.001 / 0 |
| 12% (wall $7.8T$) | 27-28 | 1.85-1.94 | 1.32-1.41 | 0.29 / 0.24 | 0.90 / 0.85 | 0.009 / 0 |
| 26% (wall $3.2T$) | 37-40 | 2.54-2.71 | 1.60-1.74 | 0.34 / 0.28 | 0.94 / 0.90 | 0.044 / 0.002 |
| 55% (wall $T$) | 50-56 | 3.45-3.81 | 2.04-2.27 | 0.38 / 0.32 | 0.97 / 0.94 | 0.11 / 0.02 |
| 106% (wall at the object) | 66-81 | 4.53-5.52 | 2.56-3.10 | 0.41 / 0.35 | 0.98 / 0.97 | 0.23 / 0.11 |

At the flatness the pilot outputs actually have, $\kappa\approx1.4$-2.7 (i.i.d.) or 1.1-1.7 (multi-res). The values 3.5-5.5 in the table above need the object to fill 55-106% of the range, and nothing shows that Marigold does that.

**What survives.** $\kappa>1$ in every row. So in Marigold's latent space even a *perfect* model's first step is a noisy vote rather than the average: the seed-to-seed sd of the one-step weight is 0.2-0.4, and 82-98% of seeds end on the object their first step favoured. **The claim "one-step models average whatever the seed" (Sec. 0.2) is false for Stable Diffusion's schedule, even for a perfect model.** It is exactly true only for a model with no noise input (Lotus-D returns $\mathbb E[z\mid c]$). A noise-input model needs $\kappa\ll1$. Even $\kappa<1$ requires $\bar\alpha_T\lesssim(\sigma_T^2/\lVert z_j-z_k\rVert)^2$, i.e. $2.3\times10^{-3}$ at the flattest geometry and $1.5\times10^{-4}$ at the widest, below SD's 0.0047; a zero-terminal-SNR schedule (Lin et al., WACV 2024, arXiv:2305.08891) meets this. **Marigold v1.1 has one:** its scheduler config (prs-eth/marigold-depth-v1-1, `scheduler_config.json`) sets `rescale_betas_zero_snr: true`, `timestep_spacing: trailing` and `prediction_type: v_prediction`, so $\bar\alpha_T=0$, $x_T$ carries no information about $z$, and the exact model's one-step output is $\mathbb E[z\mid c]$ whatever the seed. Any seed dependence of v1.1's one-step output is therefore model error. v1.0 keeps SD2's scaled-linear schedule ($\bar\alpha_T=0.0047$, no rescaling, `steps_offset: 1`). **What does not survive:** the single numbers "$`\kappa\approx4`$, sd 0.4, 97-98% keep their choice, 10-25% valid at one step". They hold only for an object that fills the range and for the i.i.d. law. The statement "the tighter crop raises $\kappa$" is only a prediction. It holds if Marigold stretches a tightly cropped object over more of its output range, and that has to be measured (P5).

*Pilot material of the earlier framing, not used by the paper: the next two paragraphs, the last sentence on Lotus-G, and the last two sentences of the second bullet and of "What survives" above. The paper does not test P5 or P6 (Section 5).*

**A tension in the pilot.** The four seeds' one-step outputs on the impossible image are alike, with median-step splits (0.15, 0.45, 0.40), (-0.23, 0.57, 0.65), (-0.19, 0.69, 0.50), (0.17, 0.35, 0.48), and none is near a one-hot vote. At the flat geometries a vote with max weight $>0.9$ is uncommon anyway (3-28% of seeds, check11), so four seeds cannot refute the vote picture. But their small spread is also what a network that largely ignores $x_T$ would produce (effective $\kappa\ll1$). P5 decides it with 100 seeds.

Caveat: all of these are distances between *our* encoded objects. The model's own atoms may differ; P5 and P6 measure them from its samples.

**Lotus-G** is trained only at $t=999$ on the same SD2 schedule, with input $\sqrt{\bar\alpha_{999}}z+\sqrt{1-\bar\alpha_{999}}\varepsilon$ (from the Lotus code). Its Bayes-optimal output is therefore the same vote, $\sum_k\pi_k(z_T)z_k$. The argument "SNR 0.0047 per element, so $z_T$ carries almost no information and Lotus-G averages" overlooks that the log-odds add up over all 16,384 latent elements along $z_j-z_k$: for our objects $\kappa\approx1.4$-5.5 depending on how flat the object is (i.i.d. law, which Lotus-G is trained with if its code adds plain Gaussian noise; not verified), not $\approx0$. Whether the *trained* Lotus-G reads $z_T$ at all is an empirical question. On unambiguous training images it never needs to. The seed-to-seed spread of its split (P5) answers it.

### 4b. Latent diffusion: the mean lives in latent space

Proposition 8 gives $\mathbb E[z\mid x_T,c]$ for the VAE latent $z$. The depth the pipeline returns is $\mathrm{Dec}(\hat z)$, channel-averaged and affinely rescaled. Tears are linear in depth, so "one-step tears $=T\pi$" requires $\tau\circ\mathrm{Dec}$ to be affine on $\operatorname{conv}\lbrace z_k\rbrace$.

* **What can be claimed.** (i) For an exact model, the latent one-step output is a convex combination $\sum\pi_kz_k$ of the model's own three latent atoms. (ii) Its decoded tear split is $\phi(\pi)$, where the map $\phi$ is fixed by the decoder and the atoms and can be *measured* by decoding convex combinations of encoded ground truth. Measured (check4a): the VAE round trip keeps each object (split of $\mathrm{Dec}(z_k)$ is one-hot to 0.003). Decoding convex combinations gives a split within 0.04 of $w$ for the wall targets, e.g. $w=(0.6,0.3,0.1)\to(0.572,0.336,0.092)$ and uniform $\to(0.358,0.338,0.304)$, and within 0.06 for the tight targets (uniform $`\to(0.396,0.327,0.277)`$). At the pilot's flatness (object 12% of the range, check10b) it is within 0.02-0.06, e.g. $(0.5,0.5,0)\to(0.449,0.563,-0.012)$ and $(0.6,0.3,0.1)\to(0.575,0.345,0.081)$. These are biases for *our* encoded atoms; the model's own atoms may behave differently. So $\phi$ is close to the identity, stays within 0.02 of the simplex, and has a bias of at most 0.06. Decoding still does not commute with averaging: $|\mathrm{Dec}(\bar z)-\overline{\mathrm{Dec}(z)}|$ averages 0.03-0.09 over the foreground (2-6% of the object's depth range), and reaches 43-82% of the range near the corners. (iii) Seed-averaging should be done on latents before decoding ($`\mathrm{Dec}(\mathbb E\hat z)`$), or by projecting $\hat z$ onto the encoded objects. This is for estimating $\pi$ from the split. The paper's seed-averaged totals read the average of the decoded maps (`scripts/run_marigold.py`), which differs from $\mathrm{Dec}(\mathbb E\hat z)$ by the non-commutation above.
* **What cannot be claimed.** That the model's atoms are our encoded ground truths. The model normalises depth over the whole frame using its own guess of the background, it may flatten the object, and its VAE round trip is imperfect. So a negative split entry in a decoded one-step output does not by itself violate Tweedie. If a latent-space projection leaves a large residual, the model's atoms differ from ours.

### 4c. Deterministic sampling: what "q versus the one-step split" can show

**Proposition 9 (exact probability flow).** With the exact score, the probability-flow ODE (Song et al., ICLR 2021, arXiv:2011.13456) transports $p_T$ to $p_\epsilon$ for every $\epsilon>0$. DDIM with $\eta=0$ (Song, Meng, Ermon, ICLR 2021, arXiv:2010.02502) is its first-order discretisation. So a sampler started from $x_T\sim p_T$ ends near $z_k$ with probability $p_k$. For a discrete posterior, both $\pi$ and the DDIM update depend on $x$ only through $Z^\top x$. The basins are therefore cylinders over the 3-d span of the atoms, and everything depends on the atoms only through their Gram matrix, which is why the simulation below is exact. A sampler started from $x_T\sim\mathcal N(0,I)$ has $q_k=\mathcal N(0,I)(B_k)$, and $|q_k-p_k|\le\mathrm{TV}\big(\mathcal N(0,I),p_T\big)$ restricted to that span. $\square$
**This bound is vacuous for Marigold.** The atoms share a large common component ($\lVert z_k\rVert\approx167$), so $p_T$'s mean lies $a_T\lVert\bar z\rVert/\sigma_T=11.2$ noise-sd from the origin along it. The TV distance is then $1-10^{-8}$ (check12). In general the bound is useless whenever $\bar\alpha_T\lVert\bar z\rVert^2\gg1$. How far $q$ actually drifts is a property of the flow, and we only know it by simulation.

Simulation (check4b; exact denoiser, 20,000 seeds, SD schedule, trailing spacing):

$p=(0.6,0.3,0.1)$, start $x_T\sim\mathcal N(0,I)$ unless noted. The "ideal" rows use equilateral atoms that share a large common component; the Marigold rows use the Gram matrix of our encoded objects.

| geometry | $\kappa$ | seed-mean 1-step weights | sd of 1-step weight on obj 1 | $q$ at 50 steps | valid at 1 / 4 / 10 / 50 steps | P(final = 1-step argmax) |
|---|---|---|---|---|---|---|
| ideal | 0.07 | (0.600, 0.300, 0.100) | 0.015 | (0.598, 0.300, 0.102) | 0 / 0 / 0.29 / 0.98 | 0.60 |
| ideal | 0.34 | (0.595, 0.303, 0.102) | 0.072 | (0.579, 0.309, 0.112) | 0 / 0.90 / 1 / 1 | 0.60 |
| ideal | 1.03 | (0.565, 0.316, 0.118) | 0.193 | (0.538, 0.324, 0.138) | 0 / 1 / 1 / 1 | 0.78 |
| ideal | 2.74 | (0.482, 0.337, 0.181) | 0.343 | (0.466, 0.340, 0.194) | 0.04 / 1 / 1 / 1 | 0.94 |
| Marigold latent, wall (55%), i.i.d. law | 3.5-3.8 | (0.455, 0.354, 0.191) | 0.382 | (0.445, 0.357, 0.198) | 0.11 / 1 / 1 / 1 | 0.97 |
| Marigold latent, tight (106%), i.i.d. law | 4.5-5.5 | (0.388, 0.388, 0.224) | 0.408 | (0.380, 0.391, 0.229) | 0.23 / 1 / 1 / 1 | 0.98 |
| Marigold latent, wall, **start from $p_T$** | 3.5-3.8 | (0.598, 0.303, 0.099) | 0.463 | (0.597, 0.304, 0.099) | 0.65 / 1 / 1 / 1 | 0.99 |

The same exact-denoiser DDIM under both noise laws and at the pilot's flatness (check11, 50 trailing steps, $x_T\sim\mathcal N(0,I)$; $s$ = seed-mean one-step weights):

| geometry (object share) | law | $s$, $p=(0.6,0.3,0.1)$ | $q$ | $\max\lvert q-s\rvert$ | $s$, $p=(0.4,0.35,0.25)$ | $q$ | $\max\lvert q-s\rvert$ |
|---|---|---|---|---|---|---|---|
| 6.7% | iid | (0.580, 0.298, 0.122) | (0.574, 0.292, 0.134) | 0.012 | (0.422, 0.325, 0.253) | (0.437, 0.310, 0.253) | 0.015 |
| 6.7% | MR | (0.552, 0.315, 0.133) | (0.541, 0.310, 0.149) | 0.016 | (0.383, 0.339, 0.278) | (0.379, 0.330, 0.291) | 0.013 |
| 12% | iid | (0.557, 0.307, 0.136) | (0.549, 0.304, 0.147) | 0.011 | (0.418, 0.325, 0.257) | (0.428, 0.317, 0.255) | 0.010 |
| 12% | MR | (0.539, 0.323, 0.138) | (0.530, 0.321, 0.149) | 0.011 | (0.378, 0.343, 0.279) | (0.373, 0.337, 0.290) | 0.011 |
| 26% | iid | (0.518, 0.323, 0.159) | (0.510, 0.323, 0.167) | 0.008 | (0.403, 0.332, 0.264) | (0.406, 0.329, 0.264) | 0.003 |
| 26% | MR | (0.531, 0.329, 0.140) | (0.531, 0.329, 0.140) | 0.001 | (0.386, 0.345, 0.270) | (0.395, 0.342, 0.262) | 0.009 |
| 55% | iid | (0.456, 0.355, 0.189) | (0.447, 0.358, 0.195) | 0.009 | (0.368, 0.356, 0.277) | (0.367, 0.355, 0.277) | 0.001 |
| 55% | MR | (0.546, 0.324, 0.131) | (0.558, 0.322, 0.120) | 0.012 | (0.421, 0.341, 0.238) | (0.438, 0.340, 0.221) | 0.018 |
| 106% | iid | (0.390, 0.389, 0.221) | (0.386, 0.391, 0.223) | 0.005 | (0.328, 0.381, 0.291) | (0.325, 0.383, 0.293) | 0.003 |
| 106% | MR | (0.605, 0.297, 0.098) | (0.628, 0.288, 0.084) | 0.023 | (0.511, 0.317, 0.173) | (0.537, 0.311, 0.152) | 0.026 |

(With $\Sigma_t=I$, check11 reproduces check4b's i.i.d. numbers to within Monte-Carlo error.)

For $p=(0.4,0.35,0.25)$ the Marigold rows give seed-mean weights (0.366, 0.355, 0.279) and $q=(0.361, 0.358, 0.281)$ (wall), and (0.326, 0.380, 0.294) and $q=(0.322, 0.382, 0.296)$ (tight). Uniform $p$ gives $q$ within 0.01 of uniform for the ideal atoms and from $p_T$, and $q=(0.334, 0.348, 0.317)$ (wall) and $(0.304, 0.373, 0.323)$ (tight) from $\mathcal N(0,I)$. Started from $p_T$, $q=p$ to within 0.01 in every geometry: the flow itself is exact, and all the drift comes from the start distribution. Under the i.i.d. law, in the tight geometry, the order of objects 1 and 2 even flips ($q_2>q_1$ although $p_1=2p_2$). **This flip, and most of the drift toward uniform, are artefacts of the i.i.d. law**. Under the multi-resolution law the same atoms give $s=(0.605,0.297,0.098)$ and $q=(0.628,0.288,0.084)$, close to $p$. At the pilot's flatness both laws give $q_1\approx0.51$-0.57 for $p_1=0.6$. So how far $q$ drifts from $p$ depends on the geometry and on the noise law; it is not set by $\kappa$ alone.

**What the within-model test can show.** For every geometry built from our encoded objects, under both noise laws, $q$ (the frequencies of valid 50-step samples) and $s$ (the seed-averaged one-step weights) agree to within 0.001-0.026, even when both drift from $p$. **This is not a property of exact samplers in general** (check12). Take 80 random Marigold-like atom sets with $\lVert z_k\rVert=167\pm0.6$ (the measured spread is 0.2-1.15), pairwise $\kappa$ 0.2-18.7, and $p\sim$ Dirichlet(2,2,2). Then $\max\lvert q-s\rvert$ has median 0.017, 90th percentile 0.10 and maximum 0.39, and it exceeds 0.03 in 33% of cases: 88% of those with $\min\kappa<1$, 19% of those with $\min\kappa\ge1$. The worst case, $p=(0.32,0.32,0.35)$, has $s=(0.24,0.23,0.52)$ but $q=(0.015,0.07,0.92)$. Going from 50 to 1000 steps changes nothing, so this is not discretisation. The mechanism: $x_T\sim\mathcal N(0,I)$ lacks the component $a_T\bar z$ that $p_T$ has along the atoms' common direction (11 noise-sd), and norm differences of <1% decide where the flow sends that mass at later $t$. Equalising the norms brings the worst case back to 0.02, but 28% of the equalised cases still exceed 0.03 (max 0.08). Agreement between $q$ and $s$ is therefore a consequence of the geometry. It is not a calibration identity, and a fixed budget such as "$`\lvert q-s\rvert\le0.03`$" would reject a perfect sampler for some atom geometries. The usable test (P6) estimates the model's own atoms from its samples, simulates $q$ and $s$ with the exact denoiser for that geometry under both noise laws, and compares the measured $q$ with the simulated $q$. It cannot show that $p$ is "right", because an impossible image has no ground-truth $p$. And agreement does not prove exact sampling, since both sides can be biased the same way.

**Failure modes, each with a concrete signature.**
1. *Imperfect score.* If the denoisers at different $t$ are not posterior means of one joint distribution, the measured $q$ departs from the $q$ an exact sampler would give for the same atoms (in our geometries, that $q$ is within 0.03 of the seed-averaged one-step weights; in general it need not be, see above). Detecting this is the purpose of the test, but the test cannot say at which $t$ the model is wrong.
2. *Too few steps.* With the exact denoiser and a start from $\mathcal N(0,I)$, 1 step leaves 0% of samples valid (a single dominant tear) for $\kappa\le1.03$ and 2-23% for $\kappa=2.7$-5.5, and 4 trailing steps leave 0% ($\kappa=0.07$), 88-90% ($\kappa=0.34$) or $\approx100$% ($\kappa\ge1$) valid. Classifying non-valid outputs by their largest tear biases $q$ toward the majority: at small $\kappa$ a single step gives $q=(1,0,0)$ for $p=(0.6,0.3,0.1)$. Report validity and use only step counts where it is $\approx1$ (for the exact model, $S\ge10$ when $\kappa\gtrsim0.3$ and $S\approx50$ when $\kappa\approx0.07$).
3. *Non-zero terminal SNR* ($\bar\alpha_T=0.0047$). Starting from $\mathcal N(0,I)$ instead of $p_T$ flattens $q$ toward uniform by an amount set by $\kappa$. For $p_1=0.6$: $q_1=0.598,\ 0.579,\ 0.538,\ 0.466$ at $\kappa=0.07,\ 0.34,\ 1.03,\ 2.74$. The seed-averaged one-step weights flatten almost the same way ($0.600,\ 0.595,\ 0.565,\ 0.482$).
4. *Timestep spacing.* "Leading" spacing with `steps_offset=1` evaluates a single step at $t=1$. The exact denoiser then returns the atom nearest the noise (a Voronoi vote), so $q$ does not depend on $p$ at all. It is set by the atoms' geometry: $(0.33,0.34,0.33)$ for the symmetric ideal atoms, but from $(0.005,0.84,0.16)$ (object at 106% of the range) to $(0.95,0.02,0.03)$ (6.7%) for our encoded objects (check13). A real network returns noise (Martin Garcia et al., arXiv:2409.11355). With 4 steps, leading spacing starts at $t=751$ on $\mathcal N(0,I)$ input. $q$ then depends on both $p$ and the geometry, e.g. $(0.665,0.279,0.056)$ (ideal, $\lVert\Delta z\rVert=1$) or $(0.357,0.392,0.251)$ (encoded, 55%) for $p=(0.6,0.3,0.1)$. Trailing spacing is required.
5. *v- versus $`\varepsilon`$-prediction.* Irrelevant for an exact model. For a real one, $\varepsilon$-prediction at $t=T$ gives $\hat z_0=(x_T-\sigma_T\hat\varepsilon)/a_T$, which amplifies the network's error by $\sigma_T/a_T=14.6$. v-prediction (Marigold v1-0) uses $\hat z_0=a_Tx_T-\sigma_T\hat v$, which does not. The one-step output of an $\varepsilon$-model is therefore not a usable estimate of the mean.
6. *Training noise law* (multi-resolution noise, Section 4a): under the Gaussian approximation it lowers $\kappa$ by 15-45% and changes both $s$ and $q$ by up to 0.24 (tight geometry). Every simulated prediction is therefore given for both laws. *Latent decoding*: Section 4b.
7. *Ensembling.* By Proposition 5, a per-pixel median over samples returns object $k$ if $q_k>\frac12$ and otherwise the middle map with $T/3$ tears. In the installed diffusers 0.40.0, `MarigoldDepthPipeline.ensemble_depth` defaults to `reduction="median"` (checked in the source). The default ensemble therefore reintroduces the L1 behaviour, so the test must use per-seed outputs (`ensemble_size=1`, as in the pilot).

---

## 5. Theory box and testable predictions

*This section was written for the earlier framing (whether one step averages and many steps choose). The paper does not test P1-P9 as stated; it reads every network with the excess tear instead (Section 7).*

### 5.1 Theory box of the earlier framing (superseded; the paper's version is Section 0)

> **Theory.** A Penrose image is the exact image of three rigid objects $d_1,d_2,d_3$ that differ only in where the loop of bars is cut. Along the loop each object is a ramp that falls by $T$, closed by one jump of $T$ at its own corner. So every depth map whose bars keep the rigid slope has a tear vector $\tau$ with $\sum_j\tau_j=T$, and a valid object is $\tau=Te_k$. If a predictor puts posterior weights $p$ on the three objects, its training loss decides what it returns. Squared error against a target normalised independently of the prediction returns the posterior mean, $\tau=Tp$: partial tears that read out $p$. Per-pixel absolute error returns the median: object $k$ if $p_k>\frac12$, and otherwise one fixed map with $\tau=\frac T3(1,1,1)$ whatever $p$ is. Scale-and-shift-invariant losses keep these answers only while a shared background dominates the normalisation. When the object fills the frame, least-squares alignment turns the mean into a principal component, and the median/MAD loss with a gradient term that Depth Anything V2's teacher is trained with commits to the most likely object for every $p$ we tested. A diffusion model's first prediction from noise is $\mathbb E[z\mid x_T,\text{image}]$: the three objects mixed with weights $p$ perturbed in log-odds by noise of size $\kappa=\sqrt{\bar\alpha_T}\thinspace\lVert z_j-z_k\rVert/(1-\bar\alpha_T)$ (with a $\Sigma^{-1}$-weighted norm under Marigold's multi-resolution training noise). For Marigold's latents $\kappa$ lies between about 1 and 5, depending on how much of the depth range the object gets and on the noise law. It exceeds 1 in every case, so even a perfect model's first step already votes. For the atom geometries we can compute, an exact deterministic sampler picks object $k$ with a frequency close to (within 0.03 of) the seed average of its first step. That is a property of these geometries, not an identity, and neither number equals $p$ unless the terminal SNR is zero.

### 5.2 Testable predictions

Tolerances come from the one-answer control. Every read-out is compared with the same model's output on the possible control, never with the ideal 1 and 0 (Section 6). The span diagnostic's split/ramp/residual triple is used only when the model's control residual is $\le0.1$ (Section 1); above that, only one-hot vs not, inside vs outside the simplex, and in-bar steps relative to the control are interpreted. For the diffusion predictions (P5-P8) the model's own atoms are estimated from its samples, and every reference number is simulated for those atoms under **both** noise laws (Section 4a). No fixed number such as "sd 0.4" is used.

| # | Prediction | Model(s) | Exact measured quantity | Falsified if |
|---|---|---|---|---|
| P1 | An L2 model averages: its split can be **any** point of the simplex, and it moves smoothly with the image | Lotus-D (latent L2, per-target normalisation, no noise input) | Tear split at 36 rotation angles, the same on the one-answer control; span-diagnostic triple only if the control residual is $\le0.1$ | The split is within the control's error of one-hot at most angles (commitment), or leaves the simplex by more than the control's error plus the decoder bias (0.06, Sec. 4b). If the control residual is $>0.1$, "mean" cannot be told from "patchwork" (Sec. 1), and P1 then only tests one-hot vs interior |
| P2 | DAv2's split is **predicted** to sit near one of four values: $e_1,e_2,e_3$ (in any regime) or $(\frac13,\frac13,\frac13)$ (shared background, no majority). Two-corner splits such as (0.6, 0.4, 0) are not predicted by the teacher's loss or by the student-style loss with 10% trimming of the ssi term (check7). If trimming also acts on the GM term, a two-object mixture wins at a two-way tie (check7), and so do MiDaS-style 20% trimming patchworks (Sec. 3.4). **So two-corner splits are tolerated near ties**, i.e. within one angle step of a switch between committed corners. DAv1's object-only loss predicts a patchwork (Sec. 3.2), testable only through in-bar steps beyond the control's | DAv2 S/B/L | Distance of the split to the nearest of the 4 points, against its distance to the nearest simplex point, over all angles; in-bar step statistic vs the control | Splits far from all four points (beyond control error) at many angles *away from* corner switches |
| P3 | Abrupt versus smooth: along a continuous nuisance (rotation in 5° steps, or a morph toward the possible control of object $k$), the L1 model's split **jumps** between allowed points and the L2 model's split **slides** | DAv2 vs Lotus-D | Split against angle or morph level; the largest change between neighbouring settings | DAv2 changes as smoothly as Lotus-D |
| P4 | Crop regime: on a tight crop (object fills the frame) DAv2 is **predicted** not to show three partial tears. On a small object against a far background it may show $\frac13$ tears, but only when no object has a majority. This also holds with 10% ssi trimming (check7, check7b) | DAv2 | Split at object frame fractions 20%, 30%, 50%, 90% (pad or crop the same render) | Three partial tears (beyond control error) on most tight-crop renders |
| P5 | The one-step output of a noise-input diffusion model depends on the seed. Only its **seed average** estimates the mean | Marigold v1-0 (trailing, $t=999$), Lotus-G | Seed-to-seed sd of the one-step weights over 100 seeds. The weights come from projecting each one-step latent onto the model's atoms $\hat z_k$ (P6). The reference sd is simulated with the exact denoiser for the Gram matrix of $\hat z_k$ (i.i.d. law) and for $\hat z_k^\top\Sigma_t^{-1}\hat z_l$ (multi-res law). For our encoded objects it is 0.22-0.41, and 0.22-0.34 at the pilot's flatness | The measured sd lies clearly outside the interval spanned by the two simulations. Below both: the model ignores its noise (effective $\kappa\ll1$), which the pilot's four similar one-step outputs hint at. Above both: noisier than any Bayes model |
| P6 | Within-model calibration, **in latent space**: an exact sampler with the model's own atoms and its own first-step weights predicts $q$ | Marigold v1-0 | (i) *Atoms*: k-means ($k=3$, started from the three encoded GT latents) on the final 50-step latents of $\ge100$ seeds (400 for $\pm0.045$); nearest encoded GT object as a cross-check. (ii) *Validity* of each sample: distance to its nearest atom divided by the distance between that atom and the next nearest, valid if $<0.25$ (pre-registered; sensitivity reported for 0.15-0.35). (iii) $q_k$ = share of samples nearest to $\hat z_k$. (iv) $s$ = weights, constrained to sum to 1, that best regress $\operatorname{mean}_{\text{seeds}}\hat z_0^{(1)}$ onto $\lbrace\hat z_k\rbrace$, reported with the projection residual. (v) Fit $p$ so that the exact-denoiser simulation for these atoms reproduces $s$, under each noise law, then predict $q$. The decoded tear read-out is a secondary check only | Measured $q$ differs from the predicted $q$ under both laws by more than the binomial error ($\pm0.09$ at 100 seeds, $q=0.3$). Pre-registered: if fewer than half the samples are valid, $q$ uses all samples labelled by their largest projection weight, the test is reported as weak, and low validity is the finding (P8). $\lvert q-s\rvert$ is reported only as a description, with no fixed budget (Sec. 4c) |
| P7 | The seed decides at step 1 even for a perfect model | Marigold v1-0 | Agreement rate between the 1-step and 50-step choice per seed (choices = nearest atom / largest projection weight) | Rate near chance (1/3 to $p_{\max}$). Reference from the simulation for the measured atoms: 0.82-0.98 for our encoded objects, 0.82-0.94 at the pilot's flatness |
| P8 | Validity grows with steps: an exact model has few valid samples at 1 step (0-23% for our encoded objects, $\le4$% at the pilot's flatness), $\approx100$% from 4 steps on when $\kappa\ge1$, and $q$ is stable (±0.01) from 4 steps on | Marigold v1-0 | Share of valid samples (P6 definition) and $q$ for $S\in\lbrace 1,2,4,10,25,50\rbrace$ | Validity stays low at 25-50 steps (the pilot hints at this). That would mean the samples are not draws from a three-object posterior, a real finding rather than a step-count artefact |
| P9 | The default median ensemble returns object $k$ if $q_k>\frac12$ and otherwise $\approx\frac13$ tears | Marigold with `ensemble_size`$=E$ | Split of the ensembled output for $E=1,5,10,20$ | Ensemble split tracks $q$ continuously |

---

## 6. The pilot outputs through this lens (check5)

*Pilot material of the earlier framing. The pilot's network outputs are not in the repository; check5 and check9 read their loop profiles from `theory/pilot_compat/pred_profiles.json`, which `pilot_compat/make_pred_profiles.py` made from those outputs.*

The span diagnostic (Section 1), applied to every pilot prediction on `texture_a90` and its one-answer control. $w$ comes from fitting the 15-cube profile with the three object profiles and a constant.

| output | split $w/\sum w$ | ramp share | profile residual |
|---|---|---|---|
| DAv2-L, impossible | (1.73, -0.12, -0.61) | 0.41 | 0.52 |
| DAv2-L, possible control | (1.31, -0.14, -0.17) | 0.62 | 0.21 |
| DAv2-B, impossible | (1.56, -0.93, 0.36) | 0.35 | 0.56 |
| DAv2-B, possible control | (1.26, -0.22, -0.04) | 0.65 | 0.27 |
| Marigold 1 step, impossible, seeds 0-3 | e.g. (-0.74, 0.75, 0.99), (-0.96, 0.75, 1.20) | 0.34-0.77 | 0.46-0.70 |
| Marigold 1 step, control, seeds 0-3 | e.g. (0.46, 0.22, 0.32), (0.43, 0.27, 0.29) | 1.00 | 0.73-0.90 |
| Marigold 25 steps, impossible, seeds 1-2 | (-1.80, 1.15, 1.65), (-2.98, 2.54, 1.45) | 0.14-0.22 | 0.23-0.45 |

What this says:

1. **No real output is in the span, not even on the one-answer control** (residual 0.21-0.30 for DAv2 S/B/L on the control). The models bend the bars. One likely reason is that DAv2 predicts affine-invariant *inverse* depth, and negative disparity is affine in depth only for a distant camera. So every tear number must be read **relative to the control**, never against the ideal values 1 and 0. The pilot's median-step read-out (split (0.98, 0.19, -0.17) for DAv2-L) and the span fit (1.73, -0.12, -0.61) disagree for the same reason.
2. DAv2-L on the impossible image looks like DAv2-L on the control of object 1: dominant weight on corner 1, ramp share 0.4-0.6. This is the commit behaviour of Theorem 7(a) / Section 3.3, as the pilot said. This is a coarse reading ("one dominant corner, like the control"), which Section 1's rule allows at this residual (0.52 vs the control's 0.21). The finer label "object rather than principal component" is *not* supported at this residual.
3. Marigold's one-step outputs on the *control* have profile residual 0.73-0.90. The span fit explains only about 19-47% of their profile variance, so their split and ramp share are mostly fitting error and **cannot be read**. The same signature (ramp +1.00, split (0.03, 0.36, 0.61), residual 0.76) also appears on a committed but wrong *final* sample: control seed 1 at 25 steps. So it does not pick out a posterior mean. To the extent they can be read at all, four seeds giving similar broad mixtures rather than votes would mean a small *effective* $\kappa$. That is in tension with Section 4a, where an exact model has $\kappa>1$ at every flatness we tried; and on a one-answer control an exact model returns $z_1$ whatever the seed. P5 decides this. On the *impossible* image the one-step outputs (residual 0.46-0.70) have a negative split entry and ramp share 0.34-0.77, so they are not convex mixtures of our objects. The candidates are decoder nonlinearity, a flattened object whose atoms differ from ours (the far background eats the depth range), or model error. Only the latent-space projection onto the model's own atoms (Section 4b(iii), P6) can separate them. Profile-level splits of Marigold outputs are reported only together with their residual and only when that residual is near the level of a committed model's control.

---

## 7. The excess tear of every optimum (check14)

**The read-out.** The paper's read-out (`pdepth/readout.py`, `pdepth/loop.py`) takes each cube's median depth over pixels at least 3 px inside it, fits a line to each bar's four interior cubes (slope $b_j$ per cube step) and defines the jump $J_j$ at joint $j$ as the gap between the previous bar's line and bar $j$'s line there. The corner cubes are not used. For every depth map $\sum_jJ_j=-5\sum_jb_j$ (here $n-1=5$), so the steps around the loop always sum to zero. The excess tear is
$$e_j=\frac{J_j-r_j}{g_j-r_j},$$
where $r_j$ is the step at joint $j$ when it is an ordinary joint (averaged over the two objects whose gap is elsewhere; in the paper, over the two possible controls) and $g_j$ the step there when it is the gap. check14 takes $r_j$ and $g_j$ from the exact objects: $r=(0.010,0.001,0.001)$ and $g=(8.670,8.661,8.661)$, against $T=8.660$. It also divides each bar slope by the objects' common slope ($-0.578$ per cube step): $\beta_j=1$ recedes like an object, 0 is flat, $-1$ is reversed.

### 7.1 Maps built from the objects

Let $m=\sum_kc_kd_k+b$ with any real $c_k$ and $C=\sum_kc_k$. On every bar the objects differ by constants (Lemma 1), so the cube medians of $m$ are the same combination of the objects' cube medians, and $b_j$ and $J_j$ are linear in $c$: $b_j(m)=C\thinspace b_{\rm obj}$ on every bar, and $J_j(m)=c_jg_j+\sum_{k\ne j}c_kr_j^{(k)}$. When the two objects whose gap is not at $j$ get equal weights, or when their ordinary steps are equal (true here to $10^{-4}$),
$$e_j=c_j+(C-1)\thinspace\rho_j,\qquad \rho_j=\frac{r_j}{g_j-r_j}.$$
In ground-truth units $\rho_j\le0.0012$, so $e_j=c_j$, $\sum_je_j=C$ and $\beta_j=C$ on all three bars. check14 (1)-(3) and (7):

| map | $C$ | $e$ | $\sum_je_j$ | $\beta$ |
|---|---|---|---|---|
| object $k$ | 1 | one-hot | 1.000 | 1, 1, 1 |
| mean $\sum_kp_kd_k$ (Prop. 1), five values of $p$ | 1 | $p$ | 1.000 | 1, 1, 1 |
| middle of the median-centred objects (Prop. 5, Lemma 6; the paper's Prop. 2) | 1 | (0.334, 0.333, 0.333) | 1.000 | 1, 1, 1 |
| middle of the objects as rendered (= object 2) | 1 | (0, 1, 0) | 1.000 | 1, 1, 1 |
| $w\thinspace\mathrm{mean}(p)-(1-w)\thinspace\mathrm{mean}(q)$: the objects and their depth reversals | $2w-1$ | $wp-(1-w)q$ | $+0.50,+0.20,0.00,-0.20,-0.50,-1.00$ at $w=0.75,0.6,0.5,0.4,0.25,0$ | $2w-1$ on every bar |

So a posterior mean over the objects and their depth reversals has $\sum_je_j=P(\text{up})-P(\text{rev})$ and one common slope, $2w-1$ times the objects', on all three bars. Its per-joint excesses can have either sign: with $p$ uniform, $q$ = object 1 and $w=0.5$, $e=(-0.33,+0.17,+0.17)$, $\sum_je_j=0$ and the bars are flat.

### 7.2 The output scale

$e_j$ is unchanged by a shift of each output and by a scale (of either sign) shared by an image and its references: for the mean at $p=(0.5,0.3,0.2)$, $e=(0.5,0.3,0.2)$ both ways. Scaling the image alone by $a$ multiplies $e$ by $a$ (here, where $r\approx0$): $(1.00,0.60,0.40)$ for $a=2$. The scale-invariant losses (median/MAD normalisation, least-squares alignment) fix their output only up to a positive scale (least squares: any nonzero scale), so for them $\sum_je_j=1$ is not implied. A positive scale keeps the sign of $\sum_je_j$ and the ratios of the bar slopes.

At the objects' MAD, which is where a median/MAD loss compares maps, the flatter maps are stretched. The mean reads $\sum_je_j=1.99,\ 1.80,\ 2.26,\ 2.91,\ 3.06$ at $p=(0.6,0.3,0.1),\ (0.5,0.5,0),\ (0.45,0.45,0.1),\ (0.4,0.35,0.25)$ and uniform, and the middle map 3.06, with equal slopes on the three bars. The objects stay at 1.

The least-squares-aligned squared error returns the principal component $\sum_kw_kd_k$ of Theorem 3, with any nonzero scale. At the objects' MAD it reads $\sum_je_j=+0.63$ at $p=(0.6,0.3,0.1)$ and $+0.17$ at $(0.4,0.35,0.25)$, with equal slopes (0.62 and 0.17) on the three bars. At the two-way tie $(0.45,0.45,0.1)$ it reads $e=(-0.75,+0.75,0.00)$, and at uniform $p$ $e=(-0.26,+0.75,-0.49)$: $\sum_je_j=0.00$ with flat bars, whatever the scale and sign.

### 7.3 Optima that are not built from the objects

Regime A (the loss normalises over the object alone): the best maps found by check3b and check7 and the SSI-MAE patchworks of check3, read at the objects' MAD (check14 (5) and (6)).

| loss | $p$ | best map found | its loss vs the best object | $\beta$ | $\sum_je_j$ |
|---|---|---|---|---|---|
| MiDaS trimmed + 0.5 GM (DPT-L) | (0.45, 0.45, 0.1) | objects 1 and 2 stitched together | 0.3824 vs 0.4077 | 4.01, 2.54, 1.10 | 2.55 |
| same | uniform | mean-like map | 0.4834 vs 0.4856 | 2.95, 2.95, 3.31 | 3.08 |
| same | (0.4, 0.35, 0.25) | object 1, slightly bent | 0.4416 vs 0.4424 | 1.00, 1.07, 0.85 | 0.97 |
| DAv2 with the GM residual also trimmed | (0.45, 0.45, 0.1) | mixture of objects 1 and 2 (Adam from the mean) | 1.1962 vs 1.2323 | 2.74, 1.75, 1.36 | 1.95 |
| SSI-MAE (DAv1's $L_{ssi}$) | (0.45, 0.45, 0.1) | patchwork (Theorem 7) | below the objects (Sec. 3.2) | 3.27, 2.05, 2.05 | 2.46 |
| same | (0.4, 0.35, 0.25) | patchwork | below the objects | 3.06, 1.04, 2.50 | 2.20 |
| same | uniform | patchwork | below the objects | 2.51, 4.19, 1.48 | 2.73 |

For DAv2 (untrimmed, or with only the ssi term trimmed), for E2E-FT's least-squares + L1 loss and for MiDaS with a clear leader, the best map found at every $p$ tested is a single object ($\sum_je_j=1$, $\beta=1$). Every Adam end in check3b and check7, including the local minima a finite network might land in, keeps all three bars receding ($\beta$ between 0.84 and 4.1) and reads $\sum_je_j$ between 0.97 and 3.09 at the objects' MAD. In regime B the normalisation is set by the shared background, so the optima of Section 3.5 (an object or the middle map) keep the objects' scale and read $\sum_je_j=1$ (by 7.1; not run in check14).

### 7.4 In a network's own units

In the paper $r_j$ and $g_j$ come from each network's own possible controls, and there $r_j$ is not zero: every network steps toward the viewer at real joints. The median of $r_j/g_j$ over the 16 control layouts is $-0.06$ (DA-L), $-0.11$ (DA-B), $-0.19$ (DA-S), $-0.25$ (Lotus-D, Marigold v1.1), $-0.29$ (DPT-L), $-0.31$ (E2E-FT), $-0.42$ (Lotus-G) and $-0.43$ (ZoeDepth, Marigold v1.0) (check14 (8), from `results/`, set B, seed 0). The formula of 7.1 holds for combinations of a network's control profiles, with these $\rho_j$. `scripts/paper_numbers.py` prints $-\rho_j$ under the name `rho_j`. Three consequences:

1. $e=0$ means that the network makes its ordinary step at every joint, not that the map is continuous. A profile with no jump at any joint reads $\sum_je_j=-\sum_j\rho_j$, from $+0.21$ (DA-L) to $+1.29$ (DPT-L).
2. The value $-1$ for a reversed mean does not carry over. The negated average of a network's own controls reads $\sum_je_j=-1-2\sum_j\rho_j$: $+0.00$ (DA-S), $-0.28$ (DA-B), $-0.57$ (DA-L), $+1.57$ (DPT-L), $+1.17$ (ZoeDepth), $+0.01$ (Lotus-D), $+0.63$ (Lotus-G, E2E-FT), $+0.79$ (Marigold v1.0), $+0.28$ (Marigold v1.1). An even mix of upright and reversed reads $-\sum_j\rho_j$, the same as a map with no jumps. How a network would render a reversed object is not observed, so these are only what a negated control would read.
3. Comparing a network's $\sum_je_j$ with the 1 of 7.1 assumes that a network believing in object $k$ returns what it returns on object $k$'s control, and that it mixes beliefs linearly in its output (for the latent-diffusion networks, in latent space, Section 4b).

So a $\sum_je_j$ near 0 rules out a mean over the upright objects. On its own it does not separate "every joint read as an ordinary joint" from a mean that mixes upright and depth-reversed readings.

---

## 8. Scripts, logs, references

All scripts are in `theory/`. `theory/README.md` lists what each one checks, the logs and outputs it writes, the order to run them in and their runtimes on a laptop CPU. The pilot read-out `theory/pilot_compat/tears.py` is imported unchanged. The pilot's network outputs are not in the repository; check5, check9 and check10 read their loop profiles and depth-range shares from `theory/pilot_compat/pred_profiles.json`. Every check was rerun from a fresh copy of the repository on 25/09/2026 and reproduced its log, apart from a few Adam end points and Monte-Carlo rows that move in the third or fourth decimal with the number of threads (listed in `theory/README.md`).

**References.** arXiv ids and the quoted loss details were checked on the arXiv pages / HTML on 24-25/09/2026. Venues were checked for Efron, Lin et al., Penrose, DDIM and score-SDE, and taken from our earlier literature notes (checked then) for Marigold, DAv2, Lotus and Martin Garcia et al. The venues of MiDaS (TPAMI 2022) and Depth Anything V1 (CVPR 2024) were checked on Crossref (see `paper/refs.bib`).
* R. Ranftl, K. Lasinger, D. Hafner, K. Schindler, V. Koltun. Towards Robust Monocular Depth Estimation: Mixing Datasets for Zero-shot Cross-dataset Transfer. TPAMI 2022, arXiv:1907.01341. (Loss definitions read from the ar5iv HTML: $L_{ssimse}$, $L_{ssimae}$, $L_{ssitrim}$ with $U_m=0.8M$, $L_{reg}$ with $K=4$, $\alpha=0.5$.)
* L. Yang et al. Depth Anything. CVPR 2024, arXiv:2401.10891 (median/MAD affine-invariant MAE on disparity).
* L. Yang et al. Depth Anything V2. NeurIPS 2024, arXiv:2406.09414 ($L_{ssi}:L_{gm}=1:2$, "proposed by MiDaS"; top-10%-loss regions ignored on pseudo-labelled samples; feature alignment loss on pseudo-labelled images; students trained only on pseudo-labelled real images; affine-invariant inverse depth; re-read on the arXiv HTML 25/09/2026).
* B. Ke et al. Repurposing Diffusion-Based Image Generators for Monocular Depth Estimation (Marigold). CVPR 2024, arXiv:2312.02145 (2/98 percentile normalisation, v-objective, annealed multi-resolution noise). Training-noise code and config read on GitHub (prs-eth/Marigold, `src/util/multi_res_noise.py`, `config/train_marigold_depth.yaml`, `src/trainer/marigold_depth_trainer.py`) on 25/09/2026; the repository's current config is assumed to match the v1-0 training (not verified).
* J. He et al. Lotus. ICLR 2025, arXiv:2409.18124 ($x_0$-prediction at $t=T$, latent MSE, Lotus-D without noise input, disparity).
* G. Martin Garcia et al. Fine-Tuning Image-Conditional Diffusion Models is Easier than You Think. WACV 2025, arXiv:2409.11355 (DDIM leading-spacing flaw; LS alignment + L1).
* S. Lin, B. Liu, J. Li, X. Yang. Common Diffusion Noise Schedules and Sample Steps are Flawed. WACV 2024, arXiv:2305.08891.
* B. Efron. Tweedie's Formula and Selection Bias. JASA 106(496):1602-1614, 2011.
* J. Song, C. Meng, S. Ermon. Denoising Diffusion Implicit Models. ICLR 2021, arXiv:2010.02502.
* Y. Song et al. Score-Based Generative Modeling through Stochastic Differential Equations. ICLR 2021, arXiv:2011.13456.
* R. Penrose. On the Cohomology of Impossible Figures. Leonardo 25(3/4):245-247, 1992.
* diffusers 0.40.0 source (`MarigoldDepthPipeline.ensemble_depth`, default `reduction="median"`), checked locally.
* Scheduler configs of `prs-eth/marigold-depth-v1-0` and `prs-eth/marigold-depth-v1-1` (Hugging Face, `scheduler/scheduler_config.json`), read from the local cache on 25/09/2026.

**Open points (not verified).** The number of GM scales DAv2 uses. How DAv2 implements its 10% trimming (pixels or regions; ssi only or GM too), whether $L_{gm}$ is applied to pseudo-labelled images, and what the feature-alignment loss does to the output. Which noise law the real Marigold network behaves like on i.i.d. input; the multi-resolution law is modelled by a Gaussian with the right covariance, not by its exact scale mixture. Lotus's noise schedule (SD2's, presumably; this matters for Lotus-G's $\kappa$). How the real networks place the background and each object relative to it, which decides regime A vs B and the gauge. Every loss-level statement is about the population optimum; a finite network only approximates it, and its inductive bias can pull it toward smooth outputs.
