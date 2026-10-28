# Objective separability

## 1. The accepted Full objective

For robot \(k \in \{A, B\}\) with joint trajectory \(q_k\), root \(r_k\), yaw \(\psi_k\), forward kinematics
\(p_k = \mathrm{FK}(q_k, r_k, \psi_k)\), per‑robot targets \(\hat p_k\) (stationed, event‑grounded SMPL‑X keypoints),
per‑robot event windows \(w_k\) (`kick`) and plant masks \(m_k\), the accepted solver minimises

\[
L_k(q_k, r_k, \psi_k) =
\underbrace{\sum_t \| p_k - \hat p_k \|^2_{W(w_k)}}_{\text{keypoints, event‑boosted}}
+ \lambda_{\text{root}}\|\delta r_k\|^2 + \lambda_\psi \|\psi_k-\hat\psi_k\|^2 + \lambda_{\text{hip}}\|q_k^{\text{hipyaw}}\|^2
+ \underbrace{\sum_t s(w_k)\,(\lambda_v\|\dot q_k\|^2 + \lambda_a\|\ddot q_k\|^2)}_{\text{smoothness, event‑relaxed}}
+ \lambda_{\text{plant}} \sum_t m_k\|\dot p_k^{\text{ankle}}\|^2 + \lambda_{\text{sole}}\sum_t \mathrm{relu}(0.012 - p_{k,z}^{\text{ankle}})^2
+ \lambda_b\|p^{\text{pelvis}}_{k,z}-\hat p^{\text{pelvis}}_{k,z}\|^2 + \lambda_{bv}\|\dot p^{\text{pelvis}}_{k,z}-\dot{\hat p}^{\text{pelvis}}_{k,z}\|^2
+ \lambda_u\|\dot p_k^{\text{upper}}-\dot{\hat p}_k^{\text{upper}}\|^2 + \lambda_{\text{up}} + \lambda_{\text{sh}}
+ \underbrace{\lambda_{kv}\sum_t \phi(w_k)\|\dot p_k^{\text{kickfoot}}-\dot{\hat p}_k^{\text{kickfoot}}\|^2}_{\text{kick velocity}} .
\]

The complete problem solved by the pipeline is

\[
\min_{q_A, r_A, \psi_A}\ L_A \quad\text{and independently}\quad \min_{q_B, r_B, \psi_B}\ L_B,
\qquad\text{then}\qquad \text{ball} = g\big(p_A^\*, p_B^\*, \mathcal{E}, \text{lanes}\big),
\]

with \(\mathcal{E}\) the frozen annotation. There is no term containing both \((q_A, q_B)\), the ball does not
appear in \(L_A\) or \(L_B\), and \(g\) is a closed‑form post‑process. Hence

\[
L_{\text{Full}}(q_A, q_B) = L_A(q_A) + L_B(q_B)
\]

is **additively separable**; the minimiser of the sum is the pair of independent minimisers. This is exact, not
approximate.

## 2. What the targets \(\hat p_k\) share

\(\hat p_k = \mathrm{Station}_k(\text{SMPL‑X}_k; \mathcal{E}, \text{lanes})\): the lane schedule \(y(t)\) from
`_pass_lines` is the same function for both robots (it is a property of \(\mathcal{E}\)), and each robot's own
annotated foot is placed on it at its own event frames. The shared quantity is therefore a *common prescription*
(class 3 in `dependency_audit.md`), evaluated separately per robot. Nothing from robot B's pose enters \(\hat p_A\)
(probe: Δ = 0 under B perturbation).

## 3. Empirical confirmation (`separability_probe.json`)

| stage | perturbation of B | change in A | 
|---|---|---|
| stationing / grounding / step plant | source joints +N(0, 0.05 m) + 0.4 m shift | 0.0 |
| event windows | same | 0.0 |
| IK (60 it, same process, A solved alone vs after B) | — | 0.0 (joints, root) |
| post‑IK alignment | B output shifted 0.10 m | 0.0 |
| ball generation | B output shifted 0.10 m from f60 | **ball changes up to 0.099 m over f46–143** |

Re‑snapping the ball from the frozen robots reproduces the frozen `ball_cones.npz` to machine precision
(max |Δ| = 0.0), confirming that the accepted ball is exactly \(g(p_A^\*, p_B^\*)\).

## 4. Consequence for the research question

"Does the multi‑party representation provide a coordination benefit beyond applying object‑aware retargeting
independently to both actors?" — Within the accepted implementation the two cannot differ, because the accepted
method *is* two independent solvers (which are not even object‑aware at the solver level) followed by a shared
object post‑process. Any observed coordination in the output is attributable to:

1. the shared annotation and lane schedule \(\mathcal{E}\) that both independent planners obey (specification), and
2. the object path being generated from both robots' solutions (composition),

not to a coupled optimisation. The quantitative check against a stronger, genuinely object‑aware independent
baseline is in `strong_baseline_feasibility.md`.
