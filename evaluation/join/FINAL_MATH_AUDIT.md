# Final mathematical consistency audit

Read-only check of the frozen EXT-JC w10 implementation. No optimization was rerun. Loss weights, thresholds, trajectories, evaluator definitions, and reported experiment numbers were not changed.

Trajectories read:

- `018`: `outputs/paper_eval/ext_jc_visual_refinement/reference_runs_018/ext_jc_w10_staged/{raw,smooth}`
- `020`, `109`, `110`: `outputs/paper_eval/object_aware_baseline_extension/runs_multiseq/ext_jc_w10_staged/{raw,smooth}`

Frozen config (`config.json` on those runs): `joint_contact_continuity_weight = 10`, `joint_contact_tau_m = 0.04`, `joint_contact_beta_m = 0.01`, `joint_contact_separable_surrogate = false`. Surface-aware mode, soft-min temperature override, and edge taper are absent from the config, so they stay at the defaults `false` / `0` / `0`. Contact anchors use `palm_offset_m = 0.052`, `anchor_smoothing_sigma = 1.5`, `grasp_up_frac = 0.4`, `palmar_contact = true`.

---

## 1. Soft-contact proxy

### Verified equations

Object-frame rubber-hand samples \(x_{t,n}\) (both hands concatenated). Hull planes \((a_j, b_j)\) are the SciPy `ConvexHull.equations`: a point is inside iff \(a_j^\top x + b_j \le 0\) for every plane. The origin of a unit cube gives signed value \(-0.5\), which confirms the sign.

\[
d_{t,n} = \max_j \bigl(a_j^\top x_{t,n} + b_j\bigr)
\]

\[
d_t = -\beta \log \sum_{n=1}^{N} \exp(-d_{t,n}/\beta)
\]

\[
s_t = \sigma\bigl((\tau - d_t)/\beta\bigr)
\]

Frozen constants: \(\tau = 0.04\,\mathrm{m}\), \(\beta = 0.01\,\mathrm{m}\), \(N = 1024\) (512 farthest-point samples on each rubber-hand STL, both hands in one soft-min). \(d_{t,n}\) is negative inside the hull and, outside, a lower bound on Euclidean distance to the hull.

The surface-aware variant, which clamps \(d_t\) at the surface before the sigmoid, is **not** this function. It is default-off and was not set on the frozen runs.

### Code

| Piece | Location |
|---|---|
| Indicator | `src/core4d_dual_retarget/retarget/coupled.py`, `torch_soft_contact_indicator`, lines 315–332 |
| Both-hand samples in the object frame | same file, lines 1234–1241 |
| Sample construction, 512 per hand | `src/core4d_dual_retarget/retarget/balance.py`, `_rubber_hand_mesh_offsets` (count 512) and `LEFT_HAND_VOLUME_OFFSETS_M` / `RIGHT_HAND_VOLUME_OFFSETS_M`, lines 779–796 |
| Hull equations | `convex_hull_equations` in `balance.py`, lines 1020–1024 |
| Frozen call site (non-surface-aware branch) | `coupled.py`, lines 1805–1811 |
| Defaults \(\tau,\beta\), surface-aware off | `coupled.py`, lines 218–225 |

### Soft-min versus the sampled geometry

On the **smoothed** trajectories (the poses the frozen evaluator scores), the soft-min sits below the true minimum of \(d_{t,n}\) by a nearly constant bias:

| Sequence | Robot | Mean bias (cm) | Max bias (cm) | Mean \(s\) | Frames with \(s>0.5\) | Mean true min when \(s>0.5\) (cm) | Mean mesh distance when \(s>0.5\) (cm) |
|---|---|---|---|---|---|---|---|
| 018 | 1 | 3.593 | 4.384 | 0.405 | 0.411 | +2.03 | 2.82 |
| 018 | 2 | 3.508 | 4.636 | 0.540 | 0.552 | +2.51 | 2.90 |
| 020 | 1 | 3.520 | 4.456 | 0.323 | 0.326 | +2.27 | 2.65 |
| 020 | 2 | 3.499 | 4.441 | 0.519 | 0.518 | +2.39 | 2.92 |
| 109 | 1 | 3.449 | 4.604 | 0.351 | 0.357 | +2.36 | 3.07 |
| 109 | 2 | 3.575 | 4.429 | 0.503 | 0.506 | +2.26 | 2.80 |
| 110 | 1 | 3.571 | 4.503 | 0.486 | 0.492 | +2.39 | 2.87 |
| 110 | 2 | 3.501 | 4.803 | 0.596 | 0.595 | +2.78 | 3.25 |

Positive true-min means the closest sample is still **outside** the hull. So \(s>0.5\) is not “the hand has reached the surface.” On these poses the hand is about 2–3 cm outside the hull, and 2.7–3.3 cm from the nearest object vertex, while \(s\) is already above one half.

\(s = 1/2\) exactly when \(d_t = \tau = 4\,\mathrm{cm}\). With a 3.5 cm bias that is a true hull clearance of about 7.5 cm. The sigmoid is steep (\(\beta = 1\,\mathrm{cm}\)), so few frames land in \((0.45, 0.55)\). The three that do have true minima 7.21 cm (020, robot 1), 6.98 cm (109, robot 2), and 7.09 cm (110, robot 2).

### Sampling density, without moving the trajectory

The same frozen sample set was subsampled (one draw, seed 0). Bias of true min minus soft-min, in centimetres:

| \(N\) | \(\beta \ln N\) | Measured bias across the eight robot tracks |
|---|---|---|
| 1024 | 6.931 | 3.449–3.593 |
| 512 | 6.238 | 2.897–2.992 |
| 256 | 5.545 | 2.273–2.449 |
| 128 | 4.852 | 1.720–1.991 |
| 64 | 4.159 | 1.227–1.342 |

Halving \(N\) lowers the bias by about 0.55–0.70 cm. \(\beta \ln 2 = 0.693\,\mathrm{cm}\), so the sample-count term is systematic and close to the log-\(N\) law, but smaller than the identical-sample ceiling \(\beta \ln N\). The effective count at full density is \(N_{\mathrm{eff}} = \exp(\mathrm{bias}/\beta) \approx 31\)–\(36\), not 1024: most samples sit well outside the nearest one and do not all enter the log-sum-exp.

A one-sample probe (one point at the queried distance, the other 1023 at \(+2\,\mathrm{m}\)) has essentially no bias: \(s = 0.5\) at 4 cm outside, \(s = 0.029\) at 7.5 cm outside. The 3.5 cm shift on real hands is the soft-min over the cluster of nearby samples, not a bug in the sigmoid.

### Deep penetration is rewarded, then saturates

Because \(d_t\) keeps falling inside the hull and \(s = \sigma((\tau - d_t)/\beta)\) is decreasing in \(d_t\), a deeper sample raises \(s\). Isolated values, with soft-min equal to the queried depth:

| Depth | \(s\) |
|---|---|
| 4 cm outside | 0.500 |
| on the surface | 0.982 |
| 2 cm inside | 0.998 |
| 4 cm inside | 0.9997 |
| 10 cm inside | 1.0000 |

The gain from the surface to 4 cm inside is \(0.018\). After that the sigmoid is flat.

On the frozen trajectories this reward does not place the hands inside the hull. Raw EXT-JC has **zero** inside-hull frames on 018, 020, and 110 (both robots) and on 109 robot 2. 109 robot 1 has one inside frame. Smoothed poses, which are what the evaluator sees, have 0–7 inside frames per robot. On those frames \(s\) is already 0.99964–0.99984. Moving only the closest sample 2 cm further inside, leaving the saved pose untouched, raises \(s\) by \(1.7\times 10^{-5}\) to \(7.1\times 10^{-5}\).

### Soft score versus the frozen evaluator mask

The evaluator’s predicted contact is a different predicate: hysteresis on the minimum Euclidean distance from the same rubber-hand samples to object **vertices**, enter 0.04 m, exit 0.055 m (`coordination_eval.py` lines 37–38 and 216–221; `metrics.py` `robot_hand_volume_distances_to_object`, lines 173–189). It is not \(s\), and it is not the hull plane distance.

On the smoothed poses, \(s>0.5\) agrees with that mask on 88.9–97.5% of frames. Every mismatch is the same way: \(s>0.5\) while the evaluator is off. The reverse never occurs on these four takes.

| Sequence | Robot | Frames | Agree | \(s>0.5\), evaluator off | Evaluator on, \(s\le 0.5\) | Both on | Mean \(s\) when evaluator is on |
|---|---|---|---|---|---|---|---|
| 018 | 1 | 163 | 0.975 | 4 | 0 | 63 | 0.997 |
| 018 | 2 | 163 | 0.908 | 15 | 0 | 75 | 0.993 |
| 020 | 1 | 141 | 0.972 | 4 | 0 | 42 | 0.994 |
| 020 | 2 | 141 | 0.943 | 8 | 0 | 65 | 0.991 |
| 109 | 1 | 154 | 0.929 | 11 | 0 | 44 | 0.994 |
| 109 | 2 | 154 | 0.916 | 13 | 0 | 65 | 0.992 |
| 110 | 1 | 126 | 0.960 | 5 | 0 | 57 | 0.990 |
| 110 | 2 | 126 | 0.889 | 14 | 0 | 61 | 0.993 |

When the evaluator reports contact, \(s\) is already 0.99. The soft score turns on earlier, because of the 3.5 cm soft-min bias and because \(\tau = 4\,\mathrm{cm}\) is applied to that biased distance rather than to the mesh distance.

---

## 2. Anchor residual

### What the code computes

Local anchor, per hand: nearest object vertex to the closest fingertip, then \(0.052\,\mathrm{m}\) along the outward direction, then clamped in the object-up axis to \(0.4\) of the half-extent and snapped back to the mesh, then Gaussian-smoothed with \(\sigma = 1.5\).

\[
\mathbf{a}^{\mathrm{w}}_{\mathrm{core}} = R_{\mathrm{obj}}\,\mathbf{a} + \mathbf{t}_{\mathrm{obj}}
\]

Implementation, `coupled.py` lines 309–312:

```python
anchors_world_core = np.einsum("tki,tji->tkj", anchors, rotations) + translations[:, None]
return points_core4d_to_robot(anchors_world_core), masks
```

`anchors` is \((T, 2, 3)\) and `rotations` is \(R_t\) with \(R_{ji}\) in entry `(j, i)`. The einsum sums \(a_i R_{ji}\), which is \(R\mathbf{a}\), not \(R^\top\mathbf{a}\). A random rotation confirmed `einsum == R @ a`. The stored `contact_anchors` in all four smoothed `robot1.npz` files match this reconstruction with `palm_offset=0.052`, `smoothing_sigma=1.5`, `grasp_up_frac=0.4` to a maximum absolute error of `0.0`. The stored contact masks match as well (0 mismatched frames).

`points_core4d_to_robot` (`coordinates.py` lines 8–24) then applies the fixed y-up → z-up map \(C\), with robot \(= C\,\mathbf{a}^{\mathrm{w}}_{\mathrm{core}}\). The palm point is already in the robot frame. Mean \(\lVert t_{\mathrm{obj}}\rVert\) on these takes is 1.059 m (018), 1.002 m (020), 1.255 m (109), 1.303 m (110).

### Which point is subtracted

J4 / EXT-JC sets `palmar_contact = true` and `hand_volume_clearance_weight = 1200`, so the contact term takes the branch at `coupled.py` lines 1138–1141:

\[
\sum_{\mathrm{hands}} \lVert \mathbf{p}_{\mathrm{palmar}} - \mathbf{a}^{\mathrm{w}} \rVert^2
\]

masked by the source contact weights and the phase weights, then normalized by the sum of those weights (lines 1161–1162). \(\mathbf{p}_{\mathrm{palmar}}\) is one sample per hand: rubber-hand origin plus \((0.05,-0.03,0)\) on the left and \((0.05,+0.03,0)\) on the right (`balance.py` lines 673–674). It is not the wrist keypoint.

The soft-min over the 512 volume samples (`torch_softmin_sq_error`, lines 1142–1151) runs only when `palmar_contact` is false. That branch is not the frozen objective.

### The written formula is the wrong residual

Operator precedence makes

```latex
p_{\mathrm{palm}} - R(\mathrm{object}) * \mathrm{anchor} + t(\mathrm{object})
```

equal to \(p - R a + t\). The world target defined in the text is \(R a + t\). The correct residual is their difference:

```latex
\mathbf{p}_{\mathrm{palmar}} - \bigl(R(\mathrm{object})\,\mathrm{anchor} + t(\mathrm{object})\bigr)
```

Algebra: the written expression equals the correct residual **plus** \(2\,t(\mathrm{object})\). A random \((p, R, a, t)\) reproduced that identity. On these takes that is an error of about 2 m, not a typesetting detail.

Use this replacement, with \(R\) and \(t\) in the same frame as \(p\) (the robot frame, i.e. after \(C\)):

```latex
\mathbf{a}^{\mathrm{w}} = R_{\mathrm{obj}}\,\mathbf{a} + \mathbf{t}_{\mathrm{obj}},
\qquad
\mathbf{r} = \mathbf{p}_{\mathrm{palmar}} - \mathbf{a}^{\mathrm{w}}
```

If the formula stays in CORE4D coordinates before the basis change, write \(\mathbf{a}^{\mathrm{w}} = C(R_{\mathrm{core}}\mathbf{a} + \mathbf{t}_{\mathrm{core}})\) and keep \(C\) on the palm point as well. Do not drop the parentheses, and do not flip the sign of \(t\).

---

## 3. Coupling interpretation

### Verified objective

Source joint-carry frames are the framewise AND of the two persons’ hand-contact masks (`coupled.py` lines 536–541). Each mask is hysteresis on fingertip-to-mesh distance with the defaults enter 0.05 m, exit 0.07 m (`contact.py` lines 84–99). With `joint_contact_edge_taper_frames = 0`, the activation \(E_J(t)\) is that hard 0/1 mask (`joint_carry_activation`, lines 360–370).

On those frames the frozen loss (`coupled.py` lines 1812–1817 and 1863) is

\[
L_{\mathrm{JC}}
= \frac{\sum_t E_J(t)\,\bigl(1 - s_t^{(1)} s_t^{(2)}\bigr)}{\sum_t E_J(t)}.
\]

The product is the only coupling. There is no \(\Delta s\), no jerk of \(s\), and no term in the length of a contact run. A long uninterrupted span can only appear because \(s^{(1)} s^{(2)}\) stays high on consecutive source frames. The reported longest valid span is computed afterwards in `coordination_eval.py` (lines 530–531) from the evaluator’s contact mask and the 0.5 cm penetration test. It is not a training target.

The separable surrogate replaces the product by \((s^{(1)}+s^{(2)})/2\) (lines 1812–1813). It does **not** replace the partner indicator by the constant 1. The comment at lines 213–216 says that, and the comment is wrong relative to the code. The frozen EXT-JC run has `joint_contact_separable_surrogate = false`, so it uses the product.

Readout of \(1 - s^{(1)} s^{(2)}\) on the source joint-carry frames of the smoothed trajectories (this is a geometry readout, not the optimized loss value, and it is not a replacement for the reported tables):

| Sequence | Source joint frames | Mean \(s^{(1)} s^{(2)}\) | Mean \(1 - s^{(1)} s^{(2)}\) | Fraction with both \(s>0.5\) | Fraction with both evaluator masks on |
|---|---|---|---|---|---|
| 018 | 60 | 0.977 | 0.023 | 1.000 | 0.883 |
| 020 | 49 | 0.922 | 0.078 | 0.939 | 0.857 |
| 109 | 54 | 0.941 | 0.059 | 1.000 | 0.759 |
| 110 | 57 | 0.929 | 0.071 | 0.947 | 0.825 |

The product is high on the source windows. The evaluator’s simultaneous-contact rate on those same windows is lower, which is the same discrepancy as in Section 1: \(s\) saturates before the evaluator’s 4 cm hysteresis turns on.

---

## Wording that must change

1. Replace `p_palm - R(object) * anchor + t(object)` by `p_palmar - (R(object) * anchor + t(object))`. As written, the expression equals the implemented residual plus \(2 t(\mathrm{object})\) (about 2 m on these takes).
2. Call the contact point the palmar sample at the rubber-hand offsets \((0.05,\mp 0.03,0)\). The frozen contact term does not use a wrist keypoint, and it does not use the 512-sample soft-min (that branch is off because `palmar_contact` is true).
3. Do not describe \(\tau = 4\,\mathrm{cm}\) as the distance where the indicator turns on. \(s = 1/2\) when the **soft-min** is 4 cm. On these hands the soft-min is 3.5 cm below the true minimum, so \(s = 1/2\) at about 7 cm of hull clearance.
4. Do not identify \(s_t\) with the evaluator’s predicted contact mask. Agreement is 88.9–97.5%, and every mismatch on 018/020/109/110 is \(s>0.5\) with the evaluator off. On evaluator-on frames, \(s\) is already 0.99.
5. Do not say the joint-contact loss maximizes the longest uninterrupted contact span, and do not describe it as a temporal derivative. It is the mean of \(1 - s^{(1)} s^{(2)}\) on the source joint-carry mask. Span length is an evaluation statistic.
6. Do not say the soft score is flat inside the object. It increases with depth. The increase is small: 0.982 on the surface, 0.9997 at 4 cm inside, and about \(10^{-5}\) per extra 2 cm once a frozen frame is already inside. The surface-aware clamp that would remove this reward was not used.
7. If the paper says the separable surrogate replaces the partner by 1, change it to the average \((s^{(1)}+s^{(2)})/2\). That is what `coupled.py` lines 1812–1813 compute. The comment at lines 213–216 does not match the code.
