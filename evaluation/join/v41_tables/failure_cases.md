# Final-six failure cases

Thresholds used for discussion: F1 < 0.8, |onset| > 5 frames, penetration/ground > 1 cm, planted-foot p95 > 10 cm/s.

## 20231108/018

Accepted demo; shallow person-2 foot/box overlap is metric-visible.

- Robot 1: geometry violation=1.05 cm; foot-slide p95=41.4 cm/s.
- Robot 2: geometry violation=2.04 cm; foot-slide p95=82.8 cm/s.

Interpretation: these are kinematic-retargeting limitations; they are not evidence of successful dynamic tracking.

## 20231108/019

Shallow hand/foot penetrations are subtle in the preview.

- Robot 1: geometry violation=1.69 cm; foot-slide p95=54.4 cm/s.
- Robot 2: geometry violation=1.72 cm; foot-slide p95=50.1 cm/s.

Interpretation: these are kinematic-retargeting limitations; they are not evidence of successful dynamic tracking.

## 20231108/020

Person-2 joining delay and person-1 foot/box penetration are visible.

- Robot 1: geometry violation=7.64 cm; foot-slide p95=68.5 cm/s.
- Robot 2: onset error=19 frames; geometry violation=4.40 cm; foot-slide p95=97.4 cm/s.

Interpretation: these are kinematic-retargeting limitations; they are not evidence of successful dynamic tracking.

## 20231108/108

Person-1 joining delay is visually obvious.

- Robot 1: weak contact F1=0.677; onset error=22 frames; geometry violation=10.49 cm; foot-slide p95=83.1 cm/s.
- Robot 2: geometry violation=1.81 cm; foot-slide p95=74.3 cm/s.

Interpretation: these are kinematic-retargeting limitations; they are not evidence of successful dynamic tracking.

## 20231108/109

Person-2 hand and person-1 foot penetration are visually apparent.

- Robot 1: weak contact F1=0.769; onset error=14 frames; geometry violation=16.78 cm; foot-slide p95=80.5 cm/s.
- Robot 2: onset error=6 frames; geometry violation=4.33 cm; foot-slide p95=80.1 cm/s.

Interpretation: these are kinematic-retargeting limitations; they are not evidence of successful dynamic tracking.

## 20231108/110

Foot/box penetration and skating are visually obvious.

- Robot 1: weak contact F1=0.760; geometry violation=9.93 cm; foot-slide p95=188.5 cm/s.
- Robot 2: weak contact F1=0.690; geometry violation=16.52 cm; foot-slide p95=72.1 cm/s.

Interpretation: these are kinematic-retargeting limitations; they are not evidence of successful dynamic tracking.
