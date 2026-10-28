# Smoke tests

These checks do not rerun IK and do not rewrite frozen outputs.

## 1. Unit tests (no frozen data)

```bash
cd $SOCCER_ROOT
python3 -m unittest discover -s tests -v
python3 -m py_compile scripts/*.py
```

Expected: all tests OK. As of this handoff, 8 tests passed.

## 2. Frozen hashes

```bash
cd $SOCCER_ROOT
sha256sum \
  data/exports/touch_ladder_demo/g1/robot_a.npz \
  data/exports/touch_ladder_demo/g1/robot_b.npz \
  data/exports/touch_ladder_demo/ball_cones.npz \
  data/exports/touch_ladder_demo/ball_cones.json \
  data/exports/touch_ladder_demo/comparison/current_hop_timed_lateral.mp4 \
  data/exports/touch_ladder_demo/smplx_joints.npz
```

Expected:

```text
ef5277f2db390ef8119a3bcc1fe21f801e8b72e4a3843fd3b2fa9a3b9f80d96b  robot_a.npz
eb5646a331064c83772351a74cc3dc951618ee69ff00e4130ebe6c212a54568b  robot_b.npz
b608f0a9ea049d33067bdb421a0c24a0f0cdc8ee55cf454444f0de9c56fef47f  ball_cones.npz
cad51d649d353ad0918834fb0f03c89aafdbb1fbdb8c42b6d25f60be85bb2156  ball_cones.json
c904d2ed28760466df36261f35f6fd281049faf9618a1f266c51cad3dd46039e  current_hop_timed_lateral.mp4
f03a8c5bb6e3e658c92b6ac7dc3915754100c07ef5cbfb010fc149cdb0ae3b4b  smplx_joints.npz
```

`g1_ball_cones_preview.mp4` and `paper_eval/videos/full.mp4` must match the preview hash.

## 3. Event counts (read-only)

```bash
python3 - <<'PY'
import json
c=json.load(open("data/exports/touch_ladder_demo/ball_cones.json"))["contacts"]
assert len(c)==7
recv=sum(i>0 for i in range(len(c)))
print("contacts", len(c), "receives", recv, "kicks", len(c), "transfers", len(c)-1)
PY
```

Expected: contacts 7, receives 6, kicks 7, transfers 6.

## 4. Video identity of the three Full copies

```bash
cmp data/exports/touch_ladder_demo/comparison/current_hop_timed_lateral.mp4 \
    data/exports/touch_ladder_demo/g1_ball_cones_preview.mp4 \
 && cmp data/exports/touch_ladder_demo/comparison/current_hop_timed_lateral.mp4 \
    data/exports/touch_ladder_demo/paper_eval/videos/full.mp4 \
 && echo identical
```

## 5. Optional replay (writes only /tmp)

See Level 1 in `REPRODUCE_SOCCER.md`. Do not pass the canonical mp4 as `--out`.

## Not a smoke test

- `retarget_dual_g1_soccer.py` with the default `--out-dir`
- `recover_ball_cones.py` on the frozen JSON path
- Any script with `--force` under `g1/` or `ball_cones.npz`
