"""Definition tests for the post-hoc cross-agent coordination metrics."""

from __future__ import annotations

import numpy as np

from core4d_dual_retarget.coordination_eval import (
    SequenceSignals,
    _dropout_events,
    _overlap_stats,
    _runs,
    coordination_metrics,
    flatten_metrics,
)


def _signals(
    expected: dict[int, np.ndarray],
    predicted: dict[int, np.ndarray],
    *,
    penetration: dict[int, np.ndarray] | None = None,
    robot_offset_cm: dict[int, float] | None = None,
) -> SequenceSignals:
    frames = len(expected[1])
    phases = np.zeros(frames, dtype=np.int8)
    penetration = penetration or {1: np.zeros(frames), 2: np.zeros(frames)}
    robot_offset_cm = robot_offset_cm or {1: 0.0, 2: 0.0}
    expected_hand = {p: np.stack([expected[p], np.zeros(frames, bool)], axis=1) for p in (1, 2)}
    anchors = {p: np.zeros((frames, 2, 3)) for p in (1, 2)}
    robot_local = {
        p: anchors[p] + np.array([robot_offset_cm[p] / 100.0, 0.0, 0.0]) for p in (1, 2)
    }
    return SequenceSignals(
        sequence_id="test/000",
        label="unit",
        fps=15.0,
        phases=phases,
        expected=expected,
        expected_hand=expected_hand,
        predicted=predicted,
        predicted_hand_or=predicted,
        hand_distance_m={p: np.zeros((frames, 2)) for p in (1, 2)},
        hand_penetration_cm=penetration,
        hand_penetration_per_hand_cm={p: np.zeros((frames, 2)) for p in (1, 2)},
        robot_hand_local=robot_local,
        source_anchor_local=anchors,
    )


def test_runs_and_overlap_definitions() -> None:
    mask = np.array([0, 1, 1, 0, 0, 1, 0, 1, 1, 1], dtype=bool)
    assert _runs(mask) == [(1, 3), (5, 6), (7, 10)]
    expected = np.array([1, 1, 1, 1, 0, 0], dtype=bool)
    predicted = np.array([0, 1, 1, 1, 1, 0], dtype=bool)
    stats = _overlap_stats(expected, predicted)
    assert stats["true_positive_frames"] == 3
    assert stats["precision"] == 0.75
    assert stats["recall"] == 0.75
    assert stats["iou"] == 0.6
    assert abs(stats["f1"] - 0.75) < 1e-12


def test_dropouts_exclude_leading_onset_delay() -> None:
    expected = np.ones(20, dtype=bool)
    predicted = np.ones(20, dtype=bool)
    predicted[:3] = False  # onset delay, not a dropout
    predicted[8:10] = False  # dropout 1
    predicted[14] = False  # dropout 2
    result = _dropout_events(expected, predicted, (0, 20))
    assert result["missed_runs_total"] == 3
    assert result["dropout_events"] == 2
    assert result["dropout_frames_total"] == 3
    assert abs(result["maintained_fraction_in_window"] - 14 / 20) < 1e-12


def test_joint_contact_is_logical_and_of_robot_states() -> None:
    frames = 40
    e1 = np.zeros(frames, bool)
    e1[10:30] = True
    e2 = np.zeros(frames, bool)
    e2[5:35] = True
    p1 = np.zeros(frames, bool)
    p1[12:30] = True
    p2 = np.zeros(frames, bool)
    p2[5:25] = True
    metrics = coordination_metrics(_signals({1: e1, 2: e2}, {1: p1, 2: p2}))
    assert metrics["expected_joint_frames"] == 20  # frames 10..29
    assert metrics["predicted_joint_frames"] == 13  # frames 12..24
    assert metrics["joint_contact"]["precision"] == 1.0
    assert metrics["joint_contact"]["recall"] == 13 / 20
    assert metrics["timing"]["expected_joint_onset_frame"] == 10
    assert metrics["timing"]["predicted_joint_onset_frame"] == 12
    assert metrics["timing"]["joint_onset_error_frames"] == 2
    assert metrics["timing"]["expected_joint_end_frame"] == 30
    assert metrics["timing"]["predicted_joint_end_frame"] == 25
    assert metrics["timing"]["longest_predicted_joint_run_frames"] == 13


def test_bilateral_error_uses_worst_robot_per_frame() -> None:
    frames = 12
    on = np.ones(frames, bool)
    metrics = coordination_metrics(
        _signals({1: on, 2: on}, {1: on, 2: on}, robot_offset_cm={1: 2.0, 2: 6.0})
    )
    bilateral = metrics["bilateral"]
    assert abs(bilateral["robot1_error_cm"] - 2.0) < 1e-9
    assert abs(bilateral["robot2_error_cm"] - 6.0) < 1e-9
    assert abs(bilateral["bilateral_max_error_cm"] - 6.0) < 1e-9
    assert bilateral["robot1_is_worse_fraction"] == 0.0


def test_geometry_valid_proxy_uses_per_frame_penetration() -> None:
    frames = 10
    on = np.ones(frames, bool)
    pen1 = np.zeros(frames)
    pen1[:4] = 0.8  # robot 1 penetrates 0.8 cm in the first four frames only
    metrics = coordination_metrics(
        _signals({1: on, 2: on}, {1: on, 2: on}, penetration={1: pen1, 2: np.zeros(frames)})
    )
    valid = metrics["geometry_valid"]
    assert valid["tau_0cm"]["coverage_of_expected_joint"] == 0.6
    assert valid["tau_0.5cm"]["coverage_of_expected_joint"] == 0.6
    assert valid["tau_1cm"]["coverage_of_expected_joint"] == 1.0
    row = flatten_metrics(metrics)
    assert row["valid_joint_coverage_tau_0.5cm"] == 0.6
    assert row["joint_f1"] == 1.0
