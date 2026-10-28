import json

import numpy as np

from core4d_dual_retarget.aggregate import aggregate_comparison
from core4d_dual_retarget.audit import (
    audit_sequence,
    load_npz_mapping,
    project_rotations_to_so3,
)
from core4d_dual_retarget.contact import first_stable_contact, join_phase_labels
from core4d_dual_retarget.controller import (
    com_balance_joint_deltas,
    world_xy_to_yaw_frame,
)
from core4d_dual_retarget.coordinates import (
    CORE4D_TO_ROBOT,
    points_core4d_to_robot,
    points_robot_to_core4d,
    poses_core4d_to_robot,
)
from core4d_dual_retarget.core4d import (
    ActionLabel,
    infer_object_category,
    parse_action_label,
    select_sequences,
)
from core4d_dual_retarget.dataset import BOX_SEQUENCES, SPLIT, _robot_pass
from core4d_dual_retarget.metrics import (
    binary_f1,
    derivative_rms,
    shin_xy_intersection_frames,
)
from core4d_dual_retarget.paper_eval import copy_frozen_sequence
from core4d_dual_retarget.remote_archive import locate_member
from core4d_dual_retarget.retarget.balance import support_excess_xy, weighted_com
from core4d_dual_retarget.retarget.paper_presets import paper_join_config
from core4d_dual_retarget.retarget.smooth import smooth_array
from core4d_dual_retarget.robot_model import load_actuated_joint_limits


def test_parse_join_label() -> None:
    label = parse_action_label("20231108/000", "join_obs0")
    assert label.action == "join"
    assert label.obstacle == "obs0"


def test_parse_label_without_obstacle() -> None:
    label = parse_action_label("20231001/007", "raise")
    assert label.action == "raise"
    assert label.obstacle is None


def test_paper_presets_are_cumulative_and_isolate_final_v41_terms() -> None:
    j1 = paper_join_config("J1", iterations=3)
    j2 = paper_join_config("J2", iterations=3)
    j3 = paper_join_config("J3", iterations=3)
    j4 = paper_join_config("J4", iterations=3)

    assert j1.iterations == j2.iterations == j3.iterations == j4.iterations == 3
    assert j1.contact_transition_phase_weight == 1.0
    assert j1.inter_robot_weight == 0.0
    assert j2.contact_transition_phase_weight == 1.5
    assert j2.inter_robot_weight > 0
    assert j2.hand_volume_clearance_weight == 0
    assert j3.hand_volume_clearance_weight > 0
    assert j3.leg_hull_weight > 0
    assert j3.source_approach_root_weight == 0
    assert j3.swing_lift_min_root_m_s == 0
    assert j4.source_approach_root_weight == 240
    assert j4.swing_lift_min_root_m_s == 0.15
    assert j4.slip_boost_start_phase == 1


def test_derivative_rms_uses_physical_units() -> None:
    fps = 10.0
    time = np.arange(8, dtype=np.float64) / fps
    velocity = 2.5
    values = (velocity * time)[:, None]
    assert np.isclose(derivative_rms(values, order=1, fps=fps), velocity)
    assert np.isclose(derivative_rms(values, order=2, fps=fps), 0.0)


def test_shin_xy_intersection_frames() -> None:
    names = ["left_knee", "left_ankle", "right_knee", "right_ankle"]
    keypoints = np.zeros((2, 4, 3), dtype=np.float64)
    keypoints[0, :, :2] = np.array(
        [[-1, 1], [1, -1], [1, 1], [-1, -1]], dtype=np.float64
    )
    keypoints[1, :, :2] = np.array(
        [[-1, 1], [-1, -1], [1, 1], [1, -1]], dtype=np.float64
    )
    assert shin_xy_intersection_frames(keypoints, names) == 1


def test_paper_copy_refuses_to_overwrite(tmp_path) -> None:
    frozen_root = tmp_path / "frozen"
    metrics_root = tmp_path / "metrics"
    video_root = tmp_path / "videos"
    source = frozen_root / "20231108_018"
    source.mkdir(parents=True)
    for person in (1, 2):
        np.savez_compressed(source / f"robot{person}.npz", value=np.array([person]))
    metrics_root.mkdir()
    (metrics_root / "20231108_018.json").write_text("{}", encoding="utf-8")
    video_root.mkdir()
    (video_root / "018_clearance_v41.mp4").write_bytes(b"video")

    output_root = tmp_path / "paper"
    copy_frozen_sequence(
        "20231108/018", frozen_root, metrics_root, video_root, output_root
    )
    assert (output_root / "018" / "robot1.npz").exists()

    import pytest

    with pytest.raises(FileExistsError):
        copy_frozen_sequence(
            "20231108/018", frozen_root, metrics_root, video_root, output_root
        )


def test_selection_is_sorted() -> None:
    labels = [
        ActionLabel("b", "join_obs0", "join", "obs0"),
        ActionLabel("a", "join_obs0", "join", "obs0"),
        ActionLabel("c", "join_obs1", "join", "obs1"),
    ]
    selected = select_sequences(labels, action="join", obstacle="obs0")
    assert [item.sequence_id for item in selected] == ["a", "b"]


def test_infer_object_category() -> None:
    assert infer_object_category({"obj_model_path": "box/box_03_m.obj"}) == "box"
    assert infer_object_category({"object_name": "wooden_table_2"}) == "desk"
    assert infer_object_category({}) is None


def test_locate_member_with_archive_prefix() -> None:
    names = [
        "human_object_motions/20231108/000/person1_poses.npz",
        "human_object_motions/20231108/000/object_metadata.json",
    ]
    assert (
        locate_member(names, "20231108/000", "object_metadata.json")
        == names[1]
    )


def test_load_object_wrapped_npz(tmp_path) -> None:
    path = tmp_path / "pose.npz"
    expected = {"joints": np.zeros((3, 2, 3), dtype=np.float32)}
    np.savez(path, expected)
    actual = load_npz_mapping(path)
    np.testing.assert_array_equal(actual["joints"], expected["joints"])


def test_audit_valid_sequence(tmp_path) -> None:
    root = tmp_path / "20231108" / "018"
    root.mkdir(parents=True)
    pose = {"joints": np.zeros((3, 2, 3), dtype=np.float32)}
    np.savez(root / "person1_poses.npz", **pose)
    np.savez(root / "person2_poses.npz", **pose)
    np.save(root / "smooth_objposes.npy", np.repeat(np.eye(4)[None], 3, axis=0))
    (root / "object_metadata.json").write_text(json.dumps({"obj_name": "Box024"}))

    report = audit_sequence(tmp_path, "20231108/018")
    assert report.valid
    assert report.frame_counts == {"person1": 3, "person2": 3, "object": 3}


def test_project_rotations_to_so3() -> None:
    poses = np.repeat(np.eye(4)[None], 2, axis=0)
    poses[1, 0, 0] = 0.99
    repaired = project_rotations_to_so3(poses)
    rotations = repaired[:, :3, :3]
    np.testing.assert_allclose(
        np.swapaxes(rotations, 1, 2) @ rotations,
        np.repeat(np.eye(3)[None], 2, axis=0),
        atol=1e-10,
    )
    np.testing.assert_allclose(np.linalg.det(rotations), 1.0, atol=1e-10)


def test_join_phase_labels() -> None:
    incumbent = np.array([False] * 5 + [True] * 22 + [False] * 3)
    joining = np.array([False] * 20 + [True] * 7 + [False] * 3)
    phases, join_frame, approach_start, joint_end = join_phase_labels(
        incumbent,
        joining,
        approach_frames=10,
        transition_radius=1,
        min_contact_frames=3,
    )
    assert first_stable_contact(joining) == 20
    assert join_frame == 20
    assert approach_start == 10
    assert joint_end == 27
    assert phases[0] == 0
    assert phases[5] == 1
    assert phases[10] == 2
    assert phases[19] == 3
    assert phases[22] == 4
    assert phases[27] == 5


def test_core4d_to_robot_is_proper_rotation() -> None:
    np.testing.assert_allclose(CORE4D_TO_ROBOT.T @ CORE4D_TO_ROBOT, np.eye(3))
    np.testing.assert_allclose(np.linalg.det(CORE4D_TO_ROBOT), 1.0)
    point = points_core4d_to_robot(np.array([1.0, 2.0, 3.0]))
    np.testing.assert_allclose(point, np.array([1.0, -3.0, 2.0]))
    np.testing.assert_allclose(points_robot_to_core4d(point), [1.0, 2.0, 3.0])


def test_pose_basis_change() -> None:
    poses = np.eye(4)[None]
    poses[0, :3, 3] = [1.0, 2.0, 3.0]
    converted = poses_core4d_to_robot(poses)
    np.testing.assert_allclose(converted[0, :3, 3], [1.0, -3.0, 2.0])
    np.testing.assert_allclose(converted[0, :3, :3], np.eye(3))


def test_load_actuated_joint_limits(tmp_path) -> None:
    urdf = tmp_path / "robot.urdf"
    urdf.write_text(
        """
<robot name="test">
  <link name="base"/>
  <link name="arm"/>
  <joint name="arm_joint" type="revolute">
    <parent link="base"/>
    <child link="arm"/>
    <axis xyz="0 0 1"/>
    <limit lower="-1" upper="2" velocity="3" effort="4"/>
  </joint>
</robot>
""".strip()
    )
    limits = load_actuated_joint_limits(urdf)
    assert len(limits) == 1
    assert limits[0].name == "arm_joint"
    assert limits[0].lower == -1
    assert limits[0].upper == 2


def test_binary_f1() -> None:
    assert binary_f1(np.array([False]), np.array([False])) == 1.0
    assert binary_f1(
        np.array([True, True, False]),
        np.array([True, False, True]),
    ) == 0.5


def test_aggregate_paired_metrics(tmp_path) -> None:
    baseline = tmp_path / "baseline"
    coupled = tmp_path / "coupled"
    baseline.mkdir()
    coupled.mkdir()
    robot_metrics = {
        "semantic_keypoint_error_cm": 5.0,
        "contact_f1": 0.5,
        "contact_onset_error_frames": -2,
        "mean_palm_distance_during_expected_contact_cm": 10.0,
        "joint_jerk_rms_rad_s3": 20.0,
    }
    for directory, f1 in ((baseline, 0.5), (coupled, 1.0)):
        metrics = dict(robot_metrics)
        metrics["contact_f1"] = f1
        payload = {
            "sequence_id": "date/001",
            "robots": {"1": metrics, "2": metrics},
        }
        (directory / "date_001.json").write_text(json.dumps(payload))
    summary = aggregate_comparison(baseline, coupled)
    assert summary["paired_sequence_count"] == 1
    assert summary["coupled_minus_independent"]["contact_f1"] == 0.5


def test_support_excess_zero_at_stance_midpoint() -> None:
    com = np.array([[0.035, 0.0]])
    left = np.array([[0.0, 0.08]])
    right = np.array([[0.0, -0.08]])
    yaw = np.array([0.0])
    contact = np.array([True])
    assert support_excess_xy(com, left, right, contact, contact, yaw)[0] == 0.0


def test_support_excess_outside_stance() -> None:
    com = np.array([[1.0, 0.0]])
    left = np.array([[0.0, 0.08]])
    right = np.array([[0.0, -0.08]])
    yaw = np.array([0.0])
    contact = np.array([True])
    assert support_excess_xy(com, left, right, contact, contact, yaw)[0] > 0.5


def test_weighted_com_is_convex() -> None:
    points = np.zeros((2, 3, 3))
    points[0, 0] = [1.0, 0.0, 0.0]
    points[0, 1] = [0.0, 1.0, 0.0]
    masses = np.array([0.25, 0.75, 0.0])
    com = weighted_com(points, masses)
    np.testing.assert_allclose(com[0], [0.25, 0.75, 0.0])


def test_split_has_no_sequence_leakage() -> None:
    assert set(BOX_SEQUENCES).isdisjoint({"20231108/000", "20231108/036", "20231108/054"})
    box_splits = {SPLIT[seq] for seq in BOX_SEQUENCES}
    assert box_splits == {"train", "val", "test"}
    assert [SPLIT[seq] for seq in BOX_SEQUENCES].count("val") == 1


def test_quality_gate_rejects_low_contact() -> None:
    passed, failures = _robot_pass({"contact_f1": 0.2, "mean_palm_distance_during_expected_contact_cm": 1.0})
    assert passed is False
    assert "contact_f1<0.90" in failures


def test_smooth_array_reduces_jerk() -> None:
    signal = np.zeros((8, 2))
    signal[3] = 1.0
    smoothed = smooth_array(signal, 1.0)
    raw_jerk = np.diff(signal, n=3, axis=0)
    smooth_jerk = np.diff(smoothed, n=3, axis=0)
    assert np.mean(smooth_jerk**2) < np.mean(raw_jerk**2)


def test_yaw_frame_rotates_world_x() -> None:
    np.testing.assert_allclose(world_xy_to_yaw_frame([1.0, 0.0], 0.0), [1.0, 0.0])
    np.testing.assert_allclose(
        world_xy_to_yaw_frame([1.0, 0.0], np.pi / 2),
        [0.0, -1.0],
        atol=1e-12,
    )


def test_com_balance_plantarflexes_when_com_is_ahead() -> None:
    extras = com_balance_joint_deltas(
        ["left_ankle_pitch_joint", "right_ankle_roll_joint"],
        [0.05, 0.0],
        [0.0, 0.0],
        kp=400.0,
        kd=0.0,
        hip_gain=0.0,
    )
    assert extras["left_ankle_pitch_joint"] < 0.0
    assert extras["right_ankle_roll_joint"] == 0.0


def test_solve_foot_forces_matches_known_base_wrench() -> None:
    from core4d_dual_retarget.contact_id import friction_cone_flags
    from core4d_dual_retarget.controller import clip_unilateral_normals, solve_foot_forces

    jacobian_base = np.eye(6)
    residual = np.array([1.0, -2.0, 3.0, 0.1, -0.2, 0.3])
    forces = solve_foot_forces(jacobian_base, residual)
    np.testing.assert_allclose(jacobian_base.T @ forces, residual)
    pulling = np.array([0.0, 0.0, -5.0, 0.0, 0.0, 10.0])
    uni = clip_unilateral_normals(pulling)
    assert uni[2] == 0.0
    assert uni[5] == 10.0
    flags = friction_cone_flags(np.array([[0.0, 0.0, 10.0, 0.0, 0.0, 10.0]]), mu=0.8)
    assert not flags["negative_normal"].any()
    assert not flags["friction_violation"].any()
    flags_bad = friction_cone_flags(np.array([[9.0, 0.0, 1.0, 0.0, 0.0, 1.0]]), mu=0.8)
    assert flags_bad["friction_violation"][0]


def test_clip_contact_frame_forces_is_unilateral() -> None:
    from core4d_dual_retarget.controller import (
        clip_contact_frame_forces,
        solve_unilateral_contact_forces,
    )

    forces = np.array([-10.0, 3.0, 4.0, 10.0, 20.0, 0.0])
    clipped = clip_contact_frame_forces(forces, mu=0.5)
    assert clipped[0] == 0.0
    assert clipped[1] == 0.0
    assert clipped[2] == 0.0
    assert clipped[3] == 10.0
    assert np.hypot(clipped[4], clipped[5]) <= 5.0 + 1e-9

    # Two contacts, second wants tension in unconstrained LS; bounded solve keeps fn>=0.
    jacobian_base = np.array(
        [
            [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
        ]
    )
    residual = np.array([0.0, 0.0, 10.0, 0.0, 0.0, 0.0])
    solved = solve_unilateral_contact_forces(jacobian_base, residual, mu=0.8)
    packed = solved.reshape(-1, 3)
    assert (packed[:, 0] >= -1e-9).all()
    np.testing.assert_allclose(jacobian_base.T @ solved, residual, atol=1e-5)


def test_equilibrium_pack_roundtrip() -> None:
    from core4d_dual_retarget.retarget.equilibrium import _pack, _unpack

    roots = {1: np.array([0.1, 0.2, 0.8]), 2: np.array([0.3, -1.0, 0.7])}
    joints = {1: np.arange(6, dtype=np.float64), 2: np.arange(6, 12, dtype=np.float64)}
    posture = {1: [0, 2, 4], 2: [1, 3, 5]}
    vector = _pack(roots, joints, posture)
    roots2, joints2 = _unpack(vector, roots, joints, posture)
    np.testing.assert_allclose(roots2[1], roots[1])
    np.testing.assert_allclose(joints2[2][posture[2]], joints[2][posture[2]])


def test_aabb_inside_depth_detects_box_penetration() -> None:
    from core4d_dual_retarget.retarget.balance import aabb_inside_depth, torch_object_clearance_loss
    import torch

    center = np.array([0.0, 0.0, 0.0])
    half = np.array([0.2, 0.1, 0.15])
    inside = np.array([[[0.0, 0.0, 0.0]]])
    outside = np.array([[[1.0, 0.0, 0.0]]])
    np.testing.assert_allclose(aabb_inside_depth(inside, center, half, 0.0)[0, 0], 0.1)
    assert aabb_inside_depth(outside, center, half, 0.0)[0, 0] == 0.0
    loss_in = torch_object_clearance_loss(
        torch.tensor(inside), torch.tensor(center), torch.tensor(half), 0.0
    )
    loss_out = torch_object_clearance_loss(
        torch.tensor(outside), torch.tensor(center), torch.tensor(half), 0.0
    )
    assert float(loss_in) > float(loss_out)


def test_convex_hull_max_loss_pushes_interior_point() -> None:
    import torch
    from core4d_dual_retarget.retarget.balance import (
        convex_hull_equations,
        torch_convex_inside_depth,
        torch_max_inside_loss,
    )

    cube = np.array(
        [[x, y, z] for x in (-0.1, 0.1) for y in (-0.1, 0.1) for z in (-0.1, 0.1)],
        dtype=np.float64,
    )
    equations = torch.tensor(convex_hull_equations(cube), dtype=torch.float64)
    inside = torch.tensor([[[0.0, 0.0, 0.0]]], dtype=torch.float64)
    outside = torch.tensor([[[0.3, 0.0, 0.0]]], dtype=torch.float64)
    deep = torch.tensor([[[0.0, 0.0, 0.0], [0.09, 0.0, 0.0]]], dtype=torch.float64)
    shallow = torch.tensor([[[0.09, 0.0, 0.0], [0.09, 0.0, 0.0]]], dtype=torch.float64)
    din = torch_convex_inside_depth(inside, equations)
    dout = torch_convex_inside_depth(outside, equations)
    assert float(din.max()) > 0.05
    assert float(dout.max()) == 0.0
    near_outside = torch.tensor([[[0.103, 0.0, 0.0]]], dtype=torch.float64)
    assert float(torch_convex_inside_depth(near_outside, equations).max()) == 0.0
    assert (
        float(
            torch_convex_inside_depth(
                near_outside, equations, margin_m=0.006
            ).max()
        )
        > 0.0
    )
    assert float(torch_max_inside_loss(torch_convex_inside_depth(deep, equations))) > float(
        torch_max_inside_loss(torch_convex_inside_depth(shallow, equations))
    )
    deep_depth = torch_convex_inside_depth(deep, equations)
    assert float(torch_max_inside_loss(deep_depth, slack_m=0.05)) < float(
        torch_max_inside_loss(deep_depth)
    )


def test_noncontact_sample_weights_spare_the_anchor() -> None:
    import torch
    from core4d_dual_retarget.retarget.balance import torch_noncontact_sample_weights

    points = torch.tensor(
        [[[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]]],
        dtype=torch.float64,
    )
    anchor = torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float64)
    on = torch.ones(1, dtype=torch.float64)
    off = torch.zeros(1, dtype=torch.float64)
    keep_on = torch_noncontact_sample_weights(points, anchor, on)
    keep_off = torch_noncontact_sample_weights(points, anchor, off)
    assert float(keep_on[0, 0]) < 0.05
    assert float(keep_on[0, 1]) > 0.95
    torch.testing.assert_close(keep_off, torch.ones_like(keep_off))


def test_orient_only_offsets_block_origin_gradient() -> None:
    import torch
    from core4d_dual_retarget.retarget.balance import _transform_offsets_orient_only

    matrix = torch.eye(4, dtype=torch.float64).unsqueeze(0).clone().requires_grad_(True)
    offsets = torch.tensor([[0.1, 0.0, 0.0]], dtype=torch.float64)
    points = _transform_offsets_orient_only(matrix, offsets)
    points.sum().backward()
    assert matrix.grad is not None
    torch.testing.assert_close(
        matrix.grad[0, :3, 3],
        torch.zeros(3, dtype=torch.float64),
    )
    assert float(matrix.grad[0, :3, :3].abs().sum()) > 0


def test_hand_surface_points_follow_rubber_hand_x() -> None:
    import torch
    from core4d_dual_retarget.retarget.balance import torch_hand_surface_points

    identity = torch.eye(4, dtype=torch.float64).expand(2, 4, 4).clone()
    offsets = torch.tensor([[0.09, 0.0, 0.0]], dtype=torch.float64)
    points = torch_hand_surface_points(identity, identity, offsets)
    assert points.shape == (2, 2, 3)
    torch.testing.assert_close(points[0, 0], torch.tensor([0.09, 0.0, 0.0], dtype=torch.float64))
    shifted = identity.clone()
    shifted[:, 0, 3] = 0.04
    points = torch_hand_surface_points(shifted, identity, offsets)
    torch.testing.assert_close(points[0, 0], torch.tensor([0.13, 0.0, 0.0], dtype=torch.float64))


def test_hand_volume_offsets_cover_fingertips() -> None:
    from core4d_dual_retarget.retarget.balance import (
        LEFT_HAND_VOLUME_OFFSETS_M,
        RIGHT_HAND_VOLUME_OFFSETS_M,
        torch_softmin_sq_error,
    )
    import torch

    left = np.asarray(LEFT_HAND_VOLUME_OFFSETS_M)
    right = np.asarray(RIGHT_HAND_VOLUME_OFFSETS_M)
    assert left.shape[1] == 3 and right.shape[1] == 3
    assert len(left) >= 24 and len(right) >= 24
    assert 0.12 < float(left[:, 0].max()) < 0.135
    assert 0.12 < float(right[:, 0].max()) < 0.135
    assert float(left[:, 1].min()) > -0.055
    assert float(right[:, 1].max()) < 0.055
    points = torch.tensor([[[0.0, 0.0, 0.0], [0.2, 0.0, 0.0]]], dtype=torch.float64)
    anchors = torch.tensor([[0.01, 0.0, 0.0]], dtype=torch.float64)
    error = torch_softmin_sq_error(points, anchors)
    assert float(error) < 0.01


def test_hand_mesh_offsets_include_distal_vertex() -> None:
    from core4d_dual_retarget.retarget.balance import LEFT_HAND_VOLUME_OFFSETS_M

    xs = np.asarray(LEFT_HAND_VOLUME_OFFSETS_M)[:, 0]
    assert float(xs[0]) == float(xs.max())


def test_foot_volume_covers_toe_and_ground_loss() -> None:
    from core4d_dual_retarget.retarget.balance import (
        LEFT_FOOT_VOLUME_OFFSETS_M,
        torch_below_ground_loss,
        torch_near_ground_slide_loss,
    )
    import torch

    feet = np.asarray(LEFT_FOOT_VOLUME_OFFSETS_M)
    assert feet.shape[1] == 3
    assert float(feet[:, 0].max()) > 0.12
    assert float(feet[:, 2].min()) < -0.03
    z = torch.tensor([[0.02, -0.04]], dtype=torch.float64)
    loss = torch_below_ground_loss(z)
    assert float(loss) > 0
    still = torch.zeros((4, 2, 3), dtype=torch.float64)
    still[:, :, 2] = 0.0
    assert float(torch_near_ground_slide_loss(still)) == 0.0
    sliding = still.clone()
    sliding[:, :, 0] = torch.linspace(0.0, 0.3, 4)[:, None]
    assert float(torch_near_ground_slide_loss(sliding)) > 0
    left_slides = torch.zeros((4, 2, 3), dtype=torch.float64)
    left_slides[:, 0, 0] = torch.tensor([0.0, 0.1, 0.2, 0.3], dtype=torch.float64)
    assert float(left_slides[:, 1, 0].abs().sum()) == 0.0
    plant_left = torch.zeros((4, 2), dtype=torch.bool)
    plant_left[:, 0] = True
    plant_right = torch.zeros((4, 2), dtype=torch.bool)
    plant_right[:, 1] = True
    assert (
        float(
            torch_near_ground_slide_loss(
                left_slides, hard=True, plant_per_foot=plant_left, n_left=1
            )
        )
        > 0.0
    )
    assert (
        float(
            torch_near_ground_slide_loss(
                left_slides, hard=True, plant_per_foot=plant_right, n_left=1
            )
        )
        == 0.0
    )


def test_planted_foot_geometry_still_and_underground() -> None:
    from core4d_dual_retarget.metrics import apply_sole_lift_to_root, planted_foot_geometry
    from core4d_dual_retarget.retarget.balance import LEFT_FOOT_VOLUME_OFFSETS_M

    n = len(LEFT_FOOT_VOLUME_OFFSETS_M)
    frames = 8
    feet = np.zeros((frames, 2 * n, 3))
    feet[:, :, 2] = 0.0
    geom = planted_foot_geometry(feet, fps=15.0)
    assert geom["planted_foot_speed_mean_cm_s"] == 0.0
    assert geom["non_skating_frame_frac"] == 1.0
    assert geom["sole_below_ground_max_cm"] == 0.0
    feet[:, :, 0] = np.linspace(0.0, 0.2, frames)[:, None]
    moving = planted_foot_geometry(feet, fps=15.0)
    assert moving["planted_foot_speed_mean_cm_s"] > 10.0
    assert moving["non_skating_frame_frac"] < 1.0
    sunk = np.zeros((frames, 2 * n, 3))
    sunk[..., 2] = -0.03
    below = planted_foot_geometry(sunk, fps=15.0)
    assert below["sole_below_ground_max_cm"] == 3.0
    root = np.zeros((frames, 3))
    lifted, delta = apply_sole_lift_to_root(root, np.full(frames, -0.03), min_z=0.0)
    np.testing.assert_allclose(delta, 0.03)
    np.testing.assert_allclose(lifted[:, 2], 0.03)


def test_geometry_gate_rejects_skating() -> None:
    from core4d_dual_retarget.dataset import _geometry_pass

    ok = {
        "contact_f1": 0.99,
        "mean_palm_distance_during_expected_contact_cm": 0.5,
        "com_support_excess_mean_m": 0.0,
        "joint_jerk_rms_rad_s3": 10.0,
        "planted_foot_speed_mean_cm_s": 8.0,
        "sole_below_ground_max_cm": 0.4,
        "hand_in_hull_max_cm": 0.2,
        "foot_in_hull_max_cm": 0.1,
    }
    passed, _ = _geometry_pass(ok)
    assert passed is True
    failed, failures = _geometry_pass({**ok, "planted_foot_speed_mean_cm_s": 18.0})
    assert failed is False
    assert any("planted_foot_speed" in item for item in failures)


def test_source_plant_mask_ignores_jump_escape() -> None:
    from core4d_dual_retarget.retarget.balance import (
        exclusive_stance_plant,
        source_plant_mask,
    )

    slow = np.array([[True, False], [False, False], [False, True]])
    contact = np.array([[True, True], [True, True], [True, True]])
    phase = np.array([False, True, True])
    mask = source_plant_mask(slow, contact, phase)
    # Frame 0: only the slow left foot.
    assert tuple(mask[0]) == (True, False)
    # Frame 1: carry + contact locks both even if source XY is not slow.
    assert tuple(mask[1]) == (True, True)
    assert tuple(mask[2]) == (True, True)
    unlocked = source_plant_mask(slow, contact, phase, require_contact=False)
    assert tuple(unlocked[0]) == (True, False)
    assert tuple(unlocked[1]) == (True, True)
    no_contact = np.zeros_like(contact)
    assert tuple(source_plant_mask(slow, no_contact, phase, require_contact=False)[1]) == (
        True,
        True,
    )
    assert tuple(source_plant_mask(slow, no_contact, phase, require_contact=True)[1]) == (
        False,
        False,
    )
    assert tuple(source_plant_mask(slow, contact, phase, slow_only=True)[1]) == (
        False,
        False,
    )
    unlocked_walk = np.array([[False, False], [False, False]])
    z = np.array([[0.08, 0.03], [0.01, 0.04]])
    exclusive = exclusive_stance_plant(unlocked_walk, z)
    assert tuple(exclusive[0]) == (False, True)
    assert tuple(exclusive[1]) == (True, False)


def test_exclusive_stance_plant_keeps_existing() -> None:
    from core4d_dual_retarget.retarget.balance import exclusive_stance_plant

    plant = np.array([[True, True]])
    z = np.array([[0.1, 0.0]])
    assert tuple(exclusive_stance_plant(plant, z)[0]) == (True, True)


def test_forced_swing_requires_lift_when_source_moves() -> None:
    import torch

    from core4d_dual_retarget.retarget.balance import torch_forced_swing_loss

    zmin = torch.tensor([0.0, 0.05], dtype=torch.float64)
    moving = torch.tensor([True, True])
    loss = torch_forced_swing_loss(zmin, moving, lift_m=0.04)
    assert float(loss) > 0.0
    planted = torch.tensor([False, False])
    assert float(torch_forced_swing_loss(zmin, planted, lift_m=0.04)) == 0.0
    lifted = torch.tensor([0.05, 0.05], dtype=torch.float64)
    assert float(torch_forced_swing_loss(lifted, moving, lift_m=0.04)) == 0.0


def test_knee_squat_loss_only_on_planted_leg() -> None:
    import torch

    from core4d_dual_retarget.retarget.balance import torch_knee_squat_loss

    knees = torch.tensor([[1.8, 0.2], [0.5, 0.5]], dtype=torch.float64)
    plant = torch.tensor([[True, False], [True, True]])
    loss = torch_knee_squat_loss(knees, plant, max_rad=1.15)
    assert float(loss) > 0.0
    straight = torch.tensor([[0.4, 0.4], [0.3, 0.3]], dtype=torch.float64)
    assert float(torch_knee_squat_loss(straight, plant, max_rad=1.15)) == 0.0


def test_palm_facing_loss_zero_when_aligned() -> None:
    import torch

    from core4d_dual_retarget.retarget.balance import torch_palm_facing_loss

    origins = torch.zeros((2, 2, 3), dtype=torch.float64)
    axes = torch.tensor(
        [[[0.0, -1.0, 0.0], [0.0, 1.0, 0.0]]] * 2,
        dtype=torch.float64,
    )
    anchors = origins + axes
    weights = torch.ones((2, 2), dtype=torch.float64)
    assert float(torch_palm_facing_loss(origins, axes, anchors, weights)) < 1e-8
    flipped = -axes
    assert float(torch_palm_facing_loss(origins, flipped, anchors, weights)) > 1.0


def test_finger_along_face_punishes_stabbing() -> None:
    import torch

    from core4d_dual_retarget.retarget.balance import torch_finger_along_face_loss

    finger = torch.tensor([[[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]]], dtype=torch.float64)
    toward = torch.tensor([[[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]], dtype=torch.float64)
    weights = torch.ones((1, 2), dtype=torch.float64)
    up = torch.tensor((0.0, 0.0, 1.0), dtype=torch.float64)
    along = torch_finger_along_face_loss(finger, toward, weights, up=up)
    stab = torch.tensor([[[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]], dtype=torch.float64)
    into = torch_finger_along_face_loss(stab, toward, weights, up=up)
    assert float(into) > float(along)


def test_foot_slip_stop_grad_freezes_previous_frame() -> None:
    import torch

    from core4d_dual_retarget.retarget.balance import torch_foot_slip_loss

    xy = torch.tensor(
        [[[0.0, 0.0], [1.0, 0.0]], [[0.1, 0.0], [1.1, 0.0]]],
        dtype=torch.float64,
        requires_grad=True,
    )
    plant = torch.ones((2, 2), dtype=torch.bool)
    torch_foot_slip_loss(xy, plant, stop_grad_prev=True).backward()
    assert torch.allclose(xy.grad[0], torch.zeros((2, 2), dtype=torch.float64))
    assert float(xy.grad[1].abs().sum()) > 0.0


def test_freeze_stance_xy_locks_to_segment_start() -> None:
    import torch

    from core4d_dual_retarget.retarget.balance import freeze_stance_xy

    centroids = torch.tensor(
        [[[0.0, 0.0], [1.0, 0.0]], [[0.2, 0.0], [1.2, 0.0]], [[0.4, 0.0], [1.4, 0.0]]],
        dtype=torch.float64,
    )
    plant = torch.tensor([[True, False], [True, True], [True, True]])
    frozen = freeze_stance_xy(centroids, plant)
    assert float(frozen[2, 0, 0]) == 0.0
    assert float(frozen[2, 1, 0]) == 1.2


def test_narrow_stance_keeps_midpoint() -> None:
    import numpy as np

    from core4d_dual_retarget.retarget.balance import narrow_stance_to_width

    left, right = narrow_stance_to_width(
        np.array([0.0, 0.0]), np.array([0.6, 0.0]), 0.24
    )
    assert np.allclose(0.5 * (left + right), [0.3, 0.0])
    assert abs(np.linalg.norm(right - left) - 0.24) < 1e-9


def test_putdown_foot_targets_are_hip_width_and_lateral() -> None:
    import numpy as np

    from core4d_dual_retarget.retarget.balance import putdown_foot_targets

    left, right = putdown_foot_targets(np.array([1.0, 2.0]), yaw=0.0, half_width_m=0.12)
    assert np.allclose(left, [1.0, 1.88])
    assert np.allclose(right, [1.0, 2.12])
    assert abs(np.linalg.norm(right - left) - 0.24) < 1e-9


def test_gait_support_and_putdown_tail_are_walk_not_weld() -> None:
    import numpy as np
    import torch

    from core4d_dual_retarget.retarget.balance import (
        gait_support_plant,
        putdown_tail_mask,
        torch_knee_min_bend_loss,
        torch_swing_knee_bend_loss,
        torch_swing_knee_target_loss,
    )

    slow = np.zeros((4, 2), dtype=bool)
    speed = np.array([[0.4, 0.1], [0.1, 0.5], [0.6, 0.5], [0.0, 0.0]])
    support = gait_support_plant(slow, speed)
    assert support[0, 1] and not support[0, 0]
    assert support[1, 0] and not support[1, 1]
    phases = np.array([2, 4, 4, 4, 4, 5])
    tail = putdown_tail_mask(phases, phase_min=4, phase_max=4, tail_frames=2)
    assert list(tail) == [False, False, False, True, True, False]
    knees = torch.tensor([[0.1, 0.1], [0.9, 0.9]], dtype=torch.float64)
    mask = torch.tensor([True, True])
    assert float(torch_knee_min_bend_loss(knees, mask, 0.8)) > 0.0
    swing = torch.tensor([[False, True], [True, False]])
    loss = torch_swing_knee_bend_loss(knees, swing, 0.55)
    assert float(loss) > 0.0
    target = torch.tensor([[0.1, 0.5], [0.8, 0.9]], dtype=torch.float64)
    target_loss = torch_swing_knee_target_loss(knees, swing, target)
    assert float(target_loss) > 0.0


def test_leg_laterality_and_self_clearance_penalize_crossing() -> None:
    import torch

    from core4d_dual_retarget.retarget.balance import (
        torch_abs_limit_loss,
        torch_limb_pair_clearance_loss,
        torch_pair_laterality_loss,
        torch_segment_samples,
        torch_swing_outward_loss,
    )

    yaw = torch.zeros(2, dtype=torch.float64)
    left = torch.tensor([[0.0, 0.12, 0.0], [0.0, -0.05, 0.0]], dtype=torch.float64)
    right = torch.tensor([[0.0, -0.12, 0.0], [0.0, 0.05, 0.0]], dtype=torch.float64)
    loss = torch_pair_laterality_loss(left, right, yaw, 0.06)
    assert float(loss) > 0.0
    ok = torch_pair_laterality_loss(left[:1], right[:1], yaw[:1], 0.06)
    assert float(ok) == 0.0
    left_seg = torch_segment_samples(
        torch.tensor([[0.0, 0.0, 0.8]], dtype=torch.float64),
        torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float64),
    )
    right_seg = torch_segment_samples(
        torch.tensor([[0.0, 0.01, 0.8]], dtype=torch.float64),
        torch.tensor([[0.0, 0.01, 0.0]], dtype=torch.float64),
    )
    assert float(torch_limb_pair_clearance_loss(left_seg, right_seg, 0.08)) > 0.0
    assert float(torch_abs_limit_loss(torch.tensor([1.5, -1.2]), 0.80)) > 0.0
    assert float(torch_abs_limit_loss(torch.tensor([0.2, -0.3]), 0.80)) == 0.0
    wide = torch.tensor([0.40, 0.12], dtype=torch.float64)
    src = torch.tensor([0.20, 0.12], dtype=torch.float64)
    swing = torch.tensor([True, False])
    assert float(torch_swing_outward_loss(wide, src, swing, slack_m=0.04, left=True)) > 0.0
    assert (
        float(
            torch_swing_outward_loss(
                torch.tensor([0.21, 0.12]),
                src,
                swing,
                slack_m=0.04,
                left=True,
            )
        )
        == 0.0
    )


def test_xy_speed_cap_blocks_teleport_stride() -> None:
    import torch

    from core4d_dual_retarget.retarget.balance import (
        torch_one_foot_down_loss,
        torch_xy_speed_cap_loss,
    )

    teleport = torch.tensor([[0.0, 0.0], [0.33, 0.0]], dtype=torch.float64)
    cap = torch.tensor([0.9, 0.9], dtype=torch.float64)
    assert float(torch_xy_speed_cap_loss(teleport, cap, fps=15.0)) > 0.0
    walk = torch.tensor([[0.0, 0.0], [0.04, 0.0]], dtype=torch.float64)
    assert float(torch_xy_speed_cap_loss(walk, cap, fps=15.0)) == 0.0
    left = torch.tensor([0.12, 0.02], dtype=torch.float64)
    right = torch.tensor([0.11, 0.15], dtype=torch.float64)
    mask = torch.tensor([True, True])
    assert float(torch_one_foot_down_loss(left, right, mask, 0.03)) > 0.0
    assert float(torch_one_foot_down_loss(left, torch.zeros(2), mask, 0.03)) == 0.0


def test_putdown_squat_losses_penalize_lunge_and_lifted_foot() -> None:
    import torch

    from core4d_dual_retarget.retarget.balance import (
        torch_both_soles_down_loss,
        torch_knee_symmetry_loss,
        torch_stance_width_loss,
    )

    mask = torch.tensor([False, True, True])
    knees = torch.tensor([[0.2, 0.2], [0.4, 1.2], [0.5, 0.5]], dtype=torch.float64)
    assert float(torch_knee_symmetry_loss(knees, mask)) > 0.1
    left_z = torch.tensor([0.02, 0.08, 0.02], dtype=torch.float64)
    right_z = torch.tensor([0.02, 0.02, 0.02], dtype=torch.float64)
    assert float(torch_both_soles_down_loss(left_z, right_z, mask)) > 0.0
    left_xy = torch.tensor([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]], dtype=torch.float64)
    right_xy = torch.tensor([[0.24, 0.0], [0.6, 0.0], [0.24, 0.0]], dtype=torch.float64)
    wide = float(torch_stance_width_loss(left_xy, right_xy, mask, 0.24))
    assert wide > 0.05


def test_grasp_up_frac_lowers_side_contacts() -> None:
    from core4d_dual_retarget.contact import SMPLX_FINGERTIP_VERTEX_IDS
    from core4d_dual_retarget.retarget.coupled import _contact_anchors

    xs = np.linspace(-0.2, 0.2, 5)
    ys = np.linspace(-0.25, 0.25, 11)
    zs = np.linspace(-0.2, 0.2, 5)
    vertices_obj = np.stack(np.meshgrid(xs, ys, zs, indexing="ij"), axis=-1).reshape(-1, 3)
    poses = np.repeat(np.eye(4)[None], 2, axis=0)
    n_verts = int(
        max(
            SMPLX_FINGERTIP_VERTEX_IDS["left"].max(),
            SMPLX_FINGERTIP_VERTEX_IDS["right"].max(),
        )
        + 1
    )
    smpl = np.zeros((2, n_verts, 3), dtype=np.float64)
    for ids in (SMPLX_FINGERTIP_VERTEX_IDS["left"], SMPLX_FINGERTIP_VERTEX_IDS["right"]):
        smpl[:, ids, 0] = 0.21
        smpl[:, ids, 1] = 0.22
    high, _ = _contact_anchors(smpl, poses, vertices_obj, palm_offset=0.0, grasp_up_frac=1.0)
    low, _ = _contact_anchors(smpl, poses, vertices_obj, palm_offset=0.0, grasp_up_frac=0.4)
    assert float(low[..., 2].mean()) < float(high[..., 2].mean()) - 0.05


def test_leg_volume_offsets_cover_shin() -> None:
    from core4d_dual_retarget.retarget.balance import (
        LEFT_KNEE_VOLUME_OFFSETS_M,
        PELVIS_VOLUME_OFFSETS_M,
    )

    knee = np.asarray(LEFT_KNEE_VOLUME_OFFSETS_M)
    pelvis = np.asarray(PELVIS_VOLUME_OFFSETS_M)
    assert len(knee) >= 24 and len(pelvis) >= 24
    assert float(knee[:, 2].min()) < -0.2
    assert float(np.ptp(pelvis, axis=0).max()) > 0.08


def test_human_foot_xy_slow_detects_slide() -> None:
    from core4d_dual_retarget.retarget.balance import human_foot_xy_slow

    joints = np.zeros((6, 10, 3))
    joints[:, 7, 1] = 0.05
    joints[:, 8, 1] = 0.05
    still = human_foot_xy_slow(joints, fps=15.0, slow_m_s=0.05)
    assert bool(still.all())
    joints[:, 7, 0] = np.linspace(0.0, 1.0, 6)
    sliding = human_foot_xy_slow(joints, fps=15.0, slow_m_s=0.05)
    assert not bool(sliding[1:, 0].all())


def test_planted_slip_zero_when_feet_still() -> None:
    from core4d_dual_retarget.metrics import planted_slip_cm_s

    keypoints = np.zeros((8, 2, 3))
    keypoints[:, 0, 2] = 0.05
    keypoints[:, 1, 2] = 0.05
    source = np.zeros((8, 10, 3))
    source[:, 7, 1] = 0.05
    source[:, 8, 1] = 0.05
    ankle, root = planted_slip_cm_s(
        keypoints,
        ["left_ankle", "right_ankle"],
        np.zeros((8, 3)),
        source,
        fps=15.0,
    )
    assert ankle == 0.0
    assert root == 0.0


def test_arm_sigma_leaves_wrist_when_zero() -> None:
    from core4d_dual_retarget.retarget.smooth import smooth_joint_positions

    names = ["left_hip_pitch_joint", "left_wrist_roll_joint"]
    positions = np.array([[0.0, 1.0], [1.0, 2.0], [2.0, 3.0], [3.0, 4.0]], dtype=np.float64)
    out = smooth_joint_positions(positions, names, sigma=1.0, arm_sigma=0.0)
    np.testing.assert_allclose(out[:, 1], positions[:, 1])
    assert not np.allclose(out[:, 0], positions[:, 0])


def test_root_sigma_zero_leaves_translation() -> None:
    from core4d_dual_retarget.retarget.smooth import smooth_array

    root = np.array([[0.0, 0.0, 0.8], [0.1, 0.0, 0.8], [0.4, 0.2, 0.8]], dtype=np.float64)
    np.testing.assert_allclose(smooth_array(root, 0.0), root)
    assert not np.allclose(smooth_array(root, 1.0), root)


def test_clearance_recipe_defaults_leave_explicit_sigmas() -> None:
    from core4d_dual_retarget.retarget.smooth import resolve_smooth_sigmas

    assert resolve_smooth_sigmas(arm_sigma=None, root_sigma=None, clearance_recipe=False) == (None, None)
    assert resolve_smooth_sigmas(arm_sigma=None, root_sigma=None, clearance_recipe=True) == (0.35, 0.0)
    assert resolve_smooth_sigmas(arm_sigma=0.1, root_sigma=0.3, clearance_recipe=True) == (0.1, 0.3)


def test_aabb_inside_depth_keeps_layer2_jerk_gate() -> None:
    from core4d_dual_retarget.dataset import layer2_clip_row

    def robot_id(tensile: float, residual: float) -> dict[str, float]:
        return {
            "negative_normal_ratio": tensile,
            "unilateral_base_residual_rms": residual,
        }

    def robot_geom(**kwargs: float) -> dict[str, float]:
        base = {
            "contact_f1": 0.99,
            "mean_palm_distance_during_expected_contact_cm": 3.0,
            "com_support_excess_mean_m": 0.0,
            "ankle_height_mae_m": 0.01,
            "joint_jerk_rms_rad_s3": 10.0,
        }
        base.update(kwargs)
        return base

    row = layer2_clip_row(
        "20231108/018",
        {"robots": {"1": robot_id(0.5, 4000.0), "2": robot_id(0.7, 6000.0)}},
        {"robots": {"1": robot_id(0.03, 60.0), "2": robot_id(0.13, 80.0)}},
        {"robots": {"1": robot_geom(), "2": robot_geom()}},
        {"robots": {"1": robot_geom(joint_jerk_rms_rad_s3=70.0), "2": robot_geom()}},
    )
    assert row["passed"] is False
    assert "jerk>60" in row["robots"]["1"]["geometry_failures"]
    assert row["robots"]["1"]["tensile_new"] == 0.03


def test_clip_equilibrium_config_keeps_temporal_weight() -> None:
    from core4d_dual_retarget.retarget.equilibrium import (
        CLIP_EQUILIBRIUM_CONFIG,
        EquilibriumConfig,
    )

    assert CLIP_EQUILIBRIUM_CONFIG.temporal_weight > 0
    assert CLIP_EQUILIBRIUM_CONFIG.target_foot_z == 0.055
    assert CLIP_EQUILIBRIUM_CONFIG.root_z_bound_m <= 0.01
    assert EquilibriumConfig().temporal_weight == 0.0
    assert CLIP_EQUILIBRIUM_CONFIG.max_iterations < EquilibriumConfig().max_iterations


def test_refine_clip_equilibrium_two_frames() -> None:
    from pathlib import Path

    import pytest

    scene = Path("outputs/mujoco/20231108_018/dual_g1_pd.xml")
    robot1 = Path("outputs/balanced_smooth/20231108_018/robot1.npz")
    robot2 = Path("outputs/balanced_smooth/20231108_018/robot2.npz")
    if not (scene.exists() and robot1.exists() and robot2.exists()):
        pytest.skip("018 kinematic release not on disk")

    from core4d_dual_retarget.retarget.equilibrium import (
        CLIP_EQUILIBRIUM_CONFIG,
        EquilibriumConfig,
        refine_clip_equilibrium,
    )

    config = EquilibriumConfig(**{**CLIP_EQUILIBRIUM_CONFIG.__dict__, "max_iterations": 2})
    outputs, diagnostics = refine_clip_equilibrium(
        scene,
        robot1,
        robot2,
        config=config,
        max_frames=2,
    )
    assert diagnostics["mode"] == "clip"
    assert diagnostics["frames"] == 2
    assert outputs[1]["joint_positions"].shape == np.load(robot1)["joint_positions"].shape
    assert diagnostics["mean_tensile_n"]["1"] >= 0.0


def test_dual_g1_keeps_official_silver_rgba(tmp_path) -> None:
    from pathlib import Path

    from core4d_dual_retarget.mujoco_scene import build_dual_g1_scene_xml

    source = Path("third_party/unitree_ros/robots/g1_description/g1_29dof_rev_1_0.xml")
    mesh = Path("data/core4d/object_models/object_models/box/box001_m.obj")
    out = tmp_path / "scene.xml"
    build_dual_g1_scene_xml(source, mesh, out)
    text = out.read_text()
    assert "0.20 0.45 0.80" not in text
    assert "0.90 0.45 0.15" not in text
    assert 'rgba="0.7 0.7 0.7 1"' in text


def test_foot_boxes_are_added_to_scene(tmp_path) -> None:
    from pathlib import Path

    from core4d_dual_retarget.mujoco_scene import build_dual_g1_scene_xml

    source = Path("third_party/unitree_ros/robots/g1_description/g1_29dof_rev_1_0.xml")
    mesh = Path("data/core4d/object_models/object_models/box/box001_m.obj")
    out = tmp_path / "scene.xml"
    build_dual_g1_scene_xml(source, mesh, out, foot_boxes=True)
    text = out.read_text()
    assert "r1_left_ankle_roll_link_sole" in text
    assert 'friction="1.8 0.4 0.01"' in text


def test_position_stance_scene_uses_implicit_actuators(tmp_path) -> None:
    from pathlib import Path

    from core4d_dual_retarget.mujoco_scene import build_dual_g1_scene_xml

    source = Path("third_party/unitree_ros/robots/g1_description/g1_29dof_rev_1_0.xml")
    mesh = Path("data/core4d/object_models/object_models/box/box001_m.obj")
    out = tmp_path / "scene.xml"
    build_dual_g1_scene_xml(
        source,
        mesh,
        out,
        foot_boxes=True,
        position_actuators=True,
        stiff_contacts=True,
        leg_kp=200.0,
        arm_kp=40.0,
    )
    text = out.read_text()
    assert 'integrator="implicitfast"' in text
    assert "<position" in text
    assert 'kp="200"' in text
    assert "r1_left_ankle_roll_link_sole" in text
    assert 'solref="0.004 1"' in text
    assert 'noslip_iterations="3"' in text
    assert 'impratio="20"' in text


def test_r1_sticky_floor_uses_private_condim6_plane(tmp_path) -> None:
    from pathlib import Path

    from core4d_dual_retarget.mujoco_scene import build_dual_g1_scene_xml

    source = Path("third_party/unitree_ros/robots/g1_description/g1_29dof_rev_1_0.xml")
    mesh = Path("data/core4d/object_models/object_models/box/box001_m.obj")
    out = tmp_path / "scene.xml"
    build_dual_g1_scene_xml(
        source,
        mesh,
        out,
        foot_boxes=True,
        position_actuators=True,
        stiff_contacts=True,
        r1_sticky_floor=True,
    )
    text = out.read_text()
    assert 'name="floor_r1_sticky"' in text
    assert 'condim="6"' in text
    assert "contype=\"2\"" in text
    assert "r2_left_ankle_roll_link_sole" in text
    r2_idx = text.find("r2_left_ankle_roll_link_sole")
    r2_chunk = text[r2_idx : r2_idx + 280]
    assert 'contype="2"' not in r2_chunk


def test_project_com_reduces_support_error(tmp_path) -> None:
    import os
    from pathlib import Path

    os.environ.setdefault("MUJOCO_GL", "egl")
    import mujoco

    from core4d_dual_retarget.controller import project_com_onto_support
    from core4d_dual_retarget.mujoco_scene import build_dual_g1_scene_xml

    source = Path("third_party/unitree_ros/robots/g1_description/g1_29dof_rev_1_0.xml")
    mesh = Path("data/core4d/object_models/object_models/box/box001_m.obj")
    out = tmp_path / "scene.xml"
    build_dual_g1_scene_xml(source, mesh, out, foot_boxes=True, position_actuators=True)
    model = mujoco.MjModel.from_xml_path(str(out))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    pelvis = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "r1_pelvis")
    left = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "r1_left_ankle_roll_link")
    right = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "r1_right_ankle_roll_link")

    def sole_error() -> float:
        support = 0.5 * (data.xpos[left][:2] + data.xpos[right][:2])
        return float(np.linalg.norm(data.subtree_com[pelvis][:2] - support))

    before = sole_error()
    project_com_onto_support(model, data, "r1_")
    after = sole_error()
    assert after <= before + 1e-6


def test_level_soles_by_root_tilt_equalizes_feet(tmp_path) -> None:
    import os
    from pathlib import Path

    os.environ.setdefault("MUJOCO_GL", "egl")
    import mujoco

    from core4d_dual_retarget.controller import (
        level_soles_by_root_tilt,
        min_sole_center_z,
    )
    from core4d_dual_retarget.mujoco_scene import build_dual_g1_scene_xml

    source = Path("third_party/unitree_ros/robots/g1_description/g1_29dof_rev_1_0.xml")
    mesh = Path("data/core4d/object_models/object_models/box/box001_m.obj")
    out = tmp_path / "scene.xml"
    build_dual_g1_scene_xml(source, mesh, out, foot_boxes=True, position_actuators=True)
    model = mujoco.MjModel.from_xml_path(str(out))
    data = mujoco.MjData(model)
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "r2_floating_base_joint")
    address = int(model.jnt_qposadr[joint_id])
    tilt = 0.06
    data.qpos[address + 3 : address + 7] = [
        np.cos(tilt / 2),
        np.sin(tilt / 2),
        0.0,
        0.0,
    ]
    mujoco.mj_forward(model, data)
    before = abs(
        min_sole_center_z(model, data, "r2_", "right")
        - min_sole_center_z(model, data, "r2_", "left")
    )
    after = abs(level_soles_by_root_tilt(model, data, "r2_"))
    assert before > 0.01
    assert after < 0.4 * before


def test_plant_hovering_foot_lowers_high_sole(tmp_path) -> None:
    import os
    from pathlib import Path

    os.environ.setdefault("MUJOCO_GL", "egl")
    import mujoco

    from core4d_dual_retarget.controller import (
        min_sole_center_z,
        plant_hovering_foot,
    )
    from core4d_dual_retarget.mujoco_scene import build_dual_g1_scene_xml
    from core4d_dual_retarget.pd_tracking import _load_trajectory, _set_initial_state

    source = Path("third_party/unitree_ros/robots/g1_description/g1_29dof_rev_1_0.xml")
    mesh = Path("data/core4d/object_models/object_models/box/box001_m.obj")
    out = tmp_path / "scene.xml"
    build_dual_g1_scene_xml(source, mesh, out, foot_boxes=True, position_actuators=True)
    model = mujoco.MjModel.from_xml_path(str(out))
    data = mujoco.MjData(model)
    stance = Path("outputs/stance/20231108_018")
    trajectories = {
        1: _load_trajectory(stance / "robot1.npz"),
        2: _load_trajectory(stance / "robot2.npz"),
    }
    _set_initial_state(model, data, trajectories)
    before = abs(
        min_sole_center_z(model, data, "r2_", "right")
        - min_sole_center_z(model, data, "r2_", "left")
    )
    after = abs(plant_hovering_foot(model, data, "r2_"))
    assert before > 0.005
    assert after < 0.5 * before


def test_balance_offsets_restore_qpos(tmp_path) -> None:
    import os
    from pathlib import Path

    os.environ.setdefault("MUJOCO_GL", "egl")
    import mujoco

    from core4d_dual_retarget.controller import balance_joint_offsets
    from core4d_dual_retarget.mujoco_scene import build_dual_g1_scene_xml

    source = Path("third_party/unitree_ros/robots/g1_description/g1_29dof_rev_1_0.xml")
    mesh = Path("data/core4d/object_models/object_models/box/box001_m.obj")
    out = tmp_path / "scene.xml"
    build_dual_g1_scene_xml(source, mesh, out, foot_boxes=True, position_actuators=True)
    model = mujoco.MjModel.from_xml_path(str(out))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    before = np.array(data.qpos, copy=True)
    balance_joint_offsets(model, data, "r1_")
    np.testing.assert_allclose(np.asarray(data.qpos), before)


def test_residual_observations_match_frame_count() -> None:
    from core4d_dual_retarget.residual import trajectory_to_observations

    robot = {
        "joint_positions": np.zeros((5, 3)),
        "root_translation": np.zeros((5, 3)),
        "root_yaw": np.zeros(5),
        "contact_mask": np.zeros((5, 2)),
        "contact_anchors": np.zeros((5, 2, 3)),
        "robot_keypoints": np.zeros((5, 4, 3)),
        "target_keypoints": np.zeros((5, 4, 3)),
        "joint_names": np.array(["a", "b", "c"]),
        "keypoint_names": np.array(["k0", "k1", "k2", "k3"]),
    }
    obs = trajectory_to_observations(robot)
    assert obs["joint_velocities"].shape == (5, 3)
    assert obs["root_velocity"].shape == (5, 3)


def test_freeze_leg_targets_keeps_stance_ankles() -> None:
    from core4d_dual_retarget.controller import freeze_leg_targets, is_leg_joint

    names = [
        "left_hip_pitch_joint",
        "left_knee_joint",
        "left_ankle_pitch_joint",
        "left_shoulder_pitch_joint",
        "waist_yaw_joint",
    ]
    stance = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    target = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
    frozen = freeze_leg_targets(names, target, stance)
    assert frozen[0] == 0.1
    assert frozen[1] == 0.2
    assert frozen[2] == 0.3
    assert frozen[3] == 1.0
    assert frozen[4] == 1.0
    assert is_leg_joint("right_ankle_roll_joint")
    assert not is_leg_joint("right_elbow_joint")
    with_waist = freeze_leg_targets(names, target, stance, freeze_waist=True)
    assert with_waist[4] == 0.5
    names_waist = names + ["waist_pitch_joint"]
    stance_w = np.append(stance, 0.7)
    target_w = np.append(target, 1.2)
    frozen_w = freeze_leg_targets(names_waist, target_w, stance_w, freeze_waist=True)
    assert frozen_w[-1] == 0.7
    waist_only = freeze_leg_targets(
        names, target, stance, freeze_legs=False, freeze_waist=True
    )
    assert waist_only[0] == 1.0
    assert waist_only[3] == 1.0
    assert waist_only[4] == 0.5
    ankles_only = freeze_leg_targets(
        names, target, stance, freeze_legs=False, freeze_ankles=True
    )
    assert ankles_only[0] == 1.0
    assert ankles_only[2] == 0.3
    assert ankles_only[3] == 1.0


def test_planted_foot_sides_and_freeze_sides() -> None:
    from core4d_dual_retarget.controller import (
        freeze_leg_targets,
        planted_foot_sides,
    )

    assert planted_foot_sides(0.04, 0.067, 0.008) == ("left",)
    assert planted_foot_sides(0.083, 0.080, 0.008) == ("left", "right")
    assert planted_foot_sides(0.07, 0.04, 0.008) == ("right",)
    names = [
        "left_hip_pitch_joint",
        "right_hip_pitch_joint",
        "left_shoulder_pitch_joint",
    ]
    stance = np.array([0.1, 0.2, 0.3])
    target = np.array([1.0, 1.0, 1.0])
    left_only = freeze_leg_targets(
        names, target, stance, freeze_legs=False, freeze_sides=("left",)
    )
    np.testing.assert_allclose(left_only, [0.1, 1.0, 1.0])
    names_s = [
        "left_hip_pitch_joint",
        "right_hip_pitch_joint",
        "right_hip_roll_joint",
        "right_knee_joint",
    ]
    stance_s = np.array([0.1, 0.2, 0.3, 0.4])
    target_s = np.array([1.0, 1.0, 1.0, 1.0])
    sag = freeze_leg_targets(
        names_s,
        target_s,
        stance_s,
        freeze_legs=False,
        freeze_sides=("left",),
        freeze_swing_sagittal=True,
    )
    np.testing.assert_allclose(sag, [0.1, 0.2, 1.0, 0.4])


def test_com_support_xy_planted_left() -> None:
    from core4d_dual_retarget.controller import com_support_xy

    left = np.array([0.0, 0.14])
    right = np.array([0.0, -0.14])
    np.testing.assert_allclose(com_support_xy(left, right), [0.0, 0.0])
    np.testing.assert_allclose(
        com_support_xy(left, right, mode="left"), [0.0, 0.14]
    )
    np.testing.assert_allclose(
        com_support_xy(
            left, right, mode="planted", planted_sides=("left",)
        ),
        [0.0, 0.14],
    )
    np.testing.assert_allclose(
        com_support_xy(
            left, right, mode="planted", left_z=0.04, right_z=0.067, margin=0.008
        ),
        [0.0, 0.14],
    )
    np.testing.assert_allclose(
        com_support_xy(
            left, right, mode="planted", left_z=0.083, right_z=0.080, margin=0.008
        ),
        [0.0, -0.14],
    )
    np.testing.assert_allclose(
        com_support_xy(
            left, right, mode="planted", planted_sides=("left", "right")
        ),
        [0.0, 0.0],
    )


def test_apply_swing_sole_hold_skips_empty_swing() -> None:
    from core4d_dual_retarget.controller import apply_swing_sole_hold

    names = ["left_hip_pitch_joint"]
    target = np.array([0.4])
    out = apply_swing_sole_hold(
        None, None, "r2_", names, target, {}, (), {}
    )
    np.testing.assert_allclose(out, target)


def test_apply_swing_com_hold_skips_empty_swing() -> None:
    from core4d_dual_retarget.controller import apply_swing_com_hold

    names = ["right_hip_pitch_joint"]
    target = np.array([0.4])
    out = apply_swing_com_hold(
        None, None, "r2_", names, target, {}, (), np.array([0.0, 0.0]), {}
    )
    np.testing.assert_allclose(out, target)


def test_hold_com_xy_reduces_freeze_stance_error(tmp_path) -> None:
    import os
    from pathlib import Path

    os.environ.setdefault("MUJOCO_GL", "egl")
    import mujoco

    from core4d_dual_retarget.controller import (
        hold_com_xy,
        is_leg_joint,
        joint_side,
        min_sole_center_z,
        planted_foot_sides,
    )
    from core4d_dual_retarget.mujoco_scene import build_dual_g1_scene_xml
    from core4d_dual_retarget.pd_tracking import _load_trajectory
    from core4d_dual_retarget.retarget.equilibrium import _set_static_pose

    source = Path("third_party/unitree_ros/robots/g1_description/g1_29dof_rev_1_0.xml")
    mesh = Path("data/core4d/object_models/object_models/box/box001_m.obj")
    out = tmp_path / "scene.xml"
    build_dual_g1_scene_xml(source, mesh, out, foot_boxes=True, position_actuators=True)
    model = mujoco.MjModel.from_xml_path(str(out))
    data = mujoco.MjData(model)
    stance = Path("outputs/stance/20231108_018")
    trajectories = {
        1: _load_trajectory(stance / "robot1.npz"),
        2: _load_trajectory(stance / "robot2.npz"),
    }
    names = [str(name) for name in trajectories[2]["joint_names"]]
    frame0 = {
        1: np.array(trajectories[1]["joint_positions"][0], dtype=np.float64),
        2: np.array(trajectories[2]["joint_positions"][0], dtype=np.float64),
    }
    roots0 = {
        1: np.array(trajectories[1]["root_translation"][0], dtype=np.float64),
        2: np.array(trajectories[2]["root_translation"][0], dtype=np.float64),
    }
    _set_static_pose(model, data, trajectories, 0, roots0, frame0)
    mujoco.mj_forward(model, data)
    pelvis = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "r2_pelvis")
    com0 = np.array(data.subtree_com[pelvis][:2], dtype=np.float64)
    z_left = min_sole_center_z(model, data, "r2_", "left")
    z_right = min_sole_center_z(model, data, "r2_", "right")
    planted = planted_foot_sides(z_left, z_right, 0.008)
    swing = tuple(side for side in ("left", "right") if side not in planted)[0]
    z_ref = min_sole_center_z(model, data, "r2_", swing)
    clip = np.array(trajectories[2]["joint_positions"][8], dtype=np.float64)
    mixed = np.array(frame0[2], copy=True)
    for index, name in enumerate(names):
        if is_leg_joint(name) and joint_side(name) == swing:
            mixed[index] = clip[index]
    joints = {1: frame0[1], 2: mixed}
    _set_static_pose(model, data, trajectories, 0, roots0, joints)
    mujoco.mj_forward(model, data)
    before = float(np.linalg.norm(data.subtree_com[pelvis][:2] - com0))
    z_before = min_sole_center_z(model, data, "r2_", swing)
    error, z_after = hold_com_xy(
        model, data, "r2_", com0, swing_side=swing, z_ref=z_ref
    )
    after = float(np.linalg.norm(error))
    assert before > 0.003
    assert after < before
    if z_before < z_ref - 0.001:
        assert z_after >= z_before - 1e-4
        assert z_after >= z_ref - 0.008


def test_filter_swing_com_hold_keeps_nonswing_and_frame0(tmp_path) -> None:
    import os
    from pathlib import Path

    os.environ.setdefault("MUJOCO_GL", "egl")

    from core4d_dual_retarget.controller import is_leg_joint, joint_side
    from core4d_dual_retarget.mujoco_scene import build_dual_g1_scene_xml
    from core4d_dual_retarget.pd_tracking import _load_trajectory
    from core4d_dual_retarget.retarget.equilibrium import filter_swing_com_hold

    source = Path("third_party/unitree_ros/robots/g1_description/g1_29dof_rev_1_0.xml")
    mesh = Path("data/core4d/object_models/object_models/box/box001_m.obj")
    scene = tmp_path / "scene.xml"
    build_dual_g1_scene_xml(source, mesh, scene, foot_boxes=True, position_actuators=True)
    stance = Path("outputs/stance/20231108_018")
    outputs, info = filter_swing_com_hold(
        scene, stance / "robot1.npz", stance / "robot2.npz", max_joint_step=0.08
    )
    orig = _load_trajectory(stance / "robot2.npz")
    names = [str(name) for name in orig["joint_names"]]
    swing = tuple(info["swing_sides"])
    keep = np.array(
        [not (is_leg_joint(name) and joint_side(name) in swing) for name in names]
    )
    np.testing.assert_allclose(
        outputs[2]["joint_positions"][0], orig["joint_positions"][0]
    )
    np.testing.assert_allclose(
        outputs[2]["joint_positions"][:, keep], orig["joint_positions"][:, keep]
    )
    assert float(info["peak_swing_step_rad"]) <= 0.0801
    assert float(info["dcom_after_max_m"]) < float(info["dcom_before_max_m"])


def test_synthesize_hover_keeps_frame0(tmp_path) -> None:
    import os
    from pathlib import Path

    os.environ.setdefault("MUJOCO_GL", "egl")

    from core4d_dual_retarget.mujoco_scene import build_dual_g1_scene_xml
    from core4d_dual_retarget.pd_tracking import _load_trajectory
    from core4d_dual_retarget.retarget.equilibrium import synthesize_hover

    source = Path("third_party/unitree_ros/robots/g1_description/g1_29dof_rev_1_0.xml")
    mesh = Path("data/core4d/object_models/object_models/box/box001_m.obj")
    scene = tmp_path / "scene.xml"
    build_dual_g1_scene_xml(source, mesh, scene, foot_boxes=True, position_actuators=True)
    stance = Path("outputs/stance/20231108_018")
    outputs, info = synthesize_hover(
        scene,
        stance / "robot1.npz",
        stance / "robot2.npz",
        yaw_amp=0.12,
        hold_com=False,
    )
    orig = _load_trajectory(stance / "robot2.npz")
    names = [str(name) for name in orig["joint_names"]]
    yaw = names.index("right_hip_yaw_joint")
    np.testing.assert_allclose(
        outputs[2]["joint_positions"][0], orig["joint_positions"][0]
    )
    assert float(np.ptp(outputs[2]["joint_positions"][:, yaw])) > 0.2
    assert info["swing_sides"] == ["right"]


def test_blend_arm_targets_scales_shoulders_only() -> None:
    from core4d_dual_retarget.controller import blend_arm_targets

    names = [
        "left_hip_pitch_joint",
        "left_shoulder_pitch_joint",
        "right_elbow_joint",
    ]
    frame0 = np.array([0.1, 0.2, 0.3])
    target = np.array([1.1, 1.2, 1.3])
    half = blend_arm_targets(names, target, frame0, 0.5)
    assert half[0] == 1.1
    assert half[1] == 0.7
    assert half[2] == 0.8
    frozen = blend_arm_targets(names, target, frame0, 0.0)
    assert frozen[1] == 0.2
    assert frozen[2] == 0.3


def test_blend_leg_targets_scales_hips_only() -> None:
    from core4d_dual_retarget.controller import blend_leg_targets

    names = [
        "left_hip_pitch_joint",
        "left_shoulder_pitch_joint",
        "right_knee_joint",
    ]
    frame0 = np.array([0.1, 0.2, 0.3])
    target = np.array([1.1, 1.2, 1.3])
    half = blend_leg_targets(names, target, frame0, 0.5)
    np.testing.assert_allclose(half, [0.6, 1.2, 0.8])
    frozen = blend_leg_targets(names, target, frame0, 0.0)
    assert frozen[0] == 0.1
    assert frozen[2] == 0.3


def test_hip_pitch_target_bias_signs() -> None:
    from core4d_dual_retarget.controller import hip_pitch_target_bias

    names = [
        "left_hip_pitch_joint",
        "right_hip_pitch_joint",
        "left_ankle_pitch_joint",
        "waist_pitch_joint",
    ]
    extras = hip_pitch_target_bias(names, 0.03)
    assert extras["left_hip_pitch_joint"] == 0.03
    assert extras["right_hip_pitch_joint"] == 0.03
    assert "left_ankle_pitch_joint" not in extras
    assert hip_pitch_target_bias(names, 0.0) == {}


def test_hip_roll_target_bias_opposite_sides() -> None:
    from core4d_dual_retarget.controller import hip_roll_target_bias

    extras = hip_roll_target_bias(
        ["left_hip_roll_joint", "right_hip_roll_joint", "left_hip_pitch_joint"],
        0.04,
    )
    assert extras["left_hip_roll_joint"] == 0.04
    assert extras["right_hip_roll_joint"] == -0.04
    assert "left_hip_pitch_joint" not in extras


def test_ankle_roll_target_bias_opposite_sides() -> None:
    from core4d_dual_retarget.controller import ankle_roll_target_bias

    extras = ankle_roll_target_bias(
        ["left_ankle_roll_joint", "right_ankle_roll_joint", "left_ankle_pitch_joint"],
        0.03,
    )
    assert extras["left_ankle_roll_joint"] == 0.03
    assert extras["right_ankle_roll_joint"] == -0.03
    assert "left_ankle_pitch_joint" not in extras


def test_r1_late_pitch_defaults_are_inert() -> None:
    from core4d_dual_retarget.pd_tracking import PDTrackingConfig

    config = PDTrackingConfig()
    assert config.r1_late_hip_pitch == 0.0
    assert config.r1_late_pitch_gain is None
    assert abs(config.r1_late_delay - 4.0) < 1e-12
    assert config.r1_late_pitch_delay is None
    assert config.r1_late2_hip_pitch == 0.0
    assert config.r1_late2_delay is None
    assert config.r1_late3_hip_pitch == 0.0
    assert config.r1_late3_delay is None
    assert config.r2_late_y_ref is None
    assert config.r2_late2_hip_pitch == 0.0
    assert config.r2_late2_delay is None
    assert config.r2_late3_hip_pitch == 0.0
    assert config.r2_late3_delay is None
    assert config.r2_late_ankle_both is False


def test_override_joint_targets() -> None:
    from core4d_dual_retarget.controller import override_joint_targets

    names = ["waist_roll_joint", "left_hip_pitch_joint"]
    target = np.array([0.1, 0.2])
    out = override_joint_targets(names, target, {"waist_roll_joint": -0.34})
    assert out[0] == -0.34
    assert out[1] == 0.2


def test_r2_ankle_kp_overrides_gainprm(tmp_path) -> None:
    import os
    from pathlib import Path

    os.environ.setdefault("MUJOCO_GL", "egl")
    import mujoco

    from core4d_dual_retarget.mujoco_scene import build_dual_g1_scene_xml
    from core4d_dual_retarget.pd_tracking import PDTrackingConfig

    source = Path("third_party/unitree_ros/robots/g1_description/g1_29dof_rev_1_0.xml")
    mesh = Path("data/core4d/object_models/object_models/box/box001_m.obj")
    out = tmp_path / "scene.xml"
    build_dual_g1_scene_xml(
        source, mesh, out, foot_boxes=True, position_actuators=True, stiff_contacts=True
    )
    # Short-circuit: apply the same override loop as simulate.
    model = mujoco.MjModel.from_xml_path(str(out))
    config = PDTrackingConfig(control_mode="position", r2_ankle_kp=500.0, waist_kp=80.0)
    for actuator_id in range(int(model.nu)):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_id) or ""
        if "waist" in name:
            model.actuator_gainprm[actuator_id, 0] = float(config.waist_kp)
        if (
            config.r2_ankle_kp is not None
            and name.startswith("r2_")
            and "ankle" in name
        ):
            model.actuator_gainprm[actuator_id, 0] = float(config.r2_ankle_kp)
    r2 = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "r2_left_ankle_pitch_joint")
    r1 = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "r1_left_ankle_pitch_joint")
    assert model.actuator_gainprm[r2, 0] == 500.0
    assert model.actuator_gainprm[r1, 0] == 200.0


def test_hold_sole_height_raises_depressed_foot(tmp_path) -> None:
    import os
    from pathlib import Path

    os.environ.setdefault("MUJOCO_GL", "egl")
    import mujoco

    from core4d_dual_retarget.controller import hold_sole_height, min_sole_center_z
    from core4d_dual_retarget.mujoco_scene import build_dual_g1_scene_xml

    source = Path("third_party/unitree_ros/robots/g1_description/g1_29dof_rev_1_0.xml")
    mesh = Path("data/core4d/object_models/object_models/box/box001_m.obj")
    out = tmp_path / "scene.xml"
    build_dual_g1_scene_xml(
        source, mesh, out, foot_boxes=True, position_actuators=True, stiff_contacts=True
    )
    model = mujoco.MjModel.from_xml_path(str(out))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    z0 = min_sole_center_z(model, data, "r2_", "right")
    joint = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_JOINT, "r2_right_ankle_pitch_joint"
    )
    data.qpos[int(model.jnt_qposadr[joint])] += 0.2
    mujoco.mj_forward(model, data)
    z_low = min_sole_center_z(model, data, "r2_", "right")
    z_held = hold_sole_height(model, data, "r2_", "right", z0)
    assert z_low < z0 - 0.002
    assert z_held > z_low + 0.001


def test_shorten_stagger_reduces_split(tmp_path) -> None:
    import os
    from pathlib import Path

    os.environ.setdefault("MUJOCO_GL", "egl")
    import mujoco

    from core4d_dual_retarget.controller import shorten_sagittal_stagger, yaw_from_xmat
    from core4d_dual_retarget.mujoco_scene import build_dual_g1_scene_xml

    source = Path("third_party/unitree_ros/robots/g1_description/g1_29dof_rev_1_0.xml")
    mesh = Path("data/core4d/object_models/object_models/box/box001_m.obj")
    out = tmp_path / "scene.xml"
    build_dual_g1_scene_xml(source, mesh, out, foot_boxes=True, position_actuators=True)
    model = mujoco.MjModel.from_xml_path(str(out))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    pelvis = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "r1_pelvis")
    left = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "r1_left_ankle_roll_link")
    right = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "r1_right_ankle_roll_link")
    yaw = yaw_from_xmat(data.xmat[pelvis])
    left_dir = np.array([-np.sin(yaw), np.cos(yaw)])
    before = abs(float(np.dot(data.xpos[left][:2] - data.xpos[right][:2], left_dir)))
    after_split = shorten_sagittal_stagger(model, data, "r1_", target_split=0.05)
    assert abs(after_split) <= max(before, 0.05) + 0.02
    assert np.isfinite(data.qpos).all()


def test_peak_foot_xy_drift() -> None:
    from core4d_dual_retarget.controller import peak_foot_xy_drift

    feet = np.zeros((4, 2, 2, 3))
    feet[3, 0, 0, 0] = 0.2
    feet[2, 1, 1, 1] = 0.05
    peak = peak_foot_xy_drift(feet)
    assert peak.shape == (2, 2)
    assert peak[0, 0] == 0.2
    assert peak[1, 1] == 0.05
    assert peak[0, 1] == 0.0


def test_max_foot_xy_speed() -> None:
    from core4d_dual_retarget.controller import max_foot_xy_speed

    previous = np.zeros((2, 2))
    current = np.array([[0.02, 0.0], [0.0, 0.0]])
    assert abs(max_foot_xy_speed(current, previous, 0.002) - 10.0) < 1e-9


def test_slip_hip_roll_opposes_leftward_skate() -> None:
    from core4d_dual_retarget.controller import slip_hip_roll_bias

    extras = slip_hip_roll_bias(
        ["left_hip_roll_joint", "right_hip_roll_joint", "left_ankle_pitch_joint"],
        0.2,
        0.0,
        0.5,
    )
    assert extras["left_hip_roll_joint"] == -0.08
    assert extras["right_hip_roll_joint"] == 0.0
    assert "left_ankle_pitch_joint" not in extras


def test_project_target_joints_restore_qpos(tmp_path) -> None:
    import os
    from pathlib import Path

    os.environ.setdefault("MUJOCO_GL", "egl")
    import mujoco

    from core4d_dual_retarget.controller import project_target_joints
    from core4d_dual_retarget.mujoco_scene import build_dual_g1_scene_xml

    source = Path("third_party/unitree_ros/robots/g1_description/g1_29dof_rev_1_0.xml")
    mesh = Path("data/core4d/object_models/object_models/box/box001_m.obj")
    out = tmp_path / "scene.xml"
    build_dual_g1_scene_xml(source, mesh, out, foot_boxes=True, position_actuators=True)
    model = mujoco.MjModel.from_xml_path(str(out))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    names = ["left_ankle_pitch_joint", "right_ankle_pitch_joint"]
    targets = np.array(
        [
            data.qpos[
                int(
                    model.jnt_qposadr[
                        mujoco.mj_name2id(
                            model, mujoco.mjtObj.mjOBJ_JOINT, "r1_left_ankle_pitch_joint"
                        )
                    ]
                )
            ],
            data.qpos[
                int(
                    model.jnt_qposadr[
                        mujoco.mj_name2id(
                            model, mujoco.mjtObj.mjOBJ_JOINT, "r1_right_ankle_pitch_joint"
                        )
                    ]
                )
            ],
        ],
        dtype=np.float64,
    )
    before = np.array(data.qpos, copy=True)
    project_target_joints(model, data, "r1_", names, targets)
    np.testing.assert_allclose(np.asarray(data.qpos), before)


def test_imu_ankle_residuals_clip_and_signs() -> None:
    from core4d_dual_retarget.controller import imu_ankle_residuals

    names = [
        "left_ankle_pitch_joint",
        "right_ankle_pitch_joint",
        "left_ankle_roll_joint",
        "right_ankle_roll_joint",
        "left_elbow_joint",
    ]
    extras = imu_ankle_residuals(
        names,
        np.array([0.2, -0.1]),
        np.array([0.0, 0.0]),
        (10.0, 0.0, 10.0, 0.0),
        hip_gain=0.0,
        clip=0.12,
    )
    assert extras["left_ankle_pitch_joint"] == -0.12
    assert extras["right_ankle_pitch_joint"] == extras["left_ankle_pitch_joint"]
    assert extras["right_ankle_roll_joint"] == -extras["left_ankle_roll_joint"]
    assert "left_elbow_joint" not in extras
    hips = imu_ankle_residuals(
        names + ["left_hip_pitch_joint", "right_hip_roll_joint"],
        np.array([0.2, 0.0]),
        np.array([0.0, 0.0]),
        (1.0, 0.0, 1.0, 0.0),
        apply_to="hip",
        clip=0.5,
    )
    assert "left_ankle_pitch_joint" not in hips
    assert hips["left_hip_pitch_joint"] == -0.2


def test_residual_hold_cost_penalizes_early_fall() -> None:
    from core4d_dual_retarget.residual import residual_hold_cost

    standing = {
        "robots": {
            "1": {"time_to_fall_s": None, "minimum_pelvis_height_m": 0.80},
            "2": {"time_to_fall_s": None, "minimum_pelvis_height_m": 0.78},
        }
    }
    fallen = {
        "robots": {
            "1": {"time_to_fall_s": 3.0, "minimum_pelvis_height_m": 0.05},
            "2": {"time_to_fall_s": None, "minimum_pelvis_height_m": 0.80},
        }
    }
    assert residual_hold_cost(standing, 6.0) < residual_hold_cost(fallen, 6.0)


def test_lift_seats_named_soles(tmp_path) -> None:
    import os
    from pathlib import Path

    os.environ.setdefault("MUJOCO_GL", "egl")
    import mujoco

    from core4d_dual_retarget.controller import _geom_is_sole, lift_roots_out_of_penetration
    from core4d_dual_retarget.mujoco_scene import build_dual_g1_scene_xml

    source = Path("third_party/unitree_ros/robots/g1_description/g1_29dof_rev_1_0.xml")
    mesh = Path("data/core4d/object_models/object_models/box/box001_m.obj")
    out = tmp_path / "scene.xml"
    build_dual_g1_scene_xml(
        source, mesh, out, foot_boxes=True, position_actuators=True, stiff_contacts=True
    )
    model = mujoco.MjModel.from_xml_path(str(out))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    lift_roots_out_of_penetration(model, data)
    assert np.isfinite(data.qpos).all()
    sole_ids = [
        geom_id
        for geom_id in range(int(model.ngeom))
        if _geom_is_sole(model, geom_id)
    ]
    assert len(sole_ids) == 4
