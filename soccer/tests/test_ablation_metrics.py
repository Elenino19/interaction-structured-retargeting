import importlib.util
import unittest
from pathlib import Path

import numpy as np


MODULE_PATH = (
    Path(__file__).parents[1] / "scripts" / "evaluate_ablation_experiments.py"
)
SPEC = importlib.util.spec_from_file_location("evaluate_ablation", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def robot_with_feet(left: np.ndarray, right: np.ndarray) -> dict[str, np.ndarray]:
    frames = len(left)
    names = np.asarray(
        ["pelvis", "left_ankle", "left_toe", "right_ankle", "right_toe"]
    )
    keypoints = np.zeros((frames, len(names), 3), dtype=np.float64)
    keypoints[:, 0, 2] = 0.8
    keypoints[:, 1] = left
    keypoints[:, 2] = left
    keypoints[:, 3] = right
    keypoints[:, 4] = right
    return {"keypoint_names": names, "robot_keypoints": keypoints}


class AblationMetricTests(unittest.TestCase):
    def test_contact_metric_prefers_annotated_foot(self):
        left = np.tile(np.array([0.0, 0.5, 0.11]), (3, 1))
        right = np.tile(np.array([0.0, 0.0, 0.11]), (3, 1))
        robots = {
            "a": robot_with_feet(left, right),
            "b": robot_with_feet(left, right),
        }
        ball = np.tile(np.array([0.0, 0.10, 0.11]), (3, 1))
        contacts = [
            {
                "frame": 0,
                "kick_frame": 0,
                "player": "a",
                "side": "left",
                "kick_side": "right",
            },
            {
                "frame": 2,
                "kick_frame": 2,
                "player": "b",
                "side": "right",
                "kick_side": "right",
            },
        ]

        result = MODULE.contact_metrics(robots, ball, contacts)

        self.assertGreater(result["wrong_minus_target_margin_mean_m"], 0.0)
        self.assertEqual(result["wrong_foot_farther_fraction"], 1.0)

    def test_straight_central_pass_has_zero_path_excess(self):
        left = np.tile(np.array([0.0, -0.25, 0.11]), (4, 1))
        right = left.copy()
        robots = {
            "a": robot_with_feet(left, right),
            "b": robot_with_feet(left, right),
        }
        ball = np.column_stack(
            [
                np.linspace(-1.0, 1.0, 4),
                np.full(4, -0.25),
                np.full(4, 0.11),
            ]
        )
        cones = np.array(
            [[0.0, y, 0.05] for y in (1.0, 0.5, 0.0, -0.5, -1.0)]
        )
        contacts = [
            {"frame": 0, "player": "a", "side": "left"},
            {"frame": 3, "player": "b", "side": "left"},
        ]
        original_lanes = MODULE.PASS_LANES
        MODULE.PASS_LANES = (3,)
        try:
            result = MODULE.pass_metrics(
                robots,
                ball,
                cones,
                contacts,
                np.asarray([1.25, 0.75, 0.25, -0.25, -0.75, -1.25]),
            )
        finally:
            MODULE.PASS_LANES = original_lanes

        self.assertAlmostEqual(result["pass_curvature_path_excess_mean"], 0.0)
        self.assertAlmostEqual(result["lane_crossing_error_mean_m"], 0.0)


if __name__ == "__main__":
    unittest.main()
