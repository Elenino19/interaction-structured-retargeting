import importlib.util
import unittest
from pathlib import Path

import numpy as np


MODULE_PATH = Path(__file__).parents[1] / "scripts" / "evaluate_soccer_paper.py"
SPEC = importlib.util.spec_from_file_location("evaluate_soccer_paper", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def make_robot(frames: int = 12) -> dict[str, np.ndarray]:
    names = np.asarray(
        ["pelvis", "left_ankle", "left_toe", "right_ankle", "right_toe"]
    )
    keypoints = np.zeros((frames, len(names), 3), dtype=np.float64)
    keypoints[:, 0, 2] = 0.8
    keypoints[:, 1:3, 1] = -0.4
    keypoints[:, 3:5, 1] = 0.4
    return {
        "keypoint_names": names,
        "robot_keypoints": keypoints,
        "root_translation": np.zeros((frames, 3), dtype=np.float64),
        "joint_positions": np.zeros((frames, 29), dtype=np.float64),
    }


class PaperMetricTests(unittest.TestCase):
    def test_receive_and_kick_are_reported_separately(self):
        robots = {"a": make_robot(), "b": make_robot()}
        ball = np.zeros((12, 3), dtype=np.float64)
        ball[:, 2] = 0.11
        ball[:, 1] = 0.4
        contacts = [
            {
                "frame": 2,
                "kick_frame": 2,
                "player": "a",
                "side": "right",
                "kick_side": "right",
            },
            {
                "frame": 6,
                "kick_frame": 9,
                "player": "b",
                "side": "right",
                "kick_side": "left",
            },
        ]
        result = MODULE.phase_contact_and_timing(robots, ball, contacts)

        self.assertEqual(result["receive"]["count"], 1)
        self.assertEqual(result["kick"]["count"], 2)
        self.assertEqual(result["overall"]["count"], 3)
        self.assertEqual(result["receive"]["correct_foot_association_rate"], 1.0)

    def test_constant_motion_has_zero_smoothness(self):
        result = MODULE.smoothness_metrics({"a": make_robot(), "b": make_robot()})

        self.assertEqual(result["root_acceleration_rms_m_per_s2"], 0.0)
        self.assertEqual(result["root_jerk_rms_m_per_s3"], 0.0)
        self.assertEqual(result["joint_jerk_rms_rad_per_s3"], 0.0)


if __name__ == "__main__":
    unittest.main()
