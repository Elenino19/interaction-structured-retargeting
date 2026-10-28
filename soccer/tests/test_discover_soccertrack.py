import importlib.util
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).parents[1] / "scripts" / "discover_soccertrack.py"
SPEC = importlib.util.spec_from_file_location("discover_soccertrack", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class DiscoverTests(unittest.TestCase):
    def test_visible_same_team_follow_up_is_retained(self):
        data = {
            "annotations": [
                {
                    "gameTime": "1 - 10:00",
                    "position": "600000",
                    "label": "Throw In",
                    "team": "left",
                    "player_id": "a",
                    "visibility": "visible",
                },
                {
                    "gameTime": "1 - 10:05",
                    "position": "605000",
                    "label": "Drive",
                    "team": "left",
                    "player_id": "b",
                    "visibility": "visible",
                },
                {
                    "gameTime": "1 - 10:09",
                    "position": "609000",
                    "label": "Pass",
                    "team": "left",
                    "player_id": "b",
                    "visibility": "visible",
                },
            ]
        }

        candidates = MODULE.discover(data, lookahead_seconds=12)

        self.assertEqual(len(candidates), 1)
        self.assertEqual(
            [item["label"] for item in candidates[0]["follow_up_events"]],
            ["Drive", "Pass"],
        )

    def test_other_team_and_out_of_window_events_are_excluded(self):
        data = {
            "annotations": [
                {
                    "gameTime": "2 - 01:00",
                    "position": "60000",
                    "label": "Throw In",
                    "team": "right",
                    "visibility": "visible",
                },
                {
                    "gameTime": "2 - 01:03",
                    "position": "63000",
                    "label": "Pass",
                    "team": "left",
                    "visibility": "visible",
                },
                {
                    "gameTime": "2 - 01:20",
                    "position": "80000",
                    "label": "Pass",
                    "team": "right",
                    "visibility": "visible",
                },
            ]
        }

        candidates = MODULE.discover(data, lookahead_seconds=12)

        self.assertEqual(candidates[0]["follow_up_events"], [])


if __name__ == "__main__":
    unittest.main()
