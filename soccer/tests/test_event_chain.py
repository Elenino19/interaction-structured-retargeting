import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from evaluate_event_chain import annotated_events, transfer_pairs  # noqa: E402

CONTACTS = [
    {"frame": 0, "player": "a", "side": "left", "kick_side": "right", "kick_frame": 0},
    {"frame": 15, "player": "b", "side": "right", "kick_side": "left", "kick_frame": 21},
    {"frame": 42, "player": "a", "side": "right", "kick_side": "right", "kick_frame": 45},
    {"frame": 71, "player": "b", "side": "right"},
]


class EventChainStructureTest(unittest.TestCase):
    def test_event_split_keeps_receive_and_kick_separate(self):
        events = annotated_events(CONTACTS)
        receives = [e for e in events if e["phase"] == "receive"]
        kicks = [e for e in events if e["phase"] == "kick"]
        self.assertEqual(len(receives), 3)
        self.assertEqual(len(kicks), 4)
        self.assertEqual((receives[0]["frame"], receives[0]["foot"]), (15, "right"))
        self.assertEqual((kicks[1]["frame"], kicks[1]["foot"]), (21, "left"))
        # Missing kick fields fall back to the receive annotation.
        self.assertEqual((kicks[3]["frame"], kicks[3]["foot"]), (71, "right"))

    def test_transfer_pairs_are_kick_to_next_receive_and_cross_agent(self):
        pairs = transfer_pairs(CONTACTS)
        self.assertEqual(len(pairs), 3)
        first = pairs[0]
        self.assertEqual((first["sender"], first["sender_foot"], first["kick_frame"]), ("a", "right", 0))
        self.assertEqual((first["receiver"], first["receiver_foot"], first["receive_frame"]), ("b", "right", 15))
        self.assertTrue(all(p["cross_agent"] for p in pairs))
        self.assertEqual(pairs[1]["kick_event_id"], "e01_kick")
        self.assertEqual(pairs[1]["receive_event_id"], "e02_recv")


if __name__ == "__main__":
    unittest.main()
