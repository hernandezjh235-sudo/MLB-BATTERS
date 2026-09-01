import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import member_board_sync as sync


class FakeStreamlit:
    def __init__(self, session_state=None):
        self.session_state = session_state or {}


class FakeResponse:
    status = 201

    def getcode(self):
        return self.status

    def read(self, _limit=-1):
        return b'{"ok":true}'

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False


class MemberBoardSyncTests(unittest.TestCase):
    def test_missing_configuration_is_a_safe_noop(self):
        with patch.dict(os.environ, {}, clear=True):
            result = sync.sync_streamlit_member_boards(FakeStreamlit())
        self.assertFalse(result["ok"])
        self.assertEqual(result["status"], "NOT_CONFIGURED")

    def test_sanitizer_maps_only_member_contract_fields(self):
        item = sync._member_item(
            {
                "Player": "Juan Soto",
                "Team": "NYY",
                "Opponent": "BOS",
                "Best Market": "H+R+RBI",
                "Best Pick": "HIGHER",
                "Best Line": "1.5",
                "Best Projection": 2.14,
                "Best Win/Hit %": "63.2%",
                "Risk Flags": "LINEUP NOT CONFIRMED | WEATHER",
                "private_model_weight": 9.99,
            },
            "upside",
            1,
            "FROZEN_UPSIDE_BOARD",
            "2026-09-01",
        )
        self.assertEqual(item["player"], "Juan Soto")
        self.assertEqual(item["line"], 1.5)
        self.assertEqual(item["winProbability"], 63.2)
        self.assertNotIn("private_model_weight", item)
        self.assertEqual(item["flags"], ["LINEUP NOT CONFIRMED", "WEATHER"])

    def test_posts_all_available_saved_boards_once(self):
        session = {
            "ow_manual_refresh_generation_v11": 4,
            "ow_manual_refresh_completed_at_v11": "2026-09-01T12:00:00Z",
            "ow_core_board_cache_v7": {
                "HRR": {"df": [{"Player": "A", "Line": 1.5}], "built_at": "2026-09-01T12:00:00Z"},
                "HOME_RUNS": {"df": [{"Player": "B", "HR Line": 0.5}], "built_at": "2026-09-01T12:00:00Z"},
                "BATTER_UPSIDE": {"df": [{"Player": "C", "Best Market": "Fantasy"}], "built_at": "2026-09-01T12:00:00Z"},
            },
            "ow_bfs_df": [{"UD Player": "D", "FS Projection": 8.4}],
            "ow_bfs_built_at": "2026-09-01T12:00:00Z",
        }
        fake_st = FakeStreamlit(session)
        with tempfile.TemporaryDirectory() as temp_dir:
            pick_path = Path(temp_dir) / "picks.json"
            result_path = Path(temp_dir) / "results.json"
            pick_path.write_text(json.dumps([{"Player": "E", "Market": "HRR", "pick_id": "e1"}]), encoding="utf-8")
            result_path.write_text(json.dumps([{"Player": "F", "Market": "HRR", "graded_result": "WIN"}]), encoding="utf-8")
            env = {
                "MEMBER_BOARD_INGEST_URL": "https://example.test/api/batter/ingest",
                "MEMBER_BOARD_SYNC_TOKEN": "test-token",
                "MEMBER_SITE_BYPASS_TOKEN": "site-bypass-token",
            }
            with patch.dict(os.environ, env, clear=False), patch.object(sync, "_open_request", return_value=FakeResponse()) as post:
                first = sync.sync_streamlit_member_boards(fake_st, temp_dir, pick_path, result_path, "TEST")
                second = sync.sync_streamlit_member_boards(fake_st, temp_dir, pick_path, result_path, "TEST")
        self.assertTrue(first["ok"])
        self.assertEqual(post.call_count, 7)
        self.assertEqual(set(first["posted"]), {"upside", "games", "hrr", "home-runs", "fantasy", "official", "results"})
        self.assertEqual(second["posted"], [])
        self.assertEqual(len(second["skipped"]), 7)
        for call in post.call_args_list:
            request = call.args[0]
            self.assertTrue(request.get_header("Authorization").startswith("Bearer "))
            self.assertTrue(request.get_header("Oai-sites-authorization").startswith("Bearer "))
            self.assertEqual(request.get_method(), "POST")


if __name__ == "__main__":
    unittest.main()
