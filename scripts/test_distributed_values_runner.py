#!/usr/bin/env python3
"""Check that the distributed launcher reaps peers after rank failures/timeouts."""

import json
from pathlib import Path
import tempfile
import time
import unittest

from test_distributed_values import run_two_ranks


class DistributedRunnerTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory(prefix="tyr-dist-runner-")
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.executable = self.root / "rank"
        self.logs = self.root / "logs"

    def fixture(self, code):
        self.executable.write_text("#!/usr/bin/env python3\nimport sys, time\n" + code)
        self.executable.chmod(0o755)

    def report(self):
        return json.loads((self.logs / "report.json").read_text())

    def test_success_requires_both_ranks(self):
        self.fixture('print("passed on rank", sys.argv[1])\n')
        self.assertEqual(run_two_ranks(self.executable, self.logs, 10, 29500), 0)
        self.assertEqual([rank["exit_code"] for rank in self.report()["ranks"]], [0, 0])
        self.assertIn("rank 1", (self.logs / "rank-1.log").read_text())

    def test_failed_rank_reaps_blocked_peer(self):
        self.fixture('if sys.argv[1] == "1":\n    print("fixture failure"); sys.exit(7)\ntime.sleep(60)\n')
        started = time.monotonic()
        self.assertEqual(run_two_ranks(self.executable, self.logs, 20, 29500), 1)
        self.assertLess(time.monotonic() - started, 10)
        ranks = self.report()["ranks"]
        self.assertIsNotNone(ranks[0]["exit_code"], "Blocked peer was not reaped")
        self.assertEqual(ranks[1]["exit_code"], 7)

    def test_timeout_reaps_both_ranks(self):
        self.fixture('time.sleep(60)\n')
        self.assertEqual(run_two_ranks(self.executable, self.logs, 0.2, 29500), 1)
        self.assertTrue(all(rank["exit_code"] is not None for rank in self.report()["ranks"]))
        self.assertIn("exceeded", self.report()["failure"])


if __name__ == "__main__":
    unittest.main()
