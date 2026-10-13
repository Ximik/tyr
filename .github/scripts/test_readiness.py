#!/usr/bin/env python3
"""Runner-label selection for the hosted qualification preflight."""
import json
import unittest

from readiness import configuration


class ReadinessTests(unittest.TestCase):
    def test_runner_association_is_explicit(self):
        with self.assertRaises(ValueError):
            configuration("ranvier-labs/tyr", "")
        labels = configuration("cpehle/tyr", "")
        self.assertIn("gb10", json.loads(labels))
        self.assertIn("tyr-qualification", json.loads(configuration("ranvier-labs/tyr",
            '["self-hosted", "Linux", "ARM64", "tyr-qualification", "gb10"]')))


if __name__ == "__main__":
    unittest.main()
