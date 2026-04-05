from __future__ import annotations

import unittest
from pathlib import Path

from run_generator_app_v3 import parse_args


class GeneratorAppEntrypointTests(unittest.TestCase):
    def test_parse_args_uses_v3_defaults(self) -> None:
        args = parse_args([])
        self.assertEqual(args.presets_dir, Path("presets_v3"))
        self.assertEqual(args.profiles_dir, Path("profiles_v3"))
