"""Protect third-party npm package names from Xiaoban rebranding."""

from __future__ import annotations

import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent


def test_hermes_parser_packages_keep_their_upstream_names():
    lock_path = REPO_ROOT / "package-lock.json"
    lock_text = lock_path.read_text(encoding="utf-8")
    lock = json.loads(lock_text)
    packages = lock["packages"]

    assert "xiaoban-parser" not in lock_text
    assert "xiaoban-estree" not in lock_text
    assert "node_modules/hermes-parser" in packages
    assert "node_modules/hermes-estree" in packages
    assert packages["node_modules/eslint-plugin-react-hooks"]["dependencies"][
        "hermes-parser"
    ] == "^0.25.1"
    assert packages["node_modules/hermes-parser"]["dependencies"][
        "hermes-estree"
    ] == "0.25.1"
