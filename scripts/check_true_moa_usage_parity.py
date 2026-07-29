#!/usr/bin/env python3
"""Run the language-neutral true-MoA usage projection/merge corpus."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from xiaoban.trusted_runtime.true_moa_durable_usage import (  # noqa: E402
    _merge_usage,
    project_true_moa_usage,
)

FIXTURE_PATH = (
    REPO_ROOT / "contracts" / "true-moa-usage-parity.v1.json"
)


def _mutate(value: Any, mutations: list[dict[str, Any]]) -> Any:
    result = copy.deepcopy(value)
    for mutation in mutations:
        path = list(mutation["path"])
        parent = result
        for part in path[:-1]:
            parent = parent[part]
        target = path[-1]
        if mutation["action"] == "set":
            parent[target] = copy.deepcopy(mutation.get("value"))
        elif mutation["action"] == "remove":
            if isinstance(parent, list):
                parent.pop(int(target))
            else:
                parent.pop(target, None)
        else:
            raise ValueError("invalid parity mutation")
    return result


def parity_results() -> list[dict[str, Any]]:
    fixture = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
    base = fixture["base"]
    results: list[dict[str, Any]] = []
    for case in fixture["cases"]:
        incoming = _mutate(base, case["mutations"])
        try:
            if case["operation"] == "project":
                projected = project_true_moa_usage(incoming)
            elif case["operation"] == "merge":
                projected = _merge_usage(base, incoming)
            else:
                raise ValueError("invalid parity operation")
        except ValueError:
            outcome = (
                "invalid"
                if case["operation"] == "project"
                else "conflict"
            )
            item = {"name": case["name"], "outcome": outcome}
        else:
            item = {
                "name": case["name"],
                "outcome": "ok",
                "value": projected,
            }
        if item["outcome"] != case["expected"]:
            raise AssertionError(
                f"{case['name']}: {item['outcome']} "
                f"!= {case['expected']}"
            )
        results.append(item)
    return results


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    results = parity_results()
    if args.json:
        print(
            json.dumps(
                results,
                ensure_ascii=True,
                sort_keys=True,
                separators=(",", ":"),
            )
        )
    else:
        print(f"ok true MoA usage parity ({len(results)} cases)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
