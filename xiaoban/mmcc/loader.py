"""Load MMCC manifests from JSON or YAML files."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .manifest import MMCCManifest
from .validator import validate_manifest


def load_manifest(path: str | Path, *, validate: bool = True) -> MMCCManifest:
    manifest_path = Path(path)
    raw = _read_mapping(manifest_path)
    manifest = MMCCManifest.from_dict(raw)
    return validate_manifest(manifest) if validate else manifest


def _read_mapping(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() in {".yaml", ".yml"}:
        try:
            import yaml
        except ModuleNotFoundError as exc:
            raise RuntimeError("PyYAML is required to load YAML MMCC manifests") from exc
        loaded = yaml.safe_load(text)
    else:
        loaded = json.loads(text)
    if not isinstance(loaded, dict):
        raise ValueError(f"MMCC manifest must be a mapping: {path}")
    return loaded
