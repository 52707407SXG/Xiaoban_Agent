"""Contract tests for the Docker stage2 immutable install-tree policy.

Hosted/container Xiaoban keeps user-writable state under XIAOBAN_HOME
(/opt/data). The installed source, venv, TUI bundle, and node_modules under
/opt/xiaoban must remain root-owned/non-writable by the runtime xiaoban user so
an agent session cannot self-modify the installation and brick the gateway.
"""
from __future__ import annotations

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
STAGE2_HOOK = REPO_ROOT / "docker" / "stage2-hook.sh"


@pytest.fixture(scope="module")
def stage2_text() -> str:
    if not STAGE2_HOOK.exists():
        pytest.skip("docker/stage2-hook.sh not present in this checkout")
    return STAGE2_HOOK.read_text()


def test_stage2_does_not_chown_install_tree_to_xiaoban(stage2_text: str) -> None:
    assert "Fixing ownership of build trees under $INSTALL_DIR" not in stage2_text
    assert 'chown -R xiaoban:xiaoban \\\n        "$INSTALL_DIR/.venv"' not in stage2_text

    assert "venv_owner=$(stat -c %u \"$INSTALL_DIR/.venv\"" not in stage2_text
    assert "chown of build trees failed" not in stage2_text
    for install_tree in (
        '"$INSTALL_DIR/.venv" \\',
        '"$INSTALL_DIR/ui-tui" \\',
        '"$INSTALL_DIR/gateway" \\',
        '"$INSTALL_DIR/node_modules" \\',
    ):
        assert install_tree not in stage2_text, (
            f"stage2 must not chown {install_tree} back to xiaoban; "
            "the Dockerfile keeps /opt/xiaoban immutable and writable state "
            "belongs under XIAOBAN_HOME"
        )


def test_stage2_documents_immutable_install_contract(stage2_text: str) -> None:
    assert "Immutable install tree" in stage2_text
    assert "PYTHONDONTWRITEBYTECODE" in stage2_text
    assert "XIAOBAN_DISABLE_LAZY_INSTALLS=1" in stage2_text
    assert "/opt/xiaoban" in stage2_text
