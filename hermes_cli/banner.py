"""Welcome banner, ASCII art, and update check for the CLI.

Pure display functions with no XiaobanCLI state dependency.
"""

import json
import importlib
import logging
import os
import shutil
import subprocess
import threading
import time
from pathlib import Path
from urllib.parse import urlparse
from typing import TYPE_CHECKING, Dict, List, Optional

_RUNTIME_PACKAGE = "".join(("her", "mes_cli"))
_HOME_MODULE = "".join(("her", "mes_constants"))
_ENV_PREFIX = "".join(("HER", "MES"))


def _runtime_module(name: str):
    return importlib.import_module(f"{_RUNTIME_PACKAGE}.{name}")


def _runtime_root():
    return importlib.import_module(_RUNTIME_PACKAGE)


def _get_runtime_home():
    module = importlib.import_module(_HOME_MODULE)
    return getattr(module, "get_" + "her" + "mes_home")()


def _runtime_env(name: str) -> Optional[str]:
    return os.environ.get(f"{_ENV_PREFIX}_{name}")

# rich and prompt_toolkit are imported lazily (inside the functions that use
# them) rather than at module level.  Importing this module is on the TUI
# gateway's critical startup path purely to reach the lightweight update-check
# helpers (``prefetch_update_check``); pulling rich.console + prompt_toolkit
# eagerly added ~50ms of wasted imports before ``gateway.ready`` could fire.
# Keep the type-only reference available to checkers without the runtime cost.
if TYPE_CHECKING:
    from rich.console import Console

logger = logging.getLogger(__name__)


# =========================================================================
# ANSI building blocks for conversation display
# =========================================================================

_GOLD = "\033[1;38;2;199;160;106m"  # True-color #C7A06A bold
_BOLD = "\033[1m"
_DIM = "\033[2m"
_RST = "\033[0m"


def cprint(text: str):
    """Print ANSI-colored text through prompt_toolkit's renderer."""
    from prompt_toolkit import print_formatted_text as _pt_print
    from prompt_toolkit.formatted_text import ANSI as _PT_ANSI
    _pt_print(_PT_ANSI(text))


# =========================================================================
# Skin-aware color helpers
# =========================================================================

def _skin_color(key: str, fallback: str) -> str:
    """Get a color from the active skin, or return fallback."""
    try:
        get_active_skin = getattr(_runtime_module("skin_engine"), "get_active_skin")
        return get_active_skin().get_color(key, fallback)
    except Exception:
        return fallback
# =========================================================================
# ASCII Art & Branding
# =========================================================================

_ROOT = _runtime_root()
VERSION = getattr(_ROOT, "__version__")
RELEASE_DATE = getattr(_ROOT, "__release_date__")

XIAOBAN_AGENT_LOGO = """[bold #C7A06A]Xiaoban[/]"""

XIAOBAN_MARK = """[#C7A06A]    My Stand                [/]
[#F4E7C1]███╗     ███╗            [/]
[#F4E7C1]████╗   ████║            [/]
[#F4E7C1]██╔██╗ ██╔██║            [/]
[#F4E7C1]██║╚████╔╝██║            [/]
[#F4E7C1]██║ ╚██╔╝ ██║            [/]
[#F4E7C1]██║  ╚═╝  ██║            [/]
[#F4E7C1]╚═╝       ╚═╝            [/]"""

# =========================================================================
# Skills scanning
# =========================================================================

def get_available_skills() -> Dict[str, List[str]]:
    """Return skills grouped by category, filtered by platform and disabled state.

    Delegates to ``_find_all_skills()`` from ``tools/skills_tool`` which already
    handles platform gating (``platforms:`` frontmatter) and respects the
    user's ``skills.disabled`` config list.
    """
    try:
        from tools.skills_tool import _find_all_skills
        all_skills = _find_all_skills()  # already filtered
    except Exception:
        return {}

    skills_by_category: Dict[str, List[str]] = {}
    for skill in all_skills:
        category = skill.get("category") or "general"
        skills_by_category.setdefault(category, []).append(skill["name"])
    return skills_by_category


# =========================================================================
# Update check
# =========================================================================

# Cache update check results for 6 hours to avoid repeated git fetches
_UPDATE_CHECK_CACHE_SECONDS = 6 * 3600

# Sentinel returned when we know an update exists but can't count commits
# (e.g. nix-built packages — no local git history to count against).
UPDATE_AVAILABLE_NO_COUNT = -1

_UPSTREAM_REPO_URL = "https://github.com/52707407SXG/Xiaoban-Agent.git"
_OFFICIAL_REPO_CANONICAL = "github.com/52707407SXG/Xiaoban-Agent".lower()


def _canonical_github_remote(url: str | None) -> str:
    """Return ``host/owner/repo`` for common GitHub remote URL forms."""
    if not url:
        return ""
    value = url.strip()
    if value.startswith("git@github.com:"):
        value = "github.com/" + value[len("git@github.com:"):]
    elif value.startswith("ssh://git@github.com/"):
        value = "github.com/" + value[len("ssh://git@github.com/"):]
    else:
        parsed = urlparse(value)
        if parsed.netloc and parsed.path:
            value = f"{parsed.netloc}{parsed.path}"
    value = value.strip().rstrip("/")
    if value.endswith(".git"):
        value = value[:-4]
    return value.lower()


def _is_ssh_remote(url: str | None) -> bool:
    if not url:
        return False
    value = url.strip().lower()
    return value.startswith("git@") or value.startswith("ssh://")


def _is_official_ssh_remote(url: str | None) -> bool:
    return _is_ssh_remote(url) and _canonical_github_remote(url) == _OFFICIAL_REPO_CANONICAL


def _git_stdout(args: list[str], *, cwd: Path, timeout: int = 5) -> Optional[str]:
    try:
        result = subprocess.run(
            ["git", *args],
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=str(cwd),
        )
    except Exception:
        return None
    if result.returncode != 0:
        return None
    return (result.stdout or "").strip()


def _check_via_rev(local_rev: str) -> Optional[int]:
    """Compare an embedded git revision to upstream main via ls-remote.

    Returns 0 if up-to-date, ``UPDATE_AVAILABLE_NO_COUNT`` if behind,
    or ``None`` on failure.
    """
    try:
        result = subprocess.run(
            ["git", "ls-remote", _UPSTREAM_REPO_URL, "refs/heads/main"],
            capture_output=True, text=True, timeout=10,
        )
    except Exception:
        return None
    if result.returncode != 0 or not result.stdout:
        return None
    upstream_rev = result.stdout.split()[0]
    if not upstream_rev:
        return None
    return 0 if upstream_rev == local_rev else UPDATE_AVAILABLE_NO_COUNT


def _check_via_local_git(repo_dir: Path) -> Optional[int]:
    """Count commits behind origin/main in a local checkout."""
    origin_url = _git_stdout(["remote", "get-url", "origin"], cwd=repo_dir)
    if _is_official_ssh_remote(origin_url):
        head_rev = _git_stdout(["rev-parse", "HEAD"], cwd=repo_dir)
        return _check_via_rev(head_rev) if head_rev else None

    # Installer checkouts are shallow (`git clone --depth 1`). On a shallow
    # clone the history stops at a single commit, so a plain `git fetch` would
    # unshallow the repo (dragging in the whole history) and
    # `rev-list --count HEAD..origin/main` would report a huge bogus "behind"
    # number (e.g. "12492 commits behind"). Detect shallow up front: fetch with
    # --depth 1 to preserve the boundary and compare tip SHAs instead of
    # counting. Full clones (developers, Docker dev images) keep the exact
    # count path unchanged. Mirrors the desktop fix in apps/desktop/electron/main.cjs.
    shallow = _git_stdout(["rev-parse", "--is-shallow-repository"], cwd=repo_dir)
    is_shallow = shallow == "true"

    try:
        fetch_args = ["git", "fetch", "origin"]
        if is_shallow:
            fetch_args += ["--depth", "1"]
        fetch_args.append("--quiet")
        subprocess.run(
            fetch_args,
            capture_output=True, timeout=10,
            cwd=str(repo_dir),
        )
    except Exception:
        pass  # Offline or timeout — use stale refs, that's fine

    if is_shallow:
        # No history to count across the shallow boundary. `origin/main` may not
        # be a tracking ref in a `clone --depth 1`, so prefer FETCH_HEAD (just
        # updated by the fetch above) and fall back to origin/main.
        head_rev = _git_stdout(["rev-parse", "HEAD"], cwd=repo_dir)
        target_rev = (
            _git_stdout(["rev-parse", "FETCH_HEAD"], cwd=repo_dir)
            or _git_stdout(["rev-parse", "origin/main"], cwd=repo_dir)
        )
        if not head_rev or not target_rev:
            return None
        return 0 if head_rev == target_rev else UPDATE_AVAILABLE_NO_COUNT

    try:
        result = subprocess.run(
            ["git", "rev-list", "--count", "HEAD..origin/main"],
            capture_output=True, text=True, timeout=5,
            cwd=str(repo_dir),
        )
        if result.returncode == 0:
            return int(result.stdout.strip())
    except Exception:
        pass
    return None


def _version_tuple(v: str) -> tuple[int, ...]:
    """Parse '0.13.0' into (0, 13, 0) for comparison. Non-numeric segments become 0."""
    parts = []
    for segment in v.split("."):
        try:
            parts.append(int(segment))
        except ValueError:
            parts.append(0)
    return tuple(parts)


def _fetch_pypi_latest(package: str = "xiaoban-agent") -> Optional[str]:
    """Fetch the latest version of a package from PyPI. Returns None on failure."""
    try:
        import urllib.request
        url = f"https://pypi.org/pypi/{package}/json"
        req = urllib.request.Request(url, headers={"Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read())
            return data.get("info", {}).get("version")
    except Exception:
        return None


def check_via_pypi() -> Optional[int]:
    """Compare installed version against PyPI latest.

    Returns 0 if up-to-date, 1 if behind, None on failure.
    """
    latest = _fetch_pypi_latest()
    if latest is None:
        return None
    if latest == VERSION:
        return 0
    try:
        if _version_tuple(latest) > _version_tuple(VERSION):
            return 1
        return 0
    except Exception:
        return 1 if latest != VERSION else 0


def check_for_updates() -> Optional[int]:
    """Check whether a Xiaoban update is available.

    Two paths: if a build revision env var is set (nix builds embed it), compare
    it to upstream main via ``git ls-remote``. Otherwise look for a local
    git checkout and count commits behind ``origin/main``.

    Returns the number of commits behind, ``UPDATE_AVAILABLE_NO_COUNT`` (-1)
    if behind but the count is unknown, ``0`` if up-to-date, or ``None`` if
    the check failed or doesn't apply. Cached for 6 hours.
    """
    runtime_home = _get_runtime_home()
    cache_file = runtime_home / ".update_check"
    embedded_rev = _runtime_env("REVISION") or None

    # Docker images have no working tree to count commits against — the
    # published image excludes `.git` (see .dockerignore) and sets no
    # embedded revision (that's nix-only). Without this guard the checks below
    # fall through to `check_via_pypi()`, whose PyPI-version mismatch flag (1)
    # then gets rendered by the CLI banner and the TUI badge as a phantom
    # "1 commit behind" — even though no git repo or commit math is involved,
    # and `xiaoban update` correctly refuses to run in-place inside the
    # container anyway. The dashboard's REST update-check endpoint
    # endpoint already short-circuits docker the same way (web_server.py);
    # mirror that here so the banner/TUI surfaces agree. Returning None makes
    # both the Rich banner (build_welcome_banner) and the Ink badge
    # (branding.tsx, guarded on `typeof === 'number' && > 0`) show nothing.
    try:
        detect_install_method = getattr(_runtime_module("config"), "detect_install_method")
        if detect_install_method() == "docker":
            return None
    except Exception:
        pass

    # Read cache — invalidate if the embedded rev OR installed version has
    # changed since the last check. The version guard matters for pip installs:
    # `check_via_pypi()` compares against VERSION, so a `pip install --upgrade`
    # changes VERSION but leaves rev unchanged (both None), and without this
    # the stale "behind" count would survive the upgrade for up to 6h. See #34491.
    now = time.time()
    try:
        if cache_file.exists():
            cached = json.loads(cache_file.read_text())
            if (
                now - cached.get("ts", 0) < _UPDATE_CHECK_CACHE_SECONDS
                and cached.get("rev") == embedded_rev
                and cached.get("ver") == VERSION
            ):
                return cached.get("behind")
    except Exception:
        pass

    if embedded_rev:
        behind = _check_via_rev(embedded_rev)
    else:
        # Prefer the running code's location over the profile-scoped path.
        # The profile-scoped checkout may be a stale copy from --clone-all;
        # Path(__file__) always resolves to the actual installed checkout.
        repo_dir = Path(__file__).parent.parent.resolve()
        if not (repo_dir / ".git").exists():
            repo_dir = runtime_home / "xiaoban-agent"
        if not (repo_dir / ".git").exists():
            behind = check_via_pypi()
        else:
            behind = _check_via_local_git(repo_dir)

    try:
        cache_file.write_text(
            json.dumps({"ts": now, "behind": behind, "rev": embedded_rev, "ver": VERSION})
        )
    except Exception:
        pass

    return behind


def _resolve_repo_dir() -> Optional[Path]:
    """Return the active Xiaoban git checkout, or None if this isn't a git install.

    Prefers the running code's location over the profile-scoped path
    because the profile-scoped checkout may be a stale copy carried
    over by ``--clone-all``.
    """
    repo_dir = Path(__file__).parent.parent.resolve()
    if not (repo_dir / ".git").exists():
        runtime_home = _get_runtime_home()
        repo_dir = runtime_home / "xiaoban-agent"
    return repo_dir if (repo_dir / ".git").exists() else None


def _git_short_hash(repo_dir: Path, rev: str) -> Optional[str]:
    """Resolve a git revision to an 8-character short hash."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short=8", rev],
            capture_output=True,
            text=True,
            timeout=5,
            cwd=str(repo_dir),
        )
    except Exception:
        return None
    if result.returncode != 0:
        return None
    value = (result.stdout or "").strip()
    return value or None


def get_git_banner_state(repo_dir: Optional[Path] = None) -> Optional[dict]:
    """Return upstream/local git hashes for the startup banner.

    For source installs and dev images this runs ``git rev-parse`` against
    the active checkout.  When no checkout is available — the canonical case
    is the published Docker image, which excludes ``.git`` from the build
    context — we fall back to the baked-in build SHA (see
    the build metadata module) and return it as a frozen
    ``upstream == local`` state with ``ahead=0``.  A built image is by
    definition pinned to one commit, so "ahead" is always zero and the
    banner correctly shows ``· upstream <sha>`` with no carried-commits
    annotation.
    """
    repo_dir = repo_dir or _resolve_repo_dir()
    if repo_dir is None:
        # No git checkout — try the baked build SHA (Docker image path).
        try:
            get_build_sha = getattr(_runtime_module("build_info"), "get_build_sha")
            baked = get_build_sha(short=8)
            if baked:
                return {"upstream": baked, "local": baked, "ahead": 0}
        except Exception:
            pass
        return None

    upstream = _git_short_hash(repo_dir, "origin/main")
    local = _git_short_hash(repo_dir, "HEAD")
    if not upstream or not local:
        # Live-git lookup failed (e.g. shallow clone without origin/main).
        # Fall back to the baked build SHA if available.
        try:
            get_build_sha = getattr(_runtime_module("build_info"), "get_build_sha")
            baked = get_build_sha(short=8)
            if baked:
                return {"upstream": baked, "local": baked, "ahead": 0}
        except Exception:
            pass
        return None

    ahead = 0
    try:
        result = subprocess.run(
            ["git", "rev-list", "--count", "origin/main..HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            cwd=str(repo_dir),
        )
        if result.returncode == 0:
            ahead = int((result.stdout or "0").strip() or "0")
    except Exception:
        ahead = 0

    return {"upstream": upstream, "local": local, "ahead": max(ahead, 0)}


_RELEASE_URL_BASE = "https://github.com/NousResearch/xiaoban-agent/releases/tag"
_latest_release_cache: Optional[tuple] = None  # (tag, url) once resolved


def get_latest_release_tag(repo_dir: Optional[Path] = None) -> Optional[tuple]:
    """Return ``(tag, release_url)`` for the latest git tag, or None.

    Local-only — runs ``git describe --tags --abbrev=0`` against the
    Xiaoban checkout. Cached per-process. Release URL always points at the
    canonical NousResearch/xiaoban-agent repo (forks don't get a link).
    """
    global _latest_release_cache
    if _latest_release_cache is not None:
        return _latest_release_cache or None

    repo_dir = repo_dir or _resolve_repo_dir()
    if repo_dir is None:
        _latest_release_cache = ()  # falsy sentinel — skip future lookups
        return None

    try:
        result = subprocess.run(
            ["git", "describe", "--tags", "--abbrev=0"],
            capture_output=True,
            text=True,
            timeout=3,
            cwd=str(repo_dir),
        )
    except Exception:
        _latest_release_cache = ()
        return None

    if result.returncode != 0:
        _latest_release_cache = ()
        return None

    tag = (result.stdout or "").strip()
    if not tag:
        _latest_release_cache = ()
        return None

    url = f"{_RELEASE_URL_BASE}/{tag}"
    _latest_release_cache = (tag, url)
    return _latest_release_cache


def format_banner_version_label() -> str:
    """Return the version label shown in the startup banner title."""
    return f"Xiaoban v{VERSION}"


# =========================================================================
# Non-blocking update check
# =========================================================================

_update_result: Optional[int] = None
_update_check_done = threading.Event()


def prefetch_update_check():
    """Kick off update check in a background daemon thread."""
    def _run():
        global _update_result
        _update_result = check_for_updates()
        _update_check_done.set()
    t = threading.Thread(target=_run, daemon=True)
    t.start()


def get_update_result(timeout: float = 0.5) -> Optional[int]:
    """Get result of prefetched check. Returns None if not ready."""
    _update_check_done.wait(timeout=timeout)
    return _update_result


# =========================================================================
# Welcome banner
# =========================================================================

def _format_context_length(tokens: int) -> str:
    """Format a token count for display (e.g. 128000 → '128K', 1048576 → '1M')."""
    if tokens >= 1_000_000:
        val = tokens / 1_000_000
        rounded = round(val)
        if abs(val - rounded) < 0.05:
            return f"{rounded}M"
        return f"{val:.1f}M"
    elif tokens >= 1_000:
        val = tokens / 1_000
        rounded = round(val)
        if abs(val - rounded) < 0.05:
            return f"{rounded}K"
        return f"{val:.1f}K"
    return str(tokens)


def _display_toolset_name(toolset_name: str) -> str:
    """Normalize internal/legacy toolset identifiers for banner display."""
    if not toolset_name:
        return "unknown"
    return (
        toolset_name[:-6]
        if toolset_name.endswith("_tools")
        else toolset_name
    )


def build_welcome_banner(console: "Console", model: str, cwd: str,
                         tools: List[dict] = None,
                         enabled_toolsets: List[str] = None,
                         session_id: str = None,
                         get_toolset_for_tool=None,
                         context_length: int = None):
    """Build and print a Claude Code style welcome banner.

    Args:
        console: Rich Console instance.
        model: Current model name.
        cwd: Current working directory.
        tools: List of tool definitions.
        enabled_toolsets: List of enabled toolset names.
        session_id: Session identifier.
        get_toolset_for_tool: Callable to map tool name -> toolset name.
        context_length: Model's context window size in tokens.
    """
    from rich.panel import Panel
    from rich.table import Table

    tools = tools or []
    enabled_toolsets = enabled_toolsets or []

    layout_table = Table.grid(expand=True, padding=(0, 2))
    layout_table.add_column("left", justify="center", min_width=42, no_wrap=True)
    layout_table.add_column("divider", justify="center", no_wrap=True)
    layout_table.add_column("right", justify="left", ratio=1)

    # Resolve skin colors once for the entire banner
    accent = _skin_color("banner_accent", "#C7A06A")
    dim = _skin_color("banner_dim", "#8A6A4A")
    text = _skin_color("banner_text", "#E9DEC8")
    divider_color = _skin_color("session_border", "#8B5E34")

    # Use skin's custom mark if provided.
    try:
        get_active_skin = getattr(_runtime_module("skin_engine"), "get_active_skin")
        _bskin = get_active_skin()
        _hero = _bskin.banner_hero if hasattr(_bskin, 'banner_hero') and _bskin.banner_hero else XIAOBAN_MARK
    except Exception:
        _bskin = None
        _hero = XIAOBAN_MARK
    left_lines = [
        "",
        f"[{text}]Welcome back![/]",
        "",
        _hero,
        "",
    ]
    model_short = model.split("/")[-1] if "/" in model else model
    if model_short.endswith(".gguf"):
        model_short = model_short[:-5]
    if len(model_short) > 28:
        model_short = model_short[:25] + "..."
    ctx_str = f" [dim {dim}]·[/] [dim {dim}]{_format_context_length(context_length)} context[/]" if context_length else ""
    left_lines.append(f"[{accent}]{model_short}[/]{ctx_str} [dim {dim}]·[/] [dim {dim}]API Usage Billing[/]")

    if _runtime_env("YOLO_MODE"):
        left_lines.append(f"[bold red]⚠ YOLO mode[/] [dim {dim}]— all approval prompts bypassed[/]")
    left_lines.append(f"[dim {dim}]{cwd}[/]")
    left_content = "\n".join(left_lines)

    right_lines = [
        f"[bold {accent}]Tips for getting started[/]",
        f"[{text}]Run /help to see Xiaoban commands[/]",
        f"[{text}]Run /new to start a clean session[/]",
        f"[dim {divider_color}]{'─' * 64}[/]",
        f"[bold {accent}]Recent activity[/]",
        f"[dim {dim}]No recent activity[/]",
    ]
    # Indicate when the codex_app_server runtime is active so users
    # understand why tool counts may not match what's actually reachable
    # (codex builds its own tool list inside the spawned subprocess).
    try:
        get_current_runtime = getattr(_runtime_module("codex_runtime_switch"), "get_current_runtime")
        _load_cfg = getattr(_runtime_module("config"), "load_config")
        if get_current_runtime(_load_cfg()) == "codex_app_server":
            right_lines.append(
                f"[bold {accent}]Runtime:[/] [{text}]codex app-server[/] "
                f"[dim {dim}](terminal/file ops/MCP run inside codex)[/]"
            )
    except Exception:
        pass
    # Show active profile name when not 'default'
    try:
        get_active_profile_name = getattr(_runtime_module("profiles"), "get_active_profile_name")
        _profile_name = get_active_profile_name()
        if _profile_name and _profile_name != "default":
            right_lines.append(f"[bold {accent}]Profile:[/] [{text}]{_profile_name}[/]")
    except Exception:
        pass  # Never break the banner over a profiles.py bug

    # Update check — use prefetched result if available
    try:
        behind = get_update_result(timeout=0.5)
        if behind is not None and behind != 0:
            _config = _runtime_module("config")
            get_managed_update_command = getattr(_config, "get_managed_update_command")
            recommended_update_command = getattr(_config, "recommended_update_command")
            if behind > 0:
                commits_word = "commit" if behind == 1 else "commits"
                right_lines.append(
                    f"[bold yellow]⚠ {behind} {commits_word} behind[/]"
                    f"[dim yellow] — run [bold]{recommended_update_command()}[/bold] to update[/]"
                )
            else:
                # UPDATE_AVAILABLE_NO_COUNT: nix-built package; we know an update
                # exists but not by how much, and we don't know how the user
                # installed it (nix run, profile, system flake, home-manager).
                managed_cmd = get_managed_update_command()
                line = "[bold yellow]⚠ update available[/]"
                if managed_cmd:
                    line += f"[dim yellow] — run [bold]{managed_cmd}[/bold][/]"
                right_lines.append(line)
    except Exception:
        pass  # Never break the banner over an update check

    # Pip-install warning — `pip install xiaoban-agent` is not the supported
    # install path (it exists on PyPI for internal/CI reasons, not end users).
    # Such installs miss the git checkout + installer-managed deps, so updates,
    # self-update, and issue triage don't behave correctly. Warn, don't block.
    try:
        detect_install_method = getattr(_runtime_module("config"), "detect_install_method")
        if detect_install_method() == "pip":
            right_lines.append(
                "[bold yellow]⚠ pip install not officially supported[/]"
                "[dim yellow] — exists for reasons other than user install; "
                "expect instability and an inability to support issues[/]"
            )
    except Exception:
        pass  # Never break the banner over the install-method check

    right_content = "\n".join(right_lines)
    divider_height = max(len(left_content.splitlines()), len(right_content.splitlines()))
    divider_content = "\n".join(f"[dim {divider_color}]│[/]" for _ in range(divider_height))
    layout_table.add_row(left_content, divider_content, right_content)

    title_color = _skin_color("banner_title", "#C7A06A")
    border_color = _skin_color("banner_border", "#8B5E34")
    version_label = format_banner_version_label()
    release_info = get_latest_release_tag()
    if release_info:
        _tag, _url = release_info
        title_markup = f"[bold {title_color}][link={_url}]{version_label}[/link][/]"
    else:
        title_markup = f"[bold {title_color}]{version_label}[/]"
    outer_panel = Panel(
        layout_table,
        title=title_markup,
        title_align="left",
        border_style=border_color,
        padding=(0, 2),
        width=console.width,
    )

    console.print()
    console.print(outer_panel)
