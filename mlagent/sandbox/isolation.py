"""Sandbox environment isolation helpers."""

from __future__ import annotations

import os
import sys
from pathlib import Path


def build_restricted_env(workspace: Path, extra_paths: list[Path] | None = None) -> dict[str, str]:
    """Build a minimal environment for subprocess execution."""
    paths = [str(workspace)]
    if extra_paths:
        paths.extend(str(p) for p in extra_paths)
    venv_site = Path(sys.prefix) / "lib"
    if venv_site.exists():
        for site in venv_site.glob("python*/site-packages"):
            paths.append(str(site))

    env = {
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        "PYTHONPATH": os.pathsep.join(paths),
        "PYTHONNOUSERSITE": "1",
        "HOME": str(workspace),
        "TMPDIR": str(workspace / "tmp"),
        "MLAGENT_WORKSPACE": str(workspace),
        "MLAGENT_SANDBOX": "1",
    }
    # Strip sensitive keys
    blocked_prefixes = (
        "AWS_",
        "OPENAI_",
        "OPENROUTER_",
        "ANTHROPIC_",
        "LANGCHAIN_",
        "API_KEY",
    )
    for key in list(os.environ):
        if any(key.startswith(p) for p in blocked_prefixes):
            continue
        if key not in env and key in ("LC_ALL", "LANG", "TZ"):
            env[key] = os.environ[key]
    return env


BLOCKED_IMPORTS = frozenset(
    {
        "subprocess",
        "socket",
        "http.client",
        "urllib.request",
        "ftplib",
        "smtplib",
        "telnetlib",
        "pty",
        "multiprocessing",
    }
)
