"""Loads params.yaml (single source of truth) and exposes project paths."""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
PARAMS_PATH = Path(os.getenv("PARAMS_PATH", ROOT / "params.yaml"))


@lru_cache(maxsize=1)
def load_params() -> dict:
    with open(PARAMS_PATH, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def path(rel: str | Path) -> Path:
    """Resolve a project-relative path (absolute paths pass through)."""
    p = Path(rel)
    return p if p.is_absolute() else ROOT / p


def git_commit() -> str:
    """Best-effort current git SHA (Git-based metadata for lineage)."""
    import subprocess

    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True, timeout=5
        )
        return out.stdout.strip() or "unknown"
    except Exception:
        return "unknown"
