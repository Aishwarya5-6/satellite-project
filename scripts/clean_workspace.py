#!/usr/bin/env python3
"""
clean_workspace.py — Remove generated artefacts so Phase 1 → 2 → 3 can be
re-run from a clean slate.

Deletes
-------
  data/topology_dataset.npz      Phase 1 output
  logs/*                          TensorBoard / training logs
  models/*                        Saved PPO checkpoints & best model

Usage
-----
    python scripts/clean_workspace.py          # interactive (asks y/N)
    python scripts/clean_workspace.py --yes    # skip confirmation
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

TARGETS: list[tuple[str, Path]] = [
    ("Phase 1 dataset",   PROJECT_ROOT / "data" / "topology_dataset.npz"),
    ("Training logs",     PROJECT_ROOT / "logs"),
    ("Saved models",      PROJECT_ROOT / "models"),
]


def _sizeof(p: Path) -> str:
    """Human-readable size of a file or directory."""
    if p.is_file():
        size = p.stat().st_size
    elif p.is_dir():
        size = sum(f.stat().st_size for f in p.rglob("*") if f.is_file())
    else:
        return "0 B"
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024:
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} TB"


def clean(auto_yes: bool = False) -> None:
    print("=" * 60)
    print("  clean_workspace.py — Artefact Removal")
    print("=" * 60)

    found: list[tuple[str, Path]] = []
    for label, path in TARGETS:
        if path.exists():
            found.append((label, path))
            print(f"  ✓  {label:20s}  {path.relative_to(PROJECT_ROOT)}  ({_sizeof(path)})")
        else:
            print(f"  ·  {label:20s}  (not present)")

    if not found:
        print("\n  Nothing to clean — workspace is already empty.")
        return

    if not auto_yes:
        answer = input("\n  Delete the above? [y/N] ").strip().lower()
        if answer != "y":
            print("  Aborted.")
            return

    print()
    for label, path in found:
        if path.is_file():
            path.unlink()
            print(f"  🗑  Deleted file:  {path.relative_to(PROJECT_ROOT)}")
        elif path.is_dir():
            # Remove contents but keep the directory (+ .gitkeep if present)
            for child in sorted(path.iterdir()):
                if child.name == ".gitkeep":
                    continue
                if child.is_dir():
                    shutil.rmtree(child)
                else:
                    child.unlink()
            print(f"  🗑  Cleaned dir:   {path.relative_to(PROJECT_ROOT)}/")

    print("\n  Done.  Workspace is ready for a fresh run.")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clean generated artefacts")
    parser.add_argument("--yes", "-y", action="store_true",
                        help="Skip confirmation prompt")
    args = parser.parse_args()
    clean(auto_yes=args.yes)
