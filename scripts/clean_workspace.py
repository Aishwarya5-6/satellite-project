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
    python scripts/clean_workspace.py             # interactive (asks y/N)
    python scripts/clean_workspace.py --yes       # skip confirmation
    python scripts/clean_workspace.py --smoke-only  # wipe only smoke-test artifacts
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

# Artifacts created only by the Safe Launch smoke test (100k-step warm-up).
# Wiped automatically when the gate passes so the full run starts from scratch.
SMOKE_TARGETS: list[tuple[str, Path]] = [
    ("Smoke-test model",  PROJECT_ROOT / "models" / "smoke_test_ppo.zip"),
    ("Smoke TB logs",     PROJECT_ROOT / "logs" / "smoke_test_ppo_1"),
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


def clean(auto_yes: bool = False, smoke_only: bool = False) -> None:
    print("=" * 60)
    if smoke_only:
        print("  clean_workspace.py — Smoke-Test Artifact Removal")
    else:
        print("  clean_workspace.py — Full Artefact Removal")
    print("=" * 60)

    targets = SMOKE_TARGETS if smoke_only else TARGETS
    found: list[tuple[str, Path]] = []
    for label, path in targets:
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
            if smoke_only:
                # Smoke targets are specific subdirs/files — remove entirely
                shutil.rmtree(path)
                print(f"  🗑  Removed dir:   {path.relative_to(PROJECT_ROOT)}/")
            else:
                # Full clean: remove contents but keep the directory (+ .gitkeep)
                for child in sorted(path.iterdir()):
                    if child.name == ".gitkeep":
                        continue
                    if child.is_dir():
                        shutil.rmtree(child)
                    else:
                        child.unlink()
                print(f"  🗑  Cleaned dir:   {path.relative_to(PROJECT_ROOT)}/")

    if smoke_only:
        print("\n  Done.  Smoke-test artifacts removed.  Workspace ready for full run.")
    else:
        print("\n  Done.  Workspace is ready for a fresh run.")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clean generated artefacts")
    parser.add_argument("--yes", "-y", action="store_true",
                        help="Skip confirmation prompt")
    parser.add_argument("--smoke-only", action="store_true",
                        help="Remove only smoke-test artifacts (model + TB logs)")
    args = parser.parse_args()
    clean(auto_yes=args.yes or args.smoke_only, smoke_only=args.smoke_only)
