#!/usr/bin/env python3
"""
================================================================================
Workspace Cleaner — Pre-Gold-Run Artifact Wipe
================================================================================
Removes ALL training artifacts from previous runs so the next training
starts from a completely uncontaminated state.

Usage:
    conda run -n leo_rl_env python scripts/clean_workspace.py
================================================================================
"""

import shutil
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ═══════════════════════════════════════════════════════════════════════════════
# Define all artifact targets
# ═══════════════════════════════════════════════════════════════════════════════

# Directories to completely wipe (delete all contents, recreate empty)
DIRS_TO_WIPE = [
    "logs",
    "tb_logs",
    "tensorboard",
    "models",
    "models/checkpoints",
    "eval",
    "checkpoints",
    "logs/tb",
    "logs/eval",
]

# Glob patterns for individual files to delete from project root
ROOT_FILE_PATTERNS = [
    "*.zip",                    # stray model saves
    "vec_normalize*.pkl",       # normalization stats
    "*.monitor.csv",            # SB3 Monitor logs
]

# Glob patterns for files to delete recursively across entire project
RECURSIVE_FILE_PATTERNS = [
    "**/*.zip",                 # all model checkpoints anywhere
    "**/*.pkl",                 # all pickle files (vec_normalize, etc.)
    "**/events.out.tfevents.*", # TensorBoard event files
    "**/*.monitor.csv",         # SB3 Monitor CSV logs
    "**/progress.csv",          # SB3 logger CSV
]

# Directories/files to NEVER delete (safety list)
PROTECTED = {
    "data",
    "src",
    "scripts",
    "README.md",
    ".git",
    ".gitignore",
    "environment.yml",
    "requirements.txt",
    "data/topology_dataset.npz",
}


def is_protected(path: Path) -> bool:
    """Check if a path is in the protected list."""
    rel = path.relative_to(PROJECT_ROOT)
    for part in PROTECTED:
        if str(rel) == part or str(rel).startswith(part + "/"):
            return True
    return False


def wipe_directory(rel_path: str) -> str:
    """Delete directory contents and recreate empty. Returns status string."""
    target = PROJECT_ROOT / rel_path
    if not target.exists():
        return f"  ⏭  {rel_path:40s}  (not found — skipped)"

    if is_protected(target):
        return f"  🛡  {rel_path:40s}  (PROTECTED — skipped)"

    try:
        file_count = sum(1 for _ in target.rglob("*") if _.is_file())
        shutil.rmtree(target)
        target.mkdir(parents=True, exist_ok=True)
        return f"  ✅  {rel_path:40s}  ({file_count} files removed)"
    except Exception as e:
        return f"  ❌  {rel_path:40s}  (ERROR: {e})"


def delete_files_by_pattern(pattern: str) -> list[str]:
    """Delete files matching a glob pattern. Returns status strings."""
    results = []
    matches = sorted(PROJECT_ROOT.glob(pattern))
    for f in matches:
        if not f.is_file():
            continue
        if is_protected(f):
            results.append(f"  🛡  {f.relative_to(PROJECT_ROOT)}  (PROTECTED)")
            continue
        try:
            f.unlink()
            results.append(f"  ✅  {f.relative_to(PROJECT_ROOT)}  (deleted)")
        except Exception as e:
            results.append(f"  ❌  {f.relative_to(PROJECT_ROOT)}  (ERROR: {e})")
    return results


def main():
    print()
    print("=" * 72)
    print("  🧹  WORKSPACE CLEANER — Pre-Gold-Run Artifact Wipe")
    print("=" * 72)
    print(f"  Project root: {PROJECT_ROOT}")
    print()

    # ── Phase 1: Wipe directories ────────────────────────────────────────
    print("  ── Phase 1: Directory Wipe ─────────────────────────────────")
    for d in DIRS_TO_WIPE:
        print(wipe_directory(d))
    print()

    # ── Phase 2: Delete stray files in project root ──────────────────────
    print("  ── Phase 2: Root File Cleanup ──────────────────────────────")
    root_results = []
    for pattern in ROOT_FILE_PATTERNS:
        root_results.extend(delete_files_by_pattern(pattern))
    if root_results:
        for r in root_results:
            print(r)
    else:
        print("  ⏭  No stray files found in project root")
    print()

    # ── Phase 3: Recursive deep clean ────────────────────────────────────
    print("  ── Phase 3: Recursive Deep Clean ───────────────────────────")
    deep_results = []
    for pattern in RECURSIVE_FILE_PATTERNS:
        deep_results.extend(delete_files_by_pattern(pattern))
    if deep_results:
        for r in deep_results:
            print(r)
    else:
        print("  ⏭  No residual artifacts found anywhere")
    print()

    # ── Phase 4: Recreate required empty directories ─────────────────────
    print("  ── Phase 4: Recreate Clean Directories ─────────────────────")
    for d in ["logs", "logs/tb", "models", "models/checkpoints"]:
        target = PROJECT_ROOT / d
        target.mkdir(parents=True, exist_ok=True)
        print(f"  📁  {d:40s}  (ready)")
    print()

    # ── Final verification ───────────────────────────────────────────────
    print("  ── Final Verification ─────────────────────────────────────")
    topology = PROJECT_ROOT / "data" / "topology_dataset.npz"
    if topology.exists():
        size_mb = topology.stat().st_size / (1024 * 1024)
        print(f"  ✅  data/topology_dataset.npz           ({size_mb:.0f} MB — intact)")
    else:
        print("  ❌  data/topology_dataset.npz           MISSING — run Phase 1!")

    src_files = list((PROJECT_ROOT / "src").glob("*.py"))
    print(f"  ✅  src/                                 ({len(src_files)} Python files — intact)")

    stray_zips = list(PROJECT_ROOT.rglob("*.zip"))
    stray_pkls = list(PROJECT_ROOT.rglob("*.pkl"))
    stray_tb = list(PROJECT_ROOT.rglob("events.out.tfevents.*"))

    if not stray_zips and not stray_pkls and not stray_tb:
        print()
        print("  " + "=" * 52)
        print("  ✅  [CLEARED] Workspace is 100% clean.")
        print("  ✅  [CLEARED] Zero artifacts from previous runs.")
        print("  ✅  [CLEARED] Safe to launch Gold Run.")
        print("  " + "=" * 52)
    else:
        print()
        print("  ⚠️  WARNING: Residual files detected:")
        for f in stray_zips + stray_pkls + stray_tb:
            print(f"       {f.relative_to(PROJECT_ROOT)}")
        print("  Run this script again or delete manually.")

    print()


if __name__ == "__main__":
    main()
