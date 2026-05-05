#!/usr/bin/env python3
"""Install an exported DFM model into VisoMaster."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


SUPPORTED_EXTENSIONS = {".dfm", ".onnx"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Copy an exported DFM/ONNX model into VisoMaster."
    )
    parser.add_argument("--model", required=True, help="Path to exported .dfm or .onnx.")
    parser.add_argument(
        "--repo-root",
        default=None,
        help="VisoMaster repo root. Defaults to the current working directory.",
    )
    parser.add_argument(
        "--name",
        default=None,
        help="Optional output filename. The original extension is preserved if omitted.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing model with the same filename.",
    )
    return parser.parse_args()


def resolve_repo_root(repo_root_arg: str | None) -> Path:
    if repo_root_arg:
        return Path(repo_root_arg).expanduser().resolve()
    return Path.cwd().resolve()


def model_target_name(source: Path, requested_name: str | None) -> str:
    if not requested_name:
        return source.name
    requested = Path(requested_name)
    if requested.suffix:
        return requested.name
    return f"{requested.name}{source.suffix.lower()}"


def main() -> int:
    args = parse_args()
    source = Path(args.model).expanduser().resolve()
    if not source.exists() or not source.is_file():
        raise SystemExit(f"Model file does not exist: {source}")
    if source.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise SystemExit("Model must end with .dfm or .onnx")
    if source.stat().st_size < 1024:
        raise SystemExit("Model file is unexpectedly small; check the export.")

    repo_root = resolve_repo_root(args.repo_root)
    target_dir = repo_root / "model_assets" / "dfm_models"
    if not target_dir.exists():
        raise SystemExit(f"VisoMaster DFM model directory not found: {target_dir}")

    target = target_dir / model_target_name(source, args.name)
    if target.exists() and not args.overwrite:
        raise SystemExit(
            f"Target already exists: {target}. Re-run with --overwrite to replace it."
        )

    shutil.copy2(source, target)
    print(f"Installed model: {target}")
    print("Restart VisoMaster, select DeepFaceLive (DFM), then choose this model.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
