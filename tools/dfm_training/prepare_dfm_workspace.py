#!/usr/bin/env python3
"""Prepare a DeepFaceLab-style workspace for DFM training.

This script does not train a model by itself. It organizes consented source and
destination material into the folder structure DeepFaceLab expects, extracts
frames from videos with ffmpeg, and writes a focused next-steps checklist.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare source/destination frames for DeepFaceLab DFM training."
    )
    parser.add_argument("--workspace", required=True, help="DeepFaceLab workspace path.")
    parser.add_argument(
        "--profile-name",
        default="dfm_training_pair",
        help="Name written to manifest and generated notes.",
    )
    parser.add_argument(
        "--src-video",
        action="append",
        default=[],
        help="Source identity video. Repeat for multiple videos.",
    )
    parser.add_argument(
        "--dst-video",
        action="append",
        default=[],
        help="Destination/target actor video. Repeat for multiple videos.",
    )
    parser.add_argument(
        "--src-images",
        action="append",
        default=[],
        help="Source image file or directory. Repeat for multiple paths.",
    )
    parser.add_argument(
        "--dst-images",
        action="append",
        default=[],
        help="Destination image file or directory. Repeat for multiple paths.",
    )
    parser.add_argument("--src-fps", type=float, default=8.0)
    parser.add_argument("--dst-fps", type=float, default=8.0)
    parser.add_argument(
        "--ffmpeg",
        default="ffmpeg",
        help="ffmpeg executable path. Defaults to ffmpeg on PATH.",
    )
    parser.add_argument(
        "--max-width",
        type=int,
        default=1920,
        help="Downscale extracted frames wider than this value. Use 0 to disable.",
    )
    parser.add_argument(
        "--jpeg-quality",
        type=int,
        default=2,
        help="ffmpeg JPEG quality, where 2 is high quality and 31 is low quality.",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Delete existing data_src/data_dst/model/mask_review folders first.",
    )
    return parser.parse_args()


def require_inputs(args: argparse.Namespace) -> None:
    if not args.src_video and not args.src_images:
        raise SystemExit("Provide at least one --src-video or --src-images path.")
    if not args.dst_video and not args.dst_images:
        raise SystemExit("Provide at least one --dst-video or --dst-images path.")
    if args.src_fps <= 0 or args.dst_fps <= 0:
        raise SystemExit("--src-fps and --dst-fps must be positive numbers.")
    if not 2 <= args.jpeg_quality <= 31:
        raise SystemExit("--jpeg-quality must be between 2 and 31.")


def reset_workspace_dirs(workspace: Path, clean: bool) -> dict[str, Path]:
    dirs = {
        "data_src": workspace / "data_src",
        "data_dst": workspace / "data_dst",
        "data_src_aligned": workspace / "data_src" / "aligned",
        "data_dst_aligned": workspace / "data_dst" / "aligned",
        "model": workspace / "model",
        "mask_review": workspace / "mask_review",
    }

    if clean:
        for key in ("data_src", "data_dst", "model", "mask_review"):
            if dirs[key].exists():
                shutil.rmtree(dirs[key])

    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)

    return dirs


def image_files_from_path(path_text: str) -> list[Path]:
    path = Path(path_text).expanduser().resolve()
    if not path.exists():
        raise SystemExit(f"Input path does not exist: {path}")
    if path.is_file():
        return [path] if path.suffix.lower() in IMAGE_EXTENSIONS else []
    return sorted(
        file
        for file in path.rglob("*")
        if file.is_file() and file.suffix.lower() in IMAGE_EXTENSIONS
    )


def copy_images(paths: list[str], output_dir: Path, prefix: str) -> int:
    copied = 0
    for path_text in paths:
        for image in image_files_from_path(path_text):
            copied += 1
            target = output_dir / f"{prefix}_img_{copied:06d}{image.suffix.lower()}"
            shutil.copy2(image, target)
    return copied


def ffmpeg_filter(max_width: int) -> str:
    if max_width <= 0:
        return "fps={fps}"
    return f"fps={{fps}},scale='min({max_width},iw)':-2"


def extract_video_frames(
    ffmpeg: str,
    video_paths: list[str],
    output_dir: Path,
    prefix: str,
    fps: float,
    max_width: int,
    jpeg_quality: int,
) -> int:
    extracted_total = 0
    filter_template = ffmpeg_filter(max_width)

    for index, video_text in enumerate(video_paths, start=1):
        video = Path(video_text).expanduser().resolve()
        if not video.exists():
            raise SystemExit(f"Video path does not exist: {video}")
        pattern = output_dir / f"{prefix}_vid{index:02d}_%06d.jpg"
        before = set(output_dir.glob(f"{prefix}_vid{index:02d}_*.jpg"))
        command = [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(video),
            "-vf",
            filter_template.format(fps=fps),
            "-q:v",
            str(jpeg_quality),
            str(pattern),
        ]
        try:
            subprocess.run(command, check=True)
        except FileNotFoundError as exc:
            raise SystemExit(
                f"Could not run ffmpeg executable '{ffmpeg}'. Install ffmpeg or pass --ffmpeg."
            ) from exc
        except subprocess.CalledProcessError as exc:
            raise SystemExit(f"ffmpeg failed while extracting {video}: {exc}") from exc

        after = set(output_dir.glob(f"{prefix}_vid{index:02d}_*.jpg"))
        extracted_total += len(after - before)

    return extracted_total


def write_next_steps(workspace: Path, profile_name: str, manifest: dict[str, object]) -> None:
    notes = workspace / "deepfacelab_next_steps.md"
    notes.write_text(
        f"""# DeepFaceLab Next Steps for {profile_name}

Generated: {manifest["generated_at"]}

## Prepared Frame Counts

- Source frames copied from images: {manifest["source"]["copied_images"]}
- Source frames extracted from videos: {manifest["source"]["extracted_video_frames"]}
- Destination frames copied from images: {manifest["destination"]["copied_images"]}
- Destination frames extracted from videos: {manifest["destination"]["extracted_video_frames"]}

## Train the DFM

1. Open DeepFaceLab and point it at this workspace:
   `{workspace}`
2. Run source faceset extraction for `data_src`.
3. Run destination faceset extraction for `data_dst`.
4. Sort both aligned facesets and delete bad detections.
5. Label XSeg masks for difficult occlusion frames.
6. Train XSeg, then apply trained XSeg masks to both facesets.
7. Train SAEHD or AMP. Start around 256 resolution for live streaming.
8. Export SAEHD or AMP as `.dfm`.

## XSeg Frames to Prioritize

- hand over mouth
- hand over nose
- fingers crossing lips or teeth
- hand touching cheek or chin
- hair, glasses, phone, microphone, or cup crossing the face
- fast hand motion blur

## VisoMaster Import

After export, run this from the VisoMaster repo root:

```powershell
python tools/dfm_training/install_dfm_model.py --model <path-to-exported-model.dfm>
```

Then restart VisoMaster and choose `DeepFaceLive (DFM)`.
""",
        encoding="utf-8",
    )


def main() -> int:
    args = parse_args()
    require_inputs(args)

    workspace = Path(args.workspace).expanduser().resolve()
    dirs = reset_workspace_dirs(workspace, args.clean)

    copied_src = copy_images(args.src_images, dirs["data_src"], "src")
    copied_dst = copy_images(args.dst_images, dirs["data_dst"], "dst")
    extracted_src = extract_video_frames(
        args.ffmpeg,
        args.src_video,
        dirs["data_src"],
        "src",
        args.src_fps,
        args.max_width,
        args.jpeg_quality,
    )
    extracted_dst = extract_video_frames(
        args.ffmpeg,
        args.dst_video,
        dirs["data_dst"],
        "dst",
        args.dst_fps,
        args.max_width,
        args.jpeg_quality,
    )

    manifest = {
        "profile_name": args.profile_name,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "workspace": str(workspace),
        "source": {
            "videos": [str(Path(path).expanduser().resolve()) for path in args.src_video],
            "images": [str(Path(path).expanduser().resolve()) for path in args.src_images],
            "fps": args.src_fps,
            "copied_images": copied_src,
            "extracted_video_frames": extracted_src,
        },
        "destination": {
            "videos": [str(Path(path).expanduser().resolve()) for path in args.dst_video],
            "images": [str(Path(path).expanduser().resolve()) for path in args.dst_images],
            "fps": args.dst_fps,
            "copied_images": copied_dst,
            "extracted_video_frames": extracted_dst,
        },
        "frame_extraction": {
            "ffmpeg": args.ffmpeg,
            "max_width": args.max_width,
            "jpeg_quality": args.jpeg_quality,
        },
        "expected_deepfacelab_dirs": {name: str(path) for name, path in dirs.items()},
    }
    (workspace / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    write_next_steps(workspace, args.profile_name, manifest)

    total_src = copied_src + extracted_src
    total_dst = copied_dst + extracted_dst
    print(f"Prepared workspace: {workspace}")
    print(f"Source frames: {total_src}")
    print(f"Destination frames: {total_dst}")
    print(f"Next steps: {workspace / 'deepfacelab_next_steps.md'}")
    if total_src == 0 or total_dst == 0:
        print("Warning: one side has zero frames. Check inputs before training.", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
