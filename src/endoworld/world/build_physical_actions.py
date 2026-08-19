"""Build video-level SCARED/C3VD latent--SE(3) caches.

Example:
    python -m endoworld.world.build_physical_actions --encoder vjepa2
"""

from __future__ import annotations

import argparse
from pathlib import Path
import re

import numpy as np
import torch

from endoworld.world.physical_actions import (
    PhysicalSequence,
    align_latents_and_poses,
    save_sequences,
)


def _frame_indices(paths: list[Path], n_poses: int) -> np.ndarray:
    parsed = []
    for path in paths:
        numbers = re.findall(r"\d+", path.stem)
        parsed.append(int(numbers[-1]) if numbers else -1)
    parsed = np.asarray(parsed, dtype=np.int64)
    if len(parsed) and parsed.min() >= 0 and parsed.max() < n_poses:
        return parsed
    return np.round(np.linspace(0, max(n_poses - 1, 0), len(paths))).astype(np.int64)


def _read_paths(paths: list[Path], image_size: int) -> torch.Tensor:
    from PIL import Image

    frames = [
        np.asarray(
            Image.open(path).convert("RGB").resize((image_size, image_size)),
            dtype=np.float32,
        )
        / 255.0
        for path in paths
    ]
    return torch.from_numpy(np.stack(frames).transpose(0, 3, 1, 2))


def _encode_chunks(enc, frames: torch.Tensor, device: str, chunk: int) -> torch.Tensor:
    tubelet = int(getattr(enc, "tubelet", 2))
    chunk = max(tubelet, (chunk // tubelet) * tubelet)
    outputs = []
    with torch.no_grad():
        for start in range(0, frames.size(0), chunk):
            part = frames[start : start + chunk]
            part = part[: (part.size(0) // tubelet) * tubelet]
            if part.size(0) < tubelet:
                continue
            outputs.append(
                enc.encode_temporal(part.unsqueeze(0).to(device).float())[0].cpu()
            )
    if not outputs:
        return torch.zeros(0, int(getattr(enc, "embed_dim", 1)))
    return torch.cat(outputs)


def _encode_past_only(
    enc,
    frames: torch.Tensor,
    device: str,
    lookback_frames: int,
    batch: int = 8,
) -> torch.Tensor:
    """Leakage-free encoding: latent t sees only frames up to its own time.

    For each tubelet step t we encode the window
    ``frames[2t+2-lookback : 2t+2]`` and keep only its final tubelet, so no
    retained latent can attend to future frames. This also removes the
    64-frame chunk resets of ``_encode_chunks``, because every step receives a
    full look-back window. Steps without a full window are dropped.
    """
    tubelet = int(getattr(enc, "tubelet", 2))
    n_steps = frames.size(0) // tubelet
    warm = lookback_frames // tubelet - 1
    if n_steps <= warm:
        return torch.zeros(0, int(getattr(enc, "embed_dim", 1)))
    n_windows = n_steps - warm
    outputs = []
    with torch.no_grad():
        for start in range(0, n_windows, batch):
            steps = range(start, min(start + batch, n_windows))
            part = (
                torch.stack(
                    [
                        frames[
                            (warm + s + 1) * tubelet - lookback_frames : (warm + s + 1)
                            * tubelet
                        ]
                        for s in steps
                    ]
                )
                .to(device)
                .float()
            )
            outputs.append(enc.encode_temporal(part)[:, -1].cpu())
    return torch.cat(outputs)


def _sequence_from_frames(
    enc,
    frame_tensor: torch.Tensor,
    frame_indices: np.ndarray,
    poses: np.ndarray,
    sequence_id: str,
    dataset: str,
    device: str,
    chunk: int,
    case_id: str | None = None,
    past_only: bool = False,
    lookback_tubelets: int = 8,
) -> PhysicalSequence | None:
    from endoworld.world.physical_actions import interpolate_pose_rows, pose_deltas

    tubelet = int(getattr(enc, "tubelet", 2))
    usable = min(frame_tensor.size(0), len(frame_indices))
    usable -= usable % tubelet
    if usable < tubelet * 2:
        return None
    if past_only:
        lookback_frames = lookback_tubelets * tubelet
        z = _encode_past_only(enc, frame_tensor[:usable], device, lookback_frames)
        warm = lookback_frames // tubelet - 1
        # Latent step k (0-based after warm-up) ends at sampled frame
        # (warm + k + 1) * tubelet - 1, whose pose row is integer: window-end
        # alignment requires no pose interpolation.
        end_steps = (warm + np.arange(len(z)) + 1) * tubelet - 1
        positions = frame_indices[end_steps].astype(np.float64)
        valid = positions <= len(poses) - 1
        z = z[valid]
        aligned_poses = interpolate_pose_rows(poses, positions[valid])
        actions = pose_deltas(aligned_poses)
        z = z.numpy() if torch.is_tensor(z) else z
    else:
        z = _encode_chunks(enc, frame_tensor[:usable], device, chunk)
        z, _, actions = align_latents_and_poses(
            z, poses, frame_indices[:usable], tubelet
        )
    if len(z) < 2:
        return None
    return PhysicalSequence(
        sequence_id=sequence_id,
        dataset=dataset,
        latents=torch.from_numpy(np.asarray(z, dtype=np.float32)),
        actions=torch.from_numpy(np.asarray(actions, dtype=np.float32)),
        case_id=case_id,
    )


def _part_path(parts_dir: Path | None, sequence_id: str) -> Path | None:
    if parts_dir is None:
        return None
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", sequence_id)
    return parts_dir / f"{safe}.pt"


def _save_part(path: Path, seq: PhysicalSequence) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "sequence_id": seq.sequence_id,
            "dataset": seq.dataset,
            "latents": seq.latents.cpu(),
            "actions": seq.actions.cpu(),
            "case_id": seq.case_id,
        },
        path,
    )


def _load_part(path: Path) -> PhysicalSequence:
    row = torch.load(path, map_location="cpu", weights_only=False)
    return PhysicalSequence(
        sequence_id=row["sequence_id"],
        dataset=row["dataset"],
        latents=row["latents"].float(),
        actions=row["actions"].float(),
        case_id=row.get("case_id"),
    )


def collect_scared(
    enc,
    root: str,
    device: str,
    max_frames: int,
    stride: int,
    chunk: int,
    stereo_eye: str | None = None,
    past_only: bool = False,
    lookback_tubelets: int = 8,
    parts_dir: Path | None = None,
):
    from endoworld.world.scared_actions import (
        crop_stereo_half,
        find_scared_keyframes,
        find_scared_rgb,
        load_scared_poses,
        read_video_frames,
    )

    rows = []
    for keyframe in find_scared_keyframes(root):
        sequence_id = f"scared:{keyframe.as_posix()}"
        part = _part_path(parts_dir, sequence_id)
        if part is not None and part.is_file():
            rows.append(_load_part(part))
            print(f"[scared] {sequence_id} (cached part)", flush=True)
            continue
        poses = load_scared_poses(keyframe)
        video, paths = find_scared_rgb(keyframe)
        if video is not None:
            import cv2

            cap = cv2.VideoCapture(str(video))
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            cap.release()
            indices = np.arange(0, min(total, len(poses)), stride, dtype=np.int64)[
                :max_frames
            ]
            frames = read_video_frames(
                video, indices, enc.image_size, stereo_eye=stereo_eye
            )
        elif paths:
            paths = paths[::stride][:max_frames]
            indices = _frame_indices(paths, len(poses))
            frames = _read_paths(paths, enc.image_size)
            if stereo_eye in ("top", "bottom"):
                import torch as _torch

                cropped = [
                    crop_stereo_half(frame.permute(1, 2, 0).numpy(), stereo_eye)
                    for frame in frames
                ]
                frames = _torch.from_numpy(np.stack(cropped).transpose(0, 3, 1, 2))
        else:
            continue
        case_id = next(
            (p for p in keyframe.parts if p.startswith("dataset_")),
            None,
        )
        seq = _sequence_from_frames(
            enc,
            frames,
            indices,
            poses,
            sequence_id,
            "SCARED",
            device,
            chunk,
            case_id=case_id,
            past_only=past_only,
            lookback_tubelets=lookback_tubelets,
        )
        if seq is not None:
            rows.append(seq)
            if part is not None:
                _save_part(part, seq)
        print(f"[scared] {sequence_id} -> {len(rows)} sequences", flush=True)
    return rows


def collect_c3vd(
    enc,
    root: str,
    device: str,
    max_frames: int,
    stride: int,
    chunk: int,
    past_only: bool = False,
    lookback_tubelets: int = 8,
):
    from endoworld.world.c3vd_actions import (
        find_c3vd_color_frames,
        find_c3vd_pose_files,
        load_pose_txt,
    )

    rows = []
    for pose_path in find_c3vd_pose_files(root):
        poses = load_pose_txt(pose_path)
        paths = find_c3vd_color_frames(pose_path.parent)[::stride][:max_frames]
        if len(paths) < 4:
            continue
        indices = _frame_indices(paths, len(poses))
        frames = _read_paths(paths, enc.image_size)
        sequence_id = f"c3vd:{pose_path.parent.as_posix()}"
        seq = _sequence_from_frames(
            enc,
            frames,
            indices,
            poses,
            sequence_id,
            "C3VD",
            device,
            chunk,
            past_only=past_only,
            lookback_tubelets=lookback_tubelets,
        )
        if seq is not None:
            rows.append(seq)
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--encoder", choices=["vjepa2", "scratch"], default="vjepa2")
    parser.add_argument("--vjepa2-id", default="facebook/vjepa2-vitl-fpc64-256")
    parser.add_argument("--scratch-ckpt", default="outputs/vjepa_l1/vjepa_l1_adapt.pt")
    parser.add_argument("--scared", default="datasets/SCARED")
    parser.add_argument("--c3vd", default="datasets/C3VD")
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--max-frames", type=int, default=4096)
    parser.add_argument("--chunk", type=int, default=64)
    parser.add_argument("--out", default="outputs/physical_actions/sequences.pt")
    parser.add_argument(
        "--stereo-eye",
        choices=["top", "bottom", "none"],
        default="none",
        help="SCARED rgb.mp4 stacks both cameras vertically; crop one eye so "
        "image motion corresponds to the single-camera SE(3) pose.",
    )
    parser.add_argument(
        "--past-only",
        action="store_true",
        help="Encode every tubelet from a look-back window only (no future "
        "attention, no 64-frame chunk resets).",
    )
    parser.add_argument(
        "--lookback-tubelets",
        type=int,
        default=8,
        help="Past-only window length in tubelets (8 tubelets = 16 frames).",
    )
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    from endoworld.understanding.encoders import load_any_encoder

    enc, _, _, _ = load_any_encoder(
        args.encoder, device, args.vjepa2_id, args.scratch_ckpt
    )
    stereo_eye = None if args.stereo_eye == "none" else args.stereo_eye
    parts_dir = Path(args.out).parent / "parts"
    sequences = collect_scared(
        enc,
        args.scared,
        device,
        args.max_frames,
        args.stride,
        args.chunk,
        stereo_eye=stereo_eye,
        past_only=args.past_only,
        lookback_tubelets=args.lookback_tubelets,
        parts_dir=parts_dir,
    )
    sequences += collect_c3vd(
        enc,
        args.c3vd,
        device,
        args.max_frames,
        args.stride,
        args.chunk,
        past_only=args.past_only,
        lookback_tubelets=args.lookback_tubelets,
    )
    if not sequences:
        raise RuntimeError("no aligned SCARED/C3VD RGB-pose sequences found")
    save_sequences(sequences, args.out)
    import json

    meta = {
        "encoder": args.encoder,
        "vjepa2_id": args.vjepa2_id,
        "stride": args.stride,
        "stereo_eye": stereo_eye,
        "past_only": args.past_only,
        "lookback_tubelets": args.lookback_tubelets,
        "alignment": (
            "window-end pose (integer, no interpolation)"
            if args.past_only
            else "fractional tubelet-centre SE(3) interpolation"
        ),
        "n_sequences": len(sequences),
        "n_latents": int(sum(s.latents.size(0) for s in sequences)),
    }
    meta_path = Path(args.out).with_suffix(".meta.json")
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"[physical-actions] saved {len(sequences)} video sequences -> {args.out}")


if __name__ == "__main__":
    main()
