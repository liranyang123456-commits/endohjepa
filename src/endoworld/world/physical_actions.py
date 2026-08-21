"""Video-level physical action samples for geometry-grounded world modelling.

All poses use a single convention: camera-to-world 4x4 matrices and local camera
motion ``log(inv(T_t) @ T_{t+1})`` represented as ``[v_x,v_y,v_z,w_x,w_y,w_z]``.
Splits are assigned by complete video/sequence id, never by transition.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch.utils.data import Dataset


def _so3_log(rotation: np.ndarray) -> np.ndarray:
    cos_theta = np.clip((np.trace(rotation) - 1.0) * 0.5, -1.0, 1.0)
    theta = float(np.arccos(cos_theta))
    vee = np.array([
        rotation[2, 1] - rotation[1, 2],
        rotation[0, 2] - rotation[2, 0],
        rotation[1, 0] - rotation[0, 1],
    ], dtype=np.float64)
    if theta < 1e-7:
        return 0.5 * vee
    return theta * vee / (2.0 * np.sin(theta))


def _so3_exp(omega: np.ndarray) -> np.ndarray:
    theta = float(np.linalg.norm(omega))
    omega_hat = np.array([
        [0.0, -omega[2], omega[1]],
        [omega[2], 0.0, -omega[0]],
        [-omega[1], omega[0], 0.0],
    ])
    if theta < 1e-7:
        return np.eye(3) + omega_hat + 0.5 * omega_hat @ omega_hat
    return (
        np.eye(3)
        + (np.sin(theta) / theta) * omega_hat
        + ((1.0 - np.cos(theta)) / theta**2) * (omega_hat @ omega_hat)
    )


def se3_log(transform: np.ndarray) -> np.ndarray:
    """Lie logarithm of one SE(3) transform as translation-twist then rotation."""
    transform = np.asarray(transform, dtype=np.float64)
    if transform.shape != (4, 4):
        raise ValueError(f"expected (4,4), got {transform.shape}")
    rotation, translation = transform[:3, :3], transform[:3, 3]
    omega = _so3_log(rotation)
    theta = float(np.linalg.norm(omega))
    omega_hat = np.array([
        [0.0, -omega[2], omega[1]],
        [omega[2], 0.0, -omega[0]],
        [-omega[1], omega[0], 0.0],
    ])
    if theta < 1e-7:
        v_inv = np.eye(3) - 0.5 * omega_hat + (omega_hat @ omega_hat) / 12.0
    else:
        a = (1.0 / theta**2) - (
            (1.0 + np.cos(theta)) / (2.0 * theta * np.sin(theta))
        )
        v_inv = np.eye(3) - 0.5 * omega_hat + a * (omega_hat @ omega_hat)
    return np.concatenate([v_inv @ translation, omega])


def se3_exp(twist: np.ndarray) -> np.ndarray:
    """Exponential map for canonical [v,w] camera-frame twists."""
    twist = np.asarray(twist, dtype=np.float64)
    if twist.shape != (6,):
        raise ValueError(f"expected (6,), got {twist.shape}")
    velocity, omega = twist[:3], twist[3:]
    theta = float(np.linalg.norm(omega))
    omega_hat = np.array([
        [0.0, -omega[2], omega[1]],
        [omega[2], 0.0, -omega[0]],
        [-omega[1], omega[0], 0.0],
    ])
    if theta < 1e-7:
        rotation = np.eye(3) + omega_hat + 0.5 * omega_hat @ omega_hat
        v_matrix = np.eye(3) + 0.5 * omega_hat + (omega_hat @ omega_hat) / 6.0
    else:
        rotation = (
            np.eye(3)
            + (np.sin(theta) / theta) * omega_hat
            + ((1.0 - np.cos(theta)) / theta**2) * (omega_hat @ omega_hat)
        )
        v_matrix = (
            np.eye(3)
            + ((1.0 - np.cos(theta)) / theta**2) * omega_hat
            + ((theta - np.sin(theta)) / theta**3) * (omega_hat @ omega_hat)
        )
    transform = np.eye(4)
    transform[:3, :3] = rotation
    transform[:3, 3] = v_matrix @ velocity
    return transform


def integrate_actions(actions: np.ndarray) -> np.ndarray:
    """Integrate local twists into a camera-to-world trajectory from identity."""
    trajectory = [np.eye(4)]
    for action in np.asarray(actions):
        trajectory.append(trajectory[-1] @ se3_exp(action))
    return np.stack(trajectory)


def pose_deltas(poses: np.ndarray) -> np.ndarray:
    """Consecutive camera-frame twists from camera-to-world poses."""
    poses = np.asarray(poses, dtype=np.float64)
    if poses.ndim != 3 or poses.shape[1:] != (4, 4):
        raise ValueError(f"expected (T,4,4), got {poses.shape}")
    if len(poses) < 2:
        return np.zeros((0, 6), dtype=np.float32)
    relative = np.linalg.inv(poses[:-1]) @ poses[1:]
    return np.stack([se3_log(t) for t in relative]).astype(np.float32)


def tubelet_pose_positions(
    frame_indices: np.ndarray, tubelet: int, n_latents: int | None = None,
) -> np.ndarray:
    """Map temporal encoder outputs to fractional source-pose positions."""
    frame_indices = np.asarray(frame_indices, dtype=np.int64)
    if tubelet < 1:
        raise ValueError("tubelet must be positive")
    complete = len(frame_indices) // tubelet
    count = complete if n_latents is None else min(complete, int(n_latents))
    if count <= 0:
        return np.zeros(0, dtype=np.float64)
    groups = frame_indices[: count * tubelet].reshape(count, tubelet)
    return groups.mean(axis=1, dtype=np.float64)


def tubelet_pose_indices(
    frame_indices: np.ndarray, tubelet: int, n_latents: int | None = None,
) -> np.ndarray:
    """Nearest native rows for diagnostics; alignment uses fractional centres."""
    return np.rint(
        tubelet_pose_positions(frame_indices, tubelet, n_latents)
    ).astype(np.int64)


def interpolate_pose_rows(poses: np.ndarray, positions: np.ndarray) -> np.ndarray:
    """Interpolate c2w poses using linear translation and SO(3) geodesics."""
    poses = np.asarray(poses, dtype=np.float64)
    output = []
    for position in np.asarray(positions, dtype=np.float64):
        lo = int(np.floor(position))
        hi = min(lo + 1, len(poses) - 1)
        alpha = float(position - lo)
        transform = np.eye(4)
        transform[:3, 3] = (
            (1.0 - alpha) * poses[lo, :3, 3]
            + alpha * poses[hi, :3, 3]
        )
        relative_rotation = poses[lo, :3, :3].T @ poses[hi, :3, :3]
        transform[:3, :3] = (
            poses[lo, :3, :3]
            @ _so3_exp(alpha * _so3_log(relative_rotation))
        )
        output.append(transform)
    return np.stack(output) if output else np.zeros((0, 4, 4))


def align_latents_and_poses(
    latents: np.ndarray | torch.Tensor,
    poses: np.ndarray,
    frame_indices: np.ndarray,
    tubelet: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return equal-length latents/poses and actions between adjacent latents."""
    z = latents.detach().cpu().numpy() if torch.is_tensor(latents) else np.asarray(latents)
    positions = tubelet_pose_positions(frame_indices, tubelet, len(z))
    valid = (positions >= 0) & (positions <= len(poses) - 1)
    positions = positions[valid]
    z = z[:len(valid)][valid]
    aligned_poses = interpolate_pose_rows(poses, positions)
    return z.astype(np.float32), aligned_poses.astype(np.float64), pose_deltas(aligned_poses)


def video_split(
    sequence_id: str,
    train: float = 0.7,
    val: float = 0.15,
    case_id: str | None = None,
    dataset: str | None = None,
) -> str:
    """Stable video-level split independent of filesystem order."""
    if dataset == "SCARED" and case_id is not None:
        case_number = int(case_id.replace("dataset_", ""))
        if case_number in {1, 2, 3, 5}:
            return "train"
        if case_number == 6:
            return "val"
        if case_number == 7:
            return "test"
        raise ValueError(f"unrecognised SCARED case: {case_id}")
    if dataset == "C3VD":
        return "test"
    u = int(hashlib.sha1(sequence_id.encode("utf-8")).hexdigest()[:8], 16) / 2**32
    if u < train:
        return "train"
    if u < train + val:
        return "val"
    return "test"


@dataclass
class PhysicalSequence:
    sequence_id: str
    dataset: str
    latents: torch.Tensor
    actions: torch.Tensor
    depth_or_risk: torch.Tensor | None = None
    case_id: str | None = None

    def validate(self) -> None:
        if self.latents.ndim != 2:
            raise ValueError("latents must be (T,D)")
        if self.actions.shape != (self.latents.size(0) - 1, 6):
            raise ValueError(
                f"actions must be (T-1,6), got {tuple(self.actions.shape)} "
                f"for T={self.latents.size(0)}")
        if self.depth_or_risk is not None and self.depth_or_risk.size(0) != self.latents.size(0):
            raise ValueError("depth_or_risk must have one row per latent")


class PhysicalActionDataset(Dataset):
    """Sliding windows from whole-video split physical sequences."""

    def __init__(
        self,
        sequences: Iterable[PhysicalSequence],
        history: int = 4,
        horizon: int = 4,
        split: str = "train",
    ):
        self.sequences = []
        self.windows: list[tuple[int, int]] = []
        self._sequence_windows: dict[int, list[int]] | None = None
        self.history, self.horizon = int(history), int(horizon)
        for sequence in sequences:
            sequence.validate()
            if video_split(
                sequence.sequence_id,
                case_id=sequence.case_id,
                dataset=sequence.dataset,
            ) != split:
                continue
            seq_index = len(self.sequences)
            self.sequences.append(sequence)
            stop = sequence.latents.size(0) - self.history - self.horizon + 1
            self.windows.extend((seq_index, start) for start in range(max(stop, 0)))

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        sequence_index, start = self.windows[index]
        seq = self.sequences[sequence_index]
        split = start + self.history
        end = split + self.horizon
        item: dict[str, torch.Tensor | str] = {
            "history": seq.latents[start:split],
            "actions": seq.actions[split - 1:end - 1],
            "future": seq.latents[split:end],
            "sequence_id": seq.sequence_id,
            "dataset": seq.dataset,
            "start_index": torch.tensor(start, dtype=torch.long),
            "index": torch.tensor(index, dtype=torch.long),
        }
        if seq.depth_or_risk is not None:
            item["depth_or_risk"] = seq.depth_or_risk[split:end]
        return item

    def sequence_windows(self) -> dict[int, list[int]]:
        """Map sequence index -> dataset row indices, sorted by start time."""
        if self._sequence_windows is None:
            grouped: dict[int, list[int]] = {}
            for index, (sequence_index, _) in enumerate(self.windows):
                grouped.setdefault(sequence_index, []).append(index)
            self._sequence_windows = grouped
        return self._sequence_windows

    def hard_negative_index(
        self, index: int, radius: int = 64, rng=None,
    ) -> int:
        """A same-sequence, temporally adjacent counterfactual (hard negative).

        Evaluation batches follow temporal order, so the canonical shuffled
        action of a batch is usually a neighbouring window. Training must
        sample negatives from the same local distribution, otherwise the model
        is trained on easy global shuffles but tested on hard local ones.
        """
        rng = rng or np.random.default_rng()
        sequence_index, start = self.windows[index]
        same = self.sequence_windows()[sequence_index]
        starts = np.asarray([self.windows[i][1] for i in same])
        mask = (np.abs(starts - start) <= radius) & (starts != start)
        choices = np.asarray(same)[mask]
        if len(choices) == 0:
            others = np.asarray(same)
            others = others[others != index]
            if len(others) == 0:
                return index
            return int(rng.choice(others))
        return int(rng.choice(choices))


def save_sequences(sequences: Iterable[PhysicalSequence], path: str | Path) -> None:
    rows = []
    for seq in sequences:
        seq.validate()
        rows.append({
            "sequence_id": seq.sequence_id,
            "dataset": seq.dataset,
            "latents": seq.latents.cpu(),
            "actions": seq.actions.cpu(),
            "depth_or_risk": None if seq.depth_or_risk is None else seq.depth_or_risk.cpu(),
            "case_id": seq.case_id,
            "split": video_split(
                seq.sequence_id, case_id=seq.case_id, dataset=seq.dataset),
        })
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"pose_convention": "c2w_local_log_se3_v_then_w", "sequences": rows}, path)


def load_sequences(path: str | Path) -> list[PhysicalSequence]:
    pack = torch.load(path, map_location="cpu", weights_only=False)
    if pack.get("pose_convention") != "c2w_local_log_se3_v_then_w":
        raise ValueError("unsupported or missing pose convention")
    return [
        PhysicalSequence(
            sequence_id=row["sequence_id"],
            dataset=row["dataset"],
            latents=row["latents"].float(),
            actions=row["actions"].float(),
            depth_or_risk=None if row.get("depth_or_risk") is None else row["depth_or_risk"].float(),
            case_id=row.get("case_id"),
        )
        for row in pack["sequences"]
    ]
