"""Load FoundationPose recordings into the simulator world frame."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml


def _pose_matrix(value: object, label: str) -> np.ndarray:
    matrix = np.asarray(value, dtype=float)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise ValueError(f"{label} must be a finite 4x4 matrix")
    if not np.allclose(matrix[3], (0, 0, 0, 1), atol=1e-5):
        raise ValueError(f"{label} must have homogeneous last row [0, 0, 0, 1]")
    rotation = matrix[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=5e-3) or np.linalg.det(rotation) < 0.99:
        raise ValueError(f"{label} must contain a proper rigid rotation")
    return matrix


def load_tracking(path: Path) -> tuple[Path, np.ndarray, np.ndarray]:
    """Return item preset, frame numbers, and world-from-item matrices."""
    path = path.expanduser().resolve(strict=True)
    with path.open() as file:
        config = yaml.safe_load(file)
    if not isinstance(config, dict) or not isinstance(config.get("item_config"), str):
        raise ValueError("Tracking YAML needs item_config")
    if not isinstance(config.get("pose_dir"), str):
        raise ValueError("Tracking YAML needs pose_dir")
    if config.get("world_from_camera") is None:
        raise ValueError("Tracking requires a non-null world_from_camera transform")
    world_from_camera = _pose_matrix(config["world_from_camera"], "world_from_camera")
    item_path = (path.parent / config["item_config"]).resolve(strict=True)
    pose_dir = (path.parent / config["pose_dir"]).resolve(strict=True)
    if not pose_dir.is_dir():
        raise ValueError(f"Not a pose directory: {pose_dir}")
    files = list(pose_dir.glob("*.txt"))
    if len(files) < 2 or any(not file.stem.isdigit() for file in files):
        raise ValueError(f"Expected at least two numbered pose files in {pose_dir}")
    files.sort(key=lambda file: int(file.stem))
    frame_numbers = np.array([int(file.stem) for file in files], dtype=float)
    if len(set(frame_numbers)) != len(files):
        raise ValueError("Pose frame numbers must be unique")
    poses = [world_from_camera @ _pose_matrix(np.loadtxt(file), str(file)) for file in files]
    return item_path, frame_numbers, np.stack(poses)
