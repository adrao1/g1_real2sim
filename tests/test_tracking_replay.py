"""Checks that the complete tracked motion is preserved when replay is slowed."""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest


def test_resampling_keeps_end_pose_and_scales_duration(monkeypatch):
    script = Path(__file__).resolve().parents[1] / "scripts/replay_tracked_pick_place.py"
    monkeypatch.syspath_prepend(str(script.parent))
    spec = importlib.util.spec_from_file_location("replay_tracked_pick_place", script)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    poses = np.repeat(np.eye(4)[None], 2, axis=0)
    poses[1, :3, :3] = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
    poses[1, 0, 3] = 1.0
    frames = np.array([0, 30], dtype=float)

    normal_positions, _ = module.resample_poses(poses, frames, 30.0, 1.0, 0.5)
    slow_positions, slow_quaternions = module.resample_poses(poses, frames, 30.0, 0.5, 0.5)

    assert len(normal_positions) == 3
    assert slow_positions[:, 0] == pytest.approx([0.0, 0.25, 0.5, 0.75, 1.0])
    assert slow_positions[-1] == pytest.approx(poses[-1, :3, 3])
    assert slow_quaternions[-1] == pytest.approx([0.0, 0.0, 2**-0.5, 2**-0.5])
    assert np.linalg.norm(slow_quaternions, axis=1) == pytest.approx(np.ones(len(slow_quaternions)))
