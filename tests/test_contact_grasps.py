"""Checks for item/grasp pairing and saved contact-evaluation outputs."""

import argparse
import csv
import importlib.util
import sys
from pathlib import Path

import pytest
import yaml

from g1_real2sim.tasks.pick_place.config.g1.env_cfg import PickPlaceEnvCfg


@pytest.fixture(scope="module")
def evaluator():
    script = Path(__file__).resolve().parents[1] / "scripts/evaluate_contact_grasps.py"
    spec = importlib.util.spec_from_file_location("evaluate_contact_grasps", script)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _grasp_file(path: Path, ids=(4, 9)) -> Path:
    grasps = {
        f"grasp_{index}": {
            "position": [0.1, 0.0, 0.0],
            "orientation": {"xyz": [0.0, 0.0, 0.0], "w": 1.0},
        }
        for index in ids
    }
    path.write_text(yaml.safe_dump({"format": "isaac_grasp", "format_version": 1.0, "grasps": grasps}))
    return path


def test_item_preset_and_selection(evaluator, tmp_path):
    usd = tmp_path / "item.usd"
    usd.write_text("#usda 1.0\n")
    _grasp_file(tmp_path / "poses.yml")
    preset = tmp_path / "item.yml"
    preset.write_text(
        yaml.safe_dump(
            {
                "name": "test_item",
                "usd_path": "item.usd",
                "scale": 1.2,
                "position": [0.0, 0.5, 0.9],
                "rotation_xyzw": [0.0, 0.0, 0.0, 1.0],
                "grasp_file": "poses.yml",
            }
        )
    )
    cfg = PickPlaceEnvCfg()
    item = evaluator.configure_item(
        cfg,
        argparse.Namespace(item_config=preset, item_position=[0.1, 0.6, 0.9], item_rotation=None),
    )
    assert cfg.scene.object.spawn.usd_path == str(usd)
    assert cfg.scene.object.spawn.scale == (1.2, 1.2, 1.2)
    assert cfg.scene.object.init_state.pos == (0.1, 0.6, 0.9)
    assert item.grasp_file == tmp_path / "poses.yml"
    grasps = evaluator.load_grasps(item.grasp_file)
    assert evaluator.select_grasps(argparse.Namespace(grasps=None, successful_from=None), grasps, item) == [4, 9]

    results = tmp_path / "results.csv"
    with results.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["item_name", "grasp_id", "success"])
        writer.writeheader()
        writer.writerows(
            [
                {"item_name": "test_item", "grasp_id": "grasp_4", "success": "True"},
                {"item_name": "test_item", "grasp_id": "grasp_4", "success": "True"},
                {"item_name": "test_item", "grasp_id": "grasp_9", "success": "False"},
            ]
        )
    assert evaluator.select_grasps(argparse.Namespace(grasps=None, successful_from=results), grasps, item) == [4]


def test_tracking_config_uses_first_world_pose_and_replay_grasps(evaluator, tmp_path):
    usd = tmp_path / "item.usd"
    usd.write_text("#usda 1.0\n")
    (tmp_path / "item.yml").write_text(
        yaml.safe_dump(
            {
                "name": "cracker_box",
                "usd_path": "item.usd",
                "scale": 1.0,
                "position": [0.0, 0.4, 0.87],
                "rotation_xyzw": [0.0, 0.0, 0.0, 1.0],
            }
        )
    )
    poses = tmp_path / "poses"
    poses.mkdir()
    (poses / "000010.txt").write_text("0 -1 0 0.1\n1 0 0 0.2\n0 0 1 0.3\n0 0 0 1\n")
    (poses / "000011.txt").write_text("1 0 0 0.2\n0 1 0 0.3\n0 0 1 0.4\n0 0 0 1\n")
    tracking = tmp_path / "tracking.yml"
    tracking.write_text(
        yaml.safe_dump(
            {
                "item_config": "item.yml",
                "pose_dir": "poses",
                "world_from_camera": [
                    [1, 0, 0, 0.3],
                    [0, 1, 0, -0.4],
                    [0, 0, 1, 1.0],
                    [0, 0, 0, 1],
                ],
            }
        )
    )
    cfg = PickPlaceEnvCfg()
    item, frame = evaluator.configure_tracked_item(cfg, tracking)
    assert frame == 10
    assert item.position == pytest.approx((0.4, -0.2, 1.3))
    assert item.rotation_xyzw == pytest.approx((0.0, 0.0, 2**-0.5, 2**-0.5))
    assert cfg.scene.object.init_state.pos == item.position
    assert item.grasp_file == evaluator.DEFAULT_GRASPS


def test_generated_grasps_use_scene_asset_and_scale(evaluator, tmp_path, monkeypatch):
    usd = tmp_path / "item.usd"
    usd.write_text("#usda 1.0\n")
    root = tmp_path / "GraspGenX"
    (root / ".venv/bin").mkdir(parents=True)
    (root / "scripts").mkdir()
    (root / ".venv/bin/python").write_text("python")
    (root / "scripts/demo_object_mesh.py").write_text("script")
    monkeypatch.chdir(tmp_path)
    calls = []

    def fake_run(command, *, cwd, check):
        assert check and cwd == root
        calls.append(command)
        assert command[command.index("--mesh_file") + 1] == str(usd)
        assert command[command.index("--mesh_scale") + 1] == "1.2"
        _grasp_file(Path(command[command.index("--output_file") + 1]))

    monkeypatch.setattr(evaluator.subprocess, "run", fake_run)
    item = evaluator.Item("test_item", str(usd), 1.2, (0.0, 0.5, 0.9), (0.0, 0.0, 0.0, 1.0), None)
    args = argparse.Namespace(grasp_file=None, regenerate_grasps=False, graspgenx_root=root)
    generated = evaluator.obtain_grasps(item, args)
    assert generated.is_file() and len(calls) == 1
    assert evaluator.obtain_grasps(item, args) == generated and len(calls) == 1


def test_successful_video_is_kept_and_failure_video_removed(evaluator, tmp_path):
    (tmp_path / "slot_00_0000.mp4").write_bytes(b"success")
    (tmp_path / "slot_01_0000.mp4").write_bytes(b"failure")
    rows = [
        {"grasp_id": "grasp_4", "repeat": 1, "success": True, "video_file": ""},
        {"grasp_id": "grasp_9", "repeat": 1, "success": False, "video_file": ""},
    ]
    evaluator.finalize_videos(tmp_path, rows, num_envs=2, clip_index=0)
    assert Path(rows[0]["video_file"]).read_bytes() == b"success"
    assert rows[1]["video_file"] == ""
    assert sorted(path.name for path in tmp_path.glob("*.mp4")) == ["grasp_4_repeat_1.mp4"]


def test_step_count_matches_video_interval(evaluator):
    assert evaluator.TOTAL_STEPS == 375


def test_empty_grasp_file_fails_before_simulation(evaluator, tmp_path):
    path = tmp_path / "empty.yml"
    path.write_text("format: isaac_grasp\nformat_version: 1.0\ngrasps: {}\n")
    with pytest.raises(ValueError, match="nonempty grasps"):
        evaluator.load_grasps(path)


def test_wrong_grasp_format_fails_before_simulation(evaluator, tmp_path):
    path = tmp_path / "wrong.yml"
    path.write_text("format: other\nformat_version: 1.0\ngrasps: {grasp_4: {}}\n")
    with pytest.raises(ValueError, match="Isaac grasp format"):
        evaluator.load_grasps(path)
