# G1 tabletop grasp runs

Run commands from this repository root.

```bash
# Inspect the registered scene in a viewport.
uv run --extra isaacsim isaaclab zero_agent \
  --task G1Real2sim-PickPlace-G1 --num_envs 1 --device cpu --viz kit

# Save a screenshot with a visual-only magenta sphere at the selected item's start position.
uv run --extra isaacsim python scripts/visualize_waypoints.py \
  --item-config configs/items/banana.yml

# Save one screenshot of the mustard-bottle scene without generating grasps.
uv run --extra isaacsim python scripts/screenshot_scene.py \
  --item-config configs/items/mustard_bottle.yml

# Evaluate every grasp in the bundled cracker-box YAML. Video is on by default;
# 25 grasps run in parallel per batch. Only successful clips are retained.
uv run --extra isaacsim --extra video python scripts/evaluate_contact_grasps.py

# Evaluate grasp 22 five times, resetting the item and hand each time.
uv run --extra isaacsim --extra video python scripts/evaluate_contact_grasps.py \
  --grasps 22 --repeats 5

# Evaluate all 100 cracker-box grasps simultaneously without video.
uv run --extra isaacsim python scripts/evaluate_contact_grasps.py \
  --no-video --batch-size 100

# Re-evaluate the successes in an earlier result CSV.
uv run --extra isaacsim --extra video python scripts/evaluate_contact_grasps.py \
  --successful-from outputs/grasp_evaluation/contact_closure_all100/trials.csv

# Evaluate another physics-ready USD item. A matching grasp YAML is generated
# with the separate GraspGenX installation and cached if the preset omits one.
uv run --extra isaacsim --extra video python scripts/evaluate_contact_grasps.py \
  --item-config configs/items/my_item.yml

# Change only the item's start position; object-relative grasps are reused.
uv run --extra isaacsim --extra video python scripts/evaluate_contact_grasps.py \
  --item-position 0.0 0.45 0.90 --grasps 22
```

The item-preset format is described in `configs/items/README.md`. Use
`--graspgenx-root /path/to/GraspGenX` if the checkout is not beside this repo.
Use `--grasp-file /path/to/grasps.yml` to supply existing poses, or
`--regenerate-grasps` to refresh generated poses for a local USD.

Each run gets a new directory under `outputs/grasp_evaluation/contact_grasps/<item>/`.
Its `run.json` records the item, pose, scale, and grasp source; `trials.csv` records
every attempt. The `video_file` column points to retained successful MP4s.
