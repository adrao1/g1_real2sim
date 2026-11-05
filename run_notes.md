# G1 tabletop grasp runs

Run commands from this repository root.
See `configs/tracking/README.md` for the camera and world frame conventions.

```bash
# Inspect the registered scene in a viewport.
uv run --extra isaacsim isaaclab zero_agent \
  --task G1Real2sim-PickPlace-G1 --num_envs 1 --device cpu --viz kit

# After tracking the cracker box with assets/ycb/cracker_box/tracking/textured.obj
# and --debug_dir debug_cracker_box, view the approximately world-aligned path
# (cyan to orange, green start, red end), sampled XYZ orientation arrows
# (red X, green Y, blue Z), and the estimated camera pose (blue sphere + axes).
uv run --extra isaacsim python scripts/visualize_waypoints.py \
  --tracking-config configs/tracking/cracker_box_d415_04.yml

# Save one fixed-camera PNG per recorded item pose, independent of the grasp.
# The script prints a new directory under outputs/tracking_pose_screenshots/.
uv run --extra isaacsim python scripts/screenshot_tracked_poses.py \
  --tracking-config configs/tracking/cracker_box_d415_05.yml

# Save grasp-pose images at recording 03's first object pose. Omit --grasps
# to render every grasp. The script prints its new output directory under
# outputs/grasp_visualization/cracker_box_d415_03/. These images show final
# wrist poses, not grasp outcomes.
uv run --extra isaacsim python scripts/visualize_grasps.py \
  --tracking-config configs/tracking/cracker_box_d415_03.yml \
  --grasps 56 68 88

# Grasp the cracker box at recording 05's first pose, lift it, and move it to
# frame 86's full object pose. The fingers stay closed at the target.
uv run --extra isaacsim python scripts/replay_tracked_pick_place.py \
  --tracking-config configs/tracking/cracker_box_d415_05.yml \
  --grasp-id 22 --target-frame 86 --viz kit

# Save the grasp and move as an MP4 without a viewport.
# The script prints the output path under outputs/tracked_moves/.
uv run --extra isaacsim --extra video python scripts/replay_tracked_pick_place.py \
  --tracking-config configs/tracking/cracker_box_d415_05.yml \
  --grasp-id 22 --target-frame 86 --record-video

# Save one screenshot of the mustard-bottle scene without generating grasps.
uv run --extra isaacsim python scripts/screenshot_scene.py \
  --item-config configs/items/mustard_bottle.yml

# Re-export FoundationPose meshes from the simulator's visual USDs.
.venv/bin/python scripts/export_ycb_tracking_meshes.py

# Evaluate grasps at recording 04's first object pose.
uv run --extra isaacsim python scripts/evaluate_contact_grasps.py \
  --tracking-config configs/tracking/cracker_box_d415_04.yml
```

Each run gets a new directory under `outputs/grasp_evaluation/contact_grasps/<item>/`.
Its `run.json` records the item, pose, scale, and grasp source. Tracking-based
runs also record the tracking YAML and first frame number. `trials.csv` records
every attempt. The `video_file` column points to retained successful MP4s.
