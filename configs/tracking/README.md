# Tracking coordinate frames

Each YAML file in this directory pairs one item preset with a FoundationPose
`ob_in_cam` pose sequence. Positions and translations are in meters.

`world_from_camera` is a row-major 4×4 transform from the **OpenCV camera
frame** to the **simulator world frame**. In the camera frame, +X points right
in the image, +Y points down, and +Z points forward. In this tabletop scene,
world +X points to the robot's right, +Y points from the robot toward the
table, and +Z points up. The matrix's last column is the camera optical
center's position relative to **world (0, 0, 0)**, not relative to the table.

The table's USD root is placed at world `(0, 0.55, 0)` in
`TabletopSceneCfg`. That root is at ground level under the table's center. The
table asset spans about ±0.401 m in local Y and its top is about 0.758 m above
the ground. For the D415 setup, the measured camera offsets give:

```text
world X = +0.330 m                     # right of the table centerline
world Y = 0.55 - 0.401 - 0.587 = -0.438 m  # behind the near table edge
world Z = 0.758 + 0.404 = 1.162 m     # above the tabletop
```

These values assume the measurements refer to the camera's optical center.
The rotation in `cracker_box_d415_01.yml` is an initial look-at estimate aimed
at the simulated item's starting position `(0, 0.40, 0.87)` with zero roll; it
is not a measured camera orientation.

FoundationPose pose files are `camera_from_mesh` transforms. For a tracking
OBJ exported in the simulator USD's object-root frame, the waypoint script
computes `world_from_item = world_from_camera @ camera_from_mesh`. It plots
the complete position path and XYZ orientation arrows at selected frames.
When `world_from_camera` is `null`, it plots raw camera-frame poses instead.

The blue camera marker uses the translation column of `world_from_camera`.
Its larger XYZ arrows show the camera orientation. Red, green, and blue arrows
indicate positive X, Y, and Z for both item and camera poses. When the matrix
is `null`, the camera marker and arrows are omitted. The camera orientation is
still an estimate; visualizing it does not calibrate it.

`scripts/replay_tracked_pick_place.py` requires `world_from_camera` and uses
the first full object pose, including its rotation, to initialize the simulated
item and approach an object-relative grasp. It closes the fingers, checks for
sustained thumb and opposing-finger contact, and stops. It does not lift or
follow the recorded trajectory.
To choose a grasp for a recording, run `scripts/evaluate_contact_grasps.py`
with `--tracking-config` pointing to this YAML. It places the item at the
first tracked world pose and runs the existing approach, closure, and lift
evaluation there. Its `run.json` records the YAML path and first frame number.
For the cracker box, evaluation and replay use the same bundled grasp YAML by
default, so a selected `grasp_id` can be passed to replay. If you supply
`--grasp-file` to either script, use the same file in both.
The current cracker-box camera rotation is still an estimate, so the starting
pose should be treated as a simulation smoke test until that rotation is
calibrated. Pass `--record-video` to save the grasp attempt as one MP4 in a new
directory under `outputs/tracked_grasps/`. Use `--video-dir` to choose the directory;
video recording requires the `video` extra.
