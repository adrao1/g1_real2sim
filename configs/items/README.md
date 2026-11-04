# Item presets

The contact evaluator keeps the G1 robot and table from `TabletopSceneCfg` and
replaces only the rigid item at `{ENV_REGEX_NS}/Object`. Copy this example into
`configs/items/my_item.yml` and use your own **local, physics-ready USD**:

```yaml
name: my_item
usd_path: /absolute/path/to/my_item.usd
scale: 1.0
position: [0.0, 0.40, 0.87]
rotation_xyzw: [0.0, 0.0, 0.0, 1.0]
# grasp_file: /absolute/path/to/existing_grasps.yml
```

The position is the item's reset position in each tabletop environment, in
meters. Set it so the item starts above the table. Rotation uses **xyzw** order.
`usd_path` and `grasp_file` may be relative to this preset file. Scale must be a
positive uniform value and is applied to both the simulated item and GraspGenX.

If `grasp_file` is omitted, the evaluator generates top-100 `unitree_g1` grasps
with the sibling `GraspGenX` checkout before launching Isaac Sim and caches the
YAML under `outputs/grasp_evaluation/grasp_cache/`. Use `--regenerate-grasps`
after changing externally referenced USD geometry or GraspGenX checkpoints.
Changing only the item's position or rotation does not require regeneration:
grasp poses are expressed in the item's local frame.
If you supply `grasp_file` or `--grasp-file`, use a YAML generated for this
USD at this scale; the grasp format itself does not record that pairing.
