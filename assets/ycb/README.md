# Local YCB assets

These are the YCB assets used by `configs/items/banana.yml`,
`configs/items/cracker_box.yml`, and `configs/items/mustard_bottle.yml`.
Each item has a simulator USD and a `tracking/` folder containing the same
visual mesh exported as `textured.obj`, `textured.mtl`, and `textured.png` for
FoundationPose. Keep USD references and textures in their relative layout.

- `banana/banana.usd` and `banana/textures/obj_000010.png` come from
  [NVlabs RoboLab](https://github.com/NVlabs/RoboLab/tree/main/assets/objects/ycb).
  Its YCB license is preserved in `banana/LICENSE`. The USD has a rigid body
  and a collision mesh.
- `mustard_bottle/Axis_Aligned_Physics/006_mustard_bottle.usd`, its referenced
  `mustard_bottle/Axis_Aligned/006_mustard_bottle.usd`, and
  `mustard_bottle/Axis_Aligned/Materials/Textures/006_mustard_bottle_COLOR.png`
  come from NVIDIA's
  [Isaac Sim YCB assets](https://docs.isaacsim.omniverse.nvidia.com/latest/assets/usd_assets_overview.html).
  The USD is from `Props/YCB/Axis_Aligned_Physics` and has a rigid body and a
  collision mesh. The `mustard_bottle/Axis_Aligned_Physics/Materials` link lets
  that USD resolve the shared texture without a duplicate copy.
- `cracker_box/Axis_Aligned_Physics/003_cracker_box.usd`, its referenced
  `cracker_box/Axis_Aligned/003_cracker_box.usd`, and
  `cracker_box/Axis_Aligned/Materials/Textures/003_cracker_box_COLOR.png`
  come from the Isaac Sim 6.2 `Props/YCB` asset catalog. The scene's default
  cracker box now uses this local physics USD.

Rebuild the three FoundationPose meshes from the USDs with:

```bash
.venv/bin/python scripts/export_ycb_tracking_meshes.py
```

The exporter selects each textured visual mesh, applies its transform relative
to the USD's object root, triangulates it, and writes its UVs and texture. In
particular, the mustard and cracker USDs scale their visual meshes by 0.01;
the OBJ vertices are exported in meters in the object-root frame. The OBJ
coordinates are suitable for matching a FoundationPose run against the
corresponding simulator USD. Tracking poses must use the matching exported mesh.

Grasp YAMLs must match the specific USD and scale used for evaluation.
