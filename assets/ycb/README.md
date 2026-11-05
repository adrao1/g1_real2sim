# Local YCB assets

These are the YCB assets used by `configs/items/banana.yml` and
`configs/items/mustard_bottle.yml`. Each item has its own folder. Keep the USD
files and their texture paths together when moving an item folder.

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

Grasp YAMLs must match the specific USD and scale used for evaluation.
