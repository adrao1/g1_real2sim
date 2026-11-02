cd ~/code/g1_real2sim

run pick and place with zero agent
uv run --extra isaacsim isaaclab zero_agent \
  --task G1Real2sim-PickPlace-G1 \
  --num_envs 1 --device cpu --viz kit

print initial hand poses
uv run --extra isaacsim python scripts/inspect_initial_hand_poses.py




cd ~/code/GraspGenX
generate grasp poses for an object
uv run python scripts/demo_object_mesh.py \
  --mesh_file /tmp/https/omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/6.2/Isaac/Props/YCB/Axis_Aligned_Physics/003_cracker_box.usd \
  --mesh_scale 1.0 \
  --gripper_name unitree_g1 \
  --grasp_threshold -1.0 \
  --return_topk \
  --topk_num_grasps 100 \
  --plot_top_mesh \
  --output_file /tmp/cracker_box_grasps.yml