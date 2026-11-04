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


capture screenshots of 100 grasps generated from GraspGenX
cd ~/code/g1_real2sim_ik
uv run --extra isaacsim python scripts/visualize_grasps.py

evaluate all 100 grasps in parallel with box collisions and gravity enabled
uv run --extra isaacsim python scripts/evaluate_grasps.py

rerun successful grasps from the 100-grasp results and save one MP4 per grasp
uv run --extra isaacsim --extra video python scripts/evaluate_grasps.py \
  --successful-from outputs/grasp_evaluation/results.csv --record-video

rerun chosen grasps and save one MP4 per grasp
uv run --extra isaacsim --extra video python scripts/evaluate_grasps.py \
  --grasps 22 36 39 --record-video

videos are saved under outputs/grasp_evaluation/videos and selected results under
outputs/grasp_evaluation/selected_results.csv
