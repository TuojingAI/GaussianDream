"""
Script for converting a Robocasa dataset to LeRobot format.

Usage:
uv run examples/robocasa/convert_robocasa_data_to_lerobot.py --data_dir /path/to/robocasa/data

If you want to push your dataset to the Hugging Face Hub, you can use the following command:
uv run examples/robocasa/convert_robocasa_data_to_lerobot.py --data_dir /path/to/robocasa/data --push_to_hub
"""

from pathlib import Path
import shutil

import h5py
from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import numpy as np
import tyro

# Define output dataset name
REPO_NAME = "your_username/robocasa_converted"


def main(data_dir: str, *, push_to_hub: bool = False):
    # Clean up any existing dataset in the output directory
    output_path = HF_LEROBOT_HOME / REPO_NAME
    if output_path.exists():
        shutil.rmtree(output_path)

    # Create LeRobot dataset structure
    # Note: You may need to adjust the shape based on the actual observation space of Robocasa
    dataset = LeRobotDataset.create(
        repo_id=REPO_NAME,
        robot_type="panda",  # Or other robot type
        fps=20,  # Robocasa default is usually 20Hz
        features={
            "image": {  # agentview
                "dtype": "image",
                "shape": (128, 128, 3),
                "names": ["height", "width", "channel"],
            },
            "wrist_image": {  # eye_in_hand
                "dtype": "image",
                "shape": (128, 128, 3),
                "names": ["height", "width", "channel"],
            },
            "state": {  # joint positions + gripper
                "dtype": "float32",
                "shape": (9,),  # 7 joints + 2 gripper
                "names": ["state"],
            },
            "actions": {
                "dtype": "float32",
                "shape": (12,),  # Matches dataset action dim
                "names": ["actions"],
            },
        },
    )

    data_path = Path(data_dir)
    # Robocasa datasets are typically HDF5 files
    demo_files = list(data_path.glob("**/*.hdf5"))

    print(f"Found {len(demo_files)} demos in {data_dir}")

    for demo_file in demo_files:
        print(f"Processing {demo_file}...")
        try:
            with h5py.File(demo_file, "r") as f:
                # Read HDF5 data
                # Note: The structure of Robocasa HDF5 files might vary.
                # Usually it is under "data/demo_X/..."

                # This is a simplified reading logic, adjust according to actual HDF5 structure
                if "data" in f:
                    demos_group = f["data"]
                    demo_keys = list(demos_group.keys())
                else:
                    # Some formats might have the root as the demo
                    demos_group = {"demo": f}
                    demo_keys = ["demo"]

                for demo_key in demo_keys:
                    demo = demos_group[demo_key]

                    # Check if required keys exist
                    if "obs" not in demo or "actions" not in demo:
                        print(f"Skipping {demo_key} in {demo_file}: missing obs or actions")
                        continue

                    obs = demo["obs"]
                    actions = demo["actions"][:]

                    num_steps = actions.shape[0]

                    # Get task description if available
                    task_description = f.attrs.get("env_name", "unknown_task")
                    # Try to find language instruction in the file if available
                    # if "model_file" in f.attrs: ...

                    for i in range(num_steps):
                        # Extract images and state
                        # Note: Robocasa images are usually (H, W, C) and might be flipped
                        # Adjust keys based on your camera names

                        # Example keys, might need to be changed to match your config
                        img_key = "robot0_agentview_left_image"
                        wrist_key = "robot0_eye_in_hand_image"

                        if img_key not in obs or wrist_key not in obs:
                            # Fallback or skip if keys don't match
                            # print(f"Missing image keys in {demo_key}")
                            pass

                        img = obs[img_key][i]
                        wrist_img = obs[wrist_key][i]

                        # State: usually joint positions + gripper
                        # Adjust keys based on robot
                        joint_pos = obs["robot0_joint_pos"][i]
                        gripper_pos = obs["robot0_gripper_qpos"][i]
                        state = np.concatenate([joint_pos, gripper_pos]).astype(np.float32)

                        dataset.add_frame(
                            {
                                "image": img,
                                "wrist_image": wrist_img,
                                "state": state,
                                "actions": actions[i].astype(np.float32),
                                "task": str(task_description),
                            }
                        )

                    dataset.save_episode()
        except Exception as e:
            print(f"Error processing {demo_file}: {e}")

    if push_to_hub:
        dataset.push_to_hub(tags=["robocasa", "panda"], private=True)


if __name__ == "__main__":
    tyro.cli(main)
