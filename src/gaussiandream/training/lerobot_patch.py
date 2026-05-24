"""
Monkey patch for LeRobot to handle dict-format images from LIBERO dataset.
"""
import torch
from lerobot.common.datasets import utils as lerobot_utils


# Save original function
_original_hf_transform_to_torch = lerobot_utils.hf_transform_to_torch


def patched_hf_transform_to_torch(items_dict):
    """
    Patched version that handles dict-format data (e.g., {'bytes': ..., 'path': ...}).
    """
    for key in items_dict:
        # Skip dict values - they will be handled by our transforms later
        if isinstance(items_dict[key], list) and len(items_dict[key]) > 0:
            if isinstance(items_dict[key][0], dict):
                # Keep dict format as-is, don't try to convert to tensor
                continue

        # For non-dict data, use original conversion
        items_dict[key] = [
            x if isinstance(x, (str, dict)) else torch.tensor(x)
            for x in items_dict[key]
        ]

    return items_dict


def apply_patch():
    """Apply the monkey patch to LeRobot."""
    lerobot_utils.hf_transform_to_torch = patched_hf_transform_to_torch
    print("Applied LeRobot patch for dict-format images")
