import dataclasses

import einops
import numpy as np

from gaussiandream import transforms
from gaussiandream.models import model as _model


def make_libero_example() -> dict:
    """Creates a random input example for the Libero policy."""
    return {
        "observation/state": np.random.rand(8),
        "observation/image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/wrist_image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "prompt": "do something",
    }


def _parse_image(image) -> np.ndarray:
    # Handle dict format from LIBERO dataset: {'bytes': ..., 'path': ...}
    if isinstance(image, dict):
        if 'bytes' in image:
            from PIL import Image
            import io
            image = Image.open(io.BytesIO(image['bytes']))
            image = np.array(image)
        else:
            raise ValueError(f"Unexpected dict format for image: {image.keys()}")
    # Handle list of dicts (temporal frames)
    elif isinstance(image, (list, tuple)) and len(image) > 0 and isinstance(image[0], dict):
        from PIL import Image
        import io
        frames = []
        for img_dict in image:
            if 'bytes' in img_dict:
                img = Image.open(io.BytesIO(img_dict['bytes']))
                frames.append(np.array(img))
            else:
                raise ValueError(f"Unexpected dict format for image: {img_dict.keys()}")
        image = np.stack(frames, axis=0)  # Stack to (T, H, W, C)
        return image  # Already in correct format
    else:
        image = np.asarray(image)

    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)

    # Handle single frame (C, H, W) -> (H, W, C)
    if image.ndim == 3 and image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    # Handle temporal frames (T, C, H, W) -> (T, H, W, C)
    elif image.ndim == 4 and image.shape[1] == 3:
        image = einops.rearrange(image, "t c h w -> t h w c")

    return image


@dataclasses.dataclass(frozen=True)
class LiberoInputs(transforms.DataTransformFn):
    """
    This class is used to convert inputs to the model to the expected format. It is used for both training and inference.

    For your own dataset, you can copy this class and modify the keys based on the comments below to pipe
    the correct elements of your dataset into the model.
    """

    # Determines which model will be used.
    # Do not change this for your own dataset.
    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        # Possibly need to parse images to uint8 (H,W,C) since LeRobot automatically
        # stores as float32 (C,H,W), gets skipped for policy inference.
        # Keep this for your own dataset, but if your dataset stores the images
        # in a different key than "observation/image" or "observation/wrist_image",
        # you should change it below.
        # Pi0 models support three image inputs at the moment: one third-person view,
        # and two wrist views (left and right). If your dataset does not have a particular type
        # of image, e.g. wrist images, you can comment it out here and replace it with zeros like we do for the
        # right wrist image below.
        base_image = _parse_image(data["observation/image"])
        wrist_image = _parse_image(data["observation/wrist_image"])

        # Match temporal agent view (T, H, W, C) with a single wrist frame at current time.
        if base_image.ndim == 4 and wrist_image.ndim == 3:
            wrist_image = np.stack([wrist_image] * base_image.shape[0], axis=0)

        # Create inputs dict. Do not change the keys in the dict below.
        inputs = {
            "state": data["observation/state"],
            "image": {
                "base_0_rgb": base_image,
                "left_wrist_0_rgb": wrist_image,
                # Pad any non-existent images with zero-arrays of the appropriate shape.
                "right_wrist_0_rgb": np.zeros_like(base_image),
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                # We only mask padding images for pi0 model, not pi0-FAST. Do not change this for your own dataset.
                "right_wrist_0_rgb": np.True_ if self.model_type == _model.ModelType.PI0_FAST else np.False_,
            },
        }

        # Pad actions to the model action dimension. Keep this for your own dataset.
        # Actions are only available during training.
        if "actions" in data:
            inputs["actions"] = data["actions"]

        # Pass the prompt (aka language instruction) to the model.
        # Keep this for your own dataset (but modify the key if the instruction is not
        # stored in "prompt"; the output dict always needs to have the key "prompt").
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        # Add depth data if available (for World Model depth supervision)
        if "observation/depth" in data:
            inputs["depth"] = data["observation/depth"]
            # Debug: log depth info
            import logging
            if hasattr(data["observation/depth"], 'shape'):
                logging.info(f"[LiberoInputs] depth shape: {data['observation/depth'].shape}")
            else:
                logging.info(f"[LiberoInputs] depth type: {type(data['observation/depth'])}")

        if "observation/flow_3d" in data:
            inputs["flow_3d"] = data["observation/flow_3d"]
        if "observation/flow_valid_mask" in data:
            inputs["flow_valid_mask"] = data["observation/flow_valid_mask"]

        return inputs


@dataclasses.dataclass(frozen=True)
class LiberoOutputs(transforms.DataTransformFn):
    """
    This class is used to convert outputs from the model back the the dataset specific format. It is
    used for inference only.

    For your own dataset, you can copy this class and modify the action dimension based on the comments below.
    """

    action_dim: int = 7

    def __call__(self, data: dict) -> dict:
        # Only return the first N actions -- since we padded actions above to fit the model action
        # dimension, we need to now parse out the correct number of actions in the return dict.
        # For Libero, we only return the first 7 actions (since the rest is padding).
        # For your own dataset, replace `7` with the action dimension of your dataset.
        return {"actions": np.asarray(data["actions"][:, : self.action_dim])}
