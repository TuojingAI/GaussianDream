import dataclasses

import einops
import numpy as np

from gaussiandream import transforms
from gaussiandream.models import model as _model


def make_robocasa_example() -> dict:
    """Creates a random input example for the Robocasa policy."""
    return {
        "observation/state": np.random.rand(16),
        "observation/image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/wrist_image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "prompt": "do something in robocasa",
    }


def _parse_image(image) -> np.ndarray:
    if isinstance(image, dict):
        if "bytes" in image:
            from PIL import Image
            import io

            image = Image.open(io.BytesIO(image["bytes"]))
            image = np.array(image)
        else:
            raise ValueError(f"Unexpected dict format for image: {image.keys()}")
    elif isinstance(image, (list, tuple)) and len(image) > 0 and isinstance(image[0], dict):
        from PIL import Image
        import io

        frames = []
        for img_dict in image:
            if "bytes" not in img_dict:
                raise ValueError(f"Unexpected dict format for image: {img_dict.keys()}")
            img = Image.open(io.BytesIO(img_dict["bytes"]))
            frames.append(np.array(img))
        image = np.stack(frames, axis=0)
        return image

    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.ndim == 3 and image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    elif image.ndim == 4 and image.shape[1] == 3:
        image = einops.rearrange(image, "t c h w -> t h w c")
    return image


@dataclasses.dataclass(frozen=True)
class RobocasaInputs(transforms.DataTransformFn):
    """
    Transforms Robocasa data to the format expected by the model.
    """

    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        # Parse images
        # Robocasa typically provides "agentview" (base) and "robot0_eye_in_hand" (wrist)
        base_image = _parse_image(data["observation/image"])
        wrist_image = _parse_image(data["observation/wrist_image"])

        inputs = {
            "state": data["observation/state"],
            "image": {
                "base_0_rgb": base_image,
                "left_wrist_0_rgb": wrist_image,
                # Pad right wrist with zeros as Robocasa (Panda) usually has 1 wrist cam
                "right_wrist_0_rgb": np.zeros_like(base_image),
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": np.True_ if self.model_type == _model.ModelType.PI0_FAST else np.False_,
            },
        }

        # Handle actions (only available during training)
        if "actions" in data:
            inputs["actions"] = data["actions"]

        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        if "observation/depth" in data:
            inputs["depth"] = data["observation/depth"]

        if "observation/flow_3d" in data:
            inputs["flow_3d"] = data["observation/flow_3d"]
        if "observation/flow_valid_mask" in data:
            inputs["flow_valid_mask"] = data["observation/flow_valid_mask"]

        return inputs


@dataclasses.dataclass(frozen=True)
class RobocasaOutputs(transforms.DataTransformFn):
    """
    Transforms model outputs back to Robocasa format.
    """

    # Robocasa action dim (e.g. 12 for mobile manipulation)
    action_dim: int = 12

    def __call__(self, data: dict) -> dict:
        # Slice the actions to the correct dimension
        return {"actions": data["actions"][:, : self.action_dim]}
