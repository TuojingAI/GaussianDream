import abc
from collections.abc import Sequence
import dataclasses
import enum
import logging
import pathlib
from typing import Generic, TypeVar

import augmax
from flax import nnx
from flax import struct
from flax import traverse_util
import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp
import safetensors
import torch

from gaussiandream.models_pytorch import policy
from gaussiandream.shared import image_tools
import gaussiandream.shared.array_typing as at

logger = logging.getLogger("gaussiandream")

# Type variable for array types (JAX arrays, PyTorch tensors, or numpy arrays)
ArrayT = TypeVar("ArrayT", bound=jax.Array | torch.Tensor | np.ndarray)


class ModelType(enum.Enum):
    """Supported model types."""

    PI0 = "pi0"
    PI0_FAST = "pi0_fast"
    PI05 = "pi05"


# The model always expects these images
IMAGE_KEYS = (
    "base_0_rgb",
    "left_wrist_0_rgb",
    "right_wrist_0_rgb",
)


# This may need change if we release a small model.
IMAGE_RESOLUTION = (224, 224)


# Data format
#
# Data transforms produce the model input as a nested dictionary which is later converted
# into `Obesrvation` and `Actions` objects. See below.
#
# In the dictory form, this data should look like:
# {
#     # Observation data.
#     "image": {
#         "base_0_rgb": (float32|uint8)[*b, h, w, 3],  # RGB image in [-1, 1] or [0, 255]
#         ...  # Additional camera views
#     },
#     "image_mask": {
#         "base_0_rgb": bool[*b],  # True if image is valid
#         ...  # Masks for additional views
#     },
#     "state": float32[*b, s],  # Low-dimensional robot state
#     "tokenized_prompt": int32[*b, l],  # Optional, tokenized language prompt
#     "tokenized_prompt_mask": bool[*b, l],  # Optional, mask for tokenized prompt
#     "token_ar_mask": int32[*b, l],  # Optional, autoregressive mask for FAST model
#     "token_loss_mask": bool[*b, l],  # Optional, loss mask for FAST model
#
#      # Actions data.
#      "actions": float32[*b ah ad]
# }
# where:
#   *b = batch dimensions
#   h,w = image height/width
#   s = state dimension
#   l = sequence length
#
@at.typecheck
@struct.dataclass
class Observation(Generic[ArrayT]):
    """Holds observations, i.e., inputs to the model.

    See `Observation.from_dict` to see the expected dictionary form. This is the format
    that should be produced by the data transforms.
    """

    # Images, in [-1, 1] float32.
    images: dict[str, at.Float[ArrayT, "*b h w c"]]
    # Image masks, with same keys as images.
    image_masks: dict[str, at.Bool[ArrayT, "*b"]]
    # Low-dimensional robot state.
    state: at.Float[ArrayT, "*b s"]

    # Depth maps (optional, for depth supervision)
    # Use different dimension names (dh, dw) to avoid conflict with image dimensions (h, w)
    depth: at.Float[ArrayT, "*b 1 dh dw"] | None = None
    # Optional 3D flow supervision aligned with future horizons.
    # Use independent leading dims here because flow uses future-horizon packing,
    # which does not have to match the image temporal context dimensions.
    flow_3d: at.Float[ArrayT, "... fh fw 3"] | None = None
    flow_valid_mask: at.Bool[ArrayT, "... fh fw"] | None = None

    # Tokenized prompt.
    tokenized_prompt: at.Int[ArrayT, "*b l"] | None = None
    # Tokenized prompt mask.
    tokenized_prompt_mask: at.Bool[ArrayT, "*b l"] | None = None

    # pi0-fast model specific fields.
    # Token auto-regressive mask (for FAST autoregressive model).
    token_ar_mask: at.Int[ArrayT, "*b l"] | None = None
    # Token loss mask (for FAST autoregressive model).
    token_loss_mask: at.Bool[ArrayT, "*b l"] | None = None

    @classmethod
    def from_dict(cls, data: at.PyTree[ArrayT]) -> "Observation[ArrayT]":
        """This method defines the mapping between unstructured data (i.e., nested dict) to the structured Observation format."""
        # Ensure that tokenized_prompt and tokenized_prompt_mask are provided together.
        if ("tokenized_prompt" in data) != ("tokenized_prompt_mask" in data):
            raise ValueError("tokenized_prompt and tokenized_prompt_mask must be provided together.")

        # Determine batch dimensions from images if possible
        # We need this to check against tokenized_prompt and expand if necessary
        # Assuming all images have same batch shape
        batch_dims_from_images = None
        first_img_key = next(iter(data["image"]), None)

        # First pass: Process Images to fix dimensions
        for key in data["image"]:
            image = data["image"][key]

            if isinstance(image, torch.Tensor):
                # Convert uint8 to float32
                if image.dtype == torch.uint8:
                    image = image.to(torch.float32) / 255.0 * 2.0 - 1.0

                # PyTorch uses [B, C, H, W] but Observation expects [B, H, W, C]
                # Detect if tensor is in channels-first format and convert to channels-last
                if image.ndim == 5:
                    if image.shape[2] == 3:
                        # [B, T, C, H, W] -> [B, T, H, W, C]
                        image = image.permute(0, 1, 3, 4, 2)
                    # Else assume already [B, T, H, W, C]
                elif image.ndim == 4:
                    if image.shape[1] == 3:
                        # [B, C, H, W] -> [B, H, W, C]
                        image = image.permute(0, 2, 3, 1)
                    # Else assume already [B, H, W, C]
            elif hasattr(image, "dtype") and image.dtype == np.uint8:
                image = image.astype(np.float32) / 255.0 * 2.0 - 1.0

            data["image"][key] = image

            # Capture batch shape from the first processed image array.
            # Excluding H, W, C, so take shape[:-3].
            if batch_dims_from_images is None and hasattr(image, "shape") and len(image.shape) >= 3:
                batch_dims_from_images = image.shape[:-3]

        # Ensure image_masks are proper boolean tensors and match image dimensions
        for key in data["image_mask"]:
            mask = data["image_mask"][key]

            # Handle PyTorch tensors
            if isinstance(mask, torch.Tensor):
                # Convert to bool if needed
                if mask.dtype != torch.bool:
                    mask = mask.to(torch.bool)

                # If images have temporal dimension [B, T, H, W, C] but masks are only [B],
                # expand masks to [B, T] to match the batch dimensions
                if key in data["image"]:
                    img = data["image"][key]
                    if isinstance(img, torch.Tensor) and img.ndim == 5 and mask.ndim == 1:
                        # Image is [B, T, H, W, C], mask is [B] -> expand to [B, T]
                        T = img.shape[1]
                        mask = mask.unsqueeze(1).expand(-1, T)

            # Handle JAX/numpy arrays
            elif hasattr(mask, "ndim") and key in data["image"]:
                img = data["image"][key]
                if hasattr(img, "ndim") and img.ndim == 5 and mask.ndim == 1:
                    # Image is [B, T, H, W, C], mask is [B] -> expand to [B, T]
                    T = img.shape[1]
                    if isinstance(mask, np.ndarray):
                        mask = np.broadcast_to(mask[:, None], (mask.shape[0], T))
                    else:  # JAX array
                        mask = jnp.broadcast_to(mask[:, None], (mask.shape[0], T))

            data["image_mask"][key] = mask

        # Fix tokenized_prompt batch dimensions if mismatched with images (e.g. strict time expansion)
        # Jaxtyping complains that images have shape [16, 2, H, W, C] (*b = [16, 2])
        # but prompt has shape [16, L] (*b = [16]), missing the time dimension [2].
        # Prompts are static across the time horizon, so we should expand them.

        if batch_dims_from_images is not None and len(batch_dims_from_images) > 1:
            # We have a time dimension in images: batch_dims_from_images is likely (B, T)
            # Check prompt
            prompt = data.get("tokenized_prompt")
            if prompt is not None:
                # If prompt is [B, L], but we need [B, T, L]
                T = batch_dims_from_images[1]
                if isinstance(prompt, torch.Tensor):
                    if prompt.shape[0] == batch_dims_from_images[0] and prompt.ndim == 2:
                        data["tokenized_prompt"] = prompt.unsqueeze(1).expand(-1, T, -1)
                elif hasattr(prompt, "ndim") and prompt.shape[0] == batch_dims_from_images[0] and prompt.ndim == 2:
                    if isinstance(prompt, np.ndarray):
                        data["tokenized_prompt"] = np.broadcast_to(
                            prompt[:, None, :], (prompt.shape[0], T, prompt.shape[1])
                        )
                    else:
                        data["tokenized_prompt"] = jnp.broadcast_to(
                            prompt[:, None, :], (prompt.shape[0], T, prompt.shape[1])
                        )

            prompt_mask = data.get("tokenized_prompt_mask")
            if prompt_mask is not None:
                T = batch_dims_from_images[1]
                if isinstance(prompt_mask, torch.Tensor):
                    if prompt_mask.shape[0] == batch_dims_from_images[0] and prompt_mask.ndim == 2:
                        data["tokenized_prompt_mask"] = prompt_mask.unsqueeze(1).expand(-1, T, -1)
                elif (
                    hasattr(prompt_mask, "ndim")
                    and prompt_mask.shape[0] == batch_dims_from_images[0]
                    and prompt_mask.ndim == 2
                ):
                    if isinstance(prompt_mask, np.ndarray):
                        data["tokenized_prompt_mask"] = np.broadcast_to(
                            prompt_mask[:, None, :], (prompt_mask.shape[0], T, prompt_mask.shape[1])
                        )
                    else:
                        data["tokenized_prompt_mask"] = jnp.broadcast_to(
                            prompt_mask[:, None, :], (prompt_mask.shape[0], T, prompt_mask.shape[1])
                        )

        # Handle state with temporal dimension and dtype
        # If images have temporal dimension [B, T_img, ...], state might have different temporal dimension [B, T_state, s]
        # Type annotation expects [*b, s] where *b should match between images and state
        # We need to ensure state's temporal dimension matches images' temporal dimension
        # Also convert to float32 if needed (type annotation expects Float, not f64)
        state = data["state"]

        # Convert dtype to float32 if needed
        if isinstance(state, torch.Tensor):
            if state.dtype != torch.float32:
                state = state.to(torch.float32)
        elif isinstance(state, np.ndarray):
            if state.dtype != np.float32:
                state = state.astype(np.float32)
        # JAX arrays are typically float32 by default

        if batch_dims_from_images is not None and len(batch_dims_from_images) > 1:
            # Images have temporal dimension [B, T_img, ...]
            T_img = batch_dims_from_images[1]
            if isinstance(state, torch.Tensor):
                if state.ndim == 2 and state.shape[0] == batch_dims_from_images[0]:
                    # State is [B, s] but images are [B, T_img, ...], expand state to [B, T_img, s]
                    state = state.unsqueeze(1).expand(-1, T_img, -1)
                    data["state"] = state
                elif state.ndim == 3 and state.shape[0] == batch_dims_from_images[0]:
                    # State is [B, T_state, s] but images are [B, T_img, ...]
                    # If T_state != T_img, we need to align them
                    T_state = state.shape[1]
                    if T_state != T_img:
                        # State has fewer frames (e.g., [t, t+1] = 2 frames) but images have more (e.g., 6 frames)
                        # We'll repeat the last frame to match images' temporal dimension
                        # This is needed for World Model which only needs [t, t+1]
                        if T_state < T_img:
                            # Repeat the last frame: [B, T_state, s] -> [B, T_img, s]
                            last_frame = state[:, -1:, :]  # [B, 1, s]
                            padding = last_frame.repeat(1, T_img - T_state, 1)  # [B, T_img - T_state, s]
                            state = torch.cat([state, padding], dim=1)  # [B, T_img, s]
                            data["state"] = state
                        else:
                            # State has more frames, take first T_img frames
                            state = state[:, :T_img, :]
                            data["state"] = state
            elif hasattr(state, "ndim"):
                # Handle JAX/numpy arrays
                if state.ndim == 2 and state.shape[0] == batch_dims_from_images[0]:
                    # State is [B, s], expand to [B, T_img, s]
                    if isinstance(state, np.ndarray):
                        state = np.broadcast_to(state[:, None, :], (state.shape[0], T_img, state.shape[1]))
                    else:  # JAX array
                        state = jnp.broadcast_to(state[:, None, :], (state.shape[0], T_img, state.shape[1]))
                    data["state"] = state
                elif state.ndim == 3 and state.shape[0] == batch_dims_from_images[0]:
                    T_state = state.shape[1]
                    if T_state != T_img:
                        if T_state < T_img:
                            # Repeat last frame
                            last_frame = state[:, -1:, :]
                            if isinstance(state, np.ndarray):
                                padding = np.repeat(last_frame, T_img - T_state, axis=1)
                            else:  # JAX array
                                padding = jnp.repeat(last_frame, T_img - T_state, axis=1)
                            state = np.concatenate([state, padding], axis=1) if isinstance(state, np.ndarray) else jnp.concatenate([state, padding], axis=1)
                            data["state"] = state
                        else:
                            # Take first T_img frames
                            state = state[:, :T_img, :]
                            data["state"] = state

        return cls(
            images=data["image"],
            image_masks=data["image_mask"],
            state=data["state"],
            depth=data.get("depth"),  # Optional depth data
            flow_3d=data.get("flow_3d"),
            flow_valid_mask=data.get("flow_valid_mask"),
            tokenized_prompt=data.get("tokenized_prompt"),
            tokenized_prompt_mask=data.get("tokenized_prompt_mask"),
            token_ar_mask=data.get("token_ar_mask"),
            token_loss_mask=data.get("token_loss_mask"),
        )

    def to_dict(self) -> at.PyTree[ArrayT]:
        """Convert the Observation to a nested dict."""
        result = dataclasses.asdict(self)
        result["image"] = result.pop("images")
        result["image_mask"] = result.pop("image_masks")
        return result


# Defines the format of the actions. This field is included as "actions" inside the dictionary
# produced by the data transforms.
Actions = at.Float[ArrayT, "*b ah ad"]


def preprocess_observation(
    rng: at.KeyArrayLike | None,
    observation: Observation,
    *,
    train: bool = False,
    image_keys: Sequence[str] = IMAGE_KEYS,
    image_resolution: tuple[int, int] = IMAGE_RESOLUTION,
) -> Observation:
    """Preprocess the observations by performing image augmentations (if train=True), resizing (if necessary), and
    filling in a default image mask (if necessary).
    """

    if not set(image_keys).issubset(observation.images):
        raise ValueError(f"images dict missing keys: expected {image_keys}, got {list(observation.images)}")

    batch_shape = observation.state.shape[:-1]

    out_images = {}
    for key in image_keys:
        image = observation.images[key]
        if image.shape[1:3] != image_resolution:
            logger.info(f"Resizing image {key} from {image.shape[1:3]} to {image_resolution}")
            image = image_tools.resize_with_pad(image, *image_resolution)

        if train:
            # Convert from [-1, 1] to [0, 1] for augmax.
            image = image / 2.0 + 0.5

            transforms = []
            if "wrist" not in key:
                height, width = image.shape[1:3]
                transforms += [
                    augmax.RandomCrop(int(width * 0.95), int(height * 0.95)),
                    augmax.Resize(width, height),
                    augmax.Rotate((-5, 5)),
                ]
            transforms += [
                augmax.ColorJitter(brightness=0.3, contrast=0.4, saturation=0.5),
            ]
            sub_rngs = jax.random.split(rng, image.shape[0])
            image = jax.vmap(augmax.Chain(*transforms))(sub_rngs, image)

            # Back to [-1, 1].
            image = image * 2.0 - 1.0

        out_images[key] = image

    # obtain mask
    out_masks = {}
    for key in out_images:
        if key not in observation.image_masks:
            # do not mask by default
            out_masks[key] = jnp.ones(batch_shape, dtype=jnp.bool)
        else:
            out_masks[key] = jnp.asarray(observation.image_masks[key])

    return Observation(
        images=out_images,
        image_masks=out_masks,
        state=observation.state,
        depth=observation.depth,
        flow_3d=observation.flow_3d,
        flow_valid_mask=observation.flow_valid_mask,
        tokenized_prompt=observation.tokenized_prompt,
        tokenized_prompt_mask=observation.tokenized_prompt_mask,
        token_ar_mask=observation.token_ar_mask,
        token_loss_mask=observation.token_loss_mask,
    )


@dataclasses.dataclass(frozen=True)
class BaseModelConfig(abc.ABC):
    """Configuration shared by all models. Specific models should inherit from this class, and implement the `create`
    method to create the corresponding model.
    """

    # Action space dimension.
    action_dim: int
    # Action sequence length.
    action_horizon: int
    # Tokenized prompt maximum length.
    max_token_len: int

    @property
    @abc.abstractmethod
    def model_type(self) -> ModelType:
        """The model type."""

    @abc.abstractmethod
    def create(self, rng: at.KeyArrayLike) -> "BaseModel":
        """Create a new model, initializing parameters."""

    def load(self, params: at.Params, *, remove_extra_params: bool = True) -> "BaseModel":
        """Create a model with the given parameters."""
        model = nnx.eval_shape(self.create, jax.random.key(0))
        graphdef, state = nnx.split(model)
        if remove_extra_params:
            params = ocp.transform_utils.intersect_trees(state.to_pure_dict(), params)
        at.check_pytree_equality(expected=state.to_pure_dict(), got=params, check_shapes=True, check_dtypes=False)
        state.replace_by_pure_dict(params)
        return nnx.merge(graphdef, state)

    def load_pytorch(self, train_config, weight_path: str):
        logger.info(f"train_config: {train_config}")
        model = policy.PolicyModel(config=train_config.model)
        missing, unexpected = safetensors.torch.load_model(model, weight_path, strict=False)
        
        if missing:
            logger.warning(f"Missing keys during loading: {missing}")
        if unexpected:
            # Filter out num_batches_tracked which are often mismatched but benign
            unexpected_real = [k for k in unexpected if "num_batches_tracked" not in k]
            if unexpected_real:
                logger.warning(f"Unexpected keys during loading: {unexpected_real}")
        return model

    @abc.abstractmethod
    def inputs_spec(self, *, batch_size: int = 1) -> tuple[Observation, Actions]:
        """Returns the input specification for the model. Values are jax.ShapeDtypeStruct."""

    def fake_obs(self, batch_size: int = 1) -> Observation:
        observation_spec, _ = self.inputs_spec(batch_size=batch_size)
        return jax.tree.map(lambda x: jnp.ones(x.shape, x.dtype), observation_spec)

    def fake_act(self, batch_size: int = 1) -> Actions:
        _, action_spec = self.inputs_spec(batch_size=batch_size)
        return jax.tree.map(lambda x: jnp.ones(x.shape, x.dtype), action_spec)


@dataclasses.dataclass
class BaseModel(nnx.Module, abc.ABC):
    """Base class for all model implementations. Specific models should inherit from this class. They should call
    super().__init__() to initialize the shared attributes (action_dim, action_horizon, and max_token_len).
    """

    action_dim: int
    action_horizon: int
    max_token_len: int

    @abc.abstractmethod
    def compute_loss(
        self,
        rng: at.KeyArrayLike,
        observation: Observation,
        actions: Actions,
        *,
        train: bool = False,
    ) -> at.Float[at.Array, "*b ah"]: ...

    @abc.abstractmethod
    def sample_actions(self, rng: at.KeyArrayLike, observation: Observation, **kwargs) -> Actions: ...


def restore_params(
    params_path: pathlib.Path | str,
    *,
    restore_type: type[np.ndarray] | type[jax.Array] = jax.Array,
    dtype: jnp.dtype | None = None,
    sharding: jax.sharding.Sharding | None = None,
) -> at.Params:
    """Restores unstructured params PyTree from a checkpoint.

    This works with checkpoints saved with `save_state` during GaussianDream training (see `training/checkpoints.py`) as
    well as pre-trained checkpoints released through the upstream OpenPI assets.

    Args:
        params_path: The local path to the checkpoint directory.
        restore_type: The type to restore the params as. Can be set to `np.ndarray` to load the params as a numpy array.
        dtype: The dtype to restore all params as. If not provided, will use the original dtype from the checkpoint.
        sharding: The sharding to use for the params. If not provided, the params will be replicated across all devices.

    Returns:
        The restored params.
    """
    params_path = pathlib.Path(params_path).resolve() if not str(params_path).startswith("gs://") else params_path

    if restore_type is jax.Array and sharding is None:
        mesh = jax.sharding.Mesh(jax.devices(), ("x",))
        sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    with ocp.PyTreeCheckpointer() as ckptr:
        metadata = ckptr.metadata(params_path)
        item = {"params": metadata["params"]}

        params = ckptr.restore(
            params_path,
            ocp.args.PyTreeRestore(
                item=item,
                restore_args=jax.tree.map(
                    lambda _: ocp.ArrayRestoreArgs(sharding=sharding, restore_type=restore_type, dtype=dtype), item
                ),
            ),
        )["params"]

    # If the params were saved with `save_state` during GaussianDream training, every key path will end with "value", which is
    # added by `nnx.State`. We remove the "value" suffix here and always return what NNX calls a "pure dict".
    flat_params = traverse_util.flatten_dict(params)
    if all(kp[-1] == "value" for kp in flat_params):
        flat_params = {kp[:-1]: v for kp, v in flat_params.items()}
    return traverse_util.unflatten_dict(flat_params)
