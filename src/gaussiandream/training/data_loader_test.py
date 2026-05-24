import dataclasses
from unittest import mock

import jax

from gaussiandream.models import pi0_config
from gaussiandream.training import config as _config
from gaussiandream.training import data_loader as _data_loader


def test_create_lerobot_dataset_aligns_state_timestamps_with_image_context():
    config = pi0_config.Pi0Config(
        pi05=True,
        use_gaussian=True,
        use_world_model=True,
        use_single_frame_mode=False,
        temporal_context_offsets=(-10, -5, 0),
        future_prediction_offsets=(1, 2, 3, 4, 5),
    )
    data_config = _config.DataConfig(repo_id="fake_libero", dataset_root="/tmp/fake")

    class _FakeMeta:
        fps = 10.0
        features = {
            "observation.images.agentview_rgb": object(),
            "observation.state": object(),
        }
        tasks = {}

    captured = {}

    def _fake_dataset(repo_id, root=None, delta_timestamps=None):
        captured["repo_id"] = repo_id
        captured["root"] = root
        captured["delta_timestamps"] = delta_timestamps
        return object()

    with (
        mock.patch.object(_data_loader.lerobot_dataset, "LeRobotDatasetMetadata", return_value=_FakeMeta()),
        mock.patch.object(_data_loader.lerobot_dataset, "LeRobotDataset", side_effect=_fake_dataset),
    ):
        _data_loader.create_lerobot_dataset(
            data_config,
            config,
            action_horizon=10,
            use_single_frame_mode=False,
        )

    assert captured["delta_timestamps"]["observation.images.agentview_rgb"] == [-1.0, -0.5, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    assert captured["delta_timestamps"]["observation.state"] == [-1.0, -0.5, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5]


def test_torch_data_loader():
    config = pi0_config.Pi0Config(action_dim=24, action_horizon=50, max_token_len=48)
    dataset = _data_loader.FakeDataset(config, 16)

    loader = _data_loader.TorchDataLoader(
        dataset,
        local_batch_size=4,
        num_batches=2,
    )
    batches = list(loader)

    assert len(batches) == 2
    for batch in batches:
        assert all(x.shape[0] == 4 for x in jax.tree.leaves(batch))


def test_torch_data_loader_infinite():
    config = pi0_config.Pi0Config(action_dim=24, action_horizon=50, max_token_len=48)
    dataset = _data_loader.FakeDataset(config, 4)

    loader = _data_loader.TorchDataLoader(dataset, local_batch_size=4)
    data_iter = iter(loader)

    for _ in range(10):
        _ = next(data_iter)


def test_torch_data_loader_parallel():
    config = pi0_config.Pi0Config(action_dim=24, action_horizon=50, max_token_len=48)
    dataset = _data_loader.FakeDataset(config, 10)

    loader = _data_loader.TorchDataLoader(dataset, local_batch_size=4, num_batches=2, num_workers=2)
    batches = list(loader)

    assert len(batches) == 2

    for batch in batches:
        assert all(x.shape[0] == 4 for x in jax.tree.leaves(batch))


def test_with_fake_dataset():
    config = _config.get_config("debug")

    loader = _data_loader.create_data_loader(config, skip_norm_stats=True, num_batches=2)
    batches = list(loader)

    assert len(batches) == 2

    for batch in batches:
        assert all(x.shape[0] == config.batch_size for x in jax.tree.leaves(batch))

    for _, actions in batches:
        assert actions.shape == (config.batch_size, config.model.action_horizon, config.model.action_dim)


def test_with_real_dataset():
    config = _config.get_config("pi0_aloha_sim")
    config = dataclasses.replace(config, batch_size=4)

    loader = _data_loader.create_data_loader(
        config,
        # Skip since we may not have the data available.
        skip_norm_stats=True,
        num_batches=2,
        shuffle=True,
    )
    # Make sure that we can get the data config.
    assert loader.data_config().repo_id == config.data.repo_id

    batches = list(loader)

    assert len(batches) == 2

    for _, actions in batches:
        assert actions.shape == (config.batch_size, config.model.action_horizon, config.model.action_dim)
