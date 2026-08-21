from __future__ import annotations

import re
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from fastwam.models.wan22.fastwam import FastWAM
from fastwam.models.wan22.helpers.io import ModelConfig
from fastwam.robomme import (
    RoboMMEAbsoluteNormalizer,
    RoboMMEImagePreprocessor,
    RoboMMEModelIO,
)
from fastwam.runtime import create_robomme_fastwam_idm


def _normalizer():
    return RoboMMEAbsoluteNormalizer(
        state_mean=np.arange(8, dtype=np.float32),
        state_std=np.arange(1, 9, dtype=np.float32),
        action_mean=np.arange(10, 18, dtype=np.float32),
        action_std=np.arange(2, 10, dtype=np.float32),
    )


def _views(frames: int | None = None):
    prefix = () if frames is None else (frames,)
    front = np.zeros((*prefix, 19, 31, 3), dtype=np.uint8)
    wrist = np.full((*prefix, 23, 17, 3), 255, dtype=np.uint8)
    return front, wrist


def test_mosaic_has_fixed_shape_order_and_range():
    front, wrist = _views()
    mosaic = RoboMMEImagePreprocessor()(front_rgb=front, wrist_rgb=wrist)

    assert mosaic.shape == (3, 224, 448)
    assert mosaic.dtype == torch.float32
    torch.testing.assert_close(mosaic[:, :, :224], torch.full((3, 224, 224), -1.0))
    torch.testing.assert_close(mosaic[:, :, 224:], torch.full((3, 224, 224), 1.0))
    assert float(mosaic.min()) == -1.0
    assert float(mosaic.max()) == 1.0
    with pytest.raises(ValueError, match="camera order"):
        RoboMMEImagePreprocessor(camera_order=("wrist", "front"))


def test_history_mosaic_and_training_shapes_are_fixed():
    front, wrist = _views(frames=5)
    io = RoboMMEModelIO(backend=object(), normalizer=_normalizer())
    sample = io.prepare_training_tensors(
        front_rgb=front,
        wrist_rgb=wrist,
        state=np.zeros((5, 8), dtype=np.float32),
        action=np.zeros((16, 8), dtype=np.float32),
        task_goal="move the cube",
    )

    assert sample["video"].shape == (3, 5, 224, 448)
    assert sample["proprio"].shape == (5, 8)
    assert sample["action"].shape == (16, 8)
    assert sample["image_is_pad"].shape == (5,)
    assert sample["action_is_pad"].shape == (16,)
    assert sample["prompt"] == "move the cube"


def test_absolute_state_and_action_normalization_round_trip():
    normalizer = _normalizer()
    state = torch.arange(40, dtype=torch.float32).reshape(5, 8)
    action = torch.arange(128, dtype=torch.float32).reshape(16, 8)

    torch.testing.assert_close(
        normalizer.denormalize_state(normalizer.normalize_state(state)), state
    )
    torch.testing.assert_close(
        normalizer.denormalize_action(normalizer.normalize_action(action)), action
    )


class _ActionBackendStub:
    def __init__(self, action=None):
        self.kwargs = None
        self.action = torch.zeros(16, 8) if action is None else action

    def infer_action(self, **kwargs):
        self.kwargs = kwargs
        return {"action": self.action}


def test_action_backend_receives_fixed_contract_and_returns_16_by_8_absolute_actions():
    backend = _ActionBackendStub()
    io = RoboMMEModelIO(backend=backend, normalizer=_normalizer())
    front, wrist = _views()

    action = io.predict_actions(
        front_rgb=front,
        wrist_rgb=wrist,
        state=np.arange(8, dtype=np.float32),
        task_goal="pick the highlighted object",
        num_inference_steps=2,
    )

    assert action.shape == (16, 8)
    assert action.dtype == np.float32
    expected = np.repeat(np.arange(10, 18, dtype=np.float32)[None, :], 16, axis=0)
    expected[:, -1] = 1.0
    np.testing.assert_array_equal(action, expected)
    assert backend.kwargs["input_image"].shape == (3, 224, 448)
    assert backend.kwargs["proprio"].shape == (8,)
    assert backend.kwargs["action_horizon"] == 16
    assert backend.kwargs["num_video_frames"] == 5
    assert backend.kwargs["num_inference_steps"] == 2


def test_inference_binarizes_denormalized_gripper_commands():
    normalized = torch.zeros(16, 8)
    normalized[:8, -1] = -10.0
    backend = _ActionBackendStub(normalized)
    io = RoboMMEModelIO(backend=backend, normalizer=_normalizer())
    front, wrist = _views()

    action = io.predict_actions(
        front_rgb=front,
        wrist_rgb=wrist,
        state=np.zeros(8, dtype=np.float32),
        task_goal="move the cube",
    )

    np.testing.assert_array_equal(action[:8, -1], -1.0)
    np.testing.assert_array_equal(action[8:, -1], 1.0)


def test_vae_encoding_mode_uses_eager_by_default_for_robomme_and_can_compile(monkeypatch):
    class _Encoder:
        def encode(self, value, scale):
            return value * scale

    video = torch.ones(1, 3, 1, 2, 2)
    stub = SimpleNamespace(
        vae=SimpleNamespace(model=_Encoder(), scale=2.0),
        device=torch.device("cpu"),
        vae_encode_mode="eager",
    )
    monkeypatch.setattr(torch, "compile", lambda *args, **kwargs: pytest.fail("compiled eager VAE"))
    torch.testing.assert_close(FastWAM._encode_video_latents(stub, video), video * 2.0)

    compiled_calls = []

    def _compile(function, **kwargs):
        compiled_calls.append(kwargs)
        return function

    monkeypatch.setattr(torch, "compile", _compile)
    stub.vae_encode_mode = "compile"
    torch.testing.assert_close(FastWAM._encode_video_latents(stub, video), video * 2.0)
    assert compiled_calls == [{"backend": "cudagraphs", "fullgraph": True}]


def test_robomme_model_config_pins_huggingface_and_fixed_dimensions():
    config_path = Path(__file__).parents[1] / "configs/model/memorywam_robomme.yaml"
    config = OmegaConf.load(config_path)

    assert config._target_ == "fastwam.runtime.create_robomme_fastwam_idm"
    assert re.fullmatch(r"[0-9a-f]{40}", config.model_revision)
    assert re.fullmatch(r"[0-9a-f]{40}", config.tokenizer_revision)
    assert list(config.camera_order) == ["front", "wrist"]
    assert list(config.mosaic_size) == [224, 448]
    assert config.state_dim == config.action_dim == 8
    assert config.action_horizon == 16
    assert config.action_representation == "absolute"
    assert config.video_dit_config.action_dim == 8
    assert config.action_dit_config.action_dim == 8
    assert config.video_cond_noise_prob == 1.0
    assert config.vae_encode_mode == "eager"
    assert config.action_dit_pretrained_path is None
    assert config.model_revision == "921dbaf3f1674a56f47e83fb80a34bac8a8f203e"
    assert config.tokenizer_revision == "37ec512624d61f7aa208f7ea8140a131f93afc9a"


def test_scheduler_selection_is_deferred_to_memorywam_training_integration():
    config_path = Path(__file__).parents[1] / "configs/model/memorywam_robomme.yaml"
    config = OmegaConf.load(config_path)
    assert "logit_normal_mu" not in config.video_scheduler
    assert "logit_normal_sigma" not in config.video_scheduler
    assert "logit_normal_mu" not in config.action_scheduler
    assert "logit_normal_sigma" not in config.action_scheduler


def test_robomme_factory_forces_pinned_huggingface_path(monkeypatch):
    captured = {}
    stub = SimpleNamespace()

    def _create(**kwargs):
        captured.update(kwargs)
        return stub

    monkeypatch.setattr("fastwam.runtime.create_fastwam_idm", _create)
    create_robomme_fastwam_idm(
        model_id="Wan-AI/Wan2.2-TI2V-5B",
        model_revision="a" * 40,
        tokenizer_model_id="Wan-AI/Wan2.1-T2V-1.3B",
        tokenizer_revision="b" * 40,
        video_dit_config={"action_dim": 8},
        action_dit_config={"action_dim": 8},
        action_scheduler={
            "train_shift": 1.0,
            "infer_shift": 1.0,
            "num_train_timesteps": 1000,
        },
    )

    assert captured["huggingface_only"] is True
    assert captured["redirect_common_files"] is False
    assert captured["model_revision"] == "a" * 40
    assert captured["tokenizer_revision"] == "b" * 40
    assert captured["proprio_dim"] == 8
    assert captured["vae_encode_mode"] == "eager"
    assert captured["video_cond_noise_prob"] == 1.0
    assert captured["model_dtype"] is torch.float16
    assert stub.robomme_action_horizon == 16
    assert stub.robomme_action_representation == "absolute"


@pytest.mark.parametrize(
    ("override", "match"),
    [
        ({"video_cond_noise_prob": 0.5}, "video_cond_noise_prob=1.0"),
        ({"model_dtype": torch.bfloat16}, "requires FP16"),
        ({"model_dtype": torch.float32}, "requires FP16"),
    ],
)
def test_robomme_factory_rejects_nonpaper_noise_or_non_fp16(monkeypatch, override, match):
    monkeypatch.setattr(
        "fastwam.runtime.create_fastwam_idm",
        lambda **kwargs: pytest.fail("invalid configuration reached model construction"),
    )
    kwargs = {
        "model_id": "Wan-AI/Wan2.2-TI2V-5B",
        "model_revision": "a" * 40,
        "tokenizer_model_id": "Wan-AI/Wan2.1-T2V-1.3B",
        "tokenizer_revision": "b" * 40,
        "video_dit_config": {"action_dim": 8},
        "action_dit_config": {"action_dim": 8},
        "action_scheduler": {
            "train_shift": 1.0,
            "infer_shift": 1.0,
            "num_train_timesteps": 1000,
        },
        **override,
    }
    with pytest.raises(ValueError, match=match):
        create_robomme_fastwam_idm(**kwargs)


def test_pinned_huggingface_config_rejects_mutable_revision_before_download():
    config = ModelConfig(
        model_id="namespace/model",
        origin_file_pattern="weights.safetensors",
        revision="main",
        use_huggingface_cache=True,
        require_immutable_revision=True,
    )
    with pytest.raises(ValueError, match="immutable 40-character"):
        config.download_if_necessary()
