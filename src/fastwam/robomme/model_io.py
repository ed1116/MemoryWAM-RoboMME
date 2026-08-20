"""Fixed RoboMME tensor contract around a FastWAM/IDM backend."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F


ROBOMME_CAMERA_ORDER = ("front", "wrist")
ROBOMME_VIEW_SIZE = (224, 224)
ROBOMME_MOSAIC_SIZE = (224, 448)
ROBOMME_STATE_DIM = 8
ROBOMME_ACTION_DIM = 8
ROBOMME_ACTION_HORIZON = 16
ROBOMME_VIDEO_FRAMES = 5


def _finite_vector(value: Sequence[float] | torch.Tensor, label: str) -> torch.Tensor:
    tensor = torch.as_tensor(value, dtype=torch.float32)
    if tensor.shape != (ROBOMME_ACTION_DIM,) or not bool(torch.isfinite(tensor).all()):
        raise ValueError(f"`{label}` must be finite with shape ({ROBOMME_ACTION_DIM},).")
    return tensor.clone()


class RoboMMEAbsoluteNormalizer:
    """Standardize absolute 8-D state/action values without delta conversion."""

    def __init__(self, *, state_mean, state_std, action_mean, action_std):
        self.state_mean = _finite_vector(state_mean, "state_mean")
        self.state_std = _finite_vector(state_std, "state_std")
        self.action_mean = _finite_vector(action_mean, "action_mean")
        self.action_std = _finite_vector(action_std, "action_std")
        if bool((self.state_std <= 0).any()) or bool((self.action_std <= 0).any()):
            raise ValueError("Normalization standard deviations must be positive.")

    @classmethod
    def from_statistics(cls, statistics: Any) -> "RoboMMEAbsoluteNormalizer":
        return cls(
            state_mean=statistics.state_mean,
            state_std=statistics.state_std,
            action_mean=statistics.action_mean,
            action_std=statistics.action_std,
        )

    @staticmethod
    def _validate_value(value, label: str) -> torch.Tensor:
        tensor = torch.as_tensor(value, dtype=torch.float32)
        if tensor.ndim == 0 or tensor.shape[-1] != ROBOMME_ACTION_DIM:
            raise ValueError(
                f"`{label}` must have final dimension {ROBOMME_ACTION_DIM}, got {tuple(tensor.shape)}."
            )
        if not bool(torch.isfinite(tensor).all()):
            raise ValueError(f"`{label}` must contain only finite values.")
        return tensor

    @staticmethod
    def _on_device(reference: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        return value.to(device=reference.device, dtype=reference.dtype)

    def normalize_state(self, value) -> torch.Tensor:
        tensor = self._validate_value(value, "state")
        return (tensor - self._on_device(tensor, self.state_mean)) / self._on_device(
            tensor, self.state_std
        )

    def denormalize_state(self, value) -> torch.Tensor:
        tensor = self._validate_value(value, "state")
        return tensor * self._on_device(tensor, self.state_std) + self._on_device(
            tensor, self.state_mean
        )

    def normalize_action(self, value) -> torch.Tensor:
        tensor = self._validate_value(value, "action")
        return (tensor - self._on_device(tensor, self.action_mean)) / self._on_device(
            tensor, self.action_std
        )

    def denormalize_action(self, value) -> torch.Tensor:
        tensor = self._validate_value(value, "action")
        return tensor * self._on_device(tensor, self.action_std) + self._on_device(
            tensor, self.action_mean
        )


class RoboMMEImagePreprocessor:
    """Resize front/wrist RGB and place front left, wrist right in [-1, 1]."""

    def __init__(self, camera_order: Sequence[str] = ROBOMME_CAMERA_ORDER):
        self.camera_order = tuple(camera_order)
        if self.camera_order != ROBOMME_CAMERA_ORDER:
            raise ValueError(
                f"RoboMME camera order must be {ROBOMME_CAMERA_ORDER}, got {self.camera_order}."
            )

    @staticmethod
    def _frame_batch(value, label: str) -> tuple[torch.Tensor, bool]:
        tensor = torch.as_tensor(value)
        is_single = tensor.ndim == 3
        if is_single:
            tensor = tensor.unsqueeze(0)
        if tensor.ndim != 4 or tensor.shape[-1] != 3 or tensor.dtype != torch.uint8:
            raise ValueError(f"`{label}` must be uint8 RGB with shape [H,W,3] or [T,H,W,3].")
        return tensor, is_single

    def __call__(self, *, front_rgb, wrist_rgb) -> torch.Tensor:
        frames = {}
        front, front_single = self._frame_batch(front_rgb, "front_rgb")
        wrist, wrist_single = self._frame_batch(wrist_rgb, "wrist_rgb")
        if front_single != wrist_single or front.shape[0] != wrist.shape[0]:
            raise ValueError("Front and wrist inputs must contain the same number of frames.")
        frames["front"] = front
        frames["wrist"] = wrist

        resized = []
        for camera in self.camera_order:
            view = frames[camera].permute(0, 3, 1, 2).to(dtype=torch.float32)
            view = F.interpolate(
                view,
                size=ROBOMME_VIEW_SIZE,
                mode="bilinear",
                align_corners=False,
            )
            resized.append(view)
        mosaic = torch.cat(resized, dim=-1).mul_(2.0 / 255.0).sub_(1.0).clamp_(-1.0, 1.0)
        if front_single:
            return mosaic[0]
        return mosaic.permute(1, 0, 2, 3).contiguous()


class RoboMMEModelIO:
    """Normalize inputs and validate a fixed 16x8 FastWAM action result."""

    def __init__(self, backend, normalizer: RoboMMEAbsoluteNormalizer):
        self.backend = backend
        self.normalizer = normalizer
        self.image_preprocessor = RoboMMEImagePreprocessor()

    def prepare_training_tensors(
        self,
        *,
        front_rgb,
        wrist_rgb,
        state,
        action,
        task_goal: str,
    ) -> dict[str, Any]:
        video = self.image_preprocessor(front_rgb=front_rgb, wrist_rgb=wrist_rgb)
        if video.shape != (3, ROBOMME_VIDEO_FRAMES, *ROBOMME_MOSAIC_SIZE):
            raise ValueError(
                "Training camera history must produce "
                f"[3,{ROBOMME_VIDEO_FRAMES},{ROBOMME_MOSAIC_SIZE[0]},{ROBOMME_MOSAIC_SIZE[1]}], "
                f"got {tuple(video.shape)}."
            )
        proprio = self.normalizer.normalize_state(state)
        normalized_action = self.normalizer.normalize_action(action)
        if proprio.shape != (ROBOMME_VIDEO_FRAMES, ROBOMME_STATE_DIM):
            raise ValueError(
                f"Training state must have shape ({ROBOMME_VIDEO_FRAMES}, {ROBOMME_STATE_DIM})."
            )
        if normalized_action.shape != (ROBOMME_ACTION_HORIZON, ROBOMME_ACTION_DIM):
            raise ValueError(
                f"Training action must have shape ({ROBOMME_ACTION_HORIZON}, {ROBOMME_ACTION_DIM})."
            )
        return {
            "video": video,
            "proprio": proprio,
            "action": normalized_action,
            "image_is_pad": torch.zeros(ROBOMME_VIDEO_FRAMES, dtype=torch.bool),
            "action_is_pad": torch.zeros(ROBOMME_ACTION_HORIZON, dtype=torch.bool),
            "prompt": str(task_goal),
        }

    @torch.no_grad()
    def predict_actions(
        self,
        *,
        front_rgb,
        wrist_rgb,
        state,
        task_goal: str,
        num_video_frames: int = ROBOMME_VIDEO_FRAMES,
        **inference_kwargs,
    ) -> np.ndarray:
        reserved = {"prompt", "input_image", "action_horizon", "num_video_frames", "proprio"}
        overlap = reserved.intersection(inference_kwargs)
        if overlap:
            raise ValueError(f"Fixed RoboMME inference arguments cannot be overridden: {sorted(overlap)}")
        input_image = self.image_preprocessor(front_rgb=front_rgb, wrist_rgb=wrist_rgb)
        if input_image.shape != (3, *ROBOMME_MOSAIC_SIZE):
            raise ValueError("RoboMME inference accepts exactly one front/wrist frame pair.")
        proprio = self.normalizer.normalize_state(state)
        if proprio.shape != (ROBOMME_STATE_DIM,):
            raise ValueError(f"Inference state must have shape ({ROBOMME_STATE_DIM},).")

        output: Mapping[str, Any] = self.backend.infer_action(
            prompt=str(task_goal),
            input_image=input_image,
            action_horizon=ROBOMME_ACTION_HORIZON,
            num_video_frames=int(num_video_frames),
            proprio=proprio,
            **inference_kwargs,
        )
        if not isinstance(output, Mapping) or "action" not in output:
            raise ValueError("FastWAM backend must return a mapping containing `action`.")
        normalized_action = torch.as_tensor(output["action"], dtype=torch.float32)
        if normalized_action.shape != (ROBOMME_ACTION_HORIZON, ROBOMME_ACTION_DIM):
            raise ValueError(
                "FastWAM action output must have shape "
                f"({ROBOMME_ACTION_HORIZON}, {ROBOMME_ACTION_DIM}), got {tuple(normalized_action.shape)}."
            )
        if not bool(torch.isfinite(normalized_action).all()):
            raise ValueError("FastWAM action output must contain only finite values.")
        action = self.normalizer.denormalize_action(normalized_action)
        action[..., -1] = torch.where(action[..., -1] >= 0.0, 1.0, -1.0)
        return action.cpu().numpy().astype(np.float32, copy=False)
