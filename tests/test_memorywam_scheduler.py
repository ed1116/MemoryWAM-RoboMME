from __future__ import annotations

import pytest
import torch

from fastwam.models.wan22.schedulers.scheduler_continuous import (
    WanContinuousFlowMatchScheduler,
)


def test_uniform_sampling_remains_the_fastwam_default():
    scheduler = WanContinuousFlowMatchScheduler(
        num_train_timesteps=1000,
        shift=5.0,
    )
    actual_generator = torch.Generator().manual_seed(123)
    expected_generator = torch.Generator().manual_seed(123)

    actual = scheduler.sample_training_t(
        8,
        device=torch.device("cpu"),
        dtype=torch.float64,
        generator=actual_generator,
    )
    uniform = torch.rand(8, generator=expected_generator, dtype=torch.float32)
    expected = (5.0 * uniform / (1.0 + 4.0 * uniform)).mul(1000.0).to(torch.float64)

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_shifted_logit_normal_sampling_matches_seeded_reference():
    scheduler = WanContinuousFlowMatchScheduler(
        num_train_timesteps=1000,
        shift=5.0,
    )
    scheduler.use_shifted_logit_normal(mu=0.25, sigma=0.75)
    actual_generator = torch.Generator().manual_seed(321)
    expected_generator = torch.Generator().manual_seed(321)

    actual = scheduler.sample_training_t(
        8,
        device=torch.device("cpu"),
        dtype=torch.float32,
        generator=actual_generator,
    )
    normal = torch.randn(8, generator=expected_generator, dtype=torch.float32)
    base = torch.sigmoid(normal.mul(0.75).add(0.25))
    expected = (5.0 * base / (1.0 + 4.0 * base)).mul(1000.0)

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"logit_normal_mu": 0.0},
        {"logit_normal_sigma": 1.0},
        {"logit_normal_mu": 0.0, "logit_normal_sigma": 0.0},
        {"logit_normal_mu": float("inf"), "logit_normal_sigma": 1.0},
    ],
)
def test_shifted_logit_normal_parameters_are_validated(kwargs):
    with pytest.raises(ValueError):
        WanContinuousFlowMatchScheduler(**kwargs)
