

import pytest
from utils.schedules import get_beta


@pytest.mark.parametrize("epoch,expected", [
    (1, 0.18),     # Start of schedule: 0.1 + 0.08
    (5, 0.5),      # Midpoint
    (10, 0.9),     # End of schedule
    (15, 0.9),     # Beyond annealing period
])
def test_linear_schedule(epoch, expected):
    beta = get_beta(
        epoch=epoch,
        schedule="linear",
        beta_start=0.1,
        beta_end=0.9,
        anneal_epochs=10
    )
    assert pytest.approx(beta, rel=1e-2) == expected


@pytest.mark.parametrize("epoch", [1, 5, 10, 15])
def test_sigmoid_schedule_values(epoch):
    beta = get_beta(
        epoch=epoch,
        schedule="sigmoid",
        beta_start=0.1,
        beta_end=0.9,
        anneal_epochs=10
    )
    assert 0.1 <= beta <= 0.9


def test_sigmoid_schedule_monotonic():
    betas = [
        get_beta(epoch=e, schedule="sigmoid", beta_start=0.0, beta_end=1.0, anneal_epochs=10)
        for e in range(1, 11)
    ]
    assert all(earlier <= later for earlier, later in zip(betas, betas[1:])), "Sigmoid beta is not monotonic"


def test_schedule_clamps_after_anneal():
    beta = get_beta(
        epoch=50,
        schedule="linear",
        beta_start=0.0,
        beta_end=1.0,
        anneal_epochs=10
    )
    assert beta == 1.0


def test_invalid_schedule():
    with pytest.raises(ValueError):
        get_beta(epoch=5, schedule="unknown", beta_start=0.0, beta_end=1.0, anneal_epochs=10)