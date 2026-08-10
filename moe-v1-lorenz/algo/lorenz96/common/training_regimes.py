"""Shared observation-noise and model-drift schedules for Lorenz-96 trainers."""

import numpy as np
import torch

from common import config


def make_attractor_pool(
    step_fn,
    pool_size,
    state_dim,
    spinup_steps,
    forcing,
    dt,
    seed,
    device,
):
    """Generate immutable, independently initialized Lorenz-96 attractor states."""
    if pool_size <= 0 or state_dim <= 0 or spinup_steps <= 0:
        raise ValueError('pool_size, state_dim, and spinup_steps must be positive')

    rng = np.random.default_rng(seed)
    initial = rng.standard_normal((pool_size, state_dim)).astype(np.float32) * 0.5
    pool = torch.tensor(initial, device=device)
    print(
        f"[attractor-pool] building size={pool_size} dim={state_dim} "
        f"spinup_steps={spinup_steps} seed={seed}",
        flush=True,
    )
    report_every = max(1, spinup_steps // 10)
    with torch.no_grad():
        for step in range(spinup_steps):
            pool = step_fn(pool, dt, F=forcing)
            completed = step + 1
            if completed % report_every == 0 or completed == spinup_steps:
                print(
                    f"[attractor-pool] spinup {completed}/{spinup_steps}",
                    flush=True,
                )
    print("[attractor-pool] ready", flush=True)
    return pool.detach()


def sample_attractor_batch(pool, batch_size, seed):
    """Clone a deterministic sample of distinct attractor states."""
    if batch_size > len(pool):
        raise ValueError(f'batch_size={batch_size} exceeds attractor pool size={len(pool)}')
    indices = np.random.default_rng(seed).choice(len(pool), size=batch_size, replace=False)
    return pool[torch.as_tensor(indices, device=pool.device)].clone()


def select_attractor_state(pool, seed):
    """Select one deterministic attractor state for an evaluation seed."""
    index = int(np.random.default_rng(seed).integers(0, len(pool)))
    return pool[index:index + 1].clone()


def make_segmented_schedule(
    n_steps,
    obs_gap,
    bptt_window,
    rng,
    low,
    high,
    difficulty,
    segment_min_cycles=10,
    segment_max_cycles=20,
):
    """Create a random piecewise-constant schedule in simulator-step units."""
    if not 0.0 <= difficulty <= 10.0:
        raise ValueError(f'difficulty must be in [0, 10], got {difficulty}')
    if low > high:
        raise ValueError(f'low={low} must not exceed high={high}')
    if obs_gap <= 0 or bptt_window <= 0:
        raise ValueError('obs_gap and bptt_window must be positive')
    if not 1 <= segment_min_cycles <= segment_max_cycles:
        raise ValueError('invalid segment-cycle bounds')
    mean_segment_cycles = 0.5 * (segment_min_cycles + segment_max_cycles)
    if mean_segment_cycles > bptt_window:
        raise ValueError('mean segment length must not exceed the BPTT window')

    effective_high = low + (difficulty / 10.0) * (high - low)
    schedule = np.empty(n_steps, dtype=np.float32)
    start = 0
    while start < n_steps:
        segment_cycles = int(rng.integers(segment_min_cycles, segment_max_cycles + 1))
        segment_steps = segment_cycles * obs_gap
        value = rng.uniform(low, effective_high) if effective_high > low else low
        schedule[start:start + segment_steps] = value
        start += segment_steps
    return schedule


def make_training_schedules(
    batch_size,
    n_steps,
    difficulty,
    seed,
    epoch,
    bptt_window=config.BPTT_WINDOW,
):
    """Build independent observation-noise and forecast-state schedules.

    ``difficulty`` may be one scalar or one value per batch trajectory.
    """
    if batch_size <= 0 or n_steps <= 0:
        raise ValueError('batch_size and n_steps must be positive')
    difficulties = np.asarray(difficulty, dtype=np.float32)
    if difficulties.ndim == 0:
        difficulties = np.full(batch_size, float(difficulties), dtype=np.float32)
    if difficulties.shape != (batch_size,):
        raise ValueError(
            f'difficulty must be scalar or shape ({batch_size},), got {difficulties.shape}'
        )

    noise_schedules = []
    state_schedules = []
    for batch_index in range(batch_size):
        trajectory_difficulty = float(difficulties[batch_index])
        noise_rng = np.random.default_rng(
            np.random.SeedSequence([seed, epoch, batch_index, 0])
        )
        state_rng = np.random.default_rng(
            np.random.SeedSequence([seed, epoch, batch_index, 1])
        )
        noise_schedules.append(
            make_segmented_schedule(
                n_steps,
                config.OBS_GAP,
                bptt_window,
                noise_rng,
                config.NOISE_STD_LOW,
                config.NOISE_STD_HIGH,
                trajectory_difficulty,
                config.SEGMENT_MIN_CYCLES,
                config.SEGMENT_MAX_CYCLES,
            )
        )
        state_schedules.append(
            make_segmented_schedule(
                n_steps,
                config.OBS_GAP,
                bptt_window,
                state_rng,
                config.FORECAST_FORCING_LOW,
                config.FORECAST_FORCING_HIGH,
                trajectory_difficulty,
                config.SEGMENT_MIN_CYCLES,
                config.SEGMENT_MAX_CYCLES,
            )
        )

    return np.stack(noise_schedules), np.stack(state_schedules)


def make_balanced_difficulties(batch_size, max_difficulty, seed, epoch):
    """Distribute a batch evenly across integer difficulties from zero to the maximum."""
    if batch_size <= 0:
        raise ValueError('batch_size must be positive')
    if not 0 <= max_difficulty <= 10:
        raise ValueError('max_difficulty must be in [0, 10]')

    levels = np.arange(max_difficulty + 1, dtype=np.float32)
    repeats, remainder = divmod(batch_size, len(levels))
    values = np.tile(levels, repeats)
    rng = np.random.default_rng(np.random.SeedSequence([seed, epoch, 2]))
    if remainder:
        values = np.concatenate([values, rng.choice(levels, size=remainder, replace=False)])
    rng.shuffle(values)
    return values
