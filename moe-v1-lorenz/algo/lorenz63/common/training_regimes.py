"""Shared observation-noise and model-mismatch schedules for Lorenz-63."""

import numpy as np
import torch

from common import config


def make_attractor_pool(step_fn, pool_size, state_dim, spinup_steps, seed, device):
    if pool_size <= 0 or spinup_steps <= 0:
        raise ValueError("pool_size and spinup_steps must be positive")
    if state_dim != config.STATE_DIM:
        raise ValueError(f"Lorenz-63 state dimension must be 3, got {state_dim}")
    initial = np.random.default_rng(seed).standard_normal((pool_size, state_dim)).astype(np.float32)
    pool = torch.tensor(initial, device=device) * 0.5
    print(
        f"[attractor-pool-L63] building size={pool_size} dim={state_dim} "
        f"spinup_steps={spinup_steps} seed={seed}", flush=True,
    )
    report_every = max(1, spinup_steps // 10)
    with torch.no_grad():
        for step in range(spinup_steps):
            pool = step_fn(pool, config.DT, rho=config.TRUTH_RHO, sigma=config.TRUTH_SIGMA)
            completed = step + 1
            if completed % report_every == 0 or completed == spinup_steps:
                print(f"[attractor-pool-L63] spinup {completed}/{spinup_steps}", flush=True)
    print("[attractor-pool-L63] ready", flush=True)
    return pool.detach()


def sample_attractor_batch(pool, batch_size, seed):
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if batch_size > len(pool):
        raise ValueError(f"batch_size={batch_size} exceeds pool size={len(pool)}")
    indices = np.random.default_rng(seed).choice(len(pool), batch_size, replace=False)
    return pool[torch.as_tensor(indices, device=pool.device)].clone()


def select_attractor_state(pool, seed):
    index = int(np.random.default_rng(seed).integers(0, len(pool)))
    return pool[index:index + 1].clone()


def make_segmented_schedule(n_steps, obs_gap, rng, low, high, difficulty, bptt_window):
    if n_steps <= 0 or obs_gap <= 0 or bptt_window <= 0:
        raise ValueError("n_steps, obs_gap, and bptt_window must be positive")
    if low > high:
        raise ValueError("schedule lower bound must not exceed its upper bound")
    if not 0 <= difficulty <= 10:
        raise ValueError("difficulty must be in [0, 10]")
    if 0.5 * (config.SEGMENT_MIN_CYCLES + config.SEGMENT_MAX_CYCLES) > bptt_window:
        raise ValueError("mean segment length must not exceed the BPTT window")
    effective_high = low + difficulty / 10.0 * (high - low)
    schedule = np.empty(n_steps, dtype=np.float32)
    start = 0
    while start < n_steps:
        cycles = int(rng.integers(config.SEGMENT_MIN_CYCLES, config.SEGMENT_MAX_CYCLES + 1))
        value = rng.uniform(low, effective_high) if effective_high > low else low
        end = start + cycles * obs_gap
        schedule[start:end] = value
        start = end
    return schedule


def make_training_schedules(batch_size, n_steps, difficulty, seed, epoch,
                            bptt_window=config.BPTT_WINDOW, obs_gap=config.OBS_GAP):
    if batch_size <= 0 or n_steps <= 0:
        raise ValueError("batch_size and n_steps must be positive")
    difficulties = np.asarray(difficulty, dtype=np.float32)
    if difficulties.ndim == 0:
        difficulties = np.full(batch_size, float(difficulties), dtype=np.float32)
    if difficulties.shape != (batch_size,):
        raise ValueError(f"difficulty must be scalar or shape ({batch_size},)")
    if not np.isfinite(difficulties).all() or ((difficulties < 0) | (difficulties > 10)).any():
        raise ValueError("all difficulties must be finite and in [0, 10]")
    noise, rho, sigma = [], [], []
    for index, level in enumerate(difficulties):
        generators = [
            np.random.default_rng(np.random.SeedSequence([seed, epoch, index, channel]))
            for channel in range(3)
        ]
        noise.append(make_segmented_schedule(
            n_steps, obs_gap, generators[0], config.NOISE_STD_LOW, config.NOISE_STD_HIGH,
            float(level), bptt_window,
        ))
        rho.append(make_segmented_schedule(
            n_steps, obs_gap, generators[1], config.TRUTH_RHO, config.FORECAST_RHO_HIGH,
            float(level), bptt_window,
        ))
        sigma.append(make_segmented_schedule(
            n_steps, obs_gap, generators[2], config.TRUTH_SIGMA, config.FORECAST_SIGMA_HIGH,
            float(level), bptt_window,
        ))
    return np.stack(noise), np.stack(rho), np.stack(sigma)


def make_balanced_difficulties(
    batch_size, max_difficulty, seed, epoch, difficulty_step=1.0
):
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if not 0 <= max_difficulty <= 10:
        raise ValueError("max_difficulty must be in [0, 10]")
    if not 0 < difficulty_step <= 10:
        raise ValueError("difficulty_step must be in (0, 10]")
    levels = np.arange(
        0.0, max_difficulty + 0.5 * difficulty_step,
        difficulty_step, dtype=np.float32,
    )
    if not np.isclose(levels[-1], max_difficulty):
        levels = np.append(levels, np.float32(max_difficulty))
    repeats, remainder = divmod(batch_size, len(levels))
    values = np.tile(levels, repeats)
    rng = np.random.default_rng(np.random.SeedSequence([seed, epoch, 3]))
    if remainder:
        values = np.concatenate([values, rng.choice(levels, remainder, replace=False)])
    rng.shuffle(values)
    return values
