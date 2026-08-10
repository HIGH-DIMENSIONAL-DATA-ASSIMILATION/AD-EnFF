"""Train MNMEF on the Lorenz-63 benchmark protocol."""

import argparse
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import config
from common.training_regimes import (
    make_attractor_pool,
    make_balanced_difficulties,
    make_training_schedules,
    sample_attractor_batch,
)

from common.lorenz63_torch_simulator import lorenz63_rk4_step
from mnmef_corrector import MNMEFCorrTerms


def normalized_mean_error(ensemble, truth):
    numerator = (ensemble.mean(2) - truth).square().sum(-1)
    denominator = truth.square().sum(-1).clamp_min(1e-8)
    return (numerator / denominator).mean()


def parse_args():
    parser = argparse.ArgumentParser(description="Train MNMEF on the Lorenz-63 benchmark")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--dim", type=int, default=config.STATE_DIM)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument(
        "--max-difficulty", "--difficulty", dest="max_difficulty", type=int, default=10,
        help="balance each batch across difficulties from 0 through this value",
    )
    parser.add_argument(
        "--difficulty-step", type=float, default=1.0,
        help="spacing between balanced difficulty levels (default: 1.0)",
    )
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--window", type=int, default=config.BPTT_WINDOW)
    parser.add_argument("--output-dir", type=Path, default=Path("checkpoints"))
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(42)
    np.random.seed(42)
    device = torch.device("cpu" if args.cpu else "cuda")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable; pass --cpu to run on CPU")

    default_batch_size, ensemble_size = (
        (config.SMOKE_BATCH_SIZE, config.SMOKE_ENSEMBLE_SIZE)
        if args.smoke else (config.TRAIN_BATCH_SIZE, config.TRAIN_ENSEMBLE_SIZE)
    )
    batch_size = args.batch_size if args.batch_size is not None else default_batch_size
    window = args.window
    n_steps = config.SMOKE_STEPS if args.smoke else config.TRAIN_STEPS
    pool_size = config.SMOKE_POOL_SIZE if args.smoke else config.ATTRACTOR_POOL_SIZE
    spinup_steps = config.SMOKE_SPINUP_STEPS if args.smoke else config.ATTRACTOR_SPINUP_STEPS
    warmup = 0 if args.smoke else 10
    dt, obs_gap = config.DT, config.OBS_GAP
    if args.dim != config.STATE_DIM:
        raise ValueError("Lorenz-63 requires --dim 3")
    model = MNMEFCorrTerms(args.dim, clamp=60.0).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.0)
    pool = make_attractor_pool(
        lorenz63_rk4_step, pool_size, args.dim, spinup_steps, 42, device
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    best_path = args.output_dir / f"{args.run_id}_best.pt"
    last_path = args.output_dir / f"{args.run_id}_last.pt"
    best_objective = float("inf")

    for epoch in range(args.epochs):
        difficulties = make_balanced_difficulties(
            batch_size, args.max_difficulty, 42, epoch, args.difficulty_step
        )
        noise_np, rho_np, lorenz_sigma_np = make_training_schedules(
            batch_size, n_steps, difficulties, 42, epoch, bptt_window=window
        )
        if epoch == 0:
            levels, counts = np.unique(difficulties, return_counts=True)
            distribution = ", ".join(
                f"{level:g}:{int(count)}" for level, count in zip(levels, counts)
            )
            print(f"difficulty trajectories per batch: {distribution}")
        noise_schedule = torch.tensor(noise_np, device=device)
        rho_schedule = torch.tensor(rho_np, device=device)
        sigma_schedule = torch.tensor(lorenz_sigma_np, device=device)
        truth = sample_attractor_batch(pool, batch_size, 42 + epoch)
        ensemble = truth[:, None].expand(-1, ensemble_size, -1).clone()
        ensemble += 0.5 * torch.randn_like(ensemble)
        particles, targets = [], []
        objective_sum = rmse_sum = 0.0
        windows = observations = 0

        for step in range(n_steps):
            truth = lorenz63_rk4_step(
                truth, dt, rho=config.TRUTH_RHO, sigma=config.TRUTH_SIGMA
            )
            ensemble = lorenz63_rk4_step(
                ensemble, dt, rho=rho_schedule[:, step, None],
                sigma=sigma_schedule[:, step, None]
            )
            if (step + 1) % obs_gap:
                continue
            observation_std = noise_schedule[:, step, None, None]
            observation = torch.atan(truth) + observation_std[:, 0] * torch.randn_like(truth)
            covariance = observation_std.square() * torch.eye(args.dim, device=device)[None]
            ensemble = model(ensemble, observation[:, None], covariance, observation_std)
            particles.append(ensemble)
            targets.append(truth)
            rmse_sum += ((ensemble.mean(1) - truth).square().mean() + 1e-8).sqrt().item()
            observations += 1
            ensemble = ensemble.detach()

            if observations % window == 0 or step + obs_gap >= n_steps:
                objective = normalized_mean_error(torch.stack(particles, 1), torch.stack(targets, 1))
                optimizer.zero_grad()
                objective.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                objective_sum += objective.item()
                windows += 1
                particles, targets = [], []

        mean_objective = objective_sum / max(1, windows)
        print(f"Epoch {epoch:3d} | RMSE {rmse_sum / max(1, observations):.4f} | nl2 {mean_objective:.5f}")
        if epoch >= warmup and mean_objective < best_objective:
            best_objective = mean_objective
            torch.save(model.state_dict(), best_path)

    torch.save(model.state_dict(), last_path)


if __name__ == "__main__":
    main()
