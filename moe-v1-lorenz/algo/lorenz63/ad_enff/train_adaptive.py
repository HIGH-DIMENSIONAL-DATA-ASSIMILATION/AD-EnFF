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

from common.ensemble_losses import gaussian_nll_loss, trajectory_energy_score_loss
from common.lorenz63_torch_simulator import lorenz63_rk4_step
from moe_adaptive import AdaptiveENFF


def parse_args():
    parser = argparse.ArgumentParser(description="Train the reported Lorenz-63 AD-EnFF")
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
    parser.add_argument("--attention", action=argparse.BooleanOptionalAction, default=True)
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
    warmup = 0 if args.smoke else 20
    dt, obs_gap = config.DT, config.OBS_GAP

    if args.dim != config.STATE_DIM:
        raise ValueError("Lorenz-63 requires --dim 3")
    model = AdaptiveENFF(args.dim, attention=args.attention, state_clamp=60.0).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, args.epochs - 1), eta_min=1e-5
    )
    pool = make_attractor_pool(
        lorenz63_rk4_step, pool_size, args.dim, spinup_steps, 42, device
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    best_path = args.output_dir / f"{args.run_id}_best.pt"
    last_path = args.output_dir / f"{args.run_id}_last.pt"
    best_objective = float("inf")

    for epoch in range(args.epochs):
        model.train()
        model.reset_cache()
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
        previous = ensemble.clone()
        optimizer.zero_grad()
        particles, targets = [], []
        objective_sum = energy_sum = nll_sum = mean_error_sum = rmse_sum = 0.0
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
            noise = noise_schedule[:, step, None]
            observation = torch.atan(truth) + noise * torch.randn_like(truth)
            output = model(ensemble, previous, observation)
            ensemble = output["posterior"]
            particles.append(ensemble)
            targets.append(truth)
            rmse_sum += ((ensemble.mean(1) - truth).square().mean() + 1e-8).sqrt().item()
            observations += 1
            previous = ensemble.detach().clone()
            ensemble = ensemble.detach()

            if observations % window == 0 or step + obs_gap >= n_steps:
                particle_window = torch.stack(particles, 1)
                truth_window = torch.stack(targets, 1)
                energy = trajectory_energy_score_loss(particle_window, truth_window)
                nll = gaussian_nll_loss(particle_window, truth_window)
                mean_error = (particle_window.mean(2) - truth_window).square().mean()
                objective = energy + 0.1 * nll + 0.2 * mean_error
                objective.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
                optimizer.zero_grad()
                model.detach_memory()
                objective_sum += objective.item()
                energy_sum += energy.item()
                nll_sum += nll.item()
                mean_error_sum += mean_error.item()
                windows += 1
                particles, targets = [], []

        window_count = max(1, windows)
        mean_objective = objective_sum / window_count
        print(
            f"Epoch {epoch:3d} | RMSE {rmse_sum / max(1, observations):.4f} "
            f"| ES {energy_sum / window_count:.4f} "
            f"| NLL {nll_sum / window_count:.4f} "
            f"| mean_MSE {mean_error_sum / window_count:.4f} "
            f"| objective {mean_objective:.4f}"
        )
        if epoch >= warmup and mean_objective < best_objective:
            best_objective = mean_objective
            torch.save(model.state_dict(), best_path)
            print(f"  saved best objective: {best_path}")
        scheduler.step()

    torch.save(model.state_dict(), last_path)


if __name__ == "__main__":
    main()
