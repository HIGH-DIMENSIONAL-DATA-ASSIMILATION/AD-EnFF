#!/usr/bin/env python3
"""Evaluate analytical ENFF, AD-EnFF, and MNMEF on one Lorenz-96 protocol."""

import argparse
from pathlib import Path
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "mnmef"))
from common import config
from common.training_regimes import make_attractor_pool, make_training_schedules, select_attractor_state
from mnmef_corrector import MNMEFCorrTerms

sys.path.insert(0, str(ROOT / "ad_enff"))
from common.ensemble_losses import trajectory_energy_score_loss, variogram_score_loss
from common.ensemble_metrics import compute_all_metrics
from common.backwards_enff_analytical import BackwardsENFFAnalytical
from common.lorenz96_torch_simulator import lorenz96_rk4_step
from moe_adaptive import AdaptiveENFF


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--results-root", type=Path, default=ROOT / "results")
    parser.add_argument("--run-dir", type=Path, default=None,
                        help="Run directory containing checkpoints (default: results-root/run-id)")
    parser.add_argument("--plain-only", action="store_true",
                        help="evaluate analytical ENFF and one plain AD-EnFF checkpoint")
    parser.add_argument("--publish", action="store_true",
                        help="evaluate only AD-EnFF (plain) and MNMEF; label them 'AD-EnFF' and 'MNMEF'")
    parser.add_argument("--dim", type=int, default=40)
    parser.add_argument("--difficulty", type=float, default=10.0)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--seed-base", type=int, default=10000)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--out-dir", type=Path, default=None,
                        help="Override output directory for PNGs and logs (default: results-root/run-id/eval)")
    parser.add_argument("--plot-dim", type=int, default=0,
                        help="State dimension index to plot in trajectory figure (default: 0)")
    return parser.parse_args()


def load_models(args, device):
    checkpoint_dir = args.run_dir / "checkpoints"
    if args.publish:
        checkpoints = {
            "AD-EnFF plain": checkpoint_dir / f"{args.run_id}_plain_best.pt",
            "MNMEF": checkpoint_dir / f"{args.run_id}_mnmef_best.pt",
        }
        models = {
            "Analytical ENFF": BackwardsENFFAnalytical(Nt_ode=12, observe_fn=torch.atan),
            "AD-EnFF plain": AdaptiveENFF(args.dim, attention=False),
            "MNMEF": MNMEFCorrTerms(args.dim, clamp=40.0),
        }
    elif args.plain_only:
        checkpoints = {
            "AD-EnFF plain": checkpoint_dir / f"{args.run_id}_best.pt",
        }
        models = {
            "Analytical ENFF": BackwardsENFFAnalytical(Nt_ode=12, observe_fn=torch.atan),
            "AD-EnFF plain": AdaptiveENFF(args.dim, attention=False),
        }
    else:
        checkpoints = {
            "AD-EnFF plain": checkpoint_dir / f"{args.run_id}_plain_best.pt",
            "MNMEF": checkpoint_dir / f"{args.run_id}_mnmef_best.pt",
        }
        models = {
            "Analytical ENFF": BackwardsENFFAnalytical(Nt_ode=12, observe_fn=torch.atan),
            "AD-EnFF plain": AdaptiveENFF(args.dim, attention=False),
            "MNMEF": MNMEFCorrTerms(args.dim, clamp=40.0),
        }
    missing = [str(path) for path in checkpoints.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing checkpoints:\n" + "\n".join(missing))
    for name, model in models.items():
        if name in checkpoints:
            model.load_state_dict(torch.load(checkpoints[name], map_location=device))
        model.to(device).eval()
    return models


def run_seed(seed, models, args, device, pool):
    torch.manual_seed(seed)
    np.random.seed(seed)
    for model in models.values():
        if hasattr(model, "reset_cache"):
            model.reset_cache()

    ensemble_size = config.SMOKE_ENSEMBLE_SIZE if args.smoke else config.TRAIN_ENSEMBLE_SIZE
    truth = select_attractor_state(pool, seed)
    initial = truth[:, None].expand(-1, ensemble_size, -1).clone()
    initial += 0.5 * torch.randn_like(initial)
    ensembles = {name: initial.clone() for name in models}
    previous_posterior = {name: initial.clone() for name in models}
    sigma_np, forcing_np = make_training_schedules(1, args.steps, args.difficulty, seed, 0)
    sigma = torch.tensor(sigma_np[0], device=device)
    forcing = torch.tensor(forcing_np[0], device=device)
    full_truth = []
    full_ensembles = {name: [] for name in models}
    analysis_truth = []
    analysis_ensembles = {name: [] for name in models}
    latency_ms = {name: [] for name in models}

    def timed_analysis(name, function):
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        started = time.perf_counter()
        result = function()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        latency_ms[name].append((time.perf_counter() - started) * 1000.0)
        return result

    with torch.no_grad():
        for step in range(args.steps):
            truth = lorenz96_rk4_step(truth, config.DT, F=config.TRUTH_FORCING)
            model_forcing = forcing[step].view(1, 1, 1)
            for name in ensembles:
                ensembles[name] = lorenz96_rk4_step(
                    ensembles[name], config.DT, F=model_forcing
                )
            full_truth.append(truth.clone())
            for name in models:
                full_ensembles[name].append(ensembles[name].clone())
            if (step + 1) % config.OBS_GAP:
                continue

            observation_std = sigma[step]
            observation = torch.atan(truth) + observation_std * torch.randn_like(truth)
            observation_variance = torch.ones_like(observation) * observation_std.square()
            if "Analytical ENFF" in models:
                enff_output = timed_analysis(
                    "Analytical ENFF",
                    lambda: models["Analytical ENFF"](
                        z1=ensembles["Analytical ENFF"],
                        z0=previous_posterior["Analytical ENFF"], y=observation,
                        obs_noise_diag=observation_variance, external_scheduler=0.05,
                    ),
                )
                # Clamp to +/-40, matching AD-EnFF (state_clamp=40) and MNMEF (clamp=40) so
                # every filter shares the same divergence guard (never binds for L96 ~ +/-15).
                ensembles["Analytical ENFF"] = enff_output[:, -1].clamp(-40.0, 40.0)
                previous_posterior["Analytical ENFF"] = ensembles["Analytical ENFF"].clone()
            for name in ("AD-EnFF plain",):
                if name not in models:
                    continue
                output = timed_analysis(
                    name,
                    lambda name=name: models[name](
                        ensembles[name], previous_posterior[name], observation
                    ),
                )
                ensembles[name] = output["posterior"]
                previous_posterior[name] = ensembles[name].clone()

            if "MNMEF" in models:
                mnmef_std = observation_std.view(1, 1, 1)
                covariance = mnmef_std.square() * torch.eye(args.dim, device=device)[None]
                ensembles["MNMEF"] = timed_analysis(
                    "MNMEF",
                    lambda: models["MNMEF"](
                        ensembles["MNMEF"], observation[:, None], covariance, mnmef_std
                    ),
                )
                previous_posterior["MNMEF"] = ensembles["MNMEF"].clone()
            analysis_truth.append(truth.clone())
            for name in models:
                analysis_ensembles[name].append(ensembles[name].clone())
                full_ensembles[name][-1] = ensembles[name].clone()

    full_truth = torch.stack(full_truth, 1)
    full_ensembles = {name: torch.stack(values, 1) for name, values in full_ensembles.items()}
    analysis_truth = torch.stack(analysis_truth, 1)
    analysis_ensembles = {
        name: torch.stack(values, 1) for name, values in analysis_ensembles.items()
    }
    mean_latency = {name: float(np.mean(values)) for name, values in latency_ms.items()}
    return full_truth, full_ensembles, analysis_truth, analysis_ensembles, mean_latency


def score(ensemble, truth):
    metrics = compute_all_metrics(ensemble, truth, spatial=None, level=0.95)
    metrics["traj_es"] = trajectory_energy_score_loss(ensemble, truth).item()
    metrics["variogram"] = variogram_score_loss(ensemble, truth).item()
    return metrics


def print_table(results, names):
    keys = [
        "rmse", "energy_score", "traj_es", "variogram", "coverage",
        "climatology_mmd", "latency_ms",
    ]
    print("\n" + "=" * 88)
    print("MATCHED LORENZ-96 EVALUATION (mean +/- std across seeds)")
    print("=" * 88)
    print(f"{'metric':<28}" + "".join(f"{name:>20}" for name in names))
    for key in keys:
        row = f"{key:<28}"
        for name in names:
            values = np.asarray(results[name][key], dtype=float)
            row += f"{values.mean():.3f} +/- {values.std():.2f}".rjust(20)
        print(row)
    print("=" * 88)


def plot_trajectories(truth, trajectories, out_path, plot_dim=0):
    names = list(trajectories)
    time = np.arange(truth.shape[1]) * config.DT
    truth_values = truth[0, :, plot_dim].cpu().numpy()
    fig, axes = plt.subplots(
        len(names), 1, figsize=(3.5, 1.75 * len(names)), sharex=True,
    )
    axes = np.atleast_1d(axes)
    display_names = {
        "Analytical ENFF": "Analytical EnFF",
        "AD-EnFF plain": "AD-EnFF",
    }
    legend_handles = legend_labels = None
    for axis, name in zip(axes, names):
        values = trajectories[name][0, :, :, plot_dim].cpu().numpy()
        mean = values.mean(1)
        low, high = np.quantile(values, [0.025, 0.975], axis=1)
        axis.fill_between(
            time, low, high, color="0.78", alpha=0.75, linewidth=0,
            label="95% interval",
        )
        axis.plot(
            time, truth_values, color="black", linestyle="--", linewidth=1.25,
            label="Truth",
        )
        axis.plot(
            time, mean, color="black", linestyle="-", linewidth=1.05,
            label="Ensemble mean",
        )
        axis.set_title(display_names.get(name, name), fontsize=9.5, pad=2)
        axis.set_ylabel(rf"$x_{{{plot_dim + 1}}}$", fontsize=9)
        axis.tick_params(axis="both", labelsize=8, length=3)
        axis.grid(False)
        if legend_handles is None:
            legend_handles, legend_labels = axis.get_legend_handles_labels()
    fig.legend(
        legend_handles, legend_labels,
        ncol=3, fontsize=6.8, frameon=False, loc="upper left",
        bbox_to_anchor=(0.04, 0.995), borderaxespad=0,
        handlelength=2.2, columnspacing=0.8, handletextpad=0.4,
    )
    axes[-1].set_xlabel("Time", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.955), pad=0.35, h_pad=0.35)
    fig.savefig(out_path, dpi=600, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_metrics(results, names, out_path):
    keys = ["rmse", "energy_score", "traj_es", "variogram", "coverage", "climatology_mmd"]
    fig, axes = plt.subplots(2, 3, figsize=(14, 7))
    for axis, key in zip(axes.flat, keys):
        means = [np.mean(results[name][key]) for name in names]
        errors = [np.std(results[name][key]) for name in names]
        axis.bar(
            names, means, yerr=errors,
            color=["#6b7280", "#4c78a8", "#59a14f", "#e15759"][:len(names)],
        )
        axis.set_title(key.replace("_", " ").title())
        axis.tick_params(axis="x", rotation=15)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main():
    args = parse_args()
    if args.smoke:
        args.steps = config.SMOKE_STEPS
        args.seeds = 1
    device = torch.device("cpu" if args.cpu else "cuda")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable; pass --cpu")
    args.run_dir = args.run_dir if args.run_dir is not None else args.results_root / args.run_id
    args.out_dir = args.out_dir if args.out_dir is not None else args.run_dir / "eval"
    args.out_dir.mkdir(parents=True, exist_ok=True)
    models = load_models(args, device)
    pool_size = config.SMOKE_POOL_SIZE if args.smoke else config.ATTRACTOR_POOL_SIZE
    spinup = config.SMOKE_SPINUP_STEPS if args.smoke else config.ATTRACTOR_SPINUP_STEPS
    pool = make_attractor_pool(
        lorenz96_rk4_step, pool_size, args.dim, spinup,
        config.TRUTH_FORCING, config.DT, 42, device,
    )

    names = list(models)
    results = {name: {} for name in names}
    first_truth = first_trajectories = None
    for seed in range(args.seed_base, args.seed_base + args.seeds):
        print(f"seed {seed}", flush=True)
        full_truth, full_trajectories, truth, trajectories, latency = run_seed(
            seed, models, args, device, pool
        )
        if first_truth is None:
            first_truth, first_trajectories = full_truth, full_trajectories
        for name in names:
            for key, value in score(trajectories[name], truth).items():
                results[name].setdefault(key, []).append(value)
            results[name].setdefault("latency_ms", []).append(latency[name])

    if args.publish:
        rename = {"AD-EnFF plain": "AD-EnFF"}
        relabel = lambda d: {rename.get(k, k): v for k, v in d.items()}
        results = relabel(results)
        first_trajectories = relabel(first_trajectories)
        names = [rename.get(n, n) for n in names]

    print_table(results, names)
    plot_trajectories(first_truth, first_trajectories, args.out_dir / "trajectories.png", plot_dim=args.plot_dim)
    plot_metrics(results, names, args.out_dir / "metrics.png")
    print(f"saved {args.out_dir / 'trajectories.png'}")
    print(f"saved {args.out_dir / 'metrics.png'}")


if __name__ == "__main__":
    main()
