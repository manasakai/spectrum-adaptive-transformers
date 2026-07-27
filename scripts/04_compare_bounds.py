from __future__ import annotations

import argparse

import pandas as pd

from bert_spectrum.bounds import BoundParams, observed_bound_table
from bert_spectrum.config import ensure_output_dirs, load_config, model_slug
from bert_spectrum.plotting import plot_observed_bound_grid
from bert_spectrum.utils import save_csv_and_jsonl, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare the generalization gap bound proxies.")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--model-id", type=str, default=None)
    return parser.parse_args()


def _params_by_model(summary: pd.DataFrame, bcfg: dict) -> dict[str, BoundParams]:
    params: dict[str, BoundParams] = {}
    for model_id, g in summary.groupby("model_id", sort=False):
        params[model_id] = BoundParams(
            L=int(g["num_hidden_layers"].iloc[0]),
            N=int(g["hidden_size"].iloc[0]),
            L_phi=float(bcfg.get("L_phi", 1.13)),
        )
    return params


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    set_seed(int(cfg.get("seed", 42)))
    dirs = ensure_output_dirs(cfg)
    opt = pd.read_csv(dirs["tables"] / "schatten_opt.csv")
    summary = pd.read_csv(dirs["tables"] / "matrix_summary.csv")
    if args.model_id is not None:
        opt = opt[opt["model_id"] == args.model_id].copy()
        summary = summary[summary["model_id"] == args.model_id].copy()
    if opt.empty:
        raise ValueError("No optimized rows found. Run scripts/03_optimize_schatten.py first.")

    bcfg = cfg.get("bound", {})
    # Compute our bounds using the optimal Schatten indices, and compare them with the bounds of Edelman et al.
    obs = observed_bound_table(opt, summary, _params_by_model(summary, bcfg))

    # Attach L and N to the observed generalization bounds table
    meta = summary.groupby("model_id", as_index=False).agg(
        hidden_size=("hidden_size", "first"),
        num_hidden_layers=("num_hidden_layers", "first"),
        num_heads=("num_heads", "first"),
    )
    obs = obs.merge(meta, on="model_id", how="left")
    save_csv_and_jsonl(obs, dirs["tables"] / "observed_bounds.csv")

    model_id = args.model_id or str(cfg.get("primary_model_id"))
    obs_model = obs[obs["model_id"] == model_id]
    if not obs_model.empty:
        save_csv_and_jsonl(obs_model, dirs["tables"] / f"observed_bounds_{model_slug(model_id)}.csv")

    # Growth in N with L fixed
    plot_observed_bound_grid(
        obs,
        dirs["figures"] / "observed_bound_vs_width_fixed_L",
        x_axis="hidden_size",
        group_axis="num_hidden_layers",
    )

    # Growth in L with N fixed
    plot_observed_bound_grid(
        obs,
        dirs["figures"] / "observed_bound_vs_depth_fixed_N",
        x_axis="num_hidden_layers",
        group_axis="hidden_size",
    )

    # # Growth in N with L fixed, ours only
    # plot_observed_bound_grid(
    #     obs,
    #     dirs["figures"] / "observed_bound_ours_vs_width_fixed_L",
    #     x_axis="hidden_size",
    #     group_axis="num_hidden_layers",
    #     bounds=("ours",),
    #     yscale="linear",
    # )

    # # Growth in L with N fixed, ours only
    # plot_observed_bound_grid(
    #     obs,
    #     dirs["figures"] / "observed_bound_ours_vs_depth_fixed_N",
    #     x_axis="num_hidden_layers",
    #     group_axis="hidden_size",
    #     bounds=("ours",),
    #     yscale="linear",
    # )


if __name__ == "__main__":
    main()
