from __future__ import annotations

import argparse

import pandas as pd

from bert_spectrum.bounds import optimize_p_table
from bert_spectrum.config import ensure_output_dirs, load_config, model_slug
from bert_spectrum.utils import save_csv_and_jsonl, save_latex_ready_csv, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Optimize the post hoc Schatten indices head by head.")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--model-id", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    set_seed(int(cfg.get("seed", 42)))
    dirs = ensure_output_dirs(cfg)
    sv = pd.read_csv(dirs["tables"] / "singular_values.csv")
    summary = pd.read_csv(dirs["tables"] / "matrix_summary.csv")

    bcfg = cfg.get("bound", {})
    # The grid is P_m={0,1/m,...,2}; by default m=ceil(L+log(N))
    opt = optimize_p_table(
        sv,
        summary,
        L_phi=float(bcfg.get("L_phi", 1.13)),
        p_grid_m=bcfg.get("p_grid_m", None),
        rank_tol=float(cfg.get("rank_tol", 1e-10)),
    )
    save_csv_and_jsonl(opt, dirs["tables"] / "schatten_opt.csv")
    save_latex_ready_csv(opt, dirs["tables"] / "schatten_opt_latex.csv")


if __name__ == "__main__":
    main()
