from __future__ import annotations

import argparse

import pandas as pd

from bert_spectrum.config import ensure_output_dirs, load_config
from bert_spectrum.plotting import plot_norm_scaling
from bert_spectrum.utils import save_csv_and_jsonl, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot norm scaling of the matrices across BERT widths.")
    parser.add_argument("--config", type=str, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    set_seed(int(cfg.get("seed", 42)))
    dirs = ensure_output_dirs(cfg)
    summary = pd.read_csv(dirs["tables"] / "matrix_summary.csv")
    if "use_in_proxy" in summary.columns:
        summary = summary[summary["use_in_proxy"].astype(str).str.lower().isin(["true", "1"])]
    save_csv_and_jsonl(summary, dirs["tables"] / "norm_scaling_raw.csv")

    for norm_name in ["norm_21", "norm_11", "spectral_norm"]:
        plot_norm_scaling(summary, dirs["figures"] / f"scaling_{norm_name}_individual_L2", norm_name=norm_name, fixed_L=2,)


if __name__ == "__main__":
    main()
