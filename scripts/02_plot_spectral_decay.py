from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from bert_spectrum.config import ensure_output_dirs, load_config, model_slug
from bert_spectrum.plotting import BERT_MATRIX_TYPE_SET, plot_spectral_decay
from bert_spectrum.spectrum import compute_spectrum_table
from bert_spectrum.utils import save_csv_and_jsonl, set_seed


DEFAULT_MATRIX_TYPES = "QK,V_tilde,M_in,M_out"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute singular values and plot spectral decay.")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--model-id", type=str, default=None)
    parser.add_argument("--matrix-types", type=str, default=DEFAULT_MATRIX_TYPES)
    parser.add_argument("--layers", type=str, default="0,1,2,3")
    parser.add_argument("--force", action="store_true", help="Recompute SVD tables even if cached CSVs exist.")
    parser.add_argument("--zero-tol", type=float, default=1e-10, help="Treat singular values smaller than this threshold as zero when plotting.")
    return parser.parse_args()


def _parse_ints(text: str) -> list[int]:
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def _parse_matrix_types(text: str) -> list[str]:
    matrix_types = [x.strip() for x in text.split(",") if x.strip()]
    unsupported = sorted(set(matrix_types) - BERT_MATRIX_TYPE_SET)
    if unsupported:
        allowed = ", ".join(sorted(BERT_MATRIX_TYPE_SET))
        bad = ", ".join(unsupported)
        raise ValueError(
            f"Unsupported matrix type(s): {bad}. "
            f"This experiment only plots the proxy matrices: {allowed}."
        )
    return matrix_types


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    set_seed(int(cfg.get("seed", 42)))
    dirs = ensure_output_dirs(cfg)

    matrix_types = _parse_matrix_types(args.matrix_types)
    required_types = set(matrix_types)

    matrix_index_path = dirs["tables"] / "matrix_index.csv"
    if not matrix_index_path.exists():
        raise FileNotFoundError("Run scripts/01_extract_weights.py first.")
    matrix_index = pd.read_csv(matrix_index_path)
    if "head" not in matrix_index.columns:
        raise ValueError("matrix_index.csv was produced by the old full-layer extractor. Re-run scripts/01_extract_weights.py.")

    sv_path = dirs["tables"] / "singular_values.csv"
    summary_path = dirs["tables"] / "matrix_summary.csv"
    if args.force:
        singular_values, summary = compute_spectrum_table(
            matrix_index,
            device=str(cfg.get("svd_device", "cpu")),
            dtype=str(cfg.get("svd_dtype", "float64")),
            rank_tol=float(cfg.get("rank_tol", 1e-10)),
        )
        save_csv_and_jsonl(singular_values, sv_path)
        save_csv_and_jsonl(summary, summary_path)
    else:
        singular_values = pd.read_csv(sv_path)

    model_id = args.model_id or str(cfg.get("primary_model_id"))
    layers = _parse_ints(args.layers)
    base = dirs["figures"] / f"spectral_decay_{model_slug(model_id)}"
    plot_spectral_decay(
        singular_values,
        base,
        model_id=model_id,
        matrix_types=matrix_types,
        layers=layers,
        normalize=False,
        zero_tol=args.zero_tol,
    )
    print(f"Saved figure to {base}.pdf")


if __name__ == "__main__":
    main()
