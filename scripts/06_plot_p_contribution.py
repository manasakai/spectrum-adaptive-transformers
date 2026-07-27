from __future__ import annotations

import argparse

import pandas as pd

from bert_spectrum.bounds import aggregate_contribution_curve, contribution_curve_table
from bert_spectrum.config import ensure_output_dirs, load_config, model_slug
from bert_spectrum.plotting import (
    BERT_MATRIX_TYPE_SET,
    plot_p_contribution_by_head,
)
from bert_spectrum.utils import save_csv_and_jsonl, set_seed


DEFAULT_MATRIX_TYPES = "QK,V_tilde,M_in,M_out"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot proxy contributions as a function of the Schatten index p.")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--model-id", type=str, default=None)
    parser.add_argument("--matrix-types", type=str, default=DEFAULT_MATRIX_TYPES)
    parser.add_argument("--yscale", type=str, default="log", choices=["linear", "log"])
    parser.add_argument("--layers", type=str, default=None, help="Comma-separated layer list, e.g. 0,1,2,3")
    parser.add_argument("--no-normalize-at-p0", action="store_true", help="Disable normalization by the p=0 contribution in by-head plots.",)
    return parser.parse_args()


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


def _parse_layers(text: str | None) -> list[int] | None:
    if text is None or not text.strip():
        return None
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    set_seed(int(cfg.get("seed", 42)))
    dirs = ensure_output_dirs(cfg)

    sv_path = dirs["tables"] / "singular_values.csv"
    summary_path = dirs["tables"] / "matrix_summary.csv"
    if not sv_path.exists() or not summary_path.exists():
        raise FileNotFoundError(
            "Run scripts/02_plot_spectral_decay.py first so that singular_values.csv and matrix_summary.csv are available."
        )

    singular_values = pd.read_csv(sv_path)
    summary = pd.read_csv(summary_path)

    model_id = args.model_id or str(cfg.get("primary_model_id"))
    singular_values = singular_values[singular_values["model_id"] == model_id].copy()
    summary = summary[summary["model_id"] == model_id].copy()
    if singular_values.empty or summary.empty:
        raise ValueError(f"No rows found for model_id={model_id!r}.")

    matrix_types = _parse_matrix_types(args.matrix_types)
    layers = _parse_layers(args.layers)

    bcfg = cfg.get("bound", {})
    contribution_table = contribution_curve_table(
        singular_values,
        summary,
        L_phi=float(bcfg.get("L_phi", 1.13)),
        p_grid_m=bcfg.get("p_grid_m", None),
        rank_tol=float(cfg.get("rank_tol", 1e-10)),
    )
    contribution_table = contribution_table[
        contribution_table["matrix_type"].isin(matrix_types)
    ].copy()

    slug = model_slug(model_id)
    save_csv_and_jsonl(
        contribution_table,
        dirs["tables"] / f"p_contribution_raw_{slug}.csv",
    )

    normalize_at_p0 = not args.no_normalize_at_p0

    suffix = "_normalized" if normalize_at_p0 else "_raw"
    output_base = dirs["figures"] / f"p_contribution_by_head_{slug}{suffix}"
    plot_p_contribution_by_head(
        contribution_table,
        output_base,
        model_id=model_id,
        matrix_types=matrix_types,
        layers=layers,
        yscale=args.yscale,
        normalize_at_p0=normalize_at_p0,
    )

    print(f"Saved figure to {output_base}.pdf")


if __name__ == "__main__":
    main()