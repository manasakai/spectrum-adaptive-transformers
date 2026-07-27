from __future__ import annotations

from dataclasses import dataclass
from math import ceil, log
from typing import Iterable

import numpy as np
import pandas as pd

from .norms import schatten_p_power_from_svals


IDENT_COLS = ["model_id", "model_slug", "layer", "head", "matrix_type"]
MLP_MATRIX_TYPES = ("M_in", "M_out")


@dataclass(frozen=True)
class BoundParams:
    L: int
    N: int
    L_phi: float = 1.13


def p_m(L: int, N: int, p_grid_m: int | None = None) -> tuple[np.ndarray, int]:
    # Return the post hoc grid P_m={0,1/m,...,2}
    m = int(p_grid_m) if p_grid_m is not None else int(ceil(float(L) + log(max(float(N), 2.0))))
    m = max(m, 1)
    return np.arange(0, 2 * m + 1, dtype=float) / float(m), m


def schatten_contribution_from_svals(
    svals: np.ndarray,
    p: float,
    N: int,
    scale: float, # scale corresponds to gamma^{*,(ell,a)} alpha^{(ell)} L
    rank_tol: float = 1e-10,
) -> float:
    # Summand in our bound
    # Each matrix is optimized separately
    sc_power = schatten_p_power_from_svals(svals, p=p, tol=rank_tol)
    denom = p + 2.0
    safe_scale = max(float(scale), np.finfo(float).tiny)
    return float((sc_power ** (1.0 / denom)) * (safe_scale ** (p / denom)) * (N ** ((p + 1.0) / denom)))


def optimize_p_for_svals(
    svals: np.ndarray,
    N: int,
    scale: float, # scale corresponds to gamma^{*,(ell,a)} alpha^{(ell)} L
    p_grid: np.ndarray,
    rank_tol: float = 1e-10,
) -> dict:
    values = np.asarray([
        schatten_contribution_from_svals(svals, float(p), N=N, scale=scale, rank_tol=rank_tol)
        for p in p_grid
    ])
    idx = int(np.argmin(values)) # indices of each p that minimizes the corresponding layer's contribution to the bound

    def _at(target: float) -> float:
        return float(values[int(np.argmin(np.abs(p_grid - target)))]) # returns the contribution of p that is the closest to the target

    return {
        "p_opt": float(p_grid[idx]),
        "min_contribution": float(values[idx]),
    }


def _bool_series(series: pd.Series, default: bool = True) -> pd.Series:
    if series.empty:
        return pd.Series(dtype=bool)
    if series.dtype == bool:
        return series
    return series.map(lambda x: default if pd.isna(x) else str(x).strip().lower() in {"true", "1", "yes"})


def _selected_proxy_summary(matrix_summary: pd.DataFrame) -> pd.DataFrame:
    # Keep exactly the matrices used by the generalization gap bound proxy
    summary = matrix_summary.copy()
    if "use_in_proxy" in summary.columns:
        summary = summary[_bool_series(summary["use_in_proxy"])]
    keep = summary["matrix_type"].isin(["QK", "V_tilde", *MLP_MATRIX_TYPES])
    selected = summary[keep].copy()
    return selected


def _model_maps(selected: pd.DataFrame) -> tuple[dict, dict, dict, dict]:
    qk = selected[selected["matrix_type"] == "QK"].set_index(["layer", "head"])
    val = selected[selected["matrix_type"] == "V_tilde"].set_index(["layer", "head"])
    mlp_in = selected[selected["matrix_type"] == "M_in"].set_index("layer")
    mlp_out = selected[selected["matrix_type"] == "M_out"].set_index("layer")
    return qk.to_dict("index"), val.to_dict("index"), mlp_in.to_dict("index"), mlp_out.to_dict("index")


def _require_layer_mlp(mlp_in_map: dict, mlp_out_map: dict, layer: int) -> tuple[dict, dict]:
    if layer not in mlp_in_map or layer not in mlp_out_map:
        raise ValueError(
            f"Missing M_in or M_out row for layer {layer}. "
            "Re-run scripts/01_extract_weights.py to regenerate M_in and M_out, "
            "then re-run scripts/02_plot_spectral_decay.py --force."
        )
    return mlp_in_map[layer], mlp_out_map[layer]


def multihead_alpha_by_layer(selected: pd.DataFrame, L_phi: float = 1.13) -> dict[int, float]:
    # Compute alpha^{(ell)}
    qk_map, val_map, mlp_in_map, mlp_out_map = _model_maps(selected)
    layers = sorted(int(x) for x in selected["layer"].unique())
    num_heads = int(selected["num_heads"].dropna().iloc[0])
    alpha: dict[int, float] = {}
    for ell in layers:
        prod = 1.0
        for k in layers:
            if k <= ell:
                continue
            m_in, m_out = _require_layer_mlp(mlp_in_map, mlp_out_map, k)
            c2_m_in = float(m_in["spectral_norm"])
            c2_m_out = float(m_out["spectral_norm"])
            layer_sum = 0.0
            for h in range(num_heads):
                if (k, h) not in qk_map or (k, h) not in val_map:
                    raise ValueError(f"Missing QK or V_tilde row for layer {k}, head {h}")
                c2_qk = float(qk_map[(k, h)]["spectral_norm"])
                c2_v = float(val_map[(k, h)]["spectral_norm"])
                layer_sum += c2_v * (1.0 + 4.0 * c2_qk)
            prod *= float(L_phi) * c2_m_in * c2_m_out * layer_sum
        alpha[ell] = float(prod)
    return alpha # alpha is a dictionary that contains alpha^{(ell)}'s of L layers


def gamma_for_row(
    row: pd.Series,
    selected: pd.DataFrame,
) -> float:
    # Compute gamma^{\star,(\ell)}
    _, val_map, mlp_in_map, mlp_out_map = _model_maps(selected)
    layer = int(row["layer"])
    head = int(row["head"])
    m_in, m_out = _require_layer_mlp(mlp_in_map, mlp_out_map, layer)
    c2_m_in = float(m_in["spectral_norm"])
    c2_m_out = float(m_out["spectral_norm"])
    if row["matrix_type"] == "QK":
        c2_v = float(val_map[(layer, head)]["spectral_norm"])
        return float(2.0 * c2_v * c2_m_out * c2_m_in) # gamma_QK=2*C_2^(V)*C_2^(M_out)*C_2^(M_in)
    if row["matrix_type"] == "V_tilde":
        return float(c2_m_out * c2_m_in) # gamma_V=C_2^(M_out)*C_2^(M_in)
    if row["matrix_type"] == "M_in":
        return float(c2_m_out) # gamma_M_in=C_2^(M_out)
    if row["matrix_type"] == "M_out":
        return 1.0 # gamma_M_out=1
    raise ValueError(f"Unexpected proxy matrix_type={row['matrix_type']!r}")


def contribution_curve_table(
    singular_values: pd.DataFrame,
    matrix_summary: pd.DataFrame,
    L_phi: float = 1.13,
    p_grid_m: int | None = None,
    rank_tol: float = 1e-10,
) -> pd.DataFrame:
    # Evaluate the per-matrix contribution on the full post hoc grid P_m={0,1/m,...,2}
    rows: list[dict] = []
    sv_groups = {key: group for key, group in singular_values.groupby(IDENT_COLS, sort=False)}

    selected_all = _selected_proxy_summary(matrix_summary)
    for model_id, selected_model in selected_all.groupby("model_id", sort=False):
        selected_model = selected_model.copy()
        L = int(selected_model["num_hidden_layers"].iloc[0])
        N = int(selected_model["hidden_size"].iloc[0])
        p_grid, grid_m = p_m(L=L, N=N, p_grid_m=p_grid_m)
        alpha = multihead_alpha_by_layer(selected_model, L_phi=L_phi)

        for _, summary_row in selected_model.sort_values(["layer", "head", "matrix_type"]).iterrows():
            key = tuple(
                summary_row[col] if col not in {"layer", "head"} else int(summary_row[col])
                for col in IDENT_COLS
            )
            if key not in sv_groups:
                raise ValueError(f"Missing singular values for key={key}")

            svals = sv_groups[key].sort_values("rank_index")["sigma"].to_numpy(dtype=float)
            gamma = gamma_for_row(summary_row, selected_model)
            alpha_mh = alpha[int(summary_row["layer"])]
            scale = gamma * alpha_mh * L

            base_row = {col: summary_row[col] for col in IDENT_COLS}
            base_row.update(
                {
                    "hidden_size": N,
                    "head_dim": int(summary_row.get("head_dim", -1)),
                    "num_heads": int(summary_row["num_heads"]),
                    "num_hidden_layers": L,
                    "rows": int(summary_row["rows"]),
                    "cols": int(summary_row["cols"]),
                    "rank": int(summary_row["rank"]),
                    "spectral_norm": float(summary_row["spectral_norm"]),
                    "frobenius_norm": float(summary_row["frobenius_norm"]),
                    "nuclear_norm": float(summary_row.get("nuclear_norm", np.nan)),
                    "norm_21": float(summary_row["norm_21"]),
                    "norm_11": float(summary_row["norm_11"]),
                    "stable_rank": float(summary_row.get("stable_rank", np.nan)),
                    "alpha_mh": float(alpha_mh),
                    "gamma": float(gamma),
                    "scale_gamma_alpha_L": float(scale),
                    "p_grid_m": int(grid_m),
                }
            )

            for p in p_grid:
                rows.append(
                    {
                        **base_row,
                        "p": float(p),
                        "contribution": schatten_contribution_from_svals(
                            svals,
                            p=float(p),
                            N=N,
                            scale=scale,
                            rank_tol=rank_tol,
                        ),
                    }
                )

    return pd.DataFrame(rows)


def aggregate_contribution_curve(
    contribution_table: pd.DataFrame,
    agg: str = "sum",
) -> pd.DataFrame:
    supported = {"sum", "mean", "median", "max"}
    agg = str(agg).strip().lower()
    if agg not in supported:
        allowed = ", ".join(sorted(supported))
        raise ValueError(f"Unsupported aggregation '{agg}'. Choose one of: {allowed}")

    grouped = contribution_table.groupby(
        ["model_id", "model_slug", "hidden_size", "num_hidden_layers", "matrix_type", "p", "p_grid_m"],
        as_index=False,
    )

    out = grouped.agg(
        contribution=("contribution", agg),
        num_matrices=("contribution", "size"),
    )
    out["aggregation"] = agg
    return out


def optimize_p_table(
    singular_values: pd.DataFrame,
    matrix_summary: pd.DataFrame,
    L_phi: float = 1.13,
    p_grid_m: int | None = None,
    rank_tol: float = 1e-10,
) -> pd.DataFrame:
    # Optimize the Schatten index separately for every proxy matrix
    rows: list[dict] = []
    sv_groups = {key: group for key, group in singular_values.groupby(IDENT_COLS, sort=False)}

    selected_all = _selected_proxy_summary(matrix_summary)
    for model_id, selected_model in selected_all.groupby("model_id", sort=False):
        selected_model = selected_model.copy()
        L = int(selected_model["num_hidden_layers"].iloc[0])
        N = int(selected_model["hidden_size"].iloc[0])
        p_grid, grid_m = p_m(L=L, N=N, p_grid_m=p_grid_m)
        alpha = multihead_alpha_by_layer(selected_model, L_phi=L_phi)
        for _, summary_row in selected_model.sort_values(["layer", "head", "matrix_type"]).iterrows():
            key = tuple(summary_row[col] if col not in {"layer", "head"} else int(summary_row[col]) for col in IDENT_COLS)
            if key not in sv_groups:
                raise ValueError(f"Missing singular values for key={key}")
            svals = sv_groups[key].sort_values("rank_index")["sigma"].to_numpy(dtype=float)
            gamma = gamma_for_row(summary_row, selected_model)
            alpha_mh = alpha[int(summary_row["layer"])]
            scale = gamma * alpha_mh * L
            opt = optimize_p_for_svals(svals, N=N, scale=scale, p_grid=p_grid, rank_tol=rank_tol)
            row = {col: summary_row[col] for col in IDENT_COLS}
            row.update({
                "hidden_size": N,
                "head_dim": int(summary_row.get("head_dim", -1)),
                "num_heads": int(summary_row["num_heads"]),
                "num_hidden_layers": L,
                "rows": int(summary_row["rows"]),
                "cols": int(summary_row["cols"]),
                "rank": int(summary_row["rank"]),
                "spectral_norm": float(summary_row["spectral_norm"]),
                "frobenius_norm": float(summary_row["frobenius_norm"]),
                "nuclear_norm": float(summary_row.get("nuclear_norm", np.nan)),
                "norm_21": float(summary_row["norm_21"]),
                "norm_11": float(summary_row["norm_11"]),
                "stable_rank": float(summary_row.get("stable_rank", np.nan)),
                "alpha_mh": float(alpha_mh),
                "gamma": float(gamma),
                "scale_gamma_alpha_L": float(scale),
                "p_grid_m": int(grid_m),
                **opt, # Add p_opt, min_contribution
            })
            rows.append(row)
    return pd.DataFrame(rows)


def ours_posthoc_proxy(opt_table: pd.DataFrame, params: BoundParams) -> float:
    leading = float(np.sum(opt_table["min_contribution"].to_numpy(dtype=float)))
    b_ours = leading + np.sqrt(params.L)
    return float(b_ours)


def edelman_multihead_proxy(matrix_summary: pd.DataFrame, params: BoundParams) -> float:
    selected = _selected_proxy_summary(matrix_summary)
    if selected["model_id"].nunique() != 1:
        raise ValueError("edelman_multihead_proxy expects rows from a single model")
    alpha = multihead_alpha_by_layer(selected, L_phi=params.L_phi)
    qk_map, val_map, mlp_in_map, mlp_out_map = _model_maps(selected)
    num_heads = int(selected["num_heads"].dropna().iloc[0])
    L = int(selected["num_hidden_layers"].dropna().iloc[0])
    xi_sum = 0.0
    layer_rows: list[dict] = []
    for ell in range(L):
        m_in, m_out = _require_layer_mlp(mlp_in_map, mlp_out_map, ell)
        c2_m_in = float(m_in["spectral_norm"])
        c2_m_out = float(m_out["spectral_norm"])
        xi = 0.0
        for h in range(num_heads):
            qk = qk_map[(ell, h)]
            val = val_map[(ell, h)]
            xi += (c2_m_out * c2_m_in * float(val["spectral_norm"]) * float(qk["norm_21"])) ** (2.0 / 3.0)
            xi += (c2_m_out * c2_m_in * float(val["norm_21"])) ** (2.0 / 3.0)
        xi += (c2_m_out * float(m_in["norm_21"])) ** (2.0 / 3.0)
        xi += float(m_out["norm_21"]) ** (2.0 / 3.0)
        weighted = (float(alpha[ell]) ** (2.0 / 3.0)) * xi
        xi_sum += weighted
        layer_rows.append({"layer": ell, "alpha_mh": float(alpha[ell]), "xi_mh": float(xi), "weighted_xi": float(weighted)})
    b_edelman = (1.0 + xi_sum) ** 1.5
    return float(b_edelman)


def observed_bound_table(
    opt_table: pd.DataFrame,
    matrix_summary: pd.DataFrame,
    params_by_model: dict[str, BoundParams],
) -> pd.DataFrame:
    rows: list[dict] = []
    for model_id, opt_model in opt_table.groupby("model_id", sort=False):
        params = params_by_model[model_id]
        summary_model = matrix_summary[matrix_summary["model_id"] == model_id]
        ours = ours_posthoc_proxy(opt_model, params)
        rows.append({"model_id": model_id, "bound": "ours", "value": ours})
        edelman = edelman_multihead_proxy(summary_model, params)
        rows.append({"model_id": model_id, "bound": "edelman", "value": edelman})
    return pd.DataFrame(rows)

