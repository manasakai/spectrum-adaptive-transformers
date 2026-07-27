from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

from .norms import norm_11, norm_21, norm_fro, numerical_rank


def svdvals(matrix: np.ndarray, device: str = "cpu", dtype: str = "float64") -> np.ndarray:
    torch_dtype = torch.float64 if dtype == "float64" else torch.float32
    tensor = torch.as_tensor(matrix, dtype=torch_dtype, device=device)
    with torch.no_grad():
        svals = torch.linalg.svdvals(tensor).detach().cpu().numpy()
    return np.asarray(svals, dtype=float)


def compute_spectrum_table(
    matrix_index: pd.DataFrame,
    device: str = "cpu",
    dtype: str = "float64",
    rank_tol: float = 1e-10,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    # Compute singular values and summary norms for every saved matrix
    sv_rows: list[dict] = []
    summary_rows: list[dict] = []
    for _, row in tqdm(matrix_index.iterrows(), total=len(matrix_index), desc="SVD"):
        matrix = np.load(row["path"])
        svals = svdvals(matrix, device=device, dtype=dtype)
        total_energy = float(np.sum(svals**2))
        cumulative = np.cumsum(svals**2) / total_energy if total_energy > 0 else np.zeros_like(svals)
        sigma0 = float(svals[0]) if svals.size else 0.0
        use_in_proxy = row.get("use_in_proxy", True)
        if not isinstance(use_in_proxy, bool):
            use_in_proxy = str(use_in_proxy).strip().lower() in {"true", "1", "yes"}
        common = {
            "model_id": row["model_id"],
            "model_slug": row["model_slug"],
            "layer": int(row["layer"]),
            "head": int(row.get("head", -1)),
            "matrix_type": row["matrix_type"],
            "use_in_proxy": bool(use_in_proxy),
        }
        for rank_idx, sigma in enumerate(svals, start=1):
            sv_rows.append({
                **common,
                "rank_index": rank_idx,
                "sigma": float(sigma),
                "sigma_normalized": float(sigma / sigma0) if sigma0 > 0 else 0.0,
                "energy_fraction": float((sigma**2) / total_energy) if total_energy > 0 else 0.0,
                "cumulative_energy": float(cumulative[rank_idx - 1]) if svals.size else 0.0,
            })
        summary_rows.append({
            **common,
            "rows": int(row["rows"]),
            "cols": int(row["cols"]),
            "hidden_size": int(row["hidden_size"]),
            "head_dim": int(row.get("head_dim", -1)),
            "intermediate_size": int(row["intermediate_size"]),
            "num_heads": int(row["num_heads"]),
            "num_hidden_layers": int(row["num_hidden_layers"]),
            "rank": numerical_rank(svals, tol=rank_tol),
            "spectral_norm": float(svals[0]) if svals.size else 0.0,
            "frobenius_norm": norm_fro(matrix),
            "nuclear_norm": float(np.sum(svals)),
            "norm_21": norm_21(matrix),
            "norm_11": norm_11(matrix),
            "stable_rank": float(np.sum(svals**2) / (svals[0] ** 2)) if svals.size and svals[0] > 0 else 0.0,
        })
    return pd.DataFrame(sv_rows), pd.DataFrame(summary_rows)


def read_singular_value_table(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path)
