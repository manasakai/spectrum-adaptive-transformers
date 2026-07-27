from __future__ import annotations

import numpy as np


# singular values
def singular_values(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix)
    return np.linalg.svd(matrix, compute_uv=False)


# numerical rank
def numerical_rank(svals: np.ndarray, tol: float = 1e-10) -> int:
    svals = np.asarray(svals, dtype=float)
    if svals.size == 0:
        return 0
    threshold = tol * max(svals.shape[0], 1) * max(float(svals[0]), 1.0)
    return int(np.sum(svals > threshold))


# Schatten-p norm powered by p
def schatten_p_power_from_svals(svals: np.ndarray, p: float, tol: float = 1e-10) -> float:
    svals = np.asarray(svals, dtype=float)
    if p == 0:
        return float(numerical_rank(svals, tol=tol))
    return float(np.sum(np.power(np.maximum(svals, 0.0), p)))


# (2,1) norm
def norm_21(matrix: np.ndarray) -> float:
    matrix = np.asarray(matrix, dtype=float)
    return float(np.sum(np.linalg.norm(matrix, ord=2, axis=0)))


# (1,1) norm
def norm_11(matrix: np.ndarray) -> float:
    matrix = np.asarray(matrix, dtype=float)
    return float(np.sum(np.abs(matrix)))


# Frobenius norm
def norm_fro(matrix: np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(matrix, dtype=float), ord="fro"))


# spectral norm
def norm_spectral_from_svals(svals: np.ndarray) -> float:
    svals = np.asarray(svals, dtype=float)
    return float(svals[0]) if svals.size else 0.0


# estimate the polynomial decay order of singular values
def stable_log_slope(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    keep = (x > 0) & (y > 0) & np.isfinite(x) & np.isfinite(y)
    if int(np.sum(keep)) < 2:
        return float("nan"), float("nan")
    slope, intercept = np.polyfit(np.log(x[keep]), np.log(y[keep]), deg=1)
    return float(slope), float(intercept)
