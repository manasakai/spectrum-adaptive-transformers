#!/usr/bin/env bash
set -euo pipefail

CONFIG="${1:-configs/bert_miniature.yaml}"

echo "[1/6] Extracting BERT Miniatures matrices"
python scripts/01_extract_weights.py --config "${CONFIG}"

echo "[2/6] Computing singular values and plotting spectral decay"
python scripts/02_plot_spectral_decay.py --config "${CONFIG}" --force

echo "[3/6] Optimizing post hoc Schatten indices"
python scripts/03_optimize_schatten.py --config "${CONFIG}"

echo "[4/6] Comparing observed bound proxies"
python scripts/04_compare_bounds.py --config "${CONFIG}"

echo "[5/6] Plotting norm scaling"
python scripts/05_plot_norm_scaling.py --config "${CONFIG}"

echo "[6/6] Plotting p-contribution curves"
python scripts/06_plot_p_contribution.py --config "${CONFIG}"

echo "Done. Outputs were written under outputs/."