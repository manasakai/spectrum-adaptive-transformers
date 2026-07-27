# BERT Miniatures spectrum experiments

This repository runs the experiments for spectrum-adaptive generalization bounds on the publicly released BERT Miniatures checkpoints of Turc et al. (2019).

**Reference:** Turc et al. (2019). Well-Read Students Learn Better: On the Importance of Pre-training Compact Models. *arXiv preprint arXiv:1908.08962.*.

The theoretical model in the paper is single-head, while BERT is multi-head.  The scripts therefore do **not** treat the full QK or V matrices in a layer as one object.  Instead, for every layer `ell` and head `h`, they extract

- `QK`: the head-wise query-key proxy `W_Q^{(ell,h)T} W_K^{(ell,h)}`;
- `V_tilde`: the value-output proxy `W_V^{(ell,h)} W_O^{(ell,h)}`.

For the feed-forward sublayer, BERT applies `X -> GELU(X W_in^{(ell)} + b) W_out^{(ell)}`, so the extractor saves

- `M_in`: the layer-wise feed-forward input matrix `W_in^{(ell)}`;
- `M_out`: the layer-wise feed-forward output matrix `W_out^{(ell)}`.

The Schatten index is optimized separately for every saved `QK`, `V_tilde`, `M_in`, and `M_out` matrix on the grid `P_m={0,1/m,...,2}`, with `m=ceil(L+log(N))` by default.

## Quick start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .

bash reproduce.sh
```

This runs all scripts in the order used to generate the matrices, tables, and figures.
The equivalent manual commands are:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .

python scripts/01_extract_weights.py --config configs/bert_miniature.yaml
python scripts/02_plot_spectral_decay.py --config configs/bert_miniature.yaml --force
python scripts/03_optimize_schatten.py --config configs/bert_miniature.yaml
python scripts/04_compare_bounds.py --config configs/bert_miniature.yaml
python scripts/05_plot_norm_scaling.py --config configs/bert_miniature.yaml
python scripts/06_plot_p_contribution.py --config configs/bert_miniature.yaml
```

## Environment

The experiments were run with Python 3.12.13 on a standard laptop.
The initial weight-extraction step, `scripts/01_extract_weights.py`, may take longer than the subsequent analysis and plotting steps because it downloads and processes the BERT Miniatures checkpoints.
Once the extracted matrices have been generated, they can be reused, and scripts `02_plot_spectral_decay.py`--`06_plot_p_contribution.py` typically complete within a few minutes.


The scripts require an internet connection on the first run because the public BERT Miniatures checkpoints are downloaded from Hugging Face.
Subsequent runs may reuse the local Hugging Face cache　and the matrices stored under `outputs/matrices/`.
The generated matrices, tables, and figures are written under `outputs/`.

## Outputs

- `outputs/matrices/`: head-wise and layer-wise NumPy matrices.
- `outputs/tables/matrix_index.csv`: one row per extracted matrix, including `layer`, `head`, and `matrix_type`.
- `outputs/tables/singular_values.csv`: singular values for each matrix.
- `outputs/tables/matrix_summary.csv`: spectral norm, Frobenius norm, `(2,1)` norm, `(1,1)` norm, rank, and stable rank.
- `outputs/tables/schatten_opt.csv`: optimized post hoc Schatten index and contribution for each matrix.
- `outputs/tables/observed_bounds.csv`: observed proxies for our bound and the Edelman-style norm proxy.
- `outputs/tables/p_contribution_raw.csv`: proxy contribution of each matrix when Schatten indices vary.
- `outputs/figures/`: PDF plots.

## Notes on matrix orientation

Hugging Face stores `nn.Linear.weight` as `(out_features, in_features)`, while the formulas use row-vector multiplication.  The extractor therefore saves value and MLP matrices in row-vector orientation.  For a head with PyTorch query/key/value slices `W_Q`, `W_K`, `W_V`, the saved proxies are

```text
QK      = W_Q.T @ W_K
V_tilde = W_V.T @ W_O_head
```

where `W_O_head` is the corresponding head slice of the attention output projection in row-vector orientation.  For the feed-forward sublayer, the saved row-vector matrices are

```text
M_in  = W_in.T
M_out = W_out.T
```

and all bound/proxy scripts use `M_in` and `M_out` separately.
