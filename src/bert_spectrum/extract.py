from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from transformers import AutoConfig, AutoModel

from .config import model_slug


@dataclass(frozen=True)
class MatrixRecord:
    model_id: str
    model_slug: str
    layer: int # layer index starting from 0
    head: int # attention head index
    # (The special head value -1 denotes a layerwise matrix such as M_in or M_out)
    matrix_type: str # e.g. QK, V_raw, V_tilde, M_in, M_out
    path: str # path to the save .npy file
    rows: int
    cols: int
    hidden_size: int
    head_dim: int # Usually, it is hidden_size // num_heads
    intermediate_size: int # Intermediate dimension in the MLP
    num_heads: int
    num_hidden_layers: int
    use_in_proxy: bool
    note: str = ""


def _detach_weight(module: torch.nn.Module) -> np.ndarray:
    # Return a PyTorch Linear weight as a NumPy array in PyTorch's native orientation
    return module.weight.detach().cpu().to(torch.float64).numpy()


def _bert_base_module(model: torch.nn.Module) -> torch.nn.Module:
    # Extract BERT from a given model
    # Support both AutoModel and task-specific BERT wrappers
    if hasattr(model, "bert"):
        return model.bert
    return model


def _save_matrix(
    matrix: np.ndarray,
    out_dir: Path,
    model_id: str,
    layer_idx: int,
    head: int,
    matrix_type: str,
    cfg: Any, # model setting for Hugging Face
    use_in_proxy: bool,
    note: str = "",
) -> MatrixRecord:
    # Save one theoretical/proxy matrix
    # Headwise attention matrices are saved separately
    slug = model_slug(model_id)
    model_dir = out_dir / slug
    model_dir.mkdir(parents=True, exist_ok=True)
    head_part = "layer" if head < 0 else f"head_{head:02d}"
    filename = f"layer_{layer_idx:02d}_{head_part}_{matrix_type}.npy"
    path = model_dir / filename
    np.save(path, matrix)
    rows, cols = matrix.shape
    hidden_size = int(getattr(cfg, "hidden_size", rows))
    num_heads = int(getattr(cfg, "num_attention_heads", -1))
    head_dim = hidden_size // num_heads if num_heads > 0 else -1
    return MatrixRecord(
        model_id=model_id,
        model_slug=slug,
        layer=layer_idx,
        head=int(head),
        matrix_type=matrix_type,
        path=str(path),
        rows=int(rows),
        cols=int(cols),
        hidden_size=hidden_size,
        head_dim=head_dim,
        intermediate_size=int(getattr(cfg, "intermediate_size", -1)),
        num_heads=num_heads,
        num_hidden_layers=int(getattr(cfg, "num_hidden_layers", -1)),
        use_in_proxy=bool(use_in_proxy),
        note=note,
    )


def _head_slice(head: int, head_dim: int) -> slice:
    # Extract the head-specific Q/K/V weight from a single full Q/K/V matrix mixing all heads
    return slice(head * head_dim, (head + 1) * head_dim)


def extract_matrices_for_model(
    # Extract the matrices used in the proxy
    model_id: str,
    out_dir: str | Path,
) -> pd.DataFrame:
    # For each layer ell and attention head h, this function constructs W^{QK,(l,h)} = W_Q^{(l,h)T} W_K^{(l,h)} rather than a single full QK matrix mixing all heads
    # The raw value matrix is stored in row-vector orientation as W^{V,(l,h)} = W_V^{(l,h)T}

    out_dir = Path(out_dir)
    cfg = AutoConfig.from_pretrained(model_id) # Obtain the model setting from Hugging Face
    model = AutoModel.from_pretrained(model_id, config=cfg) # Obtain the pretrained model from Hugging Face
    model.eval() # Set the model as evaluation mode (no dropouts, etc.)
    bert = _bert_base_module(model) # Extract BERT from a given model
    records: list[MatrixRecord] = []

    hidden_size = int(cfg.hidden_size)
    num_heads = int(cfg.num_attention_heads)
    if hidden_size % num_heads != 0:
        raise ValueError(f"hidden_size={hidden_size} is not divisible by num_heads={num_heads}")
    head_dim = hidden_size // num_heads

    for layer_idx, layer in enumerate(bert.encoder.layer):
        # Hugging Face Linear weights have shape (out_features, in_features)
        # In row-vector notation, a Linear layer applies x @ weight.T
        q = _detach_weight(layer.attention.self.query)
        k = _detach_weight(layer.attention.self.key)
        v = _detach_weight(layer.attention.self.value)
        out = _detach_weight(layer.attention.output.dense)
        ffn_in = _detach_weight(layer.intermediate.dense)
        ffn_out = _detach_weight(layer.output.dense)

        for head in range(num_heads):
            hs = _head_slice(head, head_dim)
            q_h = q[hs, :]
            k_h = k[hs, :]
            v_h = v[hs, :]
            w_o_h = out.T[hs, :] # Contribution of this head to the output projection in row-vector orientation
            qk_h = q_h.T @ k_h
            records.append(
                _save_matrix(qk_h, out_dir, model_id, layer_idx, head, "QK", cfg, True, "headwise W_Q^T W_K proxy")
            )
            w_v_h = v_h.T
            v_tilde_h = w_v_h @ w_o_h
            records.append(
                _save_matrix(v_tilde_h, out_dir, model_id, layer_idx, head, "V_tilde", cfg, True, "headwise value-output proxy W_V W_O",)
            )

        # BERT feed-forward: X -> GELU(X W_in + b) W_out
        # The proxy evaluates W_in and W_out separately
        records.append(
            _save_matrix(ffn_in.T, out_dir, model_id, layer_idx, -1, "M_in", cfg, True, "row-vector feed-forward input matrix W_in")
        )
        records.append(
            _save_matrix(ffn_out.T, out_dir, model_id, layer_idx, -1, "M_out", cfg, True, "row-vector feed-forward output matrix W_out")
        )

    return pd.DataFrame([r.__dict__ for r in records])
