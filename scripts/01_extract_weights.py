from __future__ import annotations

import argparse

import pandas as pd

from bert_spectrum.config import ensure_output_dirs, load_config
from bert_spectrum.extract import extract_matrices_for_model
from bert_spectrum.utils import save_csv_and_jsonl, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract headwise BERT attention proxies and separate MLP matrices.")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--model-id", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    set_seed(int(cfg.get("seed", 42)))
    dirs = ensure_output_dirs(cfg)
    model_ids = [args.model_id] if args.model_id else list(cfg.get("model_ids", [cfg["primary_model_id"]]))

    frames = []
    for model_id in model_ids:
        # Attention matrices are extracted per head; MLP input/output matrices stay separate
        frame = extract_matrices_for_model(
            model_id=model_id,
            out_dir=dirs["matrices"],
        )
        frames.append(frame)
    matrix_index = pd.concat(frames, ignore_index=True)
    save_csv_and_jsonl(matrix_index, dirs["tables"] / "matrix_index.csv")
    print(f"Saved {len(matrix_index)} headwise/layerwise proxy matrices to {dirs['matrices']}")


if __name__ == "__main__":
    main()
