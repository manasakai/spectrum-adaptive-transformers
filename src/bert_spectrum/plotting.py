from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd

# The proxy matrices used by the experiment
BERT_MATRIX_TYPES = ("QK", "V_tilde", "M_in", "M_out")
BERT_MATRIX_TYPE_SET = set(BERT_MATRIX_TYPES)

MATRIX_COLORS = {
    "QK": "tomato",
    "V_tilde": "royalblue",
    "M_in": "darkolivegreen",
    "M_out": "darkseagreen",
}
BOUND_COLORS = {
    "ours": "orange",
    "edelman": "slateblue",
}
PRETTY_NAMES = {
    "QK": "QK head",
    "V_tilde": r"$\tilde{V}=VW^{O}$ head",
    "M_in": r"$M^{in}$",
    "M_out": r"$M^{out}$",
    "ours": "Ours",
    "edelman": "Edelman et al.",
}
PRETTY_NORM_NAMES = {
    "norm_21": "(2,1) norm",
    "norm_11": "(1,1) norm",
    "spectral_norm": "Spectral norm",
}

FIGSIZE = (8.0, 5.2)
TITLE_FONTSIZE = 26
LABEL_FONTSIZE = 22
TICK_FONTSIZE = 18
LEGEND_FONTSIZE = 18
LINEWIDTH = 1.8
MARKERSIZE = 6.0

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman"],
        "mathtext.fontset": "stix",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,

        "font.size": 13,
        "axes.titlesize": TITLE_FONTSIZE,
        "axes.labelsize": LABEL_FONTSIZE,
        "xtick.labelsize": TICK_FONTSIZE,
        "ytick.labelsize": TICK_FONTSIZE,
        "legend.fontsize": LEGEND_FONTSIZE,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)

class UnsupportedMatrixTypeError(ValueError):
    """Raised when a plot asks for matrices outside the requirements."""


def _validate_matrix_types(matrix_types: Iterable[str]) -> list[str]:
    requested = [str(x).strip() for x in matrix_types if str(x).strip()]
    unsupported = sorted(set(requested) - BERT_MATRIX_TYPE_SET)
    if unsupported:
        allowed = ", ".join(BERT_MATRIX_TYPES)
        bad = ", ".join(unsupported)
        raise UnsupportedMatrixTypeError(
            f"Unsupported matrix type(s): {bad}. "
            f"Plots only support: {allowed}. "
            "Regenerate the tables if they still contain Q, K, M, M_product, O, or V_raw."
        )
    return requested


def _BERT_proxy_rows(df: pd.DataFrame, matrix_types: Iterable[str] | None = None) -> pd.DataFrame:
    # Keep only rows used by the proxy plots
    # Old cached tables may still contain diagnostic matrices
    # The plotting layer ignores them, while explicit requests for unsupported matrix types fail fast
    requested = _validate_matrix_types(matrix_types or BERT_MATRIX_TYPES)
    out = df.copy()
    if "use_in_proxy" in out.columns:
        out = out[out["use_in_proxy"].astype(str).str.lower().isin(["true", "1", "yes"])]
    return out[out["matrix_type"].astype(str).isin(requested)].copy()


def save_current_figure(path_base: str | Path, dpi: int = 300) -> None:
    path_base = Path(path_base)
    path_base.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path_base.with_suffix(".pdf"), bbox_inches="tight")
    # plt.savefig(path_base.with_suffix(".png"), bbox_inches="tight", dpi=dpi)
    plt.close()


def _unique_legend(
    ax: plt.Axes,
    fontsize: int = LEGEND_FONTSIZE,
    ncol: int = 2,
    loc: str = "best",
    bbox_to_anchor=None
) -> None:
    handles, labels = ax.get_legend_handles_labels()
    seen: set[str] = set()
    new_handles = []
    new_labels = []
    for handle, label in zip(handles, labels):
        if label.startswith("_") or label in seen:
            continue
        seen.add(label)
        new_handles.append(handle)
        new_labels.append(label)
    if new_handles:
        ax.legend(
            new_handles,
            new_labels,
            fontsize=fontsize,
            ncol=ncol,
            frameon=False,
            loc=loc,
            bbox_to_anchor=bbox_to_anchor
        )


def plot_spectral_decay(
    singular_values: pd.DataFrame,
    output_base: str | Path,
    model_id: str | None = None,
    matrix_types: Iterable[str] | None = None,
    layers: Iterable[int] | None = None,
    normalize: bool = True,
    zero_tol: float = 1e-10,
) -> None:
    df = _BERT_proxy_rows(singular_values, matrix_types=matrix_types)
    if model_id is not None:
        df = df[df["model_id"] == model_id]
    if layers is not None:
        df = df[df["layer"].isin(list(layers))]
    if df.empty:
        raise ValueError("No singular values left after filtering.")

    ycol = "sigma_normalized" if normalize else "sigma"
    fig, ax = plt.subplots(figsize=FIGSIZE)
    min_layer = int(df["layer"].min())
    has_zero_values = False
    for (mtype, layer, head), g in df.groupby(["matrix_type", "layer", "head"], sort=True):
        g = g.sort_values("rank_index")
        color = MATRIX_COLORS[mtype]
        label = PRETTY_NAMES[mtype]
        if int(head) >= 0:
            # QK and V_tilde are headwise; show all heads lightly and label once.
            label = label if int(layer) == min_layer and int(head) == 0 else f"_{label}"
        else:
            # M_in and M_out are layerwise matrices.
            label = label if int(layer) == min_layer else f"_{label}"
        y = g[ycol].mask(g[ycol].abs() < zero_tol, 0.0)
        has_zero_values = has_zero_values or bool((y == 0.0).any())
        ax.plot(g["rank_index"], y, linewidth=LINEWIDTH, color=color, alpha=0.9, label=label)

    ax.set_xlabel("Singular value index")
    ax.set_ylabel("Normalized singular value" if normalize else "Singular value")
    title = "Spectral decay of BERT weights"
    ax.set_title(title, pad=10)
    ax.grid(True, which="major", linewidth=0.5, alpha=0.25)
    ax.grid(True, which="minor", linewidth=0.25, alpha=0.12)
    ax.tick_params(axis="both", which="major", labelsize=TICK_FONTSIZE)
    _unique_legend(ax, fontsize=LEGEND_FONTSIZE, ncol=1)
    save_current_figure(output_base)


def plot_observed_bound_grid(
    observed: pd.DataFrame,
    output_base: str | Path,
    x_axis: str = "hidden_size",
    group_axis: str | None = None,
    bounds: Iterable[str] = ("ours", "edelman"),
    yscale: str = "log",
) -> None:
    df = observed[observed["bound"].isin(list(bounds))].copy()

    if group_axis is None:
        if x_axis == "hidden_size":
            group_axis = "num_hidden_layers"
        elif x_axis == "num_hidden_layers":
            group_axis = "hidden_size"

    required_cols = [x_axis, "bound", "value"]
    if group_axis is not None:
        required_cols.append(group_axis)

    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Observed bound table is missing required columns: {missing}")

    if df.empty:
        raise ValueError("No rows left for observed bound plot.")

    fig, ax = plt.subplots(figsize=FIGSIZE)

    normalize_at = {
        "hidden_size": 128,
        "num_hidden_layers": 2,
    }.get(x_axis)

    group_markers = ["o", "s", "^", "D", "v", "P", "X"]

    if group_axis is None:
        grouped = [(None, df)]
        marker_by_group = {None: "o"}
    else:
        grouped = list(df.groupby(group_axis, sort=True))
        marker_by_group = {
            fixed_value: group_markers[i % len(group_markers)]
            for i, (fixed_value, _) in enumerate(grouped)
        }

    for fixed_value, fixed_group in grouped:
        marker = marker_by_group[fixed_value]

        for bname, g in fixed_group.groupby("bound", sort=True):
            plot_df = g.groupby(x_axis, as_index=False)["value"].median().sort_values(x_axis)

            if normalize_at is not None:
                baseline = plot_df.loc[plot_df[x_axis] == normalize_at, "value"]
                if baseline.empty:
                    raise ValueError(
                        f"Cannot normalize observed bound plot: "
                        f"{x_axis}={normalize_at} is missing for bound={bname}."
                    )

                baseline_value = float(baseline.iloc[0])
                if baseline_value == 0.0:
                    raise ValueError(
                        f"Cannot normalize observed bound plot: "
                        f"{x_axis}={normalize_at} has zero value for bound={bname}."
                    )

                plot_df["value"] = plot_df["value"] / baseline_value

            ax.plot(
                plot_df[x_axis],
                plot_df["value"],
                marker=marker,
                linestyle="-",
                linewidth=LINEWIDTH,
                markersize=MARKERSIZE,
                color=BOUND_COLORS.get(bname, "darkgrey"),
                label="_nolegend_",
            )

    ax.set_yscale(yscale)

    if x_axis == "hidden_size":
        ax.set_xlabel("Hidden dimension $N$")
        title = "Scaling with hidden dimension at fixed depth"
    elif x_axis == "num_hidden_layers":
        ax.set_xlabel("Depth $L$")
        title = "Scaling with depth at fixed hidden dimension"
    else:
        ax.set_xlabel(x_axis)
        title = "Scaling of proxy generalization bounds"

    ax.set_title(title, pad=10)
    ax.grid(True, which="major", linewidth=0.5, alpha=0.25)
    ax.grid(True, which="minor", linewidth=0.25, alpha=0.12)
    ax.tick_params(axis="both", which="major", labelsize=TICK_FONTSIZE)
    ax.tick_params(axis="both", which="minor", labelsize=TICK_FONTSIZE - 1)

    bound_handles = [
        Line2D(
            [0],
            [0],
            color=BOUND_COLORS.get(bname, "darkgrey"),
            linewidth=LINEWIDTH,
            label=PRETTY_NAMES.get(bname, bname),
        )
        for bname in bounds
    ]

    common_legend_kwargs = dict(
        fontsize=LEGEND_FONTSIZE,
        title_fontsize=LEGEND_FONTSIZE * 1.1,
        frameon=False,
        loc="upper left",
        borderaxespad=0.0,
        labelspacing=0.35,
        handletextpad=0.3,
    )

    legend_bound = ax.legend(
        handles=bound_handles,
        title="Bound",
        bbox_to_anchor=(0.03, 0.97),
        handlelength=1.8,
        **common_legend_kwargs,
    )
    legend_bound._legend_box.align = "left"

    ax.add_artist(legend_bound)

    if group_axis is not None:
        fixed_name = "$L$" if group_axis == "num_hidden_layers" else "$N$"

        group_handles = [
            Line2D(
                [0],
                [0],
                color="dimgrey",
                marker=marker_by_group[fixed_value],
                linestyle="",
                markersize=MARKERSIZE,
                label=f"{fixed_name}={fixed_value}",
            )
            for fixed_value, _ in grouped
        ]

        legend_group = ax.legend(
            handles=group_handles,
            title=f"Fixed {fixed_name}",
            bbox_to_anchor=(0.35, 0.97),
            ncol=2,
            columnspacing=0.8,
            handlelength=1.0,
            **common_legend_kwargs,
        )
        legend_group._legend_box.align = "left"

    save_current_figure(output_base)


def plot_norm_scaling(summary: pd.DataFrame, output_base: str | Path, norm_name: str = "norm_21", fixed_L: int | None = 2) -> None:
    df = _BERT_proxy_rows(summary)

    if fixed_L is not None:
        df = df[df["num_hidden_layers"] == fixed_L]

    if df.empty:
        raise ValueError("No rows left for norm scaling plot.")

    fig, ax = plt.subplots(figsize=FIGSIZE)

    min_layer = int(df["layer"].min())

    for (mtype, layer, head), g in df.groupby(["matrix_type", "layer", "head"], sort=True):
        g = g.sort_values("hidden_size")

        label = PRETTY_NAMES[mtype]
        if int(head) >= 0:
            label = label if int(layer) == min_layer and int(head) == 0 else f"_{label}"
        else:
            label = label if int(layer) == min_layer else f"_{label}"

        ax.plot(
            g["hidden_size"],
            g[norm_name],
            marker="o",
            linewidth=LINEWIDTH,
            markersize=MARKERSIZE,
            color=MATRIX_COLORS[mtype],
            alpha=0.9,
            label=label,
        )

    pretty_norm_name = PRETTY_NORM_NAMES.get(norm_name, norm_name)

    ax.set_xlabel("Hidden dimension $N$")

    title = f"{pretty_norm_name} of matrices"
    if fixed_L is not None:
        title += f" with $L$={fixed_L}"
    ax.set_title(title, pad=10)

    ax.grid(True, which="major", linewidth=0.5, alpha=0.25)
    ax.grid(True, which="minor", linewidth=0.25, alpha=0.12)
    ax.tick_params(axis="both", which="major", labelsize=TICK_FONTSIZE)
    _unique_legend(ax, fontsize=LEGEND_FONTSIZE, ncol=1)
    save_current_figure(output_base)


def plot_p_contribution_by_head(
    contribution_table: pd.DataFrame,
    output_base: str | Path,
    model_id: str | None = None,
    matrix_types: Iterable[str] | None = None,
    layers: Iterable[int] | None = None,
    yscale: str = "linear",
    normalize_at_p0: bool = False,
) -> None:
    df = _BERT_proxy_rows(contribution_table, matrix_types=matrix_types)
    if model_id is not None:
        df = df[df["model_id"] == model_id]
    if layers is not None:
        df = df[df["layer"].isin(list(layers))]
    if df.empty:
        raise ValueError("No contribution-curve rows left after filtering.")

    if normalize_at_p0:
        base = (
            df[df["p"] == 0.0][["model_id", "layer", "head", "matrix_type", "contribution"]]
            .rename(columns={"contribution": "contribution_p0"})
            .copy()
        )
        df = df.merge(base, on=["model_id", "layer", "head", "matrix_type"], how="left")
        df["contribution"] = df["contribution"] / df["contribution_p0"]

    fig, ax = plt.subplots(figsize=FIGSIZE)
    min_layer = int(df["layer"].min())

    for (mtype, layer, head), g in df.groupby(["matrix_type", "layer", "head"], sort=True):
        g = g.sort_values("p")
        color = MATRIX_COLORS[mtype]
        label = PRETTY_NAMES[mtype]

        if int(head) >= 0:
            # Headwise and layerwise QK, V_tilde
            label = label if int(layer) == min_layer and int(head) == 0 else f"_{label}"
        else:
            # Layerwise M_in, M_out
            label = label if int(layer) == min_layer else f"_{label}"

        ax.plot(
            g["p"],
            g["contribution"],
            linewidth=LINEWIDTH,
            color=color,
            alpha=0.35 if int(head) >= 0 else 0.9,
            label=label,
        )

    ax.set_xlabel("Schatten index $p$")
    ax.set_title("Proxy contribution by head/layer", pad=10)

    if yscale != "linear":
        ax.set_yscale(yscale)

    ax.grid(True, which="major", linewidth=0.5, alpha=0.25)
    ax.grid(True, which="minor", linewidth=0.25, alpha=0.12)
    ax.tick_params(axis="both", which="major", labelsize=TICK_FONTSIZE)
    _unique_legend(ax, fontsize=LEGEND_FONTSIZE, ncol=1)
    save_current_figure(output_base)
