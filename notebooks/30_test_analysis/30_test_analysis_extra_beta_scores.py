from __future__ import annotations
import os, time
from typing import List
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.auto import tqdm
import wandb

import src.analyze
import src.globals
import src.plot
import matplotlib as mpl
mpl.rcParams.update({"text.usetex": False})
# ────────────────────────────────────────────────────────────────────────────────
# Parameters (edit freely)
# ────────────────────────────────────────────────────────────────────────────────
WAND_USERNAME          = "julian-yuya-caspary-university-of-mannheim"
EVAL_PROJECT           = "universal-vlm-jailbreak-eval"
ATTACK_PROJECT         = "universal-vlm-jailbreak"

# IRIS 123 Intern
SWEEP_IDS              = [
    "3a0ltqro",
    "0o6ovw5r",
    "2n9737ha",
    "cngtdiqz",

]   

# IRIS 123 Mini
# SWEEP_IDS              = [
#     "30j5x091",
#     "oqestcn4",
#     "6976fsb1",
#     "p3s80qcr"
# ]   

# # IRIS 123 Both
# SWEEP_IDS              = [
#     "3a0ltqro",
#     "30j5x091",
#     "0o6ovw5r",
#     "oqestcn4",
#     "2n9737ha",
#     "6976fsb1",
#     "cngtdiqz",
#     "p3s80qcr"

# ]   

     # ← put your sweep(s) here
REFRESH_WANDB_DOWNLOAD = True                    # force re-download?
FINISHED_ONLY          = True                    # ignore running/failed runs?

# Metrics to plot
EVAL_SCORE_METRICS = {
    #"loss/score_model=llamaguard2": "LlamaGuard‑2 score",
    "loss/score_model=strongreject": "Strong‑Reject score",
}

ATTACK_LOSS_METRICS = {
    "loss/avg": "Average loss",
    "loss_reg_unweighted/InternVL2-8B": "Regularization factor",
    "loss_reg/InternVL2-8B": "Weighted regularization factor",
    "loss_ce/InternVL2-8B": "Cross Entropy-Loss",
    # "loss_reg_unweighted/MiniCPM-V-2_6": "Regularization factor",
    # "loss_reg/MiniCPM-V-2_6": "Weighted regularization factor",
    # "loss_ce/MiniCPM-V-2_6": "Cross Entropy-Loss"
    # add more if you care
}

# Attack‑config keys you care about – order defines legend text
ATTACK_CONFIG_KEYS = [
    "image_initialization",
    "lr",
    "epsilon",
    "attack_dataset",
    "beta",
]

# Max x‑axis (optimizer steps) for nicer plots
MAX_STEPS = 2000

# ────────────────────────────────────────────────────────────────────────────────
# Directories –> group results under a sweep‑specific sub‑folder
# ────────────────────────────────────────────────────────────────────────────────
NOTEBOOK_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR, RESULTS_DIR = src.analyze.setup_notebook_dir(
    notebook_dir=NOTEBOOK_DIR,
    refresh=False,
)

SWEEP_SUBDIR = "-".join(SWEEP_IDS) or "no‑sweep‑id"
LOSS_CURVES_DIR  = os.path.join(RESULTS_DIR, SWEEP_SUBDIR, "loss_curves")
SCORE_CURVES_DIR = os.path.join(RESULTS_DIR, SWEEP_SUBDIR, "eval_score_curves")
for _d in (LOSS_CURVES_DIR, SCORE_CURVES_DIR):
    os.makedirs(_d, exist_ok=True)

# ────────────────────────────────────────────────────────────────────────────────
# Helper: light wrapper with local caching for WandB histories
# ────────────────────────────────────────────────────────────────────────────────

def fetch_histories_by_run_ids(
    project: str,
    run_ids: List[str],
    wandb_username: str,
    samples: int = 1_000_000,
    refresh: bool = False,
    filetype: str = "csv",
    data_dir: str = ".",
) -> pd.DataFrame:
    """Download & cache histories for several runs, return concatenated DF."""
    assert filetype in {"csv", "feather", "parquet"}
    cache_name = f"runs={'-'.join(run_ids[:10])}_histories.{filetype}"
    cache_path = os.path.join(data_dir, cache_name)

    if not refresh and os.path.isfile(cache_path.replace(filetype, "csv")):
        return pd.read_csv(cache_path.replace(filetype, "csv"))

    api = wandb.Api(timeout=600)
    if wandb_username is None:
        wandb_username = api.viewer.username

    dfs: list[pd.DataFrame] = []
    for rid in tqdm(run_ids, desc="download histories"):
        try:
            run = api.run(f"{wandb_username}/{project}/{rid}")
        except Exception as e:
            print(f"⚠️  skipping {rid}: {e}")
            continue

        hist = None
        for _ in range(5):
            try:
                hist = run.history(samples=samples)
                break
            except wandb.errors.CommError:
                time.sleep(3)

        if hist is None or hist.empty:
            continue

        # strip gigantic generation columns
        hist.drop(columns=[c for c in hist.columns if "generation" in c], inplace=True, errors="ignore")
        hist["attack_run_id" if project == ATTACK_PROJECT else "eval_run_id"] = rid
        dfs.append(hist)

    if not dfs:
        raise RuntimeError("No histories fetched from WandB.")

    df = pd.concat(dfs, ignore_index=True)
    # cache CSV always (universally readable), others best‑effort
    df.to_csv(cache_path.replace(filetype, "csv"), index=False)
    try:
        df.to_feather(cache_path.replace(filetype, "feather"))
    except Exception:
        pass
    try:
        df.to_parquet(cache_path.replace(filetype, "parquet"), index=False)
    except Exception:
        pass

    return df

# ────────────────────────────────────────────────────────────────────────────────
# 1) Download EVAL runs (contain evaluation scores)
# ────────────────────────────────────────────────────────────────────────────────

# Configs (so we know which attack run each eval run refers to)
eval_cfg_df = src.analyze.download_wandb_project_runs_configs(
    wandb_project_path=EVAL_PROJECT,
    data_dir=DATA_DIR,
    sweep_ids=SWEEP_IDS,
    refresh=REFRESH_WANDB_DOWNLOAD,
    finished_only=FINISHED_ONLY,
    wandb_username=WAND_USERNAME,
    filetype="csv",
).rename(columns={"run_id": "eval_run_id", "wandb_attack_run_id": "attack_run_id"})

# Histories with the evaluation metrics
unique_eval_run_ids = eval_cfg_df["eval_run_id"].tolist()
eval_hist_df = fetch_histories_by_run_ids(
    project=EVAL_PROJECT,
    run_ids=unique_eval_run_ids,
    wandb_username=WAND_USERNAME,
    data_dir=DATA_DIR,
    refresh=REFRESH_WANDB_DOWNLOAD,
)

eval_hist_df = eval_hist_df[[
    "eval_run_id", "optimizer_step_counter_epoch", *EVAL_SCORE_METRICS.keys()
]]

# ────────────────────────────────────────────────────────────────────────────────
# 2) Download ATTACK runs – configs + histories (for loss curves)
# ────────────────────────────────────────────────────────────────────────────────

unique_attack_run_ids = eval_cfg_df["attack_run_id"].dropna().unique().tolist()
# 2‑a configs
attack_cfg_df = src.analyze.download_wandb_project_runs_configs_by_run_ids(
    wandb_project_path=ATTACK_PROJECT,
    wandb_username=WAND_USERNAME,
    data_dir=DATA_DIR,
    run_ids=unique_attack_run_ids,
    refresh=REFRESH_WANDB_DOWNLOAD,
    finished_only=FINISHED_ONLY,
    filetype="csv",
).rename(columns={"run_id": "attack_run_id"})

FIELD_SPECS = [
    ("data",               "dataset",            "attack_dataset"),
    ("image_kwargs",       "image_initialization", "image_initialization"),
    ("optimization",       "eps",               "epsilon"),
    ("optimization",       "learning_rate",     "lr"),
    ("regularization_kwargs", "beta",            "beta"),
]
for col, key, new in FIELD_SPECS:
    if col in attack_cfg_df.columns:
        attack_cfg_df = src.analyze.extract_key_value_from_df_col(
            df=attack_cfg_df, col_name=col, key_in_dict=key, new_col_name=new
        )

attack_cfg_df["image_initialization"] = attack_cfg_df["image_initialization"].map(
    src.globals.IMAGE_INITIALIZATION_TO_STRINGS_DICT
)
attack_cfg_df["attack_config_label"] = attack_cfg_df.apply(
    lambda row: ", ".join(
        f"{k.split('_')[0]}={row[k]}" for k in ATTACK_CONFIG_KEYS if k in row
    ),
    axis=1,
)

# 2‑b histories for loss curves
attack_hist_df = fetch_histories_by_run_ids(
    project=ATTACK_PROJECT,
    run_ids=unique_attack_run_ids,
    wandb_username=WAND_USERNAME,
    data_dir=DATA_DIR,
    refresh=REFRESH_WANDB_DOWNLOAD,
)

def build_attack_loss_metrics(df: pd.DataFrame) -> dict[str, str]:
    """
    Return a mapping {column_name → pretty label} for every loss-related column
    that a) exists in *df* and b) matches one of the two back-end models.

    • If the run uses *one* model → labels are short
      (e.g. 'Regularization factor').

    • If the run logs *both* models → append the model name so legend lines
      are distinguishable
      (e.g. 'Regularization factor (InternVL2-8B)').
    """
    # ------------------------------------------------------------------
    # 1)  Find every loss column we care about
    # ------------------------------------------------------------------
    pattern = re.compile(
        r"loss_(?P<kind>reg_unweighted|reg|ce)/(?P<model>InternVL2-8B|MiniCPM-V-2_6)"
    )

    columns_info = [
        (col, m["kind"], m["model"])
        for col in df.columns
        if (m := pattern.match(col))
    ]
    if not columns_info:
        raise ValueError("No recognised loss columns found in attack_hist_df")

    models_present = {model for _, _, model in columns_info}
    include_model  = len(models_present) > 1

    # ------------------------------------------------------------------
    # 2)  Human-friendly base names
    # ------------------------------------------------------------------
    BASE_LABELS = {
        "reg_unweighted": "Regularization factor",
        "reg":            "Weighted regularization factor",
        "ce":             "Cross Entropy-Loss",
    }

    # ------------------------------------------------------------------
    # 3)  Build the final dict
    # ------------------------------------------------------------------
    metrics: dict[str, str] = {
        col: (
            BASE_LABELS[kind] if not include_model
            else f"{BASE_LABELS[kind]} ({model})"
        )
        for col, kind, model in columns_info
    }

    # legacy metric that’s always present
    metrics["loss/avg"] = "Average loss"

    return metrics


# … now create the dict dynamically …
ATTACK_LOSS_METRICS = build_attack_loss_metrics(attack_hist_df)


# Keep only small subset of columns (loss metrics + step counter)
keep_cols = ["attack_run_id", "optimizer_step_counter", *ATTACK_LOSS_METRICS.keys()]
attack_hist_df = attack_hist_df[keep_cols]

# Merge metadata so we can group by hyper‑parameters
attack_hist_df = attack_hist_df.merge(
    attack_cfg_df[["attack_run_id", "attack_config_label", *ATTACK_CONFIG_KEYS]],
    on="attack_run_id",
)

# ────────────────────────────────────────────────────────────────────────────────
# 3) Build evaluation long‑form DF and merge metadata
# ────────────────────────────────────────────────────────────────────────────────

eval_hist_df = eval_hist_df.merge(
    eval_cfg_df[["eval_run_id", "attack_run_id"]], on="eval_run_id"
).merge(
    attack_cfg_df[["attack_run_id", "attack_config_label", *ATTACK_CONFIG_KEYS]],
    on="attack_run_id",
)

attack_hist_df = attack_hist_df[attack_hist_df["optimizer_step_counter"] % 10 == 0]

# ────────────────────────────────────────────────────────────────────────────────
# 5) Plot evaluation curves (both metrics) – unchanged
# ────────────────────────────────────────────────────────────────────────────────

metric_long_df = (
    eval_hist_df
    .melt(
        id_vars=["optimizer_step_counter_epoch", "attack_config_label", *ATTACK_CONFIG_KEYS],
        value_vars=list(EVAL_SCORE_METRICS.keys()),
        var_name="metric_key",
        value_name="score",
    )
)
metric_long_df["metric"] = metric_long_df["metric_key"].map(EVAL_SCORE_METRICS)

# ─────────────────────────────────────────────────────────────────────────────
# 5)  Evaluation curves grouped by β   (one panel, colour = β)
# ─────────────────────────────────────────────────────────────────────────────
if "beta" in ATTACK_CONFIG_KEYS:
    beta_df     = metric_long_df.copy()
    beta_levels = sorted(beta_df["beta"].unique())

    # ── main plot ────────────────────────────────────────────────────────────
    g = sns.relplot(
        data=beta_df,
        kind="line",
        x="optimizer_step_counter_epoch",
        y="score",
        hue="beta",
        hue_order=beta_levels,
        style="metric",
        markers=False,
        dashes=False,
        linewidth=2.0,
        palette="tab10",
        height=7,          # ↑ was 5 → now 7 in tall
        aspect=1.25,       # keeps the old ≈ 9 in width
        errorbar=None,
        legend="full",
    )

    ax = g.axes.flat[0]

    # ── manual line styling (solid thick for β=0, dashed thin otherwise) ────
    import matplotlib as mpl
    import matplotlib.lines as mlines

    palette        = sns.color_palette("tab10", n_colors=len(beta_levels))
    color_for_beta = dict(zip(beta_levels, palette))
    beta_for_color = {mpl.colors.to_hex(c): b for b, c in color_for_beta.items()}

    for ln in ax.get_lines():
        if not isinstance(ln, mlines.Line2D) or len(ln.get_xdata()) == 0:
            continue
        β = beta_for_color.get(mpl.colors.to_hex(ln.get_color()), None)
        if β == 0.0:
            ln.set_linewidth(3.5)
            ln.set_linestyle("-")
        else:
            ln.set_linewidth(1.3)
            ln.set_linestyle((0, (7, 4)))   # long dash

    # ── move legend completely outside ───────────────────────────────────────
    # make room for it
    g.fig.subplots_adjust(right=0.8)        # 80 % of fig width for axes

    leg = g._legend                           # Seaborn created this for us
    leg.set_title(r"$\beta$")
    leg.set_frame_on(False)
    leg.set_bbox_to_anchor((1.02, 0.5))     # just outside, centred vertically
    leg.set_loc("center left")

    # ── labels / limits / save ───────────────────────────────────────────────
    g.set_axis_labels("Gradient step", "Evaluation score")
    g.set(xlim=(0, MAX_STEPS), ylim=(0.0, 1.0))
    g.fig.suptitle("Evaluation scores vs steps", y=1.02)

    subdir = os.path.join(SCORE_CURVES_DIR, "beta")
    os.makedirs(subdir, exist_ok=True)
    src.plot.save_plot_with_multiple_extensions(subdir, "eval_both_metrics_vs_steps_by_beta")
    plt.close(g.fig)

    dataset_order = sorted(beta_df["attack_dataset"].unique())   # or supply your own list
DATASET_ALIAS = {
    "advbench":                                   "AdvBench",
    "advbench_intern_dir_adv_100_more_harmful":   "AdvBench Self-Labeled (InternVL2)",
    "advbench_minicpm_dir_adv_100":               "AdvBench Self-Labeled (MiniCPM-V 2.6)",
}
def pretty_ds(name: str) -> str:
    """Return display alias or fall back to a de-underscored original."""
    return DATASET_ALIAS.get(name, name.replace("_", " "))

# ----------------------------------------------------------------------
# we already have beta_levels → build colour→β lookup just once
# ----------------------------------------------------------------------
import matplotlib as mpl, matplotlib.lines as mlines
palette        = sns.color_palette("tab10", n_colors=len(beta_levels))
color_for_beta = dict(zip(beta_levels, palette))
beta_for_color = {mpl.colors.to_hex(c): b for b, c in color_for_beta.items()}

# ----------------------------------------------------------------------
# 1)  Loop over datasets and plot
# ----------------------------------------------------------------------
dataset_order = sorted(beta_df["attack_dataset"].unique())   # or your own list

for ds in dataset_order:
    ds_df = beta_df.query("attack_dataset == @ds")

    g = sns.relplot(
        data=ds_df,
        kind="line",
        x="optimizer_step_counter_epoch",
        y="score",
        hue="beta",
        hue_order=beta_levels,
        style="metric",
        markers=False,
        dashes=False,
        linewidth=2.0,
        palette="tab10",
        height=7,
        aspect=1.25,
        errorbar=None,
        legend="full",
    )

    # ── style lines: β=0 → thick solid, others → thin dashed ───────────
    for ax in g.axes.flat:                         # only one, but safe
        for ln in ax.get_lines():
            if not isinstance(ln, mlines.Line2D) or len(ln.get_xdata()) == 0:
                continue
            β = beta_for_color.get(mpl.colors.to_hex(ln.get_color()))
            if β == 0.0:
                ln.set_linewidth(3.5)
                ln.set_linestyle("-")
            else:
                ln.set_linewidth(1.3)
                ln.set_linestyle((0, (7, 4)))      # long dash

    # ── external legend exactly like the pooled plot ───────────────────
    g.fig.subplots_adjust(right=0.8)
    leg = g._legend
    leg.set_title(r"$\beta$")
    leg.set_frame_on(False)
    leg.set_bbox_to_anchor((1.02, 0.5))
    leg.set_loc("center left")

    # ── title, labels, save ─────────────────────────────────────────────
    nice_name = pretty_ds(ds)
    g.fig.suptitle(
        rf"Evaluation scores vs steps: {nice_name}",
        y=1.02,
    )
    g.set_axis_labels("Gradient step", "Evaluation score")
    g.set(xlim=(0, MAX_STEPS), ylim=(0.0, 1.0))

    subdir = os.path.join(SCORE_CURVES_DIR, "beta_by_dataset")
    os.makedirs(subdir, exist_ok=True)
    fname = f"eval_both_metrics_vs_steps_by_beta__{ds}"
    src.plot.save_plot_with_multiple_extensions(subdir, fname)

    plt.close(g.fig)


print("✅ Plots saved under:")
print(f"   • Loss curves      → {LOSS_CURVES_DIR}")
print(f"   • Evaluation curves→ {SCORE_CURVES_DIR}")
