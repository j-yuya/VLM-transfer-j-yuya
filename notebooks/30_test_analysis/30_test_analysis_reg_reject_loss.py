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


# IRIS 123 Intern advbench
# SWEEP_IDS              = [
#     "3a0ltqro",
#     #"0o6ovw5r",
#     "2n9737ha",
#     #"cngtdiqz",

#]   
# IRIS 123 Intern
# SWEEP_IDS              = [
#     "3a0ltqro",
#     "0o6ovw5r",
#     "2n9737ha",
#     "cngtdiqz",

# ]   

# IRIS 123 Mini
# SWEEP_IDS              = [
#     "30j5x091",
#     "oqestcn4",
#     "6976fsb1",
#     "p3s80qcr"

# ]   

#IRIS 123 Mini advbench
SWEEP_IDS              = [
    "30j5x091",
    "6976fsb1",
]  

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


BETA_FILTER = [0.0, 0.01, 0.2, 0.5, 0.75] 
def filter_by_beta(df, column="beta"):
    """Return df unchanged if the filter is empty, else keep only chosen βs."""
    if not BETA_FILTER:                  # []  or  None  → no filtering
        return df
    return df[df[column].isin(BETA_FILTER)].copy()


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

# ────────────────────────────────────────────────────────────────────────────────
# 4) Plot LOSS curves – only param‑specific (no combined y_vs_steps)
# ────────────────────────────────────────────────────────────────────────────────

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

# ────────────────────────────────────────────────────────────────────────────────
# 6)  Curves per (attack_dataset, β)  →   loss_reg , loss_ce , Strong-Reject
# ────────────────────────────────────────────────────────────────────────────────
COMPONENT_CURVES_DIR      = os.path.join(RESULTS_DIR, SWEEP_SUBDIR, "loss_components")
STRONGREJECT_CURVES_DIR   = os.path.join(RESULTS_DIR, SWEEP_SUBDIR, "strongreject")
os.makedirs(COMPONENT_CURVES_DIR,    exist_ok=True)
os.makedirs(STRONGREJECT_CURVES_DIR, exist_ok=True)

# ------------------------------------------------------------------
# 6-a  Undo the scaling  (recorded values are β·reg  and  (1-β)·CE)
# ------------------------------------------------------------------
attack_hist_df = attack_hist_df.copy()        # keep original intact

reg_cols = [c for c in attack_hist_df.columns
            if c.startswith("loss_reg_unweighted/")]
ce_cols  = [c for c in attack_hist_df.columns
            if c.startswith("loss_ce/")]

models_reg = [c.split("/", 1)[1] for c in reg_cols]
models_ce  = [c.split("/", 1)[1] for c in ce_cols]
models     = sorted(set(models_reg) | set(models_ce))

if not models:
    raise ValueError("No loss_reg_unweighted/… or loss_ce/… columns found!")

# ─────────────────────────────────────────────────────────────────────────────
# 2)  Generate un-scaled loss columns for every model present
# ─────────────────────────────────────────────────────────────────────────────
for model in models:

    # -------- reg term: no rescaling needed --------------------------------
    src_reg = f"loss_reg_unweighted/{model}"
    if src_reg in attack_hist_df.columns:
        dst_reg = f"loss_reg_unscaled/{model}"
        attack_hist_df[dst_reg] = attack_hist_df[src_reg]

    # -------- CE term: divide by (1-β) --------------------------------------
    src_ce  = f"loss_ce/{model}"
    if src_ce in attack_hist_df.columns:
        dst_ce = f"loss_ce_unscaled/{model}"
        attack_hist_df[dst_ce] = np.where(
            attack_hist_df["beta"] < 1,
            attack_hist_df[src_ce] / (1.0 - attack_hist_df["beta"]),
            np.nan,                      # β = 1  → undefined division
        )

# ─────────────────────────────────────────────────────────────────────────────
# 3)  Back-compatibility: if there is only ONE model, expose the old
#    generic column names so the rest of the pipeline keeps working.
# ─────────────────────────────────────────────────────────────────────────────
if len(models) == 1:
    model = models[0]
    attack_hist_df["loss_reg_unscaled"] = attack_hist_df[f"loss_reg_unscaled/{model}"]
    attack_hist_df["loss_ce_unscaled"]  = attack_hist_df[f"loss_ce_unscaled/{model}"]
from seaborn import move_legend


def place_legend_top(grid, ncol=2, dy=0.02):
    """
    Push FacetGrid legend to a single horizontal row centred under the suptitle.

    Parameters
    ----------
    grid : seaborn.axisgrid.FacetGrid
    ncol : int   number of legend columns
    dy   : float vertical offset (figure-coords) – increase if legend touches title
    """
    move_legend(
        grid,
        "upper center",
        bbox_to_anchor=(0.5, 1.0 - dy),
        bbox_transform=grid.fig.transFigure,
        ncol=ncol,
        frameon=False,
        title=None,
    )

# ------------------------------------------------------------------
# 6-b  Tidy format  &  plot  (row = β , col = dataset)
# ------------------------------------------------------------------
# ── Friendly display names for every dataset we might ever see ──────────────
DATASET_ALIAS = {
    "advbench":                                   "AdvBench",
    "advbench_intern_dir_adv_100_more_harmful":   "AdvBench Self-Labeled (InternVL2)",
    "advbench_minicpm_dir_adv_100":               "AdvBench Self-Labeled (MiniCPM-V 2.6)",
}

def pretty_ds(name: str) -> str:
    """
    Return a tidy display string for a raw `attack_dataset` value.

    • If the name is in DATASET_ALIAS → use that.
    • Otherwise fall back to the original, replacing '_' with a space.
    """
    return DATASET_ALIAS.get(name, name.replace("_", " "))

loss_long = (
    attack_hist_df
    .melt(
        id_vars=["optimizer_step_counter", "attack_dataset", "beta"],
        value_vars=["loss_reg_unscaled", "loss_ce_unscaled"],
        var_name="component",
        value_name="loss",
    )
    .dropna(subset=["loss"])                               # drop undefined rows
)
loss_long  = filter_by_beta(loss_long)   
loss_long["dataset_disp"] = loss_long["attack_dataset"].apply(pretty_ds)


import pdb
pdb.set_trace()
# g = sns.relplot(
#     data=loss_long,
#     kind="line",
#     x="optimizer_step_counter",
#     y="loss",
#     hue="component",
#     col="dataset_disp",
#     row="beta",
#     facet_kws=dict(margin_titles=True),
#     linewidth=2.2,
#     height=3.8,
#     aspect=1.4,
#     errorbar=None,
#     palette="tab10",
# )
# g.set_titles(                  # keep row titles nice too
#     col_template="{col_name}",
#     row_template=r"$\beta = {row_name}$",
# )
# g.fig.subplots_adjust(top=0.73)                  # shrink facet area
# g.fig.suptitle("Loss-components vs steps   (facet: dataset × β)", y=0.98)
# place_legend_top(g, ncol=2, dy=0.08)            # drop legend just below title
# g.set_axis_labels("Gradient step", "Unscaled loss")
# g.set(xlim=(0, MAX_STEPS), ylim=(0, 10))

# g.fig.suptitle("Loss-components vs steps   (facet: dataset ⨯ β)")
# place_legend_top(g, ncol=2, dy=0.025)              # ← NEW

# src.plot.save_plot_with_multiple_extensions(
#     COMPONENT_CURVES_DIR, "loss_components_by_dataset_and_beta_ymax10"
# )
# plt.close(g.fig)

# ------------------------------------------------------------------
# 6-c  Strong-Reject score  (same facets)
# ------------------------------------------------------------------
sr_df = metric_long_df.query(
    "metric_key == 'loss/score_model=strongreject'"
).copy()

alias = {
    "advbench": "AdvBench",
    "advbench_intern_dir_adv_100_more_harmful": "Advbench Self-Labeled (InternVL2)",
}
sr_df    = filter_by_beta(sr_df)  
sr_df["dataset_disp"] = sr_df["attack_dataset"].apply(pretty_ds)


# nicer ordering: sort β numerically, datasets alphabetically
beta_order    = sorted(sr_df["beta"].dropna().unique())
dataset_order = sorted(sr_df["dataset_disp"].dropna().unique())
plt.close("all")

legend_cols = max(len(beta_order), len(dataset_order))

# g = sns.relplot(
#     data=sr_df,
#     kind="line",
#     x="optimizer_step_counter_epoch",
#     y="score",
#     hue="beta",
#     hue_order=beta_order,
#     style="dataset_disp",
#     style_order=dataset_order,
#     linewidth=2.5,
#     palette="tab10",
#     height=5,
#     aspect=1.7,
#     errorbar=None,
# )
# g.set_titles(                  # keep row titles nice too
#     col_template="{col_name}",
#     row_template=r"$\beta = {row_name}$",
# )
# g.fig.subplots_adjust(top=0.73)                    # shrink facet area
# place_legend_top(g, ncol=2, dy=0.08)            # drop legend just below title
# g.set_axis_labels("Gradient step", "Strong-Reject score")
# g.set(xlim=(0, MAX_STEPS), ylim=(0.0, 1.0))
# g.fig.suptitle("Strong-Reject vs steps   (colour = β,  linestyle = dataset)", y=1.02)
# place_legend_top(g, ncol=legend_cols, dy=0.03)     # ← NEW

# src.plot.save_plot_with_multiple_extensions(
#     STRONGREJECT_CURVES_DIR, "strongreject_singleplot_beta_colour_dataset_style"
# )
# plt.close(g.fig)

print("   • Updated component curves (y-max 10)  →", COMPONENT_CURVES_DIR)
print("   • Combined Strong-Reject plot          →", STRONGREJECT_CURVES_DIR)

#────────────────────────────────────────────────────────────────────────────────
# 6-f  Weighted (logged) loss-components   –  no rescaling, auto y-limits
# ────────────────────────────────────────────────────────────────────────────────
WEIGHTED_CURVES_DIR = os.path.join(RESULTS_DIR, SWEEP_SUBDIR, "loss_components_weighted")
os.makedirs(WEIGHTED_CURVES_DIR, exist_ok=True)

value_vars = [
    c for c in attack_hist_df.columns
    if re.fullmatch(r"loss_(reg|ce)/(InternVL2-8B|MiniCPM-V-2_6)", c)
]
if not value_vars:
    raise ValueError("No matching loss_reg/… or loss_ce/… columns found!")

# ------------------------------------------------------------------
# 2)  Tidy (long) format
# ------------------------------------------------------------------
weighted_long = (
    attack_hist_df
    .melt(
        id_vars=["optimizer_step_counter", "attack_dataset", "beta"],
        value_vars=value_vars,
        var_name="component_raw",      # keep original name for parsing
        value_name="loss",
    )
)

# ------------------------------------------------------------------
# 3)  Build friendly legend labels
# ------------------------------------------------------------------
models_in_run = {
    col.split("/", 1)[1]              # part after the slash
    for col in value_vars
}
include_model = len(models_in_run) > 1     # only show model name if ≥2 present

def pretty(col_name: str) -> str:
    """Turn 'loss_reg/InternVL2-8B' -> 'β · reg (InternVL2-8B)'."""
    prefix, model = col_name.split("/", 1)
    base = "β · reg" if prefix.endswith("reg") else "(1-β) · CE"
    return f"{base} ({model})" if include_model else base


weighted_long = filter_by_beta(weighted_long)
weighted_long["dataset_disp"] = weighted_long["attack_dataset"].apply(pretty_ds)
weighted_long["component"] = weighted_long["component_raw"].map(pretty)
weighted_long = weighted_long.drop(columns="component_raw")   # cleanup
# g = sns.relplot(
#     data=weighted_long,
#     kind="line",
#     x="optimizer_step_counter",
#     y="loss",
#     hue="component",
#     col="dataset_disp",
#     row="beta",
#     facet_kws=dict(margin_titles=True),
#     linewidth=2.2,
#     height=3.8,
#     aspect=1.4,
#     errorbar=None,
#     palette="tab10",
# )
# g.set_titles(                  # keep row titles nice too
#     col_template="{col_name}",
#     row_template=r"$\beta = {row_name}$",
# )
# g.fig.subplots_adjust(top=0.73)             # shrink facet area
# g.fig.suptitle("Loss-components vs steps   (facet: dataset × β)", y=0.98)
# place_legend_top(g, ncol=2, dy=0.08)            # drop legend just below title
# g.set_axis_labels("Gradient step", "Weighted loss")
# g.set(xlim=(0, MAX_STEPS))          # auto y-scale
# g.fig.suptitle("Weighted loss components vs steps   (facet: dataset ⨯ β)")
# place_legend_top(g, ncol=2, dy=0.025)              # ← NEW

# src.plot.save_plot_with_multiple_extensions(
#     WEIGHTED_CURVES_DIR, "loss_components_weighted_by_dataset_and_beta"
# )
# plt.close(g.fig)

print("   • Weighted component curves →", WEIGHTED_CURVES_DIR)


import matplotlib as mpl
mpl.rcParams["figure.constrained_layout.use"] = False   # make subplots_adjust work

TWIN_CURVES_DIR = os.path.join(RESULTS_DIR, SWEEP_SUBDIR,
                               "loss_components_plus_sr")
os.makedirs(TWIN_CURVES_DIR, exist_ok=True)

# ­­­ tidy SR: one mean per (β, dataset, step) ­­­­­­­­
sr_mean_df = (
    sr_df
      .groupby(["beta", "dataset_disp", "optimizer_step_counter_epoch"],
               as_index=False)["score"]
      .mean()
      .sort_values("optimizer_step_counter_epoch")
)

# ── base FacetGrid with loss components ───────────────────────────
g = sns.relplot(
    data=loss_long,
    kind="line",
    x="optimizer_step_counter",
    y="loss",
    hue="component",
    col="dataset_disp",
    row="beta",
    
    linewidth=2.2,
    height=4.5,
    aspect=1.4,
    errorbar=None,
    palette="tab10",
    facet_kws=dict(margin_titles=True),
)
g.set(xlim=(0, MAX_STEPS), ylim=(0.0, 10.0))
g.set_titles(                  # keep row titles nice too
    col_template="{col_name}",
    row_template=r"$\beta = {row_name}$",
)
g.set_axis_labels("Gradient step", "Loss components")
# ── overlay SR: red poly-line + dots ──────────────────────────────
for (row_key, col_key), ax in g.axes_dict.items():
    β       = float(row_key)
    dataset = col_key

    sel = (
        sr_df
        .query("beta == @β and dataset_disp == @dataset")
        .groupby("optimizer_step_counter_epoch", as_index=False)["score"]
        .mean()                              # one value per eval step
        .sort_values("optimizer_step_counter_epoch")
    )
    if sel.empty:
        continue

    ax2 = ax.twinx()
    ax2.plot(
        sel["optimizer_step_counter_epoch"],
        sel["score"],
        color="red",
        marker="x",
        linestyle="None",
        linewidth=0,
        label="Strong-Reject",
        zorder=6,
        markersize=20,          # ← bigger overall size
        markeredgewidth=3,
    )

    # ── NEW: tidy second y-axis ──────────────────────────────────────────
    ax2.set_ylim(0.0, 1.0)              # fixed limits
    ax2.set_yticks([0.0, 1.0])          # only bottom & top labels
    ax2.set_yticklabels(["0.0", "1.0"])

# ---------------------------------------------------------------------------
# ❷  Remove all per-axes legends that seaborn created
# ---------------------------------------------------------------------------
# ------------------------------------------------------------------------
for ax in g.fig.axes:
    leg = ax.get_legend()
    if leg:
        leg.remove()
if getattr(g, "_legend", None) is not None:
    g._legend.remove()

# C)  Build ONE shared legend including Strong-Reject
# ---------------------------------------------------
import matplotlib.lines as mlines
sr_handle = mlines.Line2D([], [], color="red", marker="x", linestyle="None",
                          markersize=7, label="Strong-Reject")

loss_handles, loss_labels = g.axes.flat[0].get_legend_handles_labels()
g.fig.legend(
    loss_handles + [sr_handle],
    loss_labels  + ["Strong-Reject"],
    loc="upper center",
    ncol=3,
    bbox_to_anchor=(0.5, 0.94),  # closer to the title
    frameon=False,
)


# 1) Bigger, bolder super-title
g.fig.suptitle(
    "Loss components vs steps  (facet: dataset × β)  +  Strong-Reject",
    y=0.974, fontsize=35
)

# 3) Tighter grid spacing, but more room on the right so β labels clear
g.fig.set_size_inches(20, 30)                    # unchanged height
g.fig.subplots_adjust(
    left=0.08, right=0.88,                      # ← more right margin
    top=0.9, bottom=0.06,
    wspace=0.25, hspace=0.55,                   # ← closer subplots
)

plt.draw()                                      # refresh canvas before saving
for ext in ("png", "pdf"):
    g.fig.savefig(
        os.path.join(
            TWIN_CURVES_DIR,
            f"loss_components_plus_sr_by_dataset_and_beta.{ext}"
        ),
        dpi=300,
        bbox_inches=None,                       # keep the space
    )
plt.close(g.fig)

###############################################################################
# 2)  ONE-PLOT view: weighted loss (primary-y) + SR (secondary-y)
###############################################################################
# ----------------------------------------------------------------------------
# 2)  ONE-PLOT: weighted loss (primary-y) + SR (secondary-y)
# ----------------------------------------------------------------------------
fig, ax_loss = plt.subplots(figsize=(10, 6))

# ---- 2a)  weighted loss ----------------------------------------------------
sns.lineplot(
    data=weighted_long,                    # ← back to weighted_long
    x="optimizer_step_counter",
    y="loss",
    hue="component",
    linewidth=2.4,
    ax=ax_loss,
    errorbar=None,
    palette="tab10",
)
ax_loss.set_xlim(0, MAX_STEPS)
ax_loss.set_xlabel("Gradient step")
ax_loss.set_ylabel("Weighted loss")

# ---- 2b)  SR score ---------------------------------------------------------
ax_sr = ax_loss.twinx()
sns.lineplot(
    data=sr_mean_df,                       # ← use the deduplicated mean table
    x="optimizer_step_counter_epoch",
    y="score",
    hue="dataset_disp",
    style="dataset_disp",
    linewidth=2.2,
    markers=True,
    dashes=False,
    color="red",                           # one red for all SR curves
    legend=False,
    ax=ax_sr,
)
ax_sr.set_ylabel("Strong-Reject score")
ax_sr.set_ylim(0, 1)
# keep twin axis black for neutrality
ax_sr.tick_params(axis="y")
ax_sr.spines["right"].set_color("black")

# ---- 2c)  joint legend -----------------------------------------------------
handles_loss, labels_loss = ax_loss.get_legend_handles_labels()
handles_sr,   labels_sr   = ax_sr.get_legend_handles_labels()
ax_loss.legend(
    handles_loss + handles_sr,
    labels_loss  + labels_sr,
    loc="upper center",
    ncol=3,
    bbox_to_anchor=(0.5, 1.20),
)

fig.subplots_adjust(top=0.82, right=0.88)
ax_loss.set_title(
    "Weighted loss & Strong-Reject vs steps  "
    "(red = SR,  colour = loss-component)",
    pad=30,
)

src.plot.save_plot_with_multiple_extensions(
    TWIN_CURVES_DIR, "weighted_loss_plus_sr_singleplot"
)
plt.close(fig)


# ------------------------------------------------------------------
# 1)  Wide table from loss_long  (one column per loss component)
# ------------------------------------------------------------------
loss_wide = (
    loss_long
      .pivot_table(
          index=["beta", "dataset_disp", "optimizer_step_counter"],
          columns="component",
          values="loss"
      )
      .reset_index()
)

# ------------------------------------------------------------------
# 2)  Identify columns whose *prefix* (part before the “/”) ends with
#     'reg_unweighted'  or  'ce_unweighted'
# ------------------------------------------------------------------
# ------------------------------------------------------------------
# 1)  Pivot loss_long -> wide table
# ------------------------------------------------------------------
loss_wide = (
    loss_long
      .pivot_table(
          index=["beta", "dataset_disp", "optimizer_step_counter"],
          columns="component",
          values="loss"
      )
      .reset_index()
)

# ------------------------------------------------------------------
# 2)  Identify the reg- and ce-loss columns automatically
# ------------------------------------------------------------------
reg_col = next(col for col in loss_wide.columns if "loss_reg_unscaled" in col)
ce_col  = next(col for col in loss_wide.columns if ("loss_ce_unscaled"  in col))

loss_wide = loss_wide.rename(columns={
    reg_col: "Regularization Factor",
    ce_col:  "Cross-Entropy Loss",
})

# ------------------------------------------------------------------
# 3)  Merge with Strong-Reject scores
# ------------------------------------------------------------------
sr_wide = sr_mean_df.rename(
    columns={"optimizer_step_counter_epoch": "optimizer_step_counter",
             "score": "Strong-Reject"}
)

merged = (
    pd.merge(
        loss_wide,
        sr_wide[["beta", "dataset_disp", "optimizer_step_counter", "Strong-Reject"]],
        on=["beta", "dataset_disp", "optimizer_step_counter"],
        how="inner",
    )
    .dropna(subset=["Regularization Factor", "Cross-Entropy Loss", "Strong-Reject"])
)

# ------------------------------------------------------------------
# 4)  Pearson correlations
# ------------------------------------------------------------------
corr = merged[["Regularization Factor", "Cross-Entropy Loss", "Strong-Reject"]].corr(method="pearson")

print("\n=== Pearson correlations across all evaluation steps ===")
print(corr.round(4))