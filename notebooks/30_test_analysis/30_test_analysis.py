from __future__ import annotations
import os, time
from typing import List

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

SWEEP_IDS              = ["cxe4ycfx"]           # ← put your sweep(s) here
REFRESH_WANDB_DOWNLOAD = True                    # force re-download?
FINISHED_ONLY          = True                    # ignore running/failed runs?

# Metrics to plot
EVAL_SCORE_METRICS = {
    "loss/score_model=llamaguard2": "LlamaGuard‑2 score",
    "loss/score_model=strongreject": "Strong‑Reject score",
}

ATTACK_LOSS_METRICS = {
    "loss/avg": "Average loss",
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
MAX_STEPS = 4000

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

sns.set_theme(style="whitegrid", font_scale=1.3)
plt.close("all")

attack_hist_df = attack_hist_df[attack_hist_df["optimizer_step_counter"] % 10 == 0]


def plot_by_single_param(
    df_hist: pd.DataFrame,
    x_col: str,
    y_col: str,
    param: str,
    pretty_y: str,
    out_dir: str,
    max_steps: int = MAX_STEPS,
):
    """Aggregate over all *other* hyper‑params and draw mean ± CI lines."""

    df_plot = (
        df_hist[[x_col, param, y_col]]
        .groupby([x_col, param], as_index=False)
        .mean()
    )

    subdir = os.path.join(out_dir, param)
    os.makedirs(subdir, exist_ok=True)

    g = sns.relplot(
        data=df_plot,
        kind="line",
        x=x_col,
        y=y_col,
        hue=param,
        linewidth=2.5,
        aspect=1.5,
        palette="tab10",
        height=5,
    )
    g.set_axis_labels("Gradient step", pretty_y)
    g.set(xlim=(0, max_steps))
    g.fig.suptitle(f"{pretty_y} vs steps – grouped by “{param}”", y=1.02)

    fname = f"attack_{y_col}_vs_steps_by_{param}"
    src.plot.save_plot_with_multiple_extensions(subdir, fname)
    plt.close(g.fig)

for col, pretty in ATTACK_LOSS_METRICS.items():
    # rename to a generic column for plotting func
    attack_hist_df_renamed = attack_hist_df.rename(columns={col: "y"})
    for key in ATTACK_CONFIG_KEYS:
        if key not in attack_hist_df_renamed.columns:
            continue
        plot_by_single_param(
            df_hist=attack_hist_df_renamed,
            x_col="optimizer_step_counter",
            y_col="y",
            param=key,
            pretty_y=pretty,
            out_dir=LOSS_CURVES_DIR,
        )

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

for key in ATTACK_CONFIG_KEYS:
    if key not in metric_long_df.columns:
        continue

    g = sns.relplot(
        data=metric_long_df,
        kind="line",
        x="optimizer_step_counter_epoch",
        y="score",
        hue=key,
        style="metric",
        linewidth=2.5,
        aspect=1.6,
        palette="tab10",
        height=5,
    )
    g.set_axis_labels("Gradient step", "Evaluation score")
    g.set(xlim=(0, MAX_STEPS), ylim=(0.0, 1.0))
    g.fig.suptitle(
        f"Evaluation scores vs steps – grouped by “{key}”", y=1.02
    )

    subdir = os.path.join(SCORE_CURVES_DIR, key)
    os.makedirs(subdir, exist_ok=True)
    fname = f"eval_both_metrics_vs_steps_by_{key}"
    src.plot.save_plot_with_multiple_extensions(subdir, fname)
    plt.close(g.fig)

print("✅ Plots saved under:")
print(f"   • Loss curves      → {LOSS_CURVES_DIR}")
print(f"   • Evaluation curves→ {SCORE_CURVES_DIR}")
