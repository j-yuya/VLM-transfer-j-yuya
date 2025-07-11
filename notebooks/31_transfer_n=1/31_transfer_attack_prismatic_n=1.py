import ast
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import seaborn as sns
import wandb

import src.analyze
import src.globals
import src.plot


refresh = True
# refresh = False
finished_only = True

data_dir, results_dir = src.analyze.setup_notebook_dir(
    notebook_dir=os.path.dirname(os.path.abspath(__file__)),
    refresh=False,
)

# Default Attack Setup
# sweep_ids = [
#     "rvn3unko",
#     "uygm5voo",
#     "25953ngs",
#     "et1rw0nu ",
#     "2wtk3ynt",
#     "rkag4r00",
#     "ndsp0bxh",
#     "9pjh0dfq",
#     "pd0agzrv",
#     "0def2rw8",
#     "u5fhpdwh",
#     "n5wgqigz"
# ]

#IRIS 1,2,3 on advbench
sweep_ids= [
    "2n9737ha",
    #"cngtdiqz",
    "6976fsb1",
    #"p3s80qcr",
    "fxr9nu6b",
    "30j5x091",
    "xr93tlv9",
    "a0vxqfaw",
    "6w6n56jo",
    "pemca1qn",
    "mxoj3yac",
    "ra4zt9ho",
    "sdq4h9u2",
    "3a0ltqro",
    #"0o6ovw5r",
    "vhk4p7k6",
    "i1kdqkal",
    "2d2wplfn",
    "ocv3rbg7",
    "y0qoa6w0",
    "fxr9nu6b",
    "30j5x091",
    #"oqestcn4"
]


wandb_username = "julian-yuya-caspary-university-of-mannheim"
eval_runs_configs_df = src.analyze.download_wandb_project_runs_configs(
    wandb_project_path="universal-vlm-jailbreak-eval",
    data_dir=data_dir,
    sweep_ids=sweep_ids,
    refresh=refresh,
    finished_only=finished_only,
    wandb_username=wandb_username,
    filetype="csv",
)
eval_runs_configs_df = src.analyze.extract_key_value_from_df_col(
    df=eval_runs_configs_df,
    col_name="data",
    key_in_dict="dataset",
    new_col_name="eval_dataset",
)
eval_runs_configs_df = src.analyze.extract_key_value_from_df_col(
    df=eval_runs_configs_df,
    col_name="data",
    key_in_dict="split",
    new_col_name="eval_dataset_split",
)

eval_runs_configs_df.rename(
    columns={"run_id": "eval_run_id", "wandb_attack_run_id": "attack_run_id"},
    inplace=True,
)

# Switch attack_model_names and eval_model_name to nice strings.
eval_runs_configs_df["model_to_eval"] = eval_runs_configs_df["model_to_eval"].apply(
    src.analyze.map_string_set_of_models_to_nice_string
)
eval_runs_configs_df["models_to_attack"] = eval_runs_configs_df[
    "models_to_attack"
].apply(src.analyze.map_string_set_of_models_to_nice_string)

# Download attack runs.
unique_attack_run_ids = eval_runs_configs_df["attack_run_id"].unique()
print("Attack Run IDs: ", unique_attack_run_ids.tolist())
attack_runs_configs_df = src.analyze.download_wandb_project_runs_configs_by_run_ids(
    wandb_project_path="universal-vlm-jailbreak",
    wandb_username=wandb_username,
    data_dir=data_dir,
    run_ids=unique_attack_run_ids,
    refresh=refresh,
    finished_only=finished_only,
    filetype="csv",
)
attack_runs_configs_df = src.analyze.extract_key_value_from_df_col(
    df=attack_runs_configs_df,
    col_name="data",
    key_in_dict="dataset",
    new_col_name="attack_dataset",
)
attack_runs_configs_df = src.analyze.extract_key_value_from_df_col(
    df=attack_runs_configs_df,
    col_name="image_kwargs",
    key_in_dict="image_initialization",
    new_col_name="image_initialization",
)
attack_runs_configs_df.rename(
    columns={"run_id": "attack_run_id"},
    inplace=True,
)
attack_runs_configs_df["image_initialization"] = attack_runs_configs_df[
    "image_initialization"
].map(src.globals.IMAGE_INITIALIZATION_TO_STRINGS_DICT)

# Join attack run data into to evals df.
eval_runs_configs_df = eval_runs_configs_df.merge(
    right=attack_runs_configs_df[
        ["attack_run_id", "attack_dataset", "image_initialization"]
    ],
    how="left",
    left_on="attack_run_id",
    right_on="attack_run_id",
)

eval_runs_configs_df["Attacked"] = eval_runs_configs_df.apply(
    lambda row: row["model_to_eval"] in row["models_to_attack"], axis=1
)

# Load the heftier runs' histories dataframe.
eval_runs_histories_df = src.analyze.download_wandb_project_runs_histories(
    wandb_project_path="universal-vlm-jailbreak-eval",
    wandb_username=wandb_username,
    data_dir=data_dir,
    sweep_ids=sweep_ids,
    refresh=refresh,
    wandb_run_history_samples=1000000,
    # nrows_to_read=5000000,
    filetype="csv",
    # filetype="feather",
    # filetype="parquet",
)
# This col is not populated on this df.
eval_runs_histories_df.drop(columns=["models_to_attack"], inplace=True)
eval_runs_histories_df.rename(columns={"run_id": "eval_run_id"}, inplace=True)


eval_runs_histories_df = eval_runs_histories_df.merge(
    right=eval_runs_configs_df[
        [
            "eval_run_id",
            "attack_run_id",
            "model_to_eval",
            "models_to_attack",
            "attack_dataset",
            "eval_dataset",
            "image_initialization",
            "Attacked",
        ]
    ],
    how="inner",
    on="eval_run_id",
)

unique_metrics_order = [
    #"loss/score_model=llamaguard2",
    "loss/score_model=strongreject"
]

eval_runs_histories_tall_df = eval_runs_histories_df.melt(
    id_vars=[
        "eval_run_id",
        "attack_run_id",
        "attack_dataset",
        "eval_dataset",
        "model_to_eval",
        "models_to_attack",
        "optimizer_step_counter_epoch",
        "image_initialization",
        "Attacked",
    ],
    value_vars=unique_metrics_order,
    var_name="Metric",
    value_name="Score",
)

eval_runs_histories_tall_df.rename(
    columns={
        "model_to_eval": "Eval VLM",
        "image_initialization": "Image Initialization",
    },
    inplace=True,
)

sorted_unique_attacked_models = list(
    sorted(eval_runs_histories_tall_df["models_to_attack"].unique())
)

# Convert metrics to nice strings.
eval_runs_histories_tall_df["Original Metric"] = eval_runs_histories_tall_df["Metric"]
eval_runs_histories_tall_df["Metric"] = eval_runs_histories_tall_df["Metric"].map(
    lambda k: src.globals.METRICS_TO_TITLE_STRINGS_DICT.get(k, k)
)

# Obtain the first optimizer_step_counter_epoch per eval_run_id.
first_optimizer_step = (
    eval_runs_histories_tall_df.groupby("eval_run_id")["optimizer_step_counter_epoch"]
    .min()
    .reset_index()
)
last_optimizer_step = (
    eval_runs_histories_tall_df.groupby("eval_run_id")["optimizer_step_counter_epoch"]
    .max()
    .reset_index()
)


# Merge these with the original dataframe to get the corresponding rows
first_optimizer_step_rows_df = (
    pd.merge(
        eval_runs_histories_tall_df,
        first_optimizer_step,
        on=["eval_run_id", "optimizer_step_counter_epoch"],
        how="inner",
    )
    .rename(columns={"Score": "Initial Score"})
    .drop(columns=["optimizer_step_counter_epoch"])
)

last_optimizer_step_rows_df = (
    pd.merge(
        eval_runs_histories_tall_df,
        last_optimizer_step,
        on=["eval_run_id", "optimizer_step_counter_epoch"],
        how="inner",
    )
    .rename(columns={"Score": "Final Score"})
    .drop(columns=["optimizer_step_counter_epoch"])
)

# Combine first and last rows into a single dataframe
first_and_last_optimizer_step_df = pd.merge(
    first_optimizer_step_rows_df,
    last_optimizer_step_rows_df,
    on=[
        "eval_run_id",
        "attack_run_id",
        "attack_dataset",
        "eval_dataset",
        "Eval VLM",
        "models_to_attack",
        "Image Initialization",
        "Attacked",
        "Metric",
        "Original Metric",
    ],
    how="inner",
)

print(eval_runs_histories_tall_df["Eval VLM"].unique())
all_eval_vlms = eval_runs_histories_tall_df["Eval VLM"].unique()
palette = sns.color_palette("tab10", n_colors=len(all_eval_vlms))
eval_vlm_color_mapping = dict(zip(all_eval_vlms, palette))

all_attack_models = sorted_unique_attacked_models           # already defined
attack_palette     = sns.color_palette("tab10",
                                       n_colors=len(all_attack_models))
attack_model_color_mapping = dict(zip(all_attack_models, attack_palette))

# list of *every* model that ever appears as an Eval VLM
sorted_unique_eval_models = sorted(all_eval_vlms)

# import pdb
# pdb.set_trace()

all_attack_models = [
    "MiniCPM-V-2_6",
    "InternVL2-8B",
    "Llama2 7B + CLIP",
    "cogvlm2-llama3-chat-19B",
    "Llama3 Instr 8B + CLIP",
    "LLAVAv1.5 7B + CLIP (Repro)",
]

attack_model_color_mapping = {
    "MiniCPM-V-2_6": "#1f77b4",             # blue
    "InternVL2-8B": "#ff7f0e",              # orange
    "Llama2 7B + CLIP": "#2ca02c",          # green
    "cogvlm2-llama3-chat-19B": "#d62728",   # red
    "Llama3 Instr 8B + CLIP": "#9467bd",    # purple
    "LLAVAv1.5 7B + CLIP (Repro)": "#8c564b",  # brown
}
# ─── Final-score-vs-initial-score scatter grid (updated) ───────────────
plt.close()
import matplotlib as mpl
mpl.rcParams['text.usetex'] = False

g = sns.relplot(
    data=first_and_last_optimizer_step_df,
    kind="scatter",
    x="Initial Score",
    y="Final Score",
    col="Eval VLM",
    col_order=sorted_unique_eval_models,
    hue="models_to_attack",
    hue_order=all_attack_models,
    palette=attack_model_color_mapping,  # ✅ manually set colors
    style="Attacked",
    style_order=[False, True],
    size="Attacked",
    size_order=[False, True],
    sizes=[100, 400],
    col_wrap=3,
    s=250,
    aspect=0.9,
)

# identity line in every panel
line = np.linspace(0.0, 1.0, 100)
for ax in g.axes.flat:
    ax.plot(line, line, "k--")

g.set_axis_labels("Harmful-Yet-Helpful (Initial)",
                  "Harmful-Yet-Helpful (Final)")
g.set(xlim=(0.0, 1.0), ylim=(0.0, 1.0))
g.set_titles(col_template="{col_name}")

# place legend outside
sns.move_legend(g, "upper left", bbox_to_anchor=(1.02, 1.0))

g.fig.suptitle(
    "Strong-Reject scores: transfer from single VLM to new VLM",
    fontsize=40,
    y=1.02
)

plt.subplots_adjust(top=0.88)

src.plot.save_plot_with_multiple_extensions(
    plot_dir=results_dir,
    plot_title="final_score_vs_initial_score_by_eval_vlm",
)

# plt.show()


learning_curves_results_dir = os.path.join(results_dir, "learning_curves")
os.makedirs(learning_curves_results_dir, exist_ok=True)


for eval_dataset in eval_runs_histories_tall_df["eval_dataset"].unique():
    learning_curves_eval_dataset_results_dir = os.path.join(
        learning_curves_results_dir, f"eval_dataset={eval_dataset}"
    )
    os.makedirs(learning_curves_eval_dataset_results_dir, exist_ok=True)
    for attack_dataset in eval_runs_histories_tall_df["attack_dataset"].unique():
        learning_curves_eval_dataset_attack_dataset_results_dir = os.path.join(
            learning_curves_eval_dataset_results_dir,
            f"attack_dataset={attack_dataset}",
        )
        os.makedirs(
            learning_curves_eval_dataset_attack_dataset_results_dir, exist_ok=True
        )
        eval_runs_histories_tall_subset_df = eval_runs_histories_tall_df[
            (eval_runs_histories_tall_df["attack_dataset"] == attack_dataset)
            & (eval_runs_histories_tall_df["eval_dataset"] == eval_dataset)
        ]

        if len(eval_runs_histories_tall_subset_df) == 0:
            print(
                f"No data for attack_dataset={attack_dataset} and eval_dataset={eval_dataset}."
            )
            continue

        plt.close()
        g = sns.relplot(
            data=eval_runs_histories_tall_subset_df,
            kind="line",
            x="optimizer_step_counter_epoch",
            y="Score",
            col="models_to_attack",
            col_order=sorted_unique_attacked_models,
            style="Attacked",                 # dashed = attacked model
            style_order=[False, True],
            hue_order=eval_vlm_order,                  # 🔸 consistent hue order
            palette=eval_vlm_palette,     
            hue="Eval VLM",
            linewidth=2,                      # base width; we’ll tweak below
            col_wrap=3,                       # 🔸 3 columns  →  2 rows of 3 plots
            aspect=1.1,                      # keep the small footprint
            height=5,                       # same visual size as before
        )
        for ax in g.axes.flat:
            for line in ax.lines:
                if line.get_linestyle() == "--":   # Attacked=True → dashed
                    line.set_linewidth(3.5)        # thicker
                else:
                    line.set_linewidth(1.2)        # thinner for non-attacked

        # ── Axis limits & labels ────────────────────────────────────────────────
        g.set_axis_labels("Gradient Step", "Harmful-Yet-Helpful")
        g.set(xlim=(0, 2000), ylim=(0.0, 1.0))
        g.set_titles(col_template="{col_name}")

        # ── Legend & title tweaked like the scatter grid ────────────────────────
        sns.move_legend(g, "upper left", bbox_to_anchor=(1.02, 1.0))

        g.fig.suptitle(
            "Strong-Reject scores: transfer from single VLM to new VLM",
            fontsize=40,    # was 60
            y=1.03          # lift title a bit
        )

        plt.subplots_adjust(top=0.92)  # breathing room for the title

        src.plot.save_plot_with_multiple_extensions(
            plot_dir=learning_curves_eval_dataset_attack_dataset_results_dir,
            plot_title="score_vs_optimizer_step_by_attacked_split_models_to_attack",
        )
        # plt.show()


print("Finished notebooks/02_transfer_attack_prismatic_n=1!")
