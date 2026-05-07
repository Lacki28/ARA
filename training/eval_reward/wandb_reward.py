import wandb
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

ENTITY = ""
PROJECT = "Train RL for ICSOC reward"
METRIC = "total_reward_final"
WINDOW = 1

api = wandb.Api()
runs = api.runs(f"{ENTITY}/{PROJECT}")

selected_runs_non_rel = [
    run for run in runs if ("outside" in run.name.lower())
]
print(len(selected_runs_non_rel))
selected_runs_relative = [
    run for run in runs if ("relative 32" in run.name.lower())
]

def aggregate_runs(run_list, metric, window, rel):
    dfs = []

    for run in run_list:
        history = run.history(keys=[metric], pandas=True)
        if metric not in history:
            continue
        series = history[metric].dropna().reset_index(drop=True)
        if rel:
            series = series.iloc[1::2].reset_index(drop=True)
        dfs.append(series)

    if not dfs:
        return None, None

    min_len = min(len(df) for df in dfs)
    aligned = np.vstack([df[:min_len] for df in dfs])

    mean = pd.Series(aligned.mean(axis=0))
    std = pd.Series(aligned.std(axis=0))

    ma = mean.rolling(window).mean()
    ma_std = std.rolling(window).mean()

    return ma, ma_std


plt.figure(figsize=(10, 6))

ma_non_rel, std_non_rel = aggregate_runs(selected_runs_non_rel, METRIC, WINDOW, False)
if ma_non_rel is not None:
    plt.plot(ma_non_rel, label="Sparse (mean)")
    plt.fill_between(
        ma_non_rel.index,
        ma_non_rel - std_non_rel,
        ma_non_rel + std_non_rel,
        alpha=0.2
    )

ma_rel, std_rel = aggregate_runs(selected_runs_relative, METRIC, WINDOW, True)
if ma_rel is not None:
    plt.plot(ma_rel, label="Relative (mean)")
    plt.fill_between(
        ma_rel.index,
        ma_rel - std_rel,
        ma_rel + std_rel,
        alpha=0.2
    )

plt.title(f"Moving Average ({WINDOW}) of {METRIC}")
plt.xlabel("Step")
plt.ylabel(METRIC)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("reward.png")



plt.figure(figsize=(10, 6))

first_label = True
for run in selected_runs_non_rel:
    history = run.history(keys=[METRIC], pandas=True)
    if METRIC not in history:
        continue

    series = history[METRIC].dropna().reset_index(drop=True)

    ma = series.rolling(WINDOW).mean()

    label = "Sparse" if first_label else None
    first_label = False

    plt.plot(ma, color="blue", alpha=0.7, label=label)


first_label = True
for run in selected_runs_relative:
    history = run.history(keys=[METRIC], pandas=True)
    if METRIC not in history:
        continue

    series = history[METRIC].dropna().reset_index(drop=True)

    series = series.iloc[1::2].reset_index(drop=True)

    ma = series.rolling(WINDOW).mean()

    label = "Relative" if first_label else None
    first_label = False

    plt.plot(ma, color="green", alpha=0.7, label=label)


plt.title(f"Moving Average ({WINDOW}) of {METRIC}")
plt.xlabel("Step")
plt.ylabel(METRIC)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("reward_individual.png")
