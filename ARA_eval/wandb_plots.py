import pandas as pd
import wandb
import matplotlib.pyplot as plt
import numpy as np
import os

if os.path.exists("./results"):
    for f in os.listdir("results"):
        os.remove(os.path.join("results", f))
else:
    print("The file does not exist")
    os.mkdir("results")

api = wandb.Api()
names = ["MVP change users 10 to 5"] #new MVP changing SLOs"]#, "MVP change users 10 to 5"] #"MVP change users and SLOs 10 to 5",


def load_run_data(run, latency_key, action_key):
    """Load and align latency + action columns from a W&B run."""
    df = run.history(samples=10000)

    lat = df[latency_key].dropna().reset_index(drop=True)[0:142]
    act = df[action_key].dropna().reset_index(drop=True)[0:143]

    # Align action with latency (your original shift)
    act = act.iloc[1:].reset_index(drop=True)

    result = pd.concat([lat, act], axis=1)
    result.columns = ["lat", "action"]
    result2 = pd.DataFrame()
    result2["lat"] = df["total_latency.total_latency"].dropna().reset_index(drop=True)
    return result, result2


def plot_ma(key, title, df, name):
    plt.figure(figsize=(14, 7))
    values = df[key]

    plt.plot(values, label=f"{key}")

    low_idx = values[values < LOWER_BOUND].index
    high_idx = values[values > UPPER_BOUND].index

    plt.scatter(low_idx, values.loc[low_idx], color="blue", marker="x", s=50)
    plt.scatter(high_idx, values.loc[high_idx], color="orange", marker="x", s=50)

    plt.axhline(LOWER_BOUND, color="red", linestyle="--", linewidth=2)
    plt.axhline(UPPER_BOUND, color="red", linestyle="--", linewidth=2)

    # plt.title(title)
    plt.xlabel("Step")
    plt.ylabel("Latency (MA)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"scatter_1000_and_1500ms_{key}_{title}_{name}.pdf")
    plt.close()


def plot_violation_count_over_time(values, lower, upper, title, filename):
    violations = ((values < lower) | (values > upper)).astype(int)
    cumulative = violations.cumsum()

    plt.figure(figsize=(7, 3.5))
    plt.plot(cumulative, color="purple", linewidth=2)

    # plt.title(title)
    plt.xlabel("Step")
    plt.ylabel("Cumulative Violations")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


def compute_action_violations(df):
    lower1 = (
            (df["lat"][0:71] < LOWER_BOUND / 2) &
            (df["action"][0:71] == 2) &
            (df["lat"][0:71].shift(-1) > UPPER_BOUND / 2)
    ).sum()
    upper1 = (
            (df["lat"][0:71] > UPPER_BOUND / 2) &
            (df["action"][0:71] == 0) &
            (df["lat"][0:71].shift(-1) < LOWER_BOUND / 2)
    ).sum()

    lower2 = (
            (df["lat"][71:] < new_LOWER_BOUND / 2) &
            (df["action"][71:] == 2) &
            (df["lat"][71:].shift(-1) > new_UPPER_BOUND / 2)
    ).sum()

    upper2 = (
            (df["lat"][71:] > new_UPPER_BOUND / 2) &
            (df["action"][71:] == 0) &
            (df["lat"][71:].shift(-1) < new_LOWER_BOUND / 2)
    ).sum()

    return upper1 + upper2, lower1 + lower2


def compute_violations(df, devider):
    lower = (df["lat"][0:71] < LOWER_BOUND / devider).sum()

    upper = (df["lat"][0:71] > UPPER_BOUND / devider).sum()
    lower2 = (df["lat"][71:] < new_LOWER_BOUND / devider).sum()

    upper2 = (df["lat"][71:] > new_UPPER_BOUND / devider).sum()

    return upper + upper2, lower + lower2


def plot_violations(df, title, filename):
    plt.figure(figsize=(7, 4))
    plt.plot(df["lat"], label="Latency", color="blue")

    plt.scatter(df.index[df["upper_violation"]],
                df["lat"][df["upper_violation"]],
                color="red", label="Upper violation", s=50)

    plt.scatter(df.index[df["lower_violation"]],
                df["lat"][df["lower_violation"]],
                color="green", label="Lower violation", s=50)

    plt.axhline(LOWER_BOUND, color="gray", linestyle="--")
    plt.axhline(UPPER_BOUND, color="gray", linestyle="--")

    # plt.title(title)
    plt.xlabel("Timestep")
    plt.ylabel("Latency")
    plt.legend()
    plt.savefig(filename)
    plt.close()


def plot_all_runs_boxplot(all_runs_data, title, filename, devider):
    plt.figure(figsize=(7, 3.5))

    methods = [
        ("PPO", "RA", "green"),
        ("SA_PPO", "SARL", "pink"),
        ("reactive", "Reactive", "purple"),
        ("proactive", "Predictive", "orange")
    ]

    data = []
    labels = []
    colors = []

    for key, label, color in methods:

        first_half = []
        for run_name, df in all_runs_data:
            if key == run_name:
                first_half.append(df["lat"][0:71])
        data.append(pd.concat(first_half))
        labels.append(f"{label}")
        colors.append(color)
    for key, label, color in methods:
        second_half = []
        for run_name, df in all_runs_data:
            if key == run_name:
                second_half.append(df["lat"][71:])
        data.append(pd.concat(second_half))
        labels.append(f"{label}")
        colors.append(color)
    # Create boxplot
    box = plt.boxplot(
        data,
        tick_labels=labels,
        patch_artist=True,
        showfliers=True,
        medianprops=dict(color="black", linewidth=1.5)
    )

    # Color boxes
    for patch, color in zip(box["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.5)

    plt.plot([0, 4.5], [LOWER_BOUND / devider, LOWER_BOUND / devider], color="blue", linestyle="--", linewidth=1)
    plt.plot([0, 4.5], [UPPER_BOUND / devider, UPPER_BOUND / devider], color="blue", linestyle="--", linewidth=1)

    plt.plot([4.5, 9], [new_LOWER_BOUND / devider, new_LOWER_BOUND / devider], color="blue", linestyle="--",
             linewidth=1)
    plt.plot([4.5, 9], [new_UPPER_BOUND / devider, new_UPPER_BOUND / devider], color="blue", linestyle="--",
             linewidth=1)

    # Vertical event line at x=4
    plt.axvline(4.5, color="black", linestyle="--", linewidth=1)

    # Add tick at x=4 labeled "Event"
    xticks = list(range(1, len(labels) + 1)) + [4.5]
    if name == "new MVP changing SLOs":
        xlabels = labels + ["e2"]
    elif name == "MVP change users 10 to 5":
        xlabels = labels + ["e1"]

    elif name == "MVP change users and SLOs 10 to 5":
        xlabels = labels + ["e1+e2"]

    plt.xticks(xticks, xlabels, rotation=25, ha='right')
    plt.ylim(bottom=0)
    plt.xlim(left=0)
    plt.xlim(right=9)
    # plt.title(title, fontsize=16)
    plt.ylabel("Response time (ms)", fontsize=16)
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


def plot_all_runs_scatter(all_runs_data, title, filename, devider):
    plt.figure(figsize=(7, 3.5))
    added_label = {"RA": False, "SARL": False, "Reactive": False, "Predictive": False}
    method_colors = {
        "PPO": ("RA", "green"),
        "SA_PPO": ("SARL", "pink"),
        "reactive": ("Reactive", "purple"),
        "proactive": ("Predictive", "orange")
    }
    plotted_method = set()
    for run_name, df in all_runs_data:
        if "PPO" in run_name and not "SA_PPO" in run_name:
            label, color = method_colors["PPO"]
        elif "reactive" in run_name:
            label, color = method_colors["reactive"]
        elif "SA_PPO" in run_name:
            label, color = method_colors["SA_PPO"]
        else:
            label, color = method_colors["proactive"]
        if run_name in plotted_method:
            continue
        plotted_method.add(run_name)
        # Add label only once per method
        plot_label = label if not added_label[label] else None
        added_label[label] = True
        plt.plot(df.index, df["lat"], linewidth=1.2, alpha=0.8, color=color, label=plot_label)

        first_mask_low = (df.index <= 71) & (df["lat"] < LOWER_BOUND / devider)
        first_mask_high = (df.index <= 71) & (df["lat"] > UPPER_BOUND / devider)
        plt.scatter(df.index[first_mask_low], df["lat"][first_mask_low], s=10, color=color)
        plt.scatter(df.index[first_mask_high], df["lat"][first_mask_high], s=10, color=color)
        # Second segment: x = 71–end
        second_mask_low = (df.index > 71) & (df["lat"] < new_LOWER_BOUND / devider)
        second_mask_high = (df.index > 71) & (df["lat"] > new_UPPER_BOUND / devider)
        plt.scatter(df.index[second_mask_low], df["lat"][second_mask_low], s=10, color=color)
        plt.scatter(df.index[second_mask_high], df["lat"][second_mask_high], s=10, color=color)

    plt.plot([0, 71], [LOWER_BOUND / devider, LOWER_BOUND / devider], color="blue", linestyle="--", linewidth=1)
    plt.plot([0, 71], [UPPER_BOUND / devider, UPPER_BOUND / devider], color="blue", linestyle="--", linewidth=1)

    plt.plot([71, 142], [new_LOWER_BOUND / devider, new_LOWER_BOUND / devider], color="blue", linestyle="--",
             linewidth=1)
    plt.plot([71, 142], [new_UPPER_BOUND / devider, new_UPPER_BOUND / devider], color="blue", linestyle="--",
             linewidth=1)

    plt.axvline(71, color="black", linestyle="--", linewidth=1)  # , label="Upper bound")
    plt.xlim([0, 142])
    # plt.title(title)

    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)

    plt.xlabel("Timesteps (min)", fontsize=16)
    plt.ylabel("Response time (ms)", fontsize=16)
    plt.legend(fontsize=14)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


for name in names:
    PROJECT_PATH = "tuw_lacki/" + name
    NAMES = ["reactive", "proactive", "PPO", "SA_PPO"]
    LOWER_BOUND = 2000
    UPPER_BOUND = 3000
    if name == "MVP change users 10 to 5":
        new_LOWER_BOUND = 2000
        new_UPPER_BOUND = 3000
    else:
        new_LOWER_BOUND = 1000
        new_UPPER_BOUND = 2000

    runs = api.runs(PROJECT_PATH)

    # Collect all runs for combined plot
    all_network_runs = []
    all_compute_runs = []
    all_latency_runs = []

    for run in runs:
        if not any(name in run.display_name for name in NAMES) or "old" in run.display_name:
            continue
        if "new" not in run.name:
            continue
        hist = run.history(samples=100000)
        len_hist = len(hist["reward"].dropna().reset_index(drop=True))
        if len_hist < 142:
            print(run.name)
            continue
        if 'proactive' in run.name:
            run.name = 'proactive'
            run.display_name = 'proactive'
        if "SA_PPO" in run.display_name:
            run.display_name = "SA_PPO"
        elif "PPO" in run.display_name:
            run.display_name = "PPO"
        elif "reactive" in run.display_name:
            run.display_name = "reactive"

        # NETWORK LATENCY
        df_net, result2 = load_run_data(
            run,
            latency_key="total_latency.network_latency",
            action_key="network_delay action"
        )
        all_network_runs.append((run.display_name, df_net))

        upper, lower = compute_violations(df_net, 2)
        # print(f"{upper} upper violations, {lower} lower violations")
        with open("./results/network_violation.txt", "a+") as f:
            f.write(f"{run.display_name}:{lower}:{upper}\n")

        upper, lower = compute_action_violations(df_net)
        # print(f"[NETWORK] action {run.name}: {upper} upper violations, {lower} lower violations")
        with open("./results/network_action_violation.txt", "a+") as f:
            f.write(f"{run.display_name}:{lower}:{upper}\n")

        action_counts = df_net["action"].value_counts().sort_index()
        # print(f"Network action counts {action_counts}")
        with open("./results/network_action.txt", "a+") as f:
            f.write(f"{run.display_name}:{action_counts}\n")

        df_comp, global_result = load_run_data(
            run,
            latency_key="total_latency.compute_latency",
            action_key="compute_cpu action"
        )
        all_compute_runs.append((run.display_name, df_comp))
        all_latency_runs.append((run.display_name, global_result))
        action_counts = df_comp["action"].value_counts().sort_index()
        # print(f"Compute action counts {action_counts}")
        with open("./results/compute_action.txt", "a+") as f:
            f.write(f"{run.display_name}:{action_counts}\n")
        upper, lower = compute_violations(df_comp, 2)
        # print(f"{upper} upper violations, {lower} lower violations")
        with open("./results/compute_violation.txt", "a+") as f:
            f.write(f"{run.display_name}:{lower}:{upper}\n")

        upper, lower = compute_violations(result2, 1)

        # print(f"GLOBAL {upper} upper violations, {lower} lower violations")
        with open("./results/global_violation.txt", "a+") as f:
            f.write(f"{run.display_name}:{lower}:{upper}\n")

        upper, lower = compute_action_violations(df_comp)

        # print(f"[COMPUTE] action {run.name}: {upper} upper violations, {lower} lower violations")
        with open("./results/compute_action_violation.txt", "a+") as f:
            f.write(f"{run.display_name}:{lower}:{upper}\n")

        # plot_violations(
        #     df_comp,
        #     title=f"Compute Violations – {run.display_name}",
        #     filename=f"{run.name}_compute.pdf"
        # )

        # plot_ma("lat", "Moving Average of Network Latency", df_net, run.display_name)
        # plot_ma("lat", "Moving Average of Compute Latency", df_comp, run.display_name)

    plot_all_runs_scatter(
        all_network_runs,
        title="Network Response time Scatter",
        filename=f"{name}_all_runs_network_scatter.pdf", devider=2
    )

    plot_all_runs_scatter(
        all_compute_runs,
        title="Compute Response time Scatter",
        filename=f"{name}_all_runs_compute_scatter.pdf", devider=2
    )

    plot_all_runs_scatter(
        all_latency_runs,
        title="Response time Violations",
        filename=f"{name}_all_runs_total_scatter.pdf", devider=1
    )

    plot_all_runs_boxplot(all_network_runs,
                          title="Network Response time Distribution",
                          filename=f"{name}_all_runs_network_boxplot.pdf", devider=2)
    plot_all_runs_boxplot(
        all_compute_runs,
        title="Compute Response time Distribution",
        filename=f"{name}_all_runs_compute_boxplot.pdf", devider=2
    )

    plot_all_runs_boxplot(
        all_latency_runs,
        title="Response time Distribution",
        filename=f"{name}_all_runs_total_boxplot.pdf", devider=1
    )
