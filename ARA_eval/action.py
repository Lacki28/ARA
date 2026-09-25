import pandas as pd
import re



data = {}
data_in_percent = {}
total_actions = {}
file1 = "./results/compute_action.txt"
file2 = "./new/results/compute_action.txt"

with open(file1, "r") as f1, open(file2, "r") as f2:
    raw = f1.read() + f2.read()

blocks = re.split(r"\n(?=[A-Za-z].*:action)", raw.strip())
for block in blocks:
    lines = block.strip().split("\n")
    header = lines[0].replace(":action", "").strip()

    counts = {}
    for line in lines[1:]:
        m = re.match(r"(\d\.\d)\s+(\d+)", line)

        if m:
            action = int(float(m.group(1)))
            count = int(m.group(2))
            counts[action] = count

    # Convert to percentages

    if header in data:
        for k, v in counts.items():
            data[header][k] = data[header].get(k, 0) + v
    else:
        data[header] = counts




file1 = "./results/network_action.txt"
file2 = "./new/results/network_action.txt"

with open(file1, "r") as f1, open(file2, "r") as f2:
    raw = f1.read() + f2.read()

blocks = re.split(r"\n(?=[A-Za-z].*:action)", raw.strip())
for block in blocks:
    lines = block.strip().split("\n")
    header = lines[0].replace(":action", "").strip()

    counts = {}
    for line in lines[1:]:
        m = re.match(r"(\d\.\d)\s+(\d+)", line)

        if m:
            action = int(float(m.group(1)))
            count = int(m.group(2))
            counts[action] = count

    if header in data:
        for k, v in counts.items():
            data[header][k] = data[header].get(k, 0) + v
    else:
        data[header] = counts


for k, v in data.items():
    total = sum(v.values())
    if k in total_actions:
        total_actions[k] += total
    else:
        total_actions[k] = total

    percentages = {a: 100 * v.get(a, 0) / total for a in [0, 1, 2]}
    data_in_percent[k] = percentages
# print("HERE")
# print(data_in_percent)
# print(data)
# exit(0)
labels = {
    0: "0 (increase resources)",
    1: "1 (no intervention)",
    2: "2 (decrease resources)"
}

methods = list(data_in_percent.keys())
# labels = list(data_in_percent.keys())
methods=["PPO", "reactive", "SA_PPO", "proactive"]
for action in [0, 1, 2]:
    row = f"{labels[action]} & "
    row += " & ".join(f"{data_in_percent[m][action]:.2f}\\%" for m in methods)
    row += " \\\\"
    print(row)









data = {} #header:{lower, upper}
filename = "./results/compute_action_violation.txt"

with open(filename, "r") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue

        line_parts = line.split(":")
        header = line_parts[0].strip()

        nums = [int(x) for x in line_parts[1:]]

        # initialize if new
        if header not in data:
            data[header] = {"lower": 0, "upper": 0, "total": 0}

        # accumulate values
        if len(nums) > 0:
            data[header]["lower"] += nums[0]
            data[header]["total"] += nums[0]
        if len(nums) > 1:
            data[header]["upper"] += nums[1]
            data[header]["total"] += nums[1]

filename = "./results/network_action_violation.txt"

with open(filename, "r") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue

        line_parts = line.split(":")
        header = line_parts[0].strip()

        nums = [int(x) for x in line_parts[1:]]

        # initialize if new
        if header not in data:
            data[header] = {"lower": 0, "upper": 0, "total": 0}

        # accumulate values
        if len(nums) > 0:
            data[header]["lower"] += nums[0]
            data[header]["total"] += nums[0]

        if len(nums) > 1:
            data[header]["upper"] += nums[1]
            data[header]["total"] += nums[1]
data_in_percent={}
for k, v in data.items():
    total = total_actions[k]
    percentages = {a: 100 * v.get(a, 0) / total for a in ['lower', 'upper', 'total']}
    data_in_percent[k] = percentages
# methods = list(data_in_percent.keys())
methods=["PPO", "reactive", "SA_PPO", "proactive"]# todo!!
print("\n")
for violation in ["lower", "upper", "total"]:
    row = f"{violation} & "
    row += " & ".join(f"{data_in_percent[m][violation]:.2f}\\%" for m in methods)
    row += " \\\\"
    print(row)
