import requests

from central_management import increase_network_latency, decrease_k8s_latency, increase_k8s_latency, \
    decrease_network_latency
import numpy as np
import re

# === perform actions ===
def apply_network_action(action, switch_delay_ms):
    if action==0:
        switch_delay_ms = decrease_network_latency(switch_delay_ms)
    elif action == 2:
        switch_delay_ms = increase_network_latency(switch_delay_ms)
    # Actions 1 and 4 are "maintain" → no-op
    print(f"[Action] Applied action {action} {switch_delay_ms}")
    return switch_delay_ms

def apply_compute_action(action, cpu_limit):
    if action == 0:
        cpu_limit=decrease_k8s_latency(cpu_limit)
    elif action == 2:
        cpu_limit=increase_k8s_latency(cpu_limit)
    # Actions 1 and 4 are "maintain" → no-op
    print(f"[Action] Applied action {action} {cpu_limit}")
    return cpu_limit
# === state lookup functions ===

def parse_timedelta_str(t_str):
    try:
        h, m, s = t_str.split(":")
        seconds = float(s)
        minutes = int(m)
        hours = int(h)
        total_seconds = hours * 3600 + minutes * 60 + seconds
        return total_seconds * 1000  # ms
    except Exception as e:
        print(f"[!] Failed to parse time string '{t_str}': {e}")
        return None

def get_total_latency(avg_n=100):
    try:
        # response times are written with locust
        with open("../response_times.csv") as f:
            lines = f.readlines()[1:]  # Skip header
        if len(lines) < avg_n:
            return 200.0
        latencies = [float(row.strip().split(",")[3]) for row in lines[-avg_n:]]
        return np.mean(latencies)
    except Exception as e:
        print(f"[!] Failed to parse response_times.csv: {e}")
        return 200.0

def get_service_latency_from_api(avg_n=100):
    try:
        url = "http://172.16.0.1:32663/get_processing_times/microservice1"
        response = requests.get(url)
        response.raise_for_status()

        data = response.json()
        time_entries = data["processing_times"][-avg_n:]
        latencies_ms = [parse_timedelta_str(entry["processing_time"]) for entry in time_entries]
        latencies_ms = [l for l in latencies_ms if l is not None]

        return np.mean(latencies_ms) if latencies_ms else 120.0
    except Exception as e:
        print(f"[!] Failed to get service latency: {e}")
        return 120.0

def get_network_latency():
    return get_total_latency()- get_service_latency_from_api()

def get_compute_latency():
    return get_service_latency_from_api()

def get_total_latency_value():
    return get_total_latency()

def get_total_delay():
    try:
        with open("delay.txt") as f:
            last_line = f.readlines()[-1]
        return float(last_line)
    except Exception as e:
        print(f"[!] Failed to parse delay.csv: {e}")
        return 200.0

def get_real_cpu_usage():
    PROMETHEUS_METRIC_URL = "http://[ip_removed_for_submission]:8008/prometheus/metrics/ALL/ALL/txt"
    TARGET_LABELS = {
        "agent": "[ip_removed_for_submission]",  # 98
        "datasource": "2.1",
        "host": "earthtiger"  # metalrabbit
    }
    try:
        response = requests.get(PROMETHEUS_METRIC_URL)
        response.raise_for_status()
        metrics = response.text.splitlines()

        for line in metrics:
            if line.startswith("sflow_cpu_utilization"):
                match = re.match(r'sflow_cpu_utilization\{([^}]*)\}\s+([0-9.]+)', line)
                if match:
                    labels_str, value = match.groups()
                    labels = dict(item.split("=") for item in labels_str.split(","))
                    labels = {k: v.strip('"') for k, v in labels.items()}

                    if all(labels.get(k) == v for k, v in TARGET_LABELS.items()):
                        cpu = float(value)
                        return np.clip(cpu, 0.0, 100.0)

        print("[!] CPU metric not found.")
        return 1000.0
    except Exception as e:
        print(f"[!] Failed to fetch CPU metric: {e}")
        return 1000.0
