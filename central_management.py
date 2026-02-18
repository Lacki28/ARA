import requests
import numpy as np

K8S_API_URL = "http://[ip_removed_for_submission]:5050"

NETWORK_API_URL = "http://[ip_removed_for_submission]/network/delay"
NETWORK_BANDWIDTH_URL = "http://[ip_removed_for_submission]/network/bandwidth"


def increase_target(last_latency, min_range, max_range):
    # return last_latency + (max_range / 10)
    steps, idx = log_steps(last_latency, min_range, max_range)
    idx = min(idx + 1, len(steps) - 1)
    return steps[idx]


def decrease_target(last_latency, min_range, max_range):
    # return last_latency - (max_range / 10)
    steps, idx = log_steps(last_latency, min_range, max_range)
    idx = max(0, idx - 1)
    return steps[idx]


# We adjust the target in logarithmic steps
def log_steps(last_latency, min_range, max_range):
    nr_steps = 10
    steps = np.logspace(0, 1, num=nr_steps, base=10)
    # between 0 and 1
    steps = (steps - steps.min()) / (steps.max() - steps.min())
    # actual range
    steps = steps * (max_range - min_range) + min_range
    idx = (np.abs(steps - last_latency)).argmin()
    return steps, idx


def increase_k8s_latency(cpu_limit_m):
    deployments = {
        "microservice1-deployment": {
            "replicas": 1,
            "cpu_limit_m": cpu_limit_m,
            "memory_limit": "512Mi"
        }
    }
    new_cpu_limit = cpu_limit_m
    for name, config in deployments.items():
        new_cpu_limit = config["cpu_limit_m"] = decrease_target(config["cpu_limit_m"], 100, 900)
        print('Central management cpu decrease limit')
        _post_cpu_update(name, config["cpu_limit_m"], config["memory_limit"])
    return new_cpu_limit


def decrease_k8s_latency(cpu_limit_m):
    deployments = {
        "microservice1-deployment": {
            "replicas": 1,
            "cpu_limit_m": cpu_limit_m,
            "memory_limit": "512Mi"
        }
    }
    new_cpu_limit = cpu_limit_m
    for name, config in deployments.items():
        new_cpu_limit = config["cpu_limit_m"] = increase_target(config["cpu_limit_m"], 100, 900)
        print('Central management cpu increase limit')
        _post_cpu_update(name, config["cpu_limit_m"], config["memory_limit"])
    return new_cpu_limit


def _post_cpu_update(deployment_name, cpu_m, memory_limit):
    cpu_str = f"{cpu_m}m"
    print(f"new_cpu_limit {cpu_m}m")
    payload = {
        "deployment_name": deployment_name,
        "new_cpu_limit": cpu_str,  # max CPU
        "new_memory_limit": memory_limit,
        "namespace": "default"
    }
    try:
        response = requests.post(f"{K8S_API_URL}/k8s/resources", json=payload)
        if response.status_code == 200:
            print(f"[+] Updated {deployment_name} to {cpu_str} CPU")
        else:
            print(f"[!] Failed to update CPU: {response.text}")
    except Exception as e:
        print(f"[!] Error updating CPU for {deployment_name}: {e}")


def update_network_latency(delay_ms):
    try:
        response = requests.post(NETWORK_API_URL, json={"delay_ms": delay_ms})
        if response.status_code == 200:
            print(f"[+] Network delay updated to {delay_ms}ms")
        else:
            print(f"[!] Failed to update delay: {response.status_code}, {response.text}")
    except Exception as e:
        print(f"[!] HTTP error while updating delay: {e}")


def increase_network_latency(switch_delay_ms):
    switch_delay_ms = increase_target(switch_delay_ms, 0, 50)
    update_network_latency(switch_delay_ms)
    return switch_delay_ms


def decrease_network_latency(switch_delay_ms):
    switch_delay_ms = decrease_target(switch_delay_ms, 0, 50)
    update_network_latency(switch_delay_ms)
    return switch_delay_ms


bandwidth_mbps = "20mbit"
response = requests.post(
    NETWORK_API_URL,
    json={"rate": bandwidth_mbps, "burst": "64kbit", "latency": "400ms"}
)


def update_network_bandwidth(bandwidth):
    try:
        response = requests.post(NETWORK_BANDWIDTH_URL,
                                 json={"rate": bandwidth_mbps, "burst": "64kbit", "latency": "400ms"})
        if response.status_code == 200:
            print(f"[+] Network delay updated to {bandwidth_mbps}")
        else:
            print(f"[!] Failed to update bandwidth: {response.status_code}, {response.text}")
    except Exception as e:
        print(f"[!] HTTP error while updating delay: {e}")


def increase_network_bandwidth(bandwidth):
    """Increase network bandwidth by 5ms."""
    new_bandwidth = increase_target(bandwidth, 0, 100)
    update_network_latency(new_bandwidth)
    return new_bandwidth


def decrease_network_bandwidth(bandwidth):
    """Decrease network bandwidth by 5ms."""
    new_bandwidth = decrease_network_bandwidth(bandwidth, 0, 100)
    update_network_latency(new_bandwidth)
    return new_bandwidth


def get_bandwidth():
    try:
        response = requests.get(NETWORK_BANDWIDTH_URL)
        if response.status_code == 200:
            print(f"[+] {response.status_code}, {response.text}")
        else:
            print(f"[!] Failed to update delay: {response.status_code}, {response.text}")
    except Exception as e:
        print(f"[!] HTTP error while updating delay: {e}")
