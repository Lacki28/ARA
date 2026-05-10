import threading
import time
from new_agent import Agent
from utils import get_network_latency, get_compute_latency, get_total_latency_value, get_real_cpu_usage, \
    apply_compute_action, apply_network_action, get_total_delay
import wandb
from flask import Flask, request, jsonify
import argparse
import requests
LOOP_INTERVAL_SECONDS = 60

app = Flask(__name__)

class SLO():
    def __init__(self):
# =============== High level Metrics ===============
        self.rewards = {"network_latency": {"lower": 1000, "upper": 1500},
                        "compute_latency": {"lower": 1000, "upper": 1500}}
        #First thing that we update to get the current state
        self.state = {"network_latency": None,
                      "compute_latency": None,
                      "total_latency": None}
        # We get the lookup definitions
        self.state_update = {"network_latency": get_network_latency,
                             "compute_latency": get_compute_latency,
                             "total_latency": get_total_latency_value}
# =============== Low level Metrics ===============
        # We get the lookup definitions from
        self.action_space = {"network_delay": {"lower": 50, "upper": 0},
                             "compute_cpu": {"lower": 100, "upper": 900}}
        self.state_action_mapping = {"network_latency": "network_delay", "compute_latency": "compute_cpu"}
        self.action_update = {
            "network_delay": {"func": apply_network_action, "params": {"action": None, "current_resources": None}},
            "compute_cpu": {"func": apply_compute_action, "params": {"action": None, "current_resources": None}}}

    def update_state(self):
        for key, update_fn in self.state_update.items():
            self.state[key] = update_fn()

    def update_SLOs(self, high_level_resource_metric, data):
        self.rewards[high_level_resource_metric]["lower"] = data.get("lower", self.rewards[high_level_resource_metric]["lower"])
        self.rewards[high_level_resource_metric]["upper"] = data.get("upper", self.rewards[high_level_resource_metric]["upper"])


    def rule_reward(self, value, lower=None, upper=None):

        if upper is not None and lower is None:
            print("CASE 1")
            if value > upper:
                return -(value - upper) / upper #proportional
            return 2 - (value / upper)

        if lower is not None and upper is None:
            print("CASE 2")
            if value < lower:
                    return -(lower - value) / lower #proportional
            return 2 - (lower / value) #if I am more above it is slightly better

        if lower is not None and upper is not None:
            midpoint = (lower + upper) / 2
            half_range = (upper - lower) / 2
            print(f"CASE 3 {lower}<{value}<{upper}")

            if value < lower:
                print("value < lower")
                return -0.5
                # return -(lower - value) / lower #min negative reward
            if value > upper:
                print("value > upper")
                return -0.5
                # return -(value - upper) / upper #min negative reward
            return 3 - abs(value - midpoint) / half_range #Max positive reward
        return 0

    def compute_reward(self):
        total = 0
        new_rewards={}
        for key, rule in self.rewards.items():
            r = self.rule_reward(
                value=self.state[key],
                lower=rule.get("lower"),
                upper=rule.get("upper")
            )
            new_rewards[key]=r
            total += r
        return total, new_rewards

SLOs = SLO()

@app.route("/slo/update", methods=["POST"])
def update_slo():
    data = request.json
    metric = data.get("metric")

    if metric not in SLOs.rewards:
        return jsonify({"error": "Metric not found"}), 404

    SLOs.update_SLOs(metric, data)
    return jsonify({"message": "SLO updated", "rewards": SLOs.rewards})

def listen_to_SLO_updates():
    app.run(host='127.0.0.1', port=5000, threaded=True)


def main_loop(wandb_name, strategy, args):
    wandb.init(project=f'{wandb_name}', name=f"new {strategy}: bs:{args.batch_size} ep:{args.epochs} save? {args.save_model}")
    route_thread = threading.Thread(target=listen_to_SLO_updates)
    route_thread.start()

    print("starting the main loop")
    agents = []
    step = 0
    print("starting the main loop")
    SLOs.update_state()
    for action in SLOs.action_update:
        state_key = next(k for k, v in SLOs.state_action_mapping.items() if v == action)
        agent = Agent(SLOs.rewards[state_key], SLOs.state[state_key],state_key, SLOs.action_space[action],
                      SLOs.action_update[action], action, strategy,args.batch_size, args.epochs, args.save_model)
        agents.append(agent)
    while True:
        print("perform")
        global_statevector=agents[0].get_state_vector()+agents[1].get_state_vector()
        i=0
        for agent in agents:
            agent.update_limits(SLOs.rewards)
            if strategy=="SA_PPO":
                agent.perform_global_action(global_statevector, i)
            else:
                agent.perform_action()
            i+=1
        print("waiting")
        response = requests.post("http://172.16.0.102:32702/clear_data")
        time.sleep(LOOP_INTERVAL_SECONDS)
        SLOs.update_state()
        total_reward, rewards = SLOs.compute_reward()
        wandb.log({"reward": total_reward})
        print(SLOs.state)
        wandb.log({"total_latency": SLOs.state})

        for agent in agents:
            agent.add_trace(rewards, SLOs.state)
        step += 1
        print("-" * 50)

def parse_args():
    parser = argparse.ArgumentParser()

    # Optional SLO parameters
    parser.add_argument("--network-lower", type=float, default=1000)
    parser.add_argument("--network-upper", type=float, default=1500)
    parser.add_argument("--compute-lower", type=float, default=1000)
    parser.add_argument("--compute-upper", type=float, default=1500)
    parser.add_argument("--wandb_name", type=str, default="MVP tests")
    parser.add_argument("--strategy", type=str, default="PPO")
    parser.add_argument("--batch_size", type=int, default=14)
    parser.add_argument("--epochs", type=int, default=0)
    parser.add_argument("--save_model", type=int, default=0)

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    SLOs.rewards["network_latency"]["lower"] = args.network_lower
    SLOs.rewards["network_latency"]["upper"] = args.network_upper
    SLOs.rewards["compute_latency"]["lower"] = args.compute_lower
    SLOs.rewards["compute_latency"]["upper"] = args.compute_upper
    main_loop(args.wandb_name, args.strategy, args)
