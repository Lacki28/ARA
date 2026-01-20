import time
from single_agent import Agent
from utils import get_network_latency, get_compute_latency, get_total_latency_value, get_real_cpu_usage, \
    apply_compute_action, apply_network_action, get_total_delay
import wandb

# === Config ===
LOOP_INTERVAL_SECONDS = 30


class SLO():
    def __init__(self):
# =============== High level Metrics ===============
        # We get the reward definitions from inSwitch
        self.rewards = {"network_latency": {"lower": 750, "upper": 1250},
                        "compute_latency": {"lower": 250, "upper": 750}}
        # We get the state definitions from inSwitch
        #First thing that we update to get the current state
        self.state = {"network_latency": None,
                      "compute_latency": None,
                      "total_latency": None}
        # We get the lookup definitions from inSwitch
        self.state_update = {"network_latency": get_network_latency,
                             "compute_latency": get_compute_latency,
                             "total_latency": get_total_latency_value}
# =============== Low level Metrics ===============
        # We get the lookup definitions from inSwitch/InNet/IDO
        self.action_space = {"network_delay": {"lower": 0, "upper": 10},
                             "compute_cpu": {"lower": 100, "upper": 900}}
        self.state_action_mapping = {"network_latency": "network_delay", "compute_latency": "compute_cpu"}
        self.action_update = {
            "network_delay": {"func": apply_network_action, "params": {"action": None, "current_resources": None}},
            "compute_cpu": {"func": apply_compute_action, "params": {"action": None, "current_resources": None}}}

    def update_state(self):
        for key, update_fn in self.state_update.items():
            self.state[key] = update_fn()

    def rule_reward(self, value, lower=None, upper=None):
        # Case 1: Only upper bound
        if upper is not None and lower is None:
            if value > upper:
                return -(value - upper) / upper #proportional
            return 2 - (value / upper)

        # Case 2: Only lower bound
        if lower is not None and upper is None:
            if value < lower:
                    return -(lower - value) / lower #proportional
            return 2 - (lower / value) #if I am more above it is slightly better

        # Case 3: Range with midpoint target
        if lower is not None and upper is not None:
            midpoint = (lower + upper) / 2
            half_range = (upper - lower) / 2

            if value < lower:
                return -(lower - value) / lower #min negative reward
            if value > upper:
                return -(value - upper) / upper #min negative reward
            # Inside the range → reward increases near midpoint
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

run = wandb.init(project='generalized MVP deployed tests with lacki ms1', name=f"Multi Agent Approach")

# === Main loop ===
def main_loop():
    print("starting the main loop")
    agents = []
    step = 0
    SLOs = SLO()
    print("starting the main loop")
    SLOs.update_state()
    for action in SLOs.action_update:
        state_key = next(k for k, v in SLOs.state_action_mapping.items() if v == action)
        agent = Agent(SLOs.rewards[state_key], SLOs.state[state_key],state_key, SLOs.action_space[action],
                      SLOs.action_update[action], action)
        agents.append(agent)
    while True:
        print("perform")
        for agent in agents:
            agent.perform_action()
        print("waiting")
        time.sleep(LOOP_INTERVAL_SECONDS)
        SLOs.update_state()
        total_reward, rewards = SLOs.compute_reward()
        print(f"Reward {total_reward}")
        wandb.log({"reward": total_reward})
        print(SLOs.state)
        wandb.log({"total_latency": SLOs.state})

        for agent in agents:
            agent.add_trace(rewards, SLOs.state)
        step += 1
        print("-" * 50)


if __name__ == "__main__":
    main_loop()
