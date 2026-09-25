import torch
from stable_baselines3 import PPO
import wandb
from env_sa import General_Env

print("CUDA available:", torch.cuda.is_available())
print("Device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU only")


steps=[32]
learning_rates=[1e-4]
#agent will interact with env a total of this timestamps
total_timesteps=[1024]

for step in steps:
    for learning_rate in learning_rates:
        for total_timestep in total_timesteps:
            env = General_Env(total_timestep)
            wandb.init(project=f'Train RL for ICSOC tests', name=f'run {step} {learning_rate} {total_timestep}')

            model = PPO(
                "MlpPolicy",
                env,
                n_steps=step,  # The number of steps to run for each environment per update
                verbose=1,  # logs
                learning_rate=learning_rate,
                seed=42,
                device="cuda" if torch.cuda.is_available() else "cpu"
            )
            for episode in range(128):
                model.learn(total_timesteps=total_timestep)
            wandb.finish()
            model.save(f"../models/sa_new_latency_model_{step}_{learning_rate}_{total_timestep}")
