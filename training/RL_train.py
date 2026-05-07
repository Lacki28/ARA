import torch
from stable_baselines3 import PPO

from ppo_env import PPO_Env

print("CUDA available:", torch.cuda.is_available())
print("Device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU only")


steps=[32]
learning_rates=[1e-3, 1e-4, 1e-5]
#agent will interact with env a total of this timestamps
total_timesteps=[128, 256, 512]
for step in steps:
    for learning_rate in learning_rates:
        for total_timestep in total_timesteps:
            wandb.init(project=f'Train Coordinator TD3',name=f"{steps}")
            env = PPO_Env()
            model = PPO(
                "MlpPolicy",
                env,
                # How many steps we explore before update is done
                n_steps=step,  # The number of steps to run for each environment per update
                verbose=1,  # logs
                learning_rate=learning_rate,
                seed=42,
                device="cuda" if torch.cuda.is_available() else "cpu"
            )
            model.learn(total_timesteps=total_timestep) #this is the min number of samples we want to train our method on
            env.finish()
            model.save(f"../models/PPO_coord_{step}_{learning_rate}_{total_timestep}")

