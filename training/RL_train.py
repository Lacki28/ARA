import torch

from single_agent_env import General_Env

print("CUDA available:", torch.cuda.is_available())
print("Device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU only")


steps=[32]
learning_rates=[1e-4]#,1e-4] #1e-5 too small
#agent will interact with env a total of this timestamps
total_timesteps=[512] #128 too small

for step in steps:
    for learning_rate in learning_rates:
        for total_timestep in total_timesteps:
            # === Environment & Training === DroneObjectDetectionEnv
            env = General_Env(step)
            check_env(env, warn=True)

            model = PPO(
                "MlpPolicy",
                env,
                # How many steps we explore before update is done
                n_steps=step,  # The number of steps to run for each environment per update
                verbose=1,  # logs
                learning_rate=learning_rate,
                seed=42,
                # tensorboard_log="./ppo_drone_tensorboard/test_run_anna",
                device="cuda" if torch.cuda.is_available() else "cpu"
            )
            # callback = DroneLatencyTensorboardCallback(env)
            # total_timesteps is the overall number of agent-environment interactions (steps) performed during training
            for episode in range(128):
                model.learn(total_timesteps=total_timestep)
                env.finish()
            model.save(f"./models/new_latency_model_{step}_{learning_rate}_{total_timestep}")

