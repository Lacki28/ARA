# ARA
Repository for: "ARA: Adaptive Resource Agents for QoS Control in Multi-Domain Environments"


# How to Run the Experiments
## Deployment
Deployment instructions can be read in 
```bash
cat iContinuumAdjustments/README.md
```

## Training
Run the training of the RA with  
```bash
python training/RL_train.py
```


## Run Experiments
All experiment scripts are located in the **`run-scripts/`** directory.  
To start an experiment, navigate to this directory and execute the desired script:

```bash
bash run_scripts/[experiment_name].sh
```
-> Make sure the right microservice is running.

## Plot results
To plot the results you may want to use the scripts in **`ARA_eval_reward/`**.

```bash
bash run_scripts/[experiment_name].sh
```
-> Make sure the right microservice is running.

## Reward Evaluation
The evaluation of the reward is in **`training/eval_reward/`**.