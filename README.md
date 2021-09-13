# d4rl_ski

### Directory Structure

```bash
├── main.py
├── model.py
├── replay_memory.py
├── sac.py
├── utils.py
├── get_logger.py

├── sk_data_and_model (make data for training SK model)
├── model_pt (save model parameters with best performance)
├── reward_results (save train/test rewards as npy)
└── transition_model_loss (save train/test MSE loss of transition model in logger and plotting forms)
``` 

### Training Transition Model
```
e.g. python d4rl_transition_model.py --relative_gaussian_noise 0.1 --epochs 150
```

### Training
```
e.g. python main.py --env-name HalfCheetah-v2 --mode ski --max_rollout_len 50
```
