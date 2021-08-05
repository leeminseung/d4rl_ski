# d4rl_ski

### Directory Structure

```bash
├── main.py
├── model.py
├── replay_memory.py
├── sac.py
├── utils.py
├── score_results (save train/test rewards as npy)
├── result (save train/test MSE loss of transition model)
└── model_pt (save a model of best performance)
``` 
### Training
```
e.g. python main.py --env-name HalfCheetah-v2 --mode ski --max_rollout_len 50
```
