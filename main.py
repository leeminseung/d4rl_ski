import argparse
from evaluation import d4rl_evaluation
import datetime
import gym
import numpy as np
import itertools
import torch
import os

from torch.utils.data.dataset import Dataset
from sac import SAC
from replay_memory import ReplayMemory
from d4rl_transition_model import EntireEnsembleModel
from Simple1DCNN import Simple1DCNN
from sk_data_and_model.SK_model import *
from utils import ActionSpaceBox, get_dataloader
import random
import d4rl
import tqdm
from utils import StdScaler
from evaluation import *
        
parser = argparse.ArgumentParser(description='PyTorch Soft Actor-Critic Args')
parser.add_argument('--env-name', default="HalfCheetah-v2",
                    help='Mujoco Gym environment (default: HalfCheetah-v2)')
parser.add_argument('--policy', default="Gaussian",
                    help='Policy Type: Gaussian | Deterministic (default: Gaussian)')
parser.add_argument('--eval', type=bool, default=True,
                    help='Evaluates a policy a policy every 10 episode (default: True)')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor for reward (default: 0.99)')
parser.add_argument('--tau', type=float, default=0.005, metavar='G',
                    help='target smoothing coefficient(τ) (default: 0.005)')
parser.add_argument('--lr', type=float, default=0.0003, metavar='G',
                    help='learning rate (default: 0.0003)')
parser.add_argument('--alpha', type=float, default=0.2, metavar='G',
                    help='Temperature parameter α determines the relative importance of the entropy\
                            term against the reward (default: 0.2)')
parser.add_argument('--automatic_entropy_tuning', type=bool, default=False, metavar='G',
                    help='Automaically adjust α (default: False)')
parser.add_argument('--seed', type=int, default=123456, metavar='N',
                    help='random seed (default: 123456)')
parser.add_argument('--batch_size', type=int, default=256, metavar='N',
                    help='batch size (default: 256)')
parser.add_argument('--num_steps', type=int, default=1000001, metavar='N',
                    help='maximum number of steps (default: 1000000)')
parser.add_argument('--hidden_size', type=int, default=256, metavar='N',
                    help='hidden size (default: 256)')
parser.add_argument('--updates_per_step', type=int, default=1, metavar='N',
                    help='model updates per simulator step (default: 1)')
parser.add_argument('--start_steps', type=int, default=10000, metavar='N',
                    help='Steps sampling random actions (default: 10000)')
parser.add_argument('--target_update_interval', type=int, default=1, metavar='N',
                    help='Value target update per no. of updates per step (default: 1)')
parser.add_argument('--replay_size', type=int, default=1000000, metavar='N',
                    help='size of replay buffer (default: 1000000)')
parser.add_argument('--cuda', default=True,
                    help='run on CUDA (default: True)')
parser.add_argument('--mode', type=str, default='ski',
                    help='Policy Type: SAC | SKI (default: Gaussian)')
parser.add_argument('--max_rollout_len', type=int, default=50,
                    help='Maximum of rollout length')
parser.add_argument('--scaler', type=str, default='std',
                    help='Data Scaler')
parser.add_argument('--cql_epochs', type=int, default=1000,
                    help='Training epoch when using CQL')
parser.add_argument('--data', type=str, default='d4rl',
                    help='Training Data (SKI or D4RL)')
parser.add_argument('--seq_length', type=int, default=4,
                    help='Sequence length of SK Model')
args = parser.parse_args()

# Environment
# env = NormalizedActions(gym.make(args.env_name))

device = 'cuda:1' if torch.cuda.is_available() else 'cpu'
start_time = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M")

mode = args.mode

# d4rl data
if args.data == 'd4rl':
    env = gym.make('halfcheetah-medium-expert-v2')
    dataset = d4rl.qlearning_dataset(env)
    scaler = StdScaler(dataset, device=device)
    d4rl_obseravtions = dataset['observations']
    
    transition_model = EntireEnsembleModel(input_size=17+6, encoder_hidden_size=512, encoder_output_size=512, transition_model_hidden_size=512, transition_model_output_size=18, ensemble_size=5, learning_rate=0.0005, device=device)
    transition_model.load_model("2021-08-29-22-22")
    transition_model.to(device)

elif args.data == 'ski':
    dataset = np.load("/home/lms/d4rl_ski/sk_data_and_model/step_size_1,sequential_length_55.npy")
    dataset = dataset.astype(np.float32)
    
    transition_model = {}
    transition_model['encoder'] = Encoder(input_dim=6, hidden_dim=32, args=args).to(device)
    transition_model['ensemble'] = EnsembleProbabilisticTransitionModel(hidden_dim=32, output_dim=1, ensemble_size=5).to(device)

    transition_model['encoder'].load_model("sk_data_and_model/model_loss/2021-09-01-11-49")
    transition_model['ensemble'].load_model("sk_data_and_model/model_loss/2021-09-01-11-49")
    # transition_model = Simple1DCNN(input_size=6, hidden_size=30, output_size=1, sequential_length=4, device=device)
    # transition_model.to(device)

# cql dataloader
cql_dataloader = get_dataloader(dataset, device=device, data_mode=args.data)

env = gym.make(args.env_name)
env.seed(args.seed)
env.action_space.seed(args.seed)

torch.manual_seed(args.seed)
np.random.seed(args.seed)


# Agent
if args.data == 'd4rl':
    agent = SAC(env.observation_space.shape[0], env.action_space, args)
elif args.data == 'ski':
    action_space = ActionSpaceBox(shape=(1,), high=dataset[:, 0, -1].max(), low=dataset[:, 0, -1].min())
    agent = SAC(5, action_space, args)

# Memory
memory = ReplayMemory(args.replay_size, args.seed)

# Training Loop
total_numsteps = 0
updates = 0
train_avg_rewards = list()
test_avg_rewards = list()

for i_episode in itertools.count(1):
    episode_reward = 0
    episode_steps = 0
    done = False

    # 'sac' mode means vanilla SAC without any modification
    if mode == 'sac':
        rollout_len = 0
        state = env.reset()
        while (not done) and (rollout_len < args.max_rollout_len):
        # while not done:
            if args.start_steps > total_numsteps:
                action = env.action_space.sample()  # Sample random action
            else:
                action = agent.select_action(state)  # Sample action from policy

            if len(memory) > args.batch_size:
                # Number of updates per step in environment
                for i in range(args.updates_per_step):
                    # Update parameters of all the networks
                    critic_1_loss, critic_2_loss, policy_loss, ent_loss, alpha = agent.update_parameters(memory, args.batch_size, updates)
                    updates += 1

            next_state, reward, done, _ = env.step(action) # Step
            episode_steps += 1
            total_numsteps += 1
            episode_reward += reward
            mask = 1 if episode_steps == env._max_episode_steps else float(not done)
            memory.push(state, action, reward, next_state, mask) # Append transition to memory

            state = next_state
            rollout_len += 1
            
            if total_numsteps % 1000 == 0:
                agent.save_model(env_name='halfcheetah', suffix=start_time + '_' + str(args.max_rollout_len))            

        if total_numsteps > args.num_steps:
            break

        train_avg_rewards.append(episode_reward)

        print("Episode: {}, total numsteps: {}, episode steps: {}, reward: {}".format(i_episode, total_numsteps, episode_steps, round(episode_reward, 2)))

    # 'ski' mode means traning agent process continues in a trained transition model
    elif mode == 'ski':
        rollout_len = 0
        state = random.choice(dataset['observations'])

        while rollout_len < args.max_rollout_len:
            if args.start_steps > total_numsteps:
                action = env.action_space.sample()  # Sample random action
            else:
                action = agent.select_action(state)  # Sample action from policy

            if len(memory) > args.batch_size:
                # Number of updates per step in environment
                for i in range(args.updates_per_step):
                    # Update parameters of all the networks
                    critic_1_loss, critic_2_loss, policy_loss, ent_loss, alpha = agent.update_parameters(memory, args.batch_size, updates)
                    updates += 1

            state = scaler.scale(torch.tensor(state.copy(), dtype=torch.float, device=device), 'observations')
            action = scaler.scale(torch.tensor(action.copy(), dtype=torch.float, device=device), 'actions')
            
            # state = torch.tensor(state, dtype=torch.float, device=device)
            # action = torch.tensor(action, dtype=torch.float, device=device)
            next_state_reward = transition_model(state.unsqueeze(0), action.unsqueeze(0)) # Step

            next_state = next_state_reward.squeeze()[:17] # hard coded with 'HalfCheetah-v2'
            reward = next_state_reward.squeeze()[17]
            
            reward = scaler.inverse_scale(reward, 'rewards')
            state = scaler.inverse_scale(state, 'observations')
            next_state = scaler.inverse_scale(next_state, 'next_observations')
            action = scaler.inverse_scale(action, 'actions')

            state = state.cpu().detach().numpy()
            action = action.cpu().detach().numpy()
            reward = reward.cpu().detach().numpy()
            next_state = next_state.cpu().detach().numpy()

            episode_steps += 1
            total_numsteps += 1
            episode_reward += reward

            mask = 1 if episode_steps == env._max_episode_steps else float(not done)
            memory.push(state, action, reward, next_state, mask) # Append transition to memory

            state = next_state
            rollout_len += 1

        if total_numsteps > args.num_steps:
            break

        train_avg_rewards.append(episode_reward)

        print("Episode: {}, total numsteps: {}, episode steps: {}, reward: {}".format(i_episode, total_numsteps, episode_steps, round(episode_reward, 2)))

    elif mode == 'ski_with_cql':
        rollout_len = 0
        state = random.choice(dataset['observations'])

        while rollout_len < args.max_rollout_len:
            if args.start_steps > total_numsteps:
                action = env.action_space.sample()  # Sample random action
            else:
                action = agent.select_action(state)  # Sample action from policy

            if len(memory) > args.batch_size:
                for i in range(args.updates_per_step):
                    state_batch, action_batch, reward_batch, next_state_batch, mask_batch = memory.sample(batch_size=args.batch_size)

                    state_batch = torch.tensor(state_batch, dtype=torch.float32)
                    next_state_batch = torch.tensor(next_state_batch, dtype=torch.float32)
                    action_batch = torch.tensor(action_batch, dtype=torch.float32)
                    reward_batch = torch.tensor(reward_batch, dtype=torch.float32)
                    mask_batch = torch.tensor(mask_batch, dtype=torch.float32)

                    batch = state_batch, action_batch, reward_batch, next_state_batch, mask_batch
                    
                    critic_1_loss, critic_2_loss, policy_loss, ent_loss, alpha = agent.cql_update_parameters(batch, updates)
                    updates += 1

            state = scaler.scale(torch.tensor(state.copy(), dtype=torch.float, device=device), 'observations')
            action = scaler.scale(torch.tensor(action.copy(), dtype=torch.float, device=device), 'actions')
            
            # state = torch.tensor(state, dtype=torch.float, device=device)
            # action = torch.tensor(action, dtype=torch.float, device=device)
            next_state_reward = transition_model(state.unsqueeze(0), action.unsqueeze(0)) # Step

            next_state = next_state_reward.squeeze()[:17] # hard coded with 'HalfCheetah-v2'
            reward = next_state_reward.squeeze()[17]
            
            reward = scaler.inverse_scale(reward, 'rewards')
            state = scaler.inverse_scale(state, 'observations')
            next_state = scaler.inverse_scale(next_state, 'next_observations')
            action = scaler.inverse_scale(action, 'actions')

            state = state.cpu().detach().numpy()
            action = action.cpu().detach().numpy()
            reward = reward.cpu().detach().numpy()
            next_state = next_state.cpu().detach().numpy()

            episode_steps += 1
            total_numsteps += 1
            episode_reward += reward

            mask = 1 if episode_steps == env._max_episode_steps else float(not done)
            memory.push(state, action, reward, next_state, mask) # Append transition to memory

            state = next_state
            rollout_len += 1

        if total_numsteps > args.num_steps:
            break
    
    elif mode == 'offline':
        rollout_len = 0
        indice = random.choices(np.arange(dataset['observations'].shape[0]), k=1)

        while rollout_len < args.max_rollout_len:
            if args.start_steps > total_numsteps:
                action = env.action_space.sample()  # Sample random action
            else:
                action = agent.select_action(state)  # Sample action from policy

            if len(memory) > args.batch_size:
                # Number of updates per step in environment
                for i in range(args.updates_per_step):
                    # Update parameters of all the networks
                    critic_1_loss, critic_2_loss, policy_loss, ent_loss, alpha = agent.update_parameters(memory, args.batch_size, updates)
                    updates += 1

            state = dataset['observations'][indice].squeeze() # hard coded with 'HalfCheetah-v2'
            next_state = dataset['next_observations'][indice].squeeze() # hard coded with 'HalfCheetah-v2'
            reward = dataset['rewards'][indice].squeeze()
            action = dataset['actions'][indice].squeeze()
            mask = float(not dataset['terminals'][indice].squeeze())
            
            episode_steps += 1
            total_numsteps += 1
            episode_reward += reward

            # mask = 1 if episode_steps == env._max_episode_steps else float(not done)
            memory.push(state, action, reward, next_state, mask) # Append transition to memory

            state = next_state
            rollout_len += 1

        if total_numsteps > args.num_steps:
            break

        train_avg_rewards.append(episode_reward)

        print("Episode: {}, total numsteps: {}, episode steps: {}, reward: {}".format(i_episode, total_numsteps, episode_steps, round(episode_reward, 2)))

    elif mode == 'offline_sac':
        print('Current Episode:', i_episode)
        for row in cql_dataloader:
            # state_batch, action_batch, reward_batch, next_state_batch, mask_batch
            for i in range(args.updates_per_step):
                critic_1_loss, critic_2_loss, policy_loss, ent_loss, alpha = agent.update_parameters(row, args.batch_size, updates, feeding=True)
                updates += 1

    elif mode == 'cql':
        print('Current Episode:', i_episode)
        for row in cql_dataloader:
            # state_batch, action_batch, reward_batch, next_state_batch, mask_batch
            for i in range(args.updates_per_step):
                critic_1_loss, critic_2_loss, policy_loss, ent_loss, alpha = agent.cql_update_parameters(row, updates)
                updates += 1
    
    elif mode == 'combo': # ski data
        print('Current Episode:', i_episode)
        for row in cql_dataloader:
            # state_batch, action_batch, reward_batch, next_state_batch, mask_batch
            rollout_len = 0

            episode_index = random.choice(range(dataset.shape[0]))
            episode_data = dataset[episode_index]
            state_action = torch.tensor(episode_data, dtype=torch.float32, device=device)
            state_action = state_action[:4]
            state = state_action[-1,:-1].clone()

            while rollout_len < 3:    
                rollout_len += 1
                encoder_output = transition_model['encoder'](state_action.view(-1, 6*4))
                predict_mu, predict_std, _ = transition_model['ensemble'](encoder_output)
                next_current = torch.mean(predict_mu, dim=0)
                
                action = agent.select_action(state.unsqueeze(0).cpu(), evaluate=True)

                state_action = torch.roll(state_action, -1, dims=0)
                state_action[-1, -2] = next_current
                state_action[-1, :-2] = torch.tensor(episode_data[rollout_len+3, :-2].copy(), dtype=torch.float32)
                state_action[-1, -1] = torch.tensor(action.squeeze(0), dtype=torch.float32)

                next_state = state_action[-1, :-1]
                
                state_to_push = state.cpu().detach().numpy().copy()
                action_to_push = action.squeeze().copy()
                reward = -abs(next_current.item() * 0.391660 + 29.983252 - 30)
                next_state_to_push = next_state.cpu().detach().numpy().copy()
                mask = 1.

                memory.push(state_to_push, action_to_push, reward, next_state_to_push, mask) # Append transition to memory
                state = next_state
            
            if len(memory) > 20:
                state_batch_from_model, action_batch_from_model, reward_batch_from_model, next_state_batch_from_model, mask_batch_from_model = memory.sample(batch_size=20)

                state_batch_from_model = torch.tensor(state_batch_from_model, dtype=torch.float32)
                action_batch_from_model = torch.tensor(action_batch_from_model, dtype=torch.float32).unsqueeze(1)
                reward_batch_from_model = torch.tensor(reward_batch_from_model, dtype=torch.float32)
                next_state_batch_from_model = torch.tensor(next_state_batch_from_model, dtype=torch.float32)
                mask_batch_from_model = torch.tensor(mask_batch_from_model, dtype=torch.float32)
                
                state_batch, action_batch, reward_batch, next_state_batch, mask_batch = row

                state_batch = torch.cat((state_batch_from_model, state_batch), dim=0)
                action_batch = torch.cat((action_batch_from_model, action_batch), dim=0)
                reward_batch = torch.cat((reward_batch_from_model, reward_batch), dim=0)
                next_state_batch = torch.cat((next_state_batch_from_model, next_state_batch), dim=0)
                mask_batch = torch.cat((mask_batch_from_model, mask_batch), dim=0)

                row = (state_batch, action_batch, reward_batch, next_state_batch, mask_batch)

            critic_1_loss, critic_2_loss, policy_loss, ent_loss, alpha = agent.cql_update_parameters(row, updates)
            updates += 1            

    elif mode == 'offline_with_fake_env':
        rollout_len = 0
        indice = random.choices(np.arange(dataset['observations'].shape[0]), k=1)

        while rollout_len < args.max_rollout_len:
            if args.start_steps > total_numsteps:
                action = env.action_space.sample()  # Sample random action
            else:
                action = agent.select_action(state)  # Sample action from policy

            if len(memory) > args.batch_size:
                # Number of updates per step in environment
                for i in range(args.updates_per_step):
                    # Update parameters of all the networks
                    critic_1_loss, critic_2_loss, policy_loss, ent_loss, alpha = agent.update_parameters(memory, args.batch_size, updates)
                    updates += 1

            state = dataset['observations'][indice].squeeze() # hard coded with 'HalfCheetah-v2'
            next_state = dataset['next_observations'][indice].squeeze() # hard coded with 'HalfCheetah-v2'
            reward = dataset['rewards'][indice].squeeze()
            action = dataset['actions'][indice].squeeze()
            mask = float(not dataset['terminals'][indice].squeeze())
            
            episode_steps += 1
            total_numsteps += 1
            episode_reward += reward

            # mask = 1 if episode_steps == env._max_episode_steps else float(not done)
            memory.push(state, action, reward, next_state, mask) # Append transition to memory

            state = next_state
            rollout_len += 1

        print("Episode: {}, Buffer Size: {}, episode steps: {}".format(i_episode, len(memory), episode_steps))
        train_avg_rewards.append(episode_reward)

        rollout_len = 0
        state = random.choice(dataset['observations'])
        while rollout_len < 3:
            action = agent.select_action(state)  # Sample action from policy

            state = scaler.scale(torch.tensor(state.copy(), dtype=torch.float, device=device), 'observations')
            action = scaler.scale(torch.tensor(action.copy(), dtype=torch.float, device=device), 'actions')
            next_state_reward = transition_model(state.unsqueeze(0), action.unsqueeze(0)) # Step

            next_state = next_state_reward.squeeze()[:17] # hard coded with 'HalfCheetah-v2'
            reward = next_state_reward.squeeze()[17]
            
            reward = scaler.inverse_scale(reward, 'rewards')
            state = scaler.inverse_scale(state, 'observations')
            next_state = scaler.inverse_scale(next_state, 'next_observations')
            action = scaler.inverse_scale(action, 'actions')

            state = state.cpu().detach().numpy()
            action = action.cpu().detach().numpy()
            reward = reward.cpu().detach().numpy()
            next_state = next_state.cpu().detach().numpy()

            mask = 1 if episode_steps == env._max_episode_steps else float(not done)
            memory.push(state, action, reward, next_state, mask) # Append transition to memory

            state = next_state
            rollout_len += 1

        if total_numsteps > args.num_steps:
            break

    # reward results saved.
    np.save(os.path.join('reward_results', "{}_mode:_{}_rollout_step:_{}_train.npy".format(start_time, mode, args.max_rollout_len)), np.array(train_avg_rewards))

    # Evalutaion Process starts.
    if i_episode % 5 == 0 and args.eval is True:
        if args.data == 'd4rl':
            avg_reward = d4rl_evaluation(env, agent, 10)
            test_avg_rewards.append(avg_reward)
    
        elif args.data == 'ski':
            avg_reward = ski_evaluation(transition_model, agent, dataset, episodes=10, rollout_len=10, device=device)
            test_avg_rewards.append(avg_reward)
    
    # reward results saved.
    np.save(os.path.join('reward_results', "{}_mode:_{}_rollout_step:_{}_test.npy".format(start_time, mode, args.max_rollout_len)), np.array(test_avg_rewards))

env.close()

