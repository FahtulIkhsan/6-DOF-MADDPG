import torch
from torch import multiprocessing
from tensordict import TensorDict
from tensordict.nn import TensorDictModule, TensorDictSequential

from torchrl.modules import (
    AdditiveGaussianModule,
    MultiAgentMLP,
    ProbabilisticActor,
    TanhDelta
)

from torchrl.data import (
    Unbounded,
    TensorDictPrioritizedReplayBuffer,
    LazyMemmapStorage
)

from torchrl.objectives import DDPGLoss, SoftUpdate, ValueEstimators

import math
import pandas as pd
import numpy as np
import os
import json
import sys

# Seed
seed = 938
torch.manual_seed(seed)

rng = np.random.default_rng(seed)

# Devices
is_fork = multiprocessing.get_start_method() == "fork"
device = (
    torch.device(0)
    if torch.cuda.is_available() and not is_fork
    else torch.device("cpu")
)
home_dir = os.path.expanduser('~')
BUFFER_FOLDER_NAME = "torchrl_replay_buffers"
BUFFER_STORAGE_NAME = "storage"
folder_path = os.path.join(home_dir, BUFFER_FOLDER_NAME)
scratch_path = os.path.join(folder_path, BUFFER_STORAGE_NAME)

OBS_DIMS = 21
BUFFER_SIZE = 1000000
MINIMUM_BUFFER_SIZE = 50001
BATCH_SIZE = 256
NETWORK_SIZE = 512
NETWORK_DEPTH = 2
GAMMA = 0.99
POLYAK_TAU = 1e-3
ACTOR_LR = 3e-4  # Learning rate
CRITIC_LR = 3e-3
WEIGHT_DECAY = 1e-1
EXPLORE_ANNEALING_STEPS = 20000
MAX_FRAME = 100
MAX_EPISODE = 300
ALPHA = 0.6
INITIAL_BETA = 0.4
FINAL_BETA = 1.0
BETA_ANNEALING_STEPS = 21000
SIGMA_INIT = 0.9

IS_MADDPG = False
if IS_MADDPG:
    ACTION_DIM = 1
    NUM_AGENTS = 6
    CENTRALIZED = True
    SHARE_PARAMETERS_POLICY = False
else:
    ACTION_DIM = 6
    NUM_AGENTS = 1
    CENTRALIZED = False
    SHARE_PARAMETERS_POLICY = True

POPULATING_REPLAY_BUFFER = False
if not POPULATING_REPLAY_BUFFER:
    REPLAY_BUFFER_EXIST_OK = True
else:
    REPLAY_BUFFER_EXIST_OK = False

class MADDPG:
    # 1. Define the initialization function
    def _xavier_init_weights(m):
        # Check if the module 'm' is an instance of a Linear layer
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            torch.nn.init.zeros_(m.bias)
            
            print(f"Initialized weights for: {m}")

    def __init__(self, target_location=[0.456, 0.213, 0.781]) :
        self.target_location = target_location
        self.replay_buffer_initialized = False
        self.current_episode = 0
        self.anneal_frame = 0
        self.frame = 0
        self.done = False
        self.terminated = False
        self.obs_data_initialized = False
        self.obs_count = 0
        self.obs_mean = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                         0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                         0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        self.obs_m2 = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                         0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                         0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        self.obs_var = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                         1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                         1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
        self.current_episode_rewards = []
        self.reward_data = []
        self.frame_num_data = []
        self.current_episode_loss_actor = []
        self.current_episode_loss_value = []
        self.loss_actor_data = []
        self.loss_value_data = []

    MLP_kwargs = {"norm_class": torch.nn.LayerNorm,
                  "norm_kwargs": {"normalized_shape": NETWORK_SIZE}}

    policy_net = MultiAgentMLP(
        n_agent_inputs=OBS_DIMS,
        n_agent_outputs=ACTION_DIM,
        n_agents=NUM_AGENTS,
        centralized=False,
        share_params=SHARE_PARAMETERS_POLICY,
        device=device,
        depth=NETWORK_DEPTH,
        num_cells=NETWORK_SIZE,
        activation_class=torch.nn.ReLU,
        **MLP_kwargs
    )

    policy_net.apply(_xavier_init_weights)

    policy_module = TensorDictModule(
            policy_net,
            in_keys=["observation"],
            out_keys=["param"],
        )

    actor = ProbabilisticActor(
            module=policy_module,
            in_keys=["param"],
            out_keys=["action"],
            distribution_class=TanhDelta,
            return_log_prob=False,
        )

    action_spec = Unbounded(
        shape=(NUM_AGENTS, ACTION_DIM),
        device=device
    )

    exploration_module = AdditiveGaussianModule(
            spec=action_spec,
            sigma_init=SIGMA_INIT,
            annealing_num_steps=EXPLORE_ANNEALING_STEPS
        )

    exploration_policy = TensorDictSequential(
        actor, exploration_module
    )

    cat_module = TensorDictModule(
            lambda obs, action: torch.cat([obs, action], dim=-1),
            in_keys=["observation", "action"],
            out_keys=["obs_action"],
        )

    critic_net = MultiAgentMLP(
        n_agent_inputs=(OBS_DIMS + ACTION_DIM),
        n_agent_outputs=1,
        n_agents=NUM_AGENTS,
        centralized=CENTRALIZED,
        share_params=SHARE_PARAMETERS_POLICY,
        device=device,
        depth=NETWORK_DEPTH,
        num_cells=NETWORK_SIZE,
        activation_class=torch.nn.ReLU,
        **MLP_kwargs
    )

    critic_net.apply(_xavier_init_weights)

    critic_module = TensorDictModule(
        module=critic_net,
        in_keys="obs_action",
        out_keys="state_action_value"
    )

    critic = TensorDictSequential(
        cat_module, critic_module
    )

    storage=LazyMemmapStorage(
        scratch_dir=scratch_path,
        max_size=BUFFER_SIZE,
        existsok=REPLAY_BUFFER_EXIST_OK
    )

    replay_buffer = TensorDictPrioritizedReplayBuffer(
        alpha=ALPHA,
        beta=INITIAL_BETA,
        storage=storage,
        batch_size=BATCH_SIZE,
        transform=lambda x: x.to(device)
    )

    loss_module = DDPGLoss(
        actor_network=actor,
        value_network=critic,
        delay_value=True,
        loss_function="l2"
    )
    loss_module.make_value_estimator(ValueEstimators.TD0, gamma=GAMMA)

    target_updater = SoftUpdate(loss_module, tau=POLYAK_TAU)

    optimiser = {
        "loss_actor": torch.optim.AdamW(
            loss_module.actor_network_params.flatten_keys().values(), lr=ACTOR_LR, weight_decay=WEIGHT_DECAY
        ),
        "loss_value": torch.optim.AdamW(
            loss_module.value_network_params.flatten_keys().values(), lr=CRITIC_LR, weight_decay=WEIGHT_DECAY
        )
    }

    def list_to_tensor(self, observations):
        obs = []
        for i in range(NUM_AGENTS):
            obs.append(observations)

        obs = torch.tensor(obs, device=device, dtype=torch.float32)

        return obs
    
    def update_observations_running_statistics(self, observations):
        self.obs_count += 1

        for i, obs in enumerate(observations):
            delta = obs - self.obs_mean[i]
            self.obs_mean[i] += delta / self.obs_count
            delta2 = obs - self.obs_mean[i]
            self.obs_m2[i] += delta * delta2

            if self.obs_count < 2 :
                pass
            else:
                self.obs_var[i] = self.obs_m2[i]/self.obs_count

    def standardize_observations(self, observations):
        if self.obs_count < 2:
            return observations
        else:
            obs_std = []
            for i, obs in enumerate(observations):
                std_dev = math.sqrt(self.obs_var[i])
                std_obs = (obs - self.obs_mean[i]) / (std_dev + 1e-4)

                obs_std.append(std_obs)

            return obs_std

    def distance_from_observations(self, observations):
        end_effector_position = observations[18:]
        
        ee_dist = math.sqrt((end_effector_position[0] - self.target_location[0])**2 +
                            (end_effector_position[1] - self.target_location[1])**2 +
                            (end_effector_position[2] - self.target_location[2])**2)
        
        return ee_dist
    
    def calculate_reward(self, observations, result_code=None):
        current_dist = self.distance_from_observations(observations)
        print(f"Current Distance : {current_dist}")

        reward = 0.5 * (math.exp(-4 * current_dist) - 1)

        if abs(current_dist) - 0.03 <= 0:
            reward = 40
            self.terminated = True

        if not POPULATING_REPLAY_BUFFER and self.frame >= (MAX_FRAME - 1):
            self.terminated = True
        
        if not POPULATING_REPLAY_BUFFER:
            if self.current_episode == (MAX_EPISODE - 1) and self.frame == (MAX_FRAME - 1):
                self.done = True
            elif self.current_episode >= (MAX_EPISODE):
                self.done = True
        else:
            if len(self.replay_buffer) >= (MINIMUM_BUFFER_SIZE - 1):
                self.done = True

        if len(self.replay_buffer) >= MINIMUM_BUFFER_SIZE:
            self.current_episode_rewards.append(reward)

        print(f"Reward : {reward}")
        return reward

    def act(self, observations):
        if not POPULATING_REPLAY_BUFFER and not self.obs_data_initialized and self.obs_count == 0:
            self.obs_data_initialized = True
            try:
                with open("obs_data.json", "r") as file:
                    loaded_data = json.load(file)
                    self.obs_count = loaded_data["obs_count"]
                    self.obs_mean = loaded_data["obs_mean"]
                    self.obs_m2 = loaded_data["obs_m2"]
                    self.obs_var = loaded_data["obs_var"]
                    print(f"Observation Count : {self.obs_count}")
                    print(f"Observation Mean : {self.obs_mean}")
                    print(f"Observation Stnd : {self.obs_m2}")
                    print(f"Observation Stnd : {self.obs_var}")
            except Exception as e:
                print(f"Error loading observation statistics data : {e}")
                sys.exit()

        std_obs = self.standardize_observations(observations)
        std_obs = self.list_to_tensor(std_obs)
        std_obs = torch.clamp(std_obs, min = -5.0, max = 5.0)
        print(f"Standardized Observation: {std_obs[0]}")
        dict_obs = TensorDict({"observation": std_obs})
        result = self.exploration_policy(dict_obs)

        self.action_result = result
        obs = self.list_to_tensor(observations)
        print(f"Unstandardized Observation: {obs[0]}")
        self.action_result.set("observation", obs, inplace=True)

        if not POPULATING_REPLAY_BUFFER:
            action = torch.flatten(result["action"])
            print(f"Unclamped Action: {action}")
            action = torch.clamp(action, -1.0, 1.0)
            action = action.tolist()
        else:
            action = rng.uniform(low=-1.0, high=1.0, size=6)
            act = torch.tensor(action, device=device, dtype=torch.float32)
            if IS_MADDPG:
                act = act.unsqueeze(1)
            self.action_result.set("action", act, inplace=True)
            print(f"Uniform Action: {self.action_result["action"]}")

        return action
    
    def store(self, new_observations, result_code):
        if not POPULATING_REPLAY_BUFFER and not self.replay_buffer_initialized:
            dummy_data = TensorDict({
                # Keys from the general metadata
                "observation": torch.zeros((NUM_AGENTS, OBS_DIMS), dtype=torch.float32),
                "action": torch.zeros((NUM_AGENTS, ACTION_DIM), dtype=torch.float32),
                "param": torch.zeros((NUM_AGENTS, ACTION_DIM), dtype=torch.float32),
                # Nested TensorDict for the 'next' key
                "next": TensorDict({
                    # Keys from the 'next' metadata
                    "observation": torch.zeros((NUM_AGENTS, OBS_DIMS), dtype=torch.float32),
                    "reward": torch.zeros((NUM_AGENTS, 1), dtype=torch.float32),
                    "terminated": torch.zeros((NUM_AGENTS, 1), dtype=torch.bool),
                    "done": torch.zeros((NUM_AGENTS, 1), dtype=torch.bool),
                }),
            })
            self.replay_buffer.add(dummy_data)
            self.replay_buffer.loads(folder_path)
            print(f"Replay Buffer Length : {len(self.replay_buffer)}")
            print(f"First Sample : {self.replay_buffer.sample()}")
            self.replay_buffer_initialized = True

        if POPULATING_REPLAY_BUFFER:
            self.update_observations_running_statistics(new_observations)
            self.frame += 1

        reward = self.list_to_tensor([self.calculate_reward(new_observations, result_code)])

        new_obs = self.list_to_tensor(new_observations)

        next_data = TensorDict({
            "observation": new_obs,
            "reward": reward,
            "terminated": torch.tensor([[self.terminated] for i in range(NUM_AGENTS)], device=device, dtype=torch.bool),
            "done": torch.tensor([[self.done] for i in range(NUM_AGENTS)], device=device, dtype=torch.bool)
        })

        replay_buffer_data = self.action_result
        replay_buffer_data.set("next", next_data)

        self.replay_buffer.add(replay_buffer_data)

        print(f"Observation: {replay_buffer_data["observation"][0]}")
        print(f"Next Observation: {replay_buffer_data["next"]["observation"][0]}")
            
        # print(f"Added data to replay buffer : {replay_buffer_data}")
    
    def step(self):
        if len(self.replay_buffer) > MINIMUM_BUFFER_SIZE:
            self.frame += 1
            subdata = self.replay_buffer.sample(BATCH_SIZE)
            for i, data in enumerate(subdata):
                obs = self.standardize_observations(data["observation"][0].tolist())
                tens_obs = self.list_to_tensor(obs)
                data.set("observation", tens_obs, inplace=True)
                next_obs = self.standardize_observations(data["next"]["observation"][0].tolist())
                tens_next_obs = self.list_to_tensor(next_obs)
                data.set(("next", "observation"), tens_next_obs, inplace=True)
            loss_vals = self.loss_module(subdata)

            if IS_MADDPG:
                # Since there 6 agents with different td_error, find the td_error mean for the sample
                per_agent_td_error = loss_vals["td_error"]
                priority_epsilon = 1e-6
                per_transition_priority = per_agent_td_error.abs().mean(dim=1) + priority_epsilon
                per_transition_priority = per_transition_priority.unsqueeze(1)
                subdata.set(self.replay_buffer.priority_key, per_transition_priority, inplace=True)

            for loss_name in ["loss_actor", "loss_value"]:
                loss = loss_vals[loss_name]
                print(f"{loss_name} : {loss}")
                
                optimizer = self.optimiser[loss_name]

                loss.backward()
                
                if loss_name == "loss_actor":
                    loss_raw = loss.item()
                    self.current_episode_loss_actor.append(loss_raw)
                    torch.nn.utils.clip_grad_norm_(self.loss_module.actor_network_params.flatten_keys().values(), max_norm=1.0)
                elif loss_name == "loss_value":
                    loss_raw = loss.item()
                    self.current_episode_loss_value.append(loss_raw)

                    total_norm = 0
                    params_iterable = self.loss_module.value_network_params.flatten_keys().values()
                    for p in params_iterable:
                        if p.grad is not None:
                            # p is the parameter tensor (e.g., weights)
                            # p.grad is the corresponding gradient tensor
                            print(f"Parameter shape: {p.shape}\nGradient norm: {p.grad.norm()}")
                            param_norm = p.grad.data.norm(2)
                            total_norm += param_norm.item() ** 2
                    total_norm = total_norm ** 0.5
                    print(f"Total gradient norm: {total_norm}")
                    
                    torch.nn.utils.clip_grad_norm_(self.loss_module.value_network_params.flatten_keys().values(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()
            
            self.target_updater.step()

            self.replay_buffer.update_tensordict_priority(subdata)
            
            # print("\n--- Using model.named_parameters() ---")
            # for name, param in self.actor.named_parameters():
            #     print(f"Parameter Name: {name}, Shape: {param.shape}, Data: {param.data}")

            # for name, param in self.critic.named_parameters():
            #     print(f"Parameter Name: {name}, Shape: {param.shape}, Data: {param.data}")

        print(f"Episode : {self.current_episode}")
        print(f"Frame : {self.frame}")

    def end_frame_check(self):
        if POPULATING_REPLAY_BUFFER and self.frame >= (5*MAX_FRAME):
            return 3
        elif self.done:
            return 2
        elif self.terminated:
            return 1
        else:
            return 0
        
    def new_episode(self):
        self.current_episode += 1
        self.anneal_frame += MAX_FRAME

        self.reward_data.append(self.current_episode_rewards)
        self.frame_num_data.append(self.frame)
        self.loss_actor_data.append(self.current_episode_loss_actor)
        self.loss_value_data.append(self.current_episode_loss_value)

        self.exploration_module.step(self.anneal_frame)

        if self.anneal_frame < BETA_ANNEALING_STEPS:
            # Linearly increase beta from its initial value to the final value
            current_beta = INITIAL_BETA + (FINAL_BETA - INITIAL_BETA) * (self.anneal_frame / BETA_ANNEALING_STEPS)

            # Directly update the beta attribute of the underlying sampler
            self.replay_buffer.sampler.beta = current_beta
        else:
            # Keep beta at its final value after annealing is complete
            self.replay_buffer.sampler.beta = FINAL_BETA

        self.current_episode_rewards = []
        self.current_episode_loss_actor = []
        self.current_episode_loss_value = []
        self.frame = 0
        self.terminated = False

    def terminate(self):
        if not POPULATING_REPLAY_BUFFER:
            self.reward_data.append(self.current_episode_rewards)
            self.frame_num_data.append(self.frame)
            self.loss_actor_data.append(self.current_episode_loss_actor)
            self.loss_value_data.append(self.current_episode_loss_value)
            df_reward = pd.DataFrame(self.reward_data)
            df_frame = pd.DataFrame(self.frame_num_data)
            df_loss_actor = pd.DataFrame(self.loss_actor_data)
            df_loss_value = pd.DataFrame(self.loss_value_data)
            df_reward.to_parquet('reward_data.parquet', engine='pyarrow', compression='gzip')
            df_frame.to_parquet('frame_data.parquet', engine='pyarrow', compression='gzip')
            df_loss_actor.to_parquet('loss_actor.parquet', engine='pyarrow', compression='gzip')
            df_loss_value.to_parquet('loss_value.parquet', engine='pyarrow', compression='gzip')
            df_reward.to_csv('reward_data.csv', mode='a', header=False, index=False)
            df_frame.to_csv('frame_data.csv', mode='a', header=False, index=False)
            df_loss_actor.to_csv('loss_actor_data.csv', mode='a', header=False, index=False)
            df_loss_actor.to_csv('loss_value_data.csv', mode='a', header=False, index=False)
        else:
            self.replay_buffer.dumps(folder_path)

        obs_data = {
            "obs_count": self.obs_count,
            "obs_mean": self.obs_mean,
            "obs_var": self.obs_var,
            "obs_m2": self.obs_m2
        }

        print(f"Finished with IS_MADDPG : {IS_MADDPG}")
        print(f"Finished with seed : {seed}")

        with open("obs_data.json", "w") as file:
            json.dump(obs_data, file, indent=4)