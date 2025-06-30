import time

import gym
import numpy as np
import math
from gym import spaces
from irsim.env import EnvBase
from stable_baselines3.common.evaluation import evaluate_policy
import torch
import torch.nn as nn
import torch.nn.functional as F
from stable_baselines3 import SAC,PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.callbacks import CallbackList
from stable_baselines3.common.callbacks import EvalCallback
import os
from collections import deque
import random

class IRSIMEnv(gym.Env):
    def __init__(self, config_path='./easy.yaml', display=False):
        super(IRSIMEnv, self).__init__()
        self.env = EnvBase(config_path, save_ani=False, full=False, display=display)

        # 动作空间为线速度和角速度，范围 [-1, 1]
        self.action_space = spaces.Box(low=np.array([-1.0, -1.0]), high=np.array([1.0, 1.0]), dtype=np.float32)

        # 参数设置
        self.max_linear_vel = 2.0   # 最大线速度 m/s
        self.max_steering_angle = 0.523  # 最大角速度 rad/s

        self.velocity = 0.0
        self.steering_angle = 0.0

        self.lidar_max_range = 8.0
        self.goal_range = 14.0
        self.robot_radius = 1.0
        self.field_size = 14.0
        self.max_steps = 300
        self.max_goal_steps = 150
        self.step_count = 0
        self.goal_count = 0
        self.prev_distance = None

        self.obstacles = np.array([])
        self.obstacle_radius = 1.5
        self.goal_radius = 1.0

        # 观测序列长度
        self.history_len = 5
        self.obs_history = deque(maxlen=self.history_len)
        self.prev_min_distance = 999
        # 先获取单帧观测长度
        dummy_obs = self._get_obs()
        obs_dim = len(dummy_obs)

        # 修改为序列空间，shape=(history_len, obs_dim)
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(self.history_len, obs_dim), dtype=np.float32)

    def sample_goal(self):
        while True:
            x = np.random.uniform(self.goal_radius, self.field_size - self.goal_radius)
            y = np.random.uniform(self.field_size - self.goal_radius-1, self.field_size - self.goal_radius)
            goal = np.array([x, y])
            if self.obstacles.size != 0:
                dists = np.linalg.norm(self.obstacles - goal, axis=1)
                if np.all(dists >= (self.obstacle_radius + self.goal_radius)):
                    return np.array([x, y, 0, 0])
            else:
                return np.array([x, y, 0, 0])

    def _target_relative_pose(self):
        robot_state = self.env.get_robot_state()
        goal = self.env.robot._goal[0]
        px, py, pt = robot_state[0][0], robot_state[1][0], robot_state[2][0]
        tx, ty, tt = goal[0], goal[1], goal[2]

        dx, dy = tx - px, ty - py
        x_rel = np.cos(pt) * dx + np.sin(pt) * dy
        y_rel = -np.sin(pt) * dx + np.cos(pt) * dy
        theta_rel = (tt - pt + np.pi) % (2 * np.pi) - np.pi
        return [x_rel, y_rel, theta_rel]

    def _get_obs(self):
        lidar = np.array(self.env.get_lidar_scan()["ranges"], dtype=np.float32)
        lidar = np.clip(lidar, 0.0, self.lidar_max_range)
        lidar = (lidar / self.lidar_max_range) * 2.0 - 1.0  # [-1, 1]

        target_pose = np.array(self._target_relative_pose(), dtype=np.float32)
        target_pose[0] = np.clip(target_pose[0], -self.goal_range, self.goal_range) / self.goal_range
        target_pose[1] = np.clip(target_pose[1], -self.goal_range, self.goal_range) / self.goal_range
        target_pose[2] = np.clip(target_pose[2], -np.pi, np.pi) / np.pi
        target_pose = np.clip(target_pose, -1.0, 1.0)

        velocity_norm = np.clip(self.velocity, -self.max_linear_vel, self.max_linear_vel) / self.max_linear_vel
        angular_norm = np.clip(self.steering_angle, -self.max_steering_angle, self.max_steering_angle) / self.max_steering_angle
        vehicle_info = np.array([velocity_norm, angular_norm], dtype=np.float32)

        return np.concatenate([lidar, target_pose, vehicle_info], dtype=np.float32)

    def _get_obs_sequence(self):
        # 不足 history_len 时，用当前最新观测填充
        if len(self.obs_history) == 0:
            current_obs = self._get_obs()
            for _ in range(self.history_len):
                self.obs_history.append(current_obs)
        elif len(self.obs_history) < self.history_len:
            last_obs = self.obs_history[-1]
            for _ in range(self.history_len - len(self.obs_history)):
                self.obs_history.append(last_obs)
        return np.stack(self.obs_history, axis=0)

    def step(self, action):
        acc = action[0] * 5.0
        steer = action[1] * 3.1416
        self.velocity += acc * self.env.step_time
        self.steering_angle += steer * self.env.step_time
        self.velocity = np.clip(self.velocity, -2, 2)
        self.steering_angle = np.clip(self.steering_angle, -0.523, 0.523)

        self.env.step(np.array([[self.velocity], [self.steering_angle]]), 0)

        obs = self._get_obs()
        self.obs_history.append(obs)

        done = self.env.done()
        self.step_count += 1
        self.goal_count += 1

        # 距离计算
        [x_rel, y_rel, _] = self._target_relative_pose()
        current_distance = np.hypot(x_rel, y_rel)

        if self.prev_distance is None:
            self.prev_distance = current_distance

        reward_obstacle = 0
        if min(np.array(self.env.get_lidar_scan()["ranges"], dtype=np.float32)) < 0.7:
            reward_obstacle = -0.5
            if min(np.array(self.env.get_lidar_scan()["ranges"], dtype=np.float32))-self.prev_min_distance<0:
                reward_obstacle += -0.5
        delta_d = self.prev_distance - current_distance
        reward_distance = delta_d * 0.5
        reward_movement = -0.2 if abs(self.velocity) < 0.3 else 0.0


        reached_goal = current_distance <= 0.5
        collided = self.env.done() and not reached_goal
        goal_timeout = self.goal_count >= self.max_goal_steps

        if collided:
            reward_done = -15.0
            done = True
            print("episode done: collision")
        elif goal_timeout:
            reward_done = -15
            done = True
            print("episode done: timeout")
        elif reached_goal:
            reward_done = 15.0
            done = True
            self.goal_count = 0
            print("goal reached")

            new_goal = self.sample_goal()
            self.env.robot.set_goal(new_goal, init=False)
        else:
            reward_done = 0.0
            done = False

        reward = reward_distance + reward_movement + reward_obstacle + reward_done

        self.prev_distance = current_distance
        self.prev_min_distance = min(np.array(self.env.get_lidar_scan()["ranges"], dtype=np.float32))
        # print(reward)

        self.env.render()

        return self._get_obs_sequence(), reward, done, {}

    def reset(self):

        self.env.reset()

        self.env.random_obstacle_position(range_low= [2, 2.5, -3.14], range_high= [13, 11.5, 3.14], ids= [6,7,8,9,10,11,12,13,14,15], non_overlapping = True)
        self.velocity = 0.0
        self.steering_angle = 0.0
        self.prev_distance = None
        self.step_count = 0
        self.goal_count = 0
        x = np.random.uniform(2, 12)
        y = np.random.uniform(0.7, 1.3)
        y = np.random.uniform(0.7, 1.3)
        theta = np.random.uniform(0, np.pi)
        self.env.robot.set_state([x,y,theta,0])
        goal = self.sample_goal()
        self.env.robot.set_goal(goal, init=False)
        obs = self._get_obs()
        self.obs_history.clear()
        for _ in range(self.history_len):
            self.obs_history.append(obs)

        return self._get_obs_sequence()

    def close(self):
        self.env.end()


class LaserGoalFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 128):
        super().__init__(observation_space, features_dim)

        self.laser_dim = 180
        self.goal_dim = 5
        self.seq_len = 5  # Sequence length of 5 timesteps

        # CNN for laser frames (each frame: 1 channel x 180)
        self.conv1 = nn.Conv1d(in_channels=5, out_channels=32, kernel_size=5, stride=2)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=5, kernel_size=3, stride=2)

        # Calculate CNN output dimension using correct dummy input
        with torch.no_grad():
            dummy = torch.zeros(1, 5, self.laser_dim)  # [1, 1, 180]
            dummy = self.conv1(dummy)
            dummy = self.conv2(dummy)
            self.cnn_feat_dim = dummy.shape[2]

        # FC for goal frames
        self.goal_fc = nn.Linear(self.goal_dim, 32)

        # LSTMs
        self.laser_lstm = nn.LSTM(input_size=self.cnn_feat_dim, hidden_size=128, batch_first=True)
        self.goal_lstm = nn.LSTM(input_size=32, hidden_size=32, batch_first=True)

        self.multihead_attn = nn.MultiheadAttention(embed_dim=128+32, num_heads=5, batch_first=True)
        # Final projection
        self.fc = nn.Linear(128 + 32, features_dim)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:

        B = observations.size(0)

        laser_seq = observations[:, :, :self.laser_dim]  # [B, 5, 180]
        goal_seq = observations[:, :, self.laser_dim:]  # [B, 5, 5]

        x = F.relu(self.conv1(laser_seq))
        x = F.relu(self.conv2(x))
        laser_h, _ = self.laser_lstm(x)  # laser_hn: [1, B, 128]

        g = F.relu(self.goal_fc(goal_seq))  # [B*5, 32]\
        g_h, _ = self.goal_lstm(g)  # goal_hn: [1, B, 32]

        features = torch.cat((laser_h, g_h), dim=2)
        attn_output, attn_weights = self.multihead_attn(features, features, features)

        # last = attn_output[:, -1, :]
        pool = attn_output.mean(dim=1)
        action = F.relu(self.fc(pool))
        return action

    def forward_actor(self, features: torch.Tensor) -> torch.Tensor:
        return self.policy_net(features)

    def forward_critic(self, features: torch.Tensor) -> torch.Tensor:
        return self.value_net(features)


if __name__ == "__main__":
    # 创建环境
    env = IRSIMEnv('env/moving.yaml', display=True)


    log_dir = "./sequence_moving_models"
    os.makedirs(log_dir, exist_ok=True)
    callback_save_best_model = EvalCallback(env, best_model_save_path=log_dir, log_path=log_dir, eval_freq=4096,
                                            deterministic=True, render=False)
    callback_list = CallbackList([callback_save_best_model])
    policy_kwargs = dict(
        features_extractor_class=LaserGoalFeatureExtractor,
        features_extractor_kwargs=dict(features_dim=256),
        net_arch=dict(pi=[256,128,64], qf=[256,128,64])
    )

    # 初始化模型
    model = SAC(
        policy="MlpPolicy",
        env=env,
        policy_kwargs=policy_kwargs,
        verbose=1,
        learning_rate=1e-4,
        tensorboard_log="./sac_laser_goal_tensorboard/"
    )


    # model = SAC.load("sequence_moving_models/akm_best_model", env=env)
    # model.learn(total_timesteps=1000000, callback=callback_list, progress_bar=True)
    model = SAC.load("sequence_moving_models/new_best_model", env=env)
    # model.learn(total_timesteps=1000000, callback=callback_list, progress_bar=True)
    # 评估
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=50)
    print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")
