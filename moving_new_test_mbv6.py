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
from buffers import ReplayBuffer
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.callbacks import CallbackList
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.buffers import ReplayBufferSamples
import os
from collections import deque
import random
from typing import List, Dict, Tuple, Any
from datetime import datetime


# ============================ 数值仿真环境定义（注意用ir-sim最新版本） ============================
class IRSIMEnv(gym.Env):     #继承了 gym.Env
    def __init__(self, config_path='./easy.yaml', display=True):
        super(IRSIMEnv, self).__init__()
        
        self.env = EnvBase(config_path, save_ani=False, full=False, display=display)    #加载底层仿真环境

        # 参数设置
        self.robot_number = 3
        self.max_linear_vel = 2.0   # 最大线速度 m/s
        self.max_steering_angle = 0.523  # 最大角速度 rad/s



        self.lidar_max_range = 8.0      # 激光最大探测范围
        self.goal_range = 14.0          # 目标可能出现在的最大范围
        self.robot_radius = 1.0         # 机器人半径     
        self.field_size = 14.0          # 场地尺寸（用于归一化等）
        self.max_steps = 300            # 每轮最大步数
        self.max_goal_steps = 200       # 每个目标最大尝试步数

        self.prev_distance = [None for _ in range(self.robot_number)]
 
        self.init_positions = []
        self.goal_positions = []
        self.obs_all = []

        self.obstacles = np.array([])       # 障碍物列表
        self.obstacle_radius = 1.5          # 障碍物碰撞判定半径
        self.goal_radius = 1.0              # 到达目标的判定半径


        # 观测序列长度
        self.history_len = 5            # 使用过去5帧观测作为输入
        self.obs_history = [ [] for _ in range(self.robot_number) ]  # 每个机器人一个空 list
        self.robot_reached_final=[]



        self.velocity = np.zeros(self.robot_number, dtype=np.float32)   
        self.steering_angle = np.zeros(self.robot_number, dtype=np.float32)
        dummy_obs = self._get_obs()     # 获取一帧初始观测数据
        self.obs_dim = dummy_obs.shape[1]
        self.obs_total = np.zeros((self.robot_number, self.history_len, self.obs_dim), dtype=np.float32)   #总观测 （robot_number * history_len * obs_dim）

        # 定义状态和动作序列空间  会影响到网络模型的输入输出结构以及buffer的处理   因此用小包的模型推理需要建立虚拟环境以自定义模型（即使我们知道模型使用没有问题）  且buffer需要自己写，因为里面会检查这个变量结构然后将传进去的结构改为这个形式
        self.observation_space = spaces.Box(         #
            low=-1.0, 
            high=1.0, 
            shape=(self.robot_number, self.history_len, self.obs_dim),  # 增加机器人数量维度
            dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-1.0, 
            high=1.0, 
            shape=(2,),  # 增加机器人数量维度
            dtype=np.float32
        )

    def reset(self):

         #为障碍物 随机重新生成位置，避免每次都是同一个环境布局；  尤其注意后面的编号  编号不对会导致导入的环境有错位
        self.env.random_obstacle_position(range_low= [2, 2.5, -3.14], range_high= [13, 11.5, 3.14], ids= [8,9,10,11,12], non_overlapping = True)  
        #清空与初始化状态变量：

        self.prev_min_distance = [999 for _ in range(self.robot_number)]
        self.prev_distance = [None for _ in range(self.robot_number)]
        self.velocity = np.zeros(self.robot_number, dtype=np.float32)
        self.steering_angle = np.zeros(self.robot_number, dtype=np.float32)
        self.robot_reached_final: List[bool] = [False] * self.robot_number    # 到达标志
        
        self.step_count = 0
        self.goal_count = 0

        self.env.reset()  #重置底层仿真环境（irsim），清空状态、时间步等。
        
        self.init_positions.clear()
        self.goal_positions.clear()    

        #初始化观测序列：
        obs_all = self._get_obs()    ## shape = (num_robots, obs_dim)
        self.obs_history.clear()
        for _ in range(self.history_len):
            self.obs_history.append(obs_all)
        obs_sequences = []

        for robot_id in range(self.robot_number):
            # 调用方法获取当前机器人的观测序列 (history_len, obs_dim)
            self._get_init_goal_positions(robot_id)
            seq = self._get_obs_sequence(robot_id)
            obs_sequences.append(seq)

        self.obs_total = np.stack(obs_sequences, axis=0)   #robot_number, history_len, obs_dim)

        return self.obs_total          #    #(num_robots, history_len, obs_dim)  

    def step(self, action):   
        """
            执行一步环境交互。

            Args:
                action: 动作数组，shape: [num_robots, action_dim]

            Returns:
                Tuple: (obs_total, reward_total, done_total, info_per_robot)
            """
        final_actions = np.zeros_like(action, dtype=np.float32)
        # --- 1. 动作处理与状态更新 ---
        for id in range(self.robot_number):
           
            if self.robot_reached_final[id]:
                # 机器人已到达终点并停止，动作设为0，保持静止
                self.velocity[id] = 0.0
                self.steering_angle[id] = 0.0           
            else:
                # 动作缩放
                acc = action[id, 0] * 5.0
                steer = action[id, 1] * 3.1416
                # 速度和转向角积分更新  
                self.velocity[id] += acc * self.env.step_time
                self.steering_angle[id] += steer * self.env.step_time
                # 裁剪到最大/最小限制
                self.velocity[id] = np.clip(self.velocity[id], -self.max_linear_vel, self.max_linear_vel)
                self.steering_angle[id] = np.clip(self.steering_angle[id], -self.max_steering_angle, self.max_steering_angle)
                # 记录最终的动作（速度和转向角）
            final_actions[id] = [self.velocity[id], self.steering_angle[id]]
        
        # --- 2. 与环境交互 ---
        # 转换为列表以满足环境接口要求
        final_actions_list = final_actions.tolist()
        robot_ids_list = list(range(self.robot_number))
            
        self.env.step(final_actions_list, action_id=robot_ids_list)
        
         # --- 3. 获取新观测和更新全局状态 ---       
        obs_all = self._get_obs()  # shape: [num_robots, obs_dim]
        self.obs_history.append(obs_all)

        self.step_count += 1  #暂时没用到
        self.goal_count += 1

        # 初始化返回列表/数组
        obs_per_robot: List[np.ndarray] = []
        rewards_per_robot: List[float] = []
        done_per_robot: List[bool] = [False] * self.robot_number    # 到达标志
        info_per_robot: Dict[int, bool] = {}  # 重置标志 (True: 需要重置 / False: 继续)


        # --- 4. 奖励计算与状态检查 ---
        for id in range(self.robot_number):
            if self.robot_reached_final[id]:
                # --- 4.1 机器人已到达并停止 (只计算 0 奖励，跳过过程奖励) ---
                single_total_reward = 0.0
                # reward_done = 0.0 # 保持为 0，不再重复给予到达奖励
                # 保持 done_per_robot[id] = True (或根据您的RL框架需求设置为False)
                done_per_robot[id] = True # 保持终结状态
                must_reset = False # 不要求重置
    

            else:
                # --- 4.1 过程奖励计算 ---
                #距离计算
                current_distance =0.0
                [x_rel, y_rel, _] = self._target_relative_pose(id)
                current_distance = np.hypot(x_rel, y_rel)
                if self.prev_distance[id] is None:
                    self.prev_distance[id] = current_distance
                #接近障碍物惩罚
                reward_obstacle = 0
                if min(np.array(self.env.get_lidar_scan(id)["ranges"], dtype=np.float32)) < 0.7:
                    reward_obstacle = -0.5
                    if min(np.array(self.env.get_lidar_scan(id)["ranges"], dtype=np.float32))-self.prev_min_distance[id]<0:
                        reward_obstacle += -0.5
                #靠近目标奖励
                delta_d = self.prev_distance[id] - current_distance
                reward_distance = delta_d * 0.5   
                #静止惩罚
                reward_movement = -0.2 if abs(self.velocity[id]) < 0.3 else 0.0
                # 判断当前 episode 是否因为到达目标、碰撞或超时而结束       
                
                reached_goal = current_distance <= 0.5
                # 假设 self.env.robot_list[id].done() 判断机器人是否碰撞
                collided = self.env.robot_list[id].done() and not reached_goal 
                goal_timeout = self.goal_count >= self.max_goal_steps
                
            #-----------------------结果奖励-----------------------
                reward_done = 0.0
                must_reset = False

                #完成目标 / 碰撞 / 超时处理
                if collided:
                    reward_done = -15.0
                    done_per_robot[id] = False
                    must_reset = True
                    print(f"episode done:  机器人{id} collision")                
                elif reached_goal:
                    reward_done =15.0
                    self.robot_reached_final[id] = True # 标记为已到达！
                    print(f"episode info: 机器人{id} reached goal")
                    done_per_robot[id] = True     
                    must_reset = False
                elif goal_timeout:
                    reward_done = -15
                    done_per_robot[id] = False
                    must_reset = True
                    print("episode done: timeout")
         
                single_total_reward = reward_distance + reward_movement + reward_obstacle + reward_done

            #更新距离参数
            if not self.robot_reached_final[id]:
                self.prev_distance[id] = current_distance    
            self.prev_min_distance[id] = min(np.array(self.env.get_lidar_scan(id)["ranges"], dtype=np.float32))    
       
            obs_per_robot.append(self._get_obs_sequence(id))
            rewards_per_robot.append(single_total_reward)
            info_per_robot[id] = must_reset

        # --- 5. 格式化返回数据 ---
        self.obs_total = np.stack(obs_per_robot, axis=0)  # (num_robots, history_len, obs_dim)
        reward_total = np.array(rewards_per_robot, dtype=np.float32)
        done_total = np.array(done_per_robot, dtype=bool)

        self.env.render()    #可视化当前环境状态：会让环境显示画面，常用于调试/训练监控。

        return self.obs_total, reward_total, done_total, info_per_robot

    def _get_init_goal_positions(self,id):
        
        min_start_goal_dist = 3.0   # 自身初始位置到目标最小距离
        min_start_start_dist = 3.5  # 各车初始位置最小间距
        min_goal_goal_dist = 3.5    # 各车目标点最小间距
        while True:
            # --------- 随机生成初始位置 ----------
            x = np.random.uniform(2, 12)
            y = np.random.uniform(0.7, 1.3)
            theta = np.random.uniform(0, np.pi)
            init_pos = np.array([x, y])

            # 检查与已有机器人的初始位置间距
            if any(np.linalg.norm(init_pos - p) < min_start_start_dist for p in self.init_positions):
                continue  # 与其他机器人太近，重新采样

            # --------- 随机生成目标点 ----------
            goal = self.sample_goal(id)
            goal_pos = goal[:2]

            # 检查自身初始位置与目标的最小距离
            if np.linalg.norm(init_pos - goal_pos) < min_start_goal_dist:
                continue  # 太近，重新采样

            # 检查与已有机器人目标点的距离
            if any(np.linalg.norm(goal_pos - g) < min_goal_goal_dist for g in self.goal_positions):
                continue  # 与其他目标太近，重新采样

            break

        # --------- 保存位置和目标 ----------
        self.init_positions.append(init_pos)
        self.goal_positions.append(goal_pos)

        # --------- 设置机器人状态和目标 ----------
        self.env.robot_list[id].set_state([x, y, theta, 0])
        self.env.robot_list[id].set_goal(goal, init=False)

    def sample_goal(self, id):
           
        while True:
            x = np.random.uniform(self.goal_radius, self.field_size - self.goal_radius)
            y = np.random.uniform(self.field_size - self.goal_radius - 1, self.field_size - self.goal_radius)
            goal = np.array([x, y])    # 采样的 x, y 坐标点。
            
            # --- 1) 检查与障碍物的冲突 ---
            if self.obstacles.size != 0:
                # 计算目标点与所有障碍物的距离
                dists_obs = np.linalg.norm(self.obstacles - goal, axis=1)
                
                # 检查是否与 '任一' 障碍物距离小于安全距离（障碍物半径 + 目标半径）
                if np.any(dists_obs < (self.obstacle_radius + self.goal_radius)):
                    continue  # 冲突 → 重新采样

            return np.array([x, y, 0, 0])

    def _get_obs_sequence(self, id):
        """
        提取指定机器人 id 的历史观测序列，并按 history_len 补齐或截断。
        返回 shape: (history_len, obs_dim)
        """
        obs_sequence = []

        for obs_step in self.obs_history:
            if id >= obs_step.shape[0]:
                raise IndexError(f"Robot id {id} out of range in obs_step with shape {obs_step.shape}")
            obs_sequence.append(obs_step[id])

        # 历史长度控制
        if len(obs_sequence) < self.history_len:
            first_obs = obs_sequence[0] if len(obs_sequence) > 0 else np.zeros_like(self._get_obs()[id])
            padding = [first_obs] * (self.history_len - len(obs_sequence))
            obs_sequence = padding + obs_sequence
        elif len(obs_sequence) > self.history_len:
            obs_sequence = obs_sequence[-self.history_len:]

        return np.stack(obs_sequence, axis=0)

    def _get_obs(self) -> np.ndarray:       
        """
            获取当前环境中所有机器人的观测（observation）信息。

            每个机器人的观测由：激光雷达数据、相对目标位姿、自身运动信息 拼接而成。

            Returns:
                np.ndarray: 所有机器人的观测数据，形状为 (self.robot_number, obs_dim)。
            """      
        obs_all=[]
        for id in range(self.robot_number):
            # --------- 激光雷达数据 ----------
            # 获取原始数据，并确保为 np.float32 类型
            lidar = np.array(self.env.get_lidar_scan(id)["ranges"], dtype=np.float32)  # 注意：要能区分第 i 个机器人
           
            lidar = np.clip(lidar, 0.0, self.lidar_max_range)    # 裁剪到最大范围，并归一化到 [-1, 1]
            lidar = (lidar / self.lidar_max_range) * 2.0 - 1.0  # 归一化到 [-1, 1]   归一化公式: (x / max_x) * 2 - 1
 
            # --------- 相对目标位姿 ----------
            target_pose = np.array(self._target_relative_pose(id), dtype=np.float32)
            target_pose[0] = np.clip(target_pose[0], -self.goal_range, self.goal_range) / self.goal_range
            target_pose[1] = np.clip(target_pose[1], -self.goal_range, self.goal_range) / self.goal_range
            target_pose[2] = np.clip(target_pose[2], -np.pi, np.pi) / np.pi
            target_pose = np.clip(target_pose, -1.0, 1.0)

            # --------- 自身运动信息 ----------
            # print(f"Robot {id}: raw velocity = {self.velocity}")

            velocity_norm = np.clip(self.velocity[id], -self.max_linear_vel, self.max_linear_vel) / self.max_linear_vel
            angular_norm = np.clip(self.steering_angle[id], -self.max_steering_angle, self.max_steering_angle) / self.max_steering_angle
            vehicle_info = np.array([velocity_norm, angular_norm], dtype=np.float32)

            # --------- 拼接单车观测 ----------
            obs_id = np.concatenate([lidar, target_pose, vehicle_info], dtype=np.float32)

            obs_all.append(obs_id)

        # --------- 返回所有车的观测 ----------
        return np.stack(obs_all, axis=0)   # shape = (num_robots, obs_dim)

    def _target_relative_pose(self,id):   
          #计算目标点相对于机器人当前位置的相对位置（dx, dy）和相对朝向差（dθ），即把目标的位置转换为以机器人自身为坐标原点和朝向基准的局部坐标系下的位置和方向。
        robot_state = self.env.robot_list[id]._state
        # robot_state = self.env.get_robot_state()
        goal= self.env.robot_list[id]._goal
        
        # goal = self.env.robot._goal[0]
        px, py, pt = robot_state[0][0], robot_state[1][0], robot_state[2][0]
        tx, ty, tt = goal[0][0], goal[0][1], goal[0][2]

        dx, dy = tx - px, ty - py
        x_rel = np.cos(pt) * dx + np.sin(pt) * dy
        y_rel = -np.sin(pt) * dx + np.cos(pt) * dy
        theta_rel = (tt - pt + np.pi) % (2 * np.pi) - np.pi
        return [x_rel, y_rel, theta_rel]

    def close(self):   #关闭底层仿真器，释放资源；
        self.env.end()

# ============================ 特征提取器类定义 ============================
class LaserGoalFeatureExtractor(BaseFeaturesExtractor):       #继承自 SB3 的 BaseFeaturesExtractor，是自定义特征提取器的标准方式。
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 128):
        super().__init__(observation_space, features_dim)
        # print("C 的维度:", observation_space.shape)
        self.laser_dim = 180    # 每帧激光观测维度
        self.goal_dim = 5       # 每帧目标相关信息（目标位置 + 速度 + 朝向等）
        self.seq_len = 5        # 时序长度（观测历史帧数）

        # CNN for laser frames (each frame: 1 channel x 180)   # 卷积提取激光特征   
        # 在 PyTorch 的 nn.Conv1d 中，输入张量格式为 [batch_size, channels, length]：
        # channels：通道数（相当于每个“传感器”一条线）；
        # length：一条通道的序列长度（如激光雷达的一帧是180维）
        self.conv1 = nn.Conv1d(in_channels=5, out_channels=32, kernel_size=5, stride=2)    #对应的是特征（对应前后5帧  进5出32  卷积是的小范围是5）
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=5, kernel_size=3, stride=2)    #接上一行（对应上面的出的32  进32出5  卷积是的小范围是3）
 
        # Calculate CNN output dimension using correct dummy input    计算卷积输出维度
        with torch.no_grad():
            dummy = torch.zeros(1, 5, self.laser_dim)  # [1, 1, 180]      #（180 进入2个卷积层）
            dummy = self.conv1(dummy)
            dummy = self.conv2(dummy)
            self.cnn_feat_dim = dummy.shape[2]    ## 展平后为全连接层输入

        # FC for goal frames       把每帧 5 维的目标状态压缩成 32 维；   后面送入 goal LSTM 编码时序信息。
        self.goal_fc = nn.Linear(self.goal_dim, 32)         # （5个（目标位置 + 速度 + 朝向等）   进入全连接层）
  
        # LSTMs  时序建模：LSTM 编码器   把卷积后的激光特征序列 [B, 5, cnn_feat_dim] 输入 LSTM，提取时序依赖；  同理，目标状态序列也输入一个小的 LSTM。     两个都过LSTM
        self.laser_lstm = nn.LSTM(input_size=self.cnn_feat_dim, hidden_size=128, batch_first=True)
        self.goal_lstm = nn.LSTM(input_size=32, hidden_size=32, batch_first=True)

        self.multihead_attn = nn.MultiheadAttention(embed_dim=128+32, num_heads=5, batch_first=True)   #多头注意力机制融合  对激光 + 目标的融合特征进行 attention 交互；   上述数据融合 
        # Final projection  最终输出层  将注意力输出的最后一步或者聚合特征映射为指定维度（如 128）；  给策略/值网络使用。
        self.fc = nn.Linear(128 + 32, features_dim)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:    
        #把环境观测输入，经过特征提取和融合，输出用于策略或价值估计的特征 forward方法的输入必须是环境观测的张量。
        #环境观测 observations，形状 [B, T, D]，    B = batch size   T = 序列长度（这里是 5，表示取最近 5 个时间步）  D = 单步观测维度 = laser_dim + goal_dim

        # print("observations 的维度:", observations.shape)
        if observations.dim() == 4:   # 评估和训练的时候不输入纬度不一样
                observations = observations.squeeze(0)
                
        B = observations.size(0)     #取 batch 大小 B。  

        # print("B 的维度:", B)
        # 将观测拆分为激光雷达序列和目标状态序列。
        laser_seq = observations[:, :, :self.laser_dim]  # [B, 5, 180]
        goal_seq = observations[:, :, self.laser_dim:]  # [B, 5, 5]

        # print("laser_seq 的维度:", laser_seq.shape)
        # print("goal_seq 的维度:", goal_seq.shape)


    # 现在的 observations 形状应该是 [Batch, 5, 185] 或 [Batch, 5, 180] (取决于您的 laser_dim)
    

        # 激光数据先经过两个 Conv1d 卷积层提取空间特征，再通过 LSTM 捕捉时间序列依赖，得到激光序列特征。
        x = F.relu(self.conv1(laser_seq))
        x = F.relu(self.conv2(x))
        laser_h, _ = self.laser_lstm(x)  # laser_hn: [1, B, 128]
        
        #目标状态数据先用全连接层降维，再用 LSTM 提取时序信息。
        g = F.relu(self.goal_fc(goal_seq))  # [B*5, 32]\
        g_h, _ = self.goal_lstm(g)  # goal_hn: [1, B, 32]

        #将激光和目标序列的时序特征拼接融合。
        features = torch.cat((laser_h, g_h), dim=2)

        #通过多头自注意力机制（Multihead Attention）让激光和目标特征互相交互，增强融合效果。
        attn_output, attn_weights = self.multihead_attn(features, features, features)

        # last = attn_output[:, -1, :]  对序列的每个时间步特征取均值，得到一组融合特征，再经过线性变换+ReLU激活，输出该时刻的特征向量（用于后续策略或价值网络）。
        pool = attn_output.mean(dim=1)
        feature = F.relu(self.fc(pool))   #fc 在初始化的时候已经定义输出维度了  实际上是features_dim  256
        # print("特征提取器输出维度:", features.shape)

        
        return feature       #  非动作  应该是观测值 256   

# ============================ 评估函数定义 ============================
def evaluate_model(model, env, num_episodes=5, deterministic=True):
    """
    对模型进行评估，运行 num_episodes 个回合，返回平均奖励。
    """
    all_episode_rewards = []
    
    for i in range(num_episodes):
        obs_total = env.reset()
        done = False
        episode_reward = 0
        
        while not done:
            # 将观测转换为模型所需的输入格式 (batch_size=1)
            obs_input = torch.from_numpy(obs_total).float().unsqueeze(0)
            
            with torch.no_grad():
                # 使用模型的 predict 方法获取动作
                action_total, _ = model.predict(obs_input, deterministic=deterministic)
            
            # 与环境交互
            obs_total, reward_total, done_total, info_total = env.step(action_total)
            
            # 累加所有机器人的奖励
            episode_reward += reward_total.sum()
            
            # 检查回合是否结束
            has_true = True in info_total.values()
            done = has_true or all(done_total)
            
        all_episode_rewards.append(episode_reward)
        
        # print(f"评估回合 {i+1}/{num_episodes}: 奖励={episode_reward:.2f}")

    mean_reward = np.mean(all_episode_rewards)
    std_reward = np.std(all_episode_rewards)

    return mean_reward, std_reward

# ============================  主函数定义 ============================
if __name__ == "__main__":
    
    # ============================  环境初始化  ============================
    env = IRSIMEnv('/home/zhangl/DRL_project/Dynamic_obs/self_env/ir_sim_test/robot_world_mbv3.yaml', display=True)

    # ============================  模型保存目录及参数初始化  ============================



    # 获取当前时间并格式化（例如：20251029_153045）
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join("./sequence_moving_models", current_time)
    os.makedirs(log_dir, exist_ok=True)      #exist_ok=True 表示：如果目录已存在就不会报错。   
  
    MODEL_SAVE_PATH=log_dir
    BEST_MEAN_REWARD = -np.inf

    EVAL_INTERVAL_EPISODES = 50  # 每隔多少个回合进行一次评估
    EVAL_EPISODES_COUNT = 5      # 每次评估跑5个回合    
 
    
    # ============================  模型策略配置  ============================
    policy_kwargs = dict( 
        features_extractor_class=LaserGoalFeatureExtractor,    #自定义的特征提取器类，例如用于处理激光雷达+目标位置等组合输入
        features_extractor_kwargs=dict(features_dim=256),      #提取器的参数，这里指定输出特征维度为 256
        net_arch=dict(pi=[128,128,64], qf=[128,128,64])        #策略网络(pi 用于输出动作的隐藏层维度)和价值网络(qf 用于估计Q值的隐藏层维度)   3 层全连接隐藏层，每层的神经元个数依次为：256 128 64
    )
        # 特征提取
        # 环境观测 → LaserGoalFeatureExtractor → 256 维特征向量
        # 网络前向传播
        # Actor: 256 → 128 → 128 → 64 → 动作分布
        # Critic: 256 → 128 → 128 → 64 → Q 值
    
    # print("TensorBoard日志路径：", MODEL_SAVE_PATH)
    # ============================  初始化模型  ============================
    model = SAC(        #虽然显示的是多层感知机策略（MLP Policy），使用 Soft Actor-Critic 算法，是一种基于值函数的离策略方法，具有高样本效率和稳定性，适合连续动作空间任务，如机器人控制、无人驾驶等。
        policy="MlpPolicy",      #定义 Actor（策略网络）和 Critic（价值网络） 的基本框架是由全连接层（Dense Layer）组成的多层感知机（MLP）   而非 CnnPolicy（卷积神经网络策略）
        env=env,
        replay_buffer_class=ReplayBuffer, 
        policy_kwargs=policy_kwargs,    #用于传入自定义特征提取器和策略结构的配置参数，
        verbose=1,                     #控制控制台日志的输出等级； 0 表示静默，1 表示每步训练都会有摘要输出，2 是更详细的调试信息。                                                           
        learning_rate=2e-4,              # 学习率，控制策略和价值网络的梯度更新步长； 对 SAC 来说，通常设置在 3e-4 到 1e-4 都是合理的，你设置得比较保守
        tensorboard_log=MODEL_SAVE_PATH,  #表示把训练过程中的日志输出到该目录；
        learning_starts=1000,  # 先收集1000步经验再开始训练（避免初始数据不足）
        train_freq=(600, "step"),  # 每收集128步经验（n_steps=128）触发一次训练
        gradient_steps=128,
          # 每次训练执行128轮梯度更新（与n_steps相等，保证数据利用率）
        batch_size=256  # 每次采样64条经验（SB3默认，平衡效率与稳定性）
    )

    # ============================  模型推理过程（训练时需要注释）  ============================
    
    model.policy.load_state_dict(torch.load("/home/zhangl/DRL_project/Dynamic_obs/self_env/sequence_moving_models/20251029_160007/best_policy_params.pt"))
    mean_reward, std_reward = evaluate_model(model, env, num_episodes=10, deterministic=True)    


    # # ============================  模型训练过程（推理时需要注释）  ============================


    # # 初始化状态
    # current_obs_total = env.reset()   # #(num_robots, history_len, obs_dim)  
    # current_timestep = 0
    # total_timesteps=1000000  
    # start_time = time.time()
    
    # model._setup_learn(total_timesteps, None)  # 初始化SB3内部状态     
    # episode_reward_sum = 0
    # episode_step_count = 0

    # print("=" * 50)
    # print(f"开始训练：总步数={total_timesteps}，收集步长n_steps=1000，梯度更新次数gradient_steps={8}")
    # print("=" * 50)

    # while current_timestep < total_timesteps:
    
    #     # -------------------------- 提取特征+与环境交互 ------------------------------
    #     obs_input = torch.from_numpy(current_obs_total).float().unsqueeze(0)  # [1, num_robots, history_len, obs_dim]
    #     with torch.no_grad():   
    #         action_total, _ = model.predict(obs_input, deterministic=False)    #  action_total 形状: (num_robots, 2)

    #     obs_total, reward_total, done_total, info_total = env.step(action_total)

    #     episode_reward_sum += reward_total.sum()  # 累加所有机器人的奖励
    #     episode_step_count += 1
    #     current_timestep += 1

    #     # -------------------------- 循环遍历每个机器人，存储独立经验 --------------------------
    #     for robot_id in range(env.robot_number):  # env.num_robots 是你环境中机器人的数量

    #         single_current_obs = current_obs_total[robot_id]  # 单个机器人的当前观测：(history_len, obs_dim)
    #         single_new_obs = obs_total[robot_id]          # 单个机器人的下一个观测：(history_len, obs_dim)
    #         single_action = action_total[robot_id]      # 单个机器人的动作：(2,)（对应你之前的 acc 和 steer）
    #         single_reward = reward_total[robot_id]      # 单个机器人的奖励：标量
    #         single_done = done_total[robot_id]          # 单个机器人的终止标志：布尔值
    #         # single_info = info_total[robot_id]          # 单个机器人的额外信息
            
    #         info1 ={}
    #         infos1 = [info1] 

    #         if not single_done:
    #             model.replay_buffer.add(
    #                 obs=single_current_obs,                # 单个机器人的当前观测
    #                 next_obs=single_new_obs,               # 单个机器人的下一个观测
    #                 action=single_action,                  # 单个机器人的动作（若需数组，可加 np.array()）
    #                 reward=np.array([single_reward]),      # 奖励：(1,) 维度数组（匹配单智能体 buffer 要求）
    #                 done=np.array([single_done]),          # 终止标志：(1,) 维度数组
    #                 infos=infos1                            # 单个机器人的额外信息(未使用)
    #             )    
    #     current_obs_total = obs_total

                
    #     # --------------------------  判断回合结束处理（重置环境+打印日志） --------------------------
    #     has_true = True in info_total.values()
    #     should_reset = has_true or all(done_total)
    #     if should_reset:
    #         model.logger.record("rollout/episode_reward", episode_reward_sum)
    #         model.logger.record("rollout/episode_length", episode_step_count)
    #         model.logger.record("time/episodes", model._episode_num, exclude="tensorboard")
    #         model._episode_num += 1  # 更新回合数
    #         model.logger.dump(model._episode_num)
    #         # 打印回合信息
    #         elapsed_time = time.time() - start_time
    #         print(f"[回合{model._episode_num}] 步数={current_timestep:6d} | 本回合奖励={episode_reward_sum:6.2f} | "
    #               f"本回合执行步数={episode_step_count:4d} | 总耗时={elapsed_time:6.2f}s")
                            
    #         # 重置环境和统计
    #         current_obs_total = env.reset()
    #         episode_reward_sum = 0
    #         episode_step_count = 0
    #         model._on_step()  # 通知SB3回合结束，更新日志

    #     # --------------------------  模型评估并记录模型 -------------------------------------
    #     # 定义当前模型的保存路径（用于非最优时）
    #     current_policy_path = os.path.join(MODEL_SAVE_PATH, "current_policy_params.pt")
    #     # 定义最优模型的保存路径
    #     best_policy_path = os.path.join(MODEL_SAVE_PATH, "best_policy_params.pt")
        
    #     if model._episode_num % EVAL_INTERVAL_EPISODES == 0:
    #         print("\n" + "=" * 20 + " 触发评估 " + "=" * 20)
    #         mean_reward, std_reward = evaluate_model(model, env, num_episodes=EVAL_EPISODES_COUNT, deterministic=True)
            
    #         # 记录评估结果到 TensorBoard
    #         model.logger.record("eval/mean_reward", mean_reward)
    #         model.logger.record("eval/std_reward", std_reward)
    #         model.logger.record("time/total_timesteps", current_timestep) # 添加当前步数方便对应
    #         model.logger.record("time/episodes", model._episode_num)

    #         print("=" * 50 + "\n")                    
    #         print(f"[评估结果] 回合数={model._episode_num:6d} | 平均奖励={mean_reward:6.2f} ± {std_reward:5.2f}")
    #         print("=" * 50 + "\n")     
  
    #         #保存模型 写好日志
    #         torch.save(model.policy.state_dict(), current_policy_path)    
    #         info_save_path = os.path.join(MODEL_SAVE_PATH, "current_model_info.txt")
    #         with open(info_save_path, "w") as f:
    #             f.write(f"current_mean_reward: {mean_reward:.2f}\n")
    #             f.write(f"episode_num: {model._episode_num}\n")
    #             f.write(f"timestep: {current_timestep}\n")

                

    #         if mean_reward > BEST_MEAN_REWARD:    # 记录最优模型
    #             print(f"🎉 New Best Model Found! ({mean_reward:.2f} > {BEST_MEAN_REWARD:.2f})")
    #             BEST_MEAN_REWARD = mean_reward # 更新最佳奖励


    #             torch.save(model.policy.state_dict(), best_policy_path)                
    #             info_save_best_path = os.path.join(MODEL_SAVE_PATH, "best_model_info.txt")
    #             with open(info_save_best_path, "w") as f:
    #                 f.write(f"best_mean_reward: {mean_reward:.2f}\n")
    #                 f.write(f"episode_num: {model._episode_num}\n")
    #                 f.write(f"timestep: {current_timestep}\n")
                
    #         model.logger.dump(model._episode_num)

 
        
    #     # --------------------------  训练更新模型 --------------------------
        
    #     if (current_timestep >= model.learning_starts) and (current_timestep % model.train_freq[0] == 0):
    #         print(f"\n[训练触发] 总步数={current_timestep:6d} | 执行{model.gradient_steps}轮梯度更新")
    #         # 调用model.train()执行梯度更新（内部会调用ReplayBuffer.sample()）
    #         model.train(
    #             gradient_steps=model.gradient_steps,
    #             batch_size=model.batch_size
    #         )
    #         # 记录训练统计（如更新次数）
    #         model._n_updates += model.gradient_steps


    #     # --------------------------  日志更新打印 --------------------------
    #     if current_timestep % 5000 == 0:
    #         model._dump_logs()  # 输出TensorBoard日志和控制台信息
    #         fps = current_timestep / (time.time() - start_time)
    #         print(f"[进度] 总步数={current_timestep:6d}/{total_timesteps} | "
    #               f"更新次数={model._n_updates:6d} | FPS={fps:6.2f}")

    #     # --------------------------  训练结束清理 --------------------------
    # print("\n" + "=" * 50)
    # print("训练结束！")
    # print("=" * 50)
    # model.env.close()  # 关闭环境





    # # ============================  小包单车原始推理模型（正常使用时注释）  ============================
    # # class CompatEnv(gym.Env):
    # #     def __init__(self, config_path='./easy.yaml', display=False):
    # #         # 动作空间为线速度和角速度，范围 [-1, 1]
    # #         self.action_space = spaces.Box(low=np.array([-1.0, -1.0]), high=np.array([1.0, 1.0]), dtype=np.float32)     #二维连续动作空间定义
    # #         self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(5, 185), dtype=np.float32)
    # #     def step(self):
    # #         pass
    # #     def reset(self):
    # #         pass
    # # env_alter = CompatEnv()
    # # model = SAC.load("/home/zhangl/DRL_project/Dynamic_obs/self_env/sequence_moving_models/akm_best_model", env=env_alter)  
    

    # # obs = env.reset()
    # # for i in range(1000):
    # #     action, _ = model.predict(obs, deterministic=True)
    # #     obs, reward, done, info = env.step(action)
    # #     print(reward)
    # #     if done.all():
    # #         obs = env.reset()