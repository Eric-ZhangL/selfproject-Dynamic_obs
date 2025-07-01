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

class IRSIMEnv(gym.Env):     #继承了 gym.Env
    def __init__(self, config_path='./easy.yaml', display=False):
        super(IRSIMEnv, self).__init__()
        self.env = EnvBase(config_path, save_ani=False, full=False, display=display)    #加载底层仿真环境 EnvBase，该类负责实际的机器人建模、地图、障碍物等。

        # 动作空间为线速度和角速度，范围 [-1, 1]
        self.action_space = spaces.Box(low=np.array([-1.0, -1.0]), high=np.array([1.0, 1.0]), dtype=np.float32)     #二维连续动作空间定义

        # 参数设置
        self.max_linear_vel = 2.0   # 最大线速度 m/s
        self.max_steering_angle = 0.523  # 最大角速度 rad/s

        self.velocity = 0.0
        self.steering_angle = 0.0

        self.lidar_max_range = 8.0      # 激光最大探测范围
        self.goal_range = 14.0          # 目标可能出现在的最大范围
        self.robot_radius = 1.0         # 机器人半径
        self.field_size = 14.0          # 场地尺寸（用于归一化等）
        self.max_steps = 300            # 每轮最大步数
        self.max_goal_steps = 150       # 每个目标最大尝试步数
        self.step_count = 0             # 步数计数器
        self.goal_count = 0             # 成功达到目标次数
        self.prev_distance = None       # 上一步到目标的距离


        self.obstacles = np.array([])       # 障碍物列表
        self.obstacle_radius = 1.5          # 障碍物碰撞判定半径
        self.goal_radius = 1.0              # 到达目标的判定半径


        # 观测序列长度
        self.history_len = 5            # 使用过去5帧观测作为输入
        self.obs_history = deque(maxlen=self.history_len)
        self.prev_min_distance = 999    # 初始化最小距离
        # 先获取单帧观测长度   📏 状态空间定义
        dummy_obs = self._get_obs()     # 获取一帧初始观测数据
        obs_dim = len(dummy_obs)        # 单帧观测长度

        # 修改为序列空间，shape=(history_len, obs_dim)
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(self.history_len, obs_dim), dtype=np.float32)

    def sample_goal(self):       ##用来随机采样一个目标点的位置。  检测是否为有效点
        while True:       #使用一个无限循环，直到采样出一个合法的目标为止（合法指：不与障碍物冲突）。    在场地范围内随机生成一个 x, y 坐标：
            x = np.random.uniform(self.goal_radius, self.field_size - self.goal_radius)
            y = np.random.uniform(self.field_size - self.goal_radius-1, self.field_size - self.goal_radius)
            goal = np.array([x, y])    #把采样结果打包成坐标点。
            if self.obstacles.size != 0:    #如果当前环境中有障碍物（障碍物不是空），就要检查目标是否离障碍太近。
                dists = np.linalg.norm(self.obstacles - goal, axis=1)
                if np.all(dists >= (self.obstacle_radius + self.goal_radius)):
                    return np.array([x, y, 0, 0])     #如果没有障碍物，那直接返回即可，不用检查冲突。
            else:
                return np.array([x, y, 0, 0])

    def _target_relative_pose(self):     #计算目标点相对于机器人当前位置的相对位置（dx, dy）和相对朝向差（dθ），即把目标的位置转换为以机器人自身为坐标原点和朝向基准的局部坐标系下的位置和方向。
        robot_state = self.env.get_robot_state()
        goal = self.env.robot._goal[0]
        px, py, pt = robot_state[0][0], robot_state[1][0], robot_state[2][0]
        tx, ty, tt = goal[0], goal[1], goal[2]

        dx, dy = tx - px, ty - py
        x_rel = np.cos(pt) * dx + np.sin(pt) * dy
        y_rel = -np.sin(pt) * dx + np.cos(pt) * dy
        theta_rel = (tt - pt + np.pi) % (2 * np.pi) - np.pi
        return [x_rel, y_rel, theta_rel]

    def _get_obs(self):        #获取当前环境的观测（observation）信息
        lidar = np.array(self.env.get_lidar_scan()["ranges"], dtype=np.float32)    # 从仿真环境中获取激光雷达扫描结果（一个 list，包含180个方向的距离信息），转为 np.array
        lidar = np.clip(lidar, 0.0, self.lidar_max_range)                          # 把激光数据限制在 [0, lidar_max_range] 范围内，超出就裁剪。
        lidar = (lidar / self.lidar_max_range) * 2.0 - 1.0  # [-1, 1]              # 将距离数据归一化到 [-1, 1]，方便神经网络训练。

        target_pose = np.array(self._target_relative_pose(), dtype=np.float32)     #获取机器人到目标点的相对位姿：通常是 [dx, dy, dθ]，单位可能是米和弧度。
        target_pose[0] = np.clip(target_pose[0], -self.goal_range, self.goal_range) / self.goal_range     #将相对位置的 dx, dy 限制在最大范围内，并归一化到 [-1, 1]。
        target_pose[1] = np.clip(target_pose[1], -self.goal_range, self.goal_range) / self.goal_range
        target_pose[2] = np.clip(target_pose[2], -np.pi, np.pi) / np.pi                                   #将角度差（方向差）裁剪在 [-π, π] 之间，并归一化到 [-1, 1]。
        target_pose = np.clip(target_pose, -1.0, 1.0)                                                     #target_pose = np.clip(target_pose, -1.0, 1.0)
               
        velocity_norm = np.clip(self.velocity, -self.max_linear_vel, self.max_linear_vel) / self.max_linear_vel        #将当前线速度和角速度限制在范围内并归一化到 [-1, 1]。
        angular_norm = np.clip(self.steering_angle, -self.max_steering_angle, self.max_steering_angle) / self.max_steering_angle
        vehicle_info = np.array([velocity_norm, angular_norm], dtype=np.float32)            #打包成 [线速度, 角速度] 的数组。

        return np.concatenate([lidar, target_pose, vehicle_info], dtype=np.float32)          #把三个部分合成一个完整的观测向量：( 180+3+2 ) * 5    一个 (5, 185) 的张量

    def _get_obs_sequence(self):    #返回一段时间序列观测（即过去连续几帧的观测信息），用于强化学习中处理有时间依赖的任务，例如使用 LSTM、RNN、注意力等结构。
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

    def step(self, action):   # 执行一步动作，更新环境状态，计算奖励，判断是否终止，并返回新的观测序列、奖励、done标志和 info
        # 动作执行：映射并裁剪
        acc = action[0] * 5.0     #action 是 SAC 策略输出的二维向量 [-1, 1] 范围的值。    更新速度，并裁剪在设定范围内。
        steer = action[1] * 3.1416
        self.velocity += acc * self.env.step_time
        self.steering_angle += steer * self.env.step_time
        self.velocity = np.clip(self.velocity, -2, 2)
        self.steering_angle = np.clip(self.steering_angle, -0.523, 0.523)

        #与仿真环境交互     向底层仿真环境（irsim）传入实际动作：线速度 + 转角。
        self.env.step(np.array([[self.velocity], [self.steering_angle]]), 0)

        # 收集当前观测 & 加入历史队列
        obs = self._get_obs()            #获取新的观测帧（激光 + 目标 + 自车状态）并加入历史观测序列。
        self.obs_history.append(obs)     #随着时间推移，obs_history 会滚动维护最近 5 帧观测。

        # 判断是否 episode 结束
        done = self.env.done()
        self.step_count += 1
        self.goal_count += 1

        # 距离计算
        [x_rel, y_rel, _] = self._target_relative_pose()
        current_distance = np.hypot(x_rel, y_rel)

        if self.prev_distance is None:
            self.prev_distance = current_distance

        #障碍物惩罚
        reward_obstacle = 0
        if min(np.array(self.env.get_lidar_scan()["ranges"], dtype=np.float32)) < 0.7:
            reward_obstacle = -0.5
            if min(np.array(self.env.get_lidar_scan()["ranges"], dtype=np.float32))-self.prev_min_distance<0:
                reward_obstacle += -0.5
        
        #距离奖励
        delta_d = self.prev_distance - current_distance
        reward_distance = delta_d * 0.5
        
        #静止惩罚
        reward_movement = -0.2 if abs(self.velocity) < 0.3 else 0.0

        # 判断当前 episode 是否因为到达目标、碰撞或超时而结束       
        reached_goal = current_distance <= 0.5
        collided = self.env.done() and not reached_goal
        goal_timeout = self.goal_count >= self.max_goal_steps

        #完成目标 / 碰撞 / 超时处理
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

        #最终总奖励计算    → 向目标靠近的奖励 → 静止的惩罚 → 接近障碍的惩罚 → 达标/失败终止的奖励
        reward = reward_distance + reward_movement + reward_obstacle + reward_done

        self.prev_distance = current_distance    #更新上一时刻的距离（用于下次计算距离变化）：
        self.prev_min_distance = min(np.array(self.env.get_lidar_scan()["ranges"], dtype=np.float32))    #记录当前时刻最小的激光距离（用于判断是否越来越靠近障碍物）：
        # print(reward)

        self.env.render()    #可视化当前环境状态：会让环境显示画面，常用于调试/训练监控。

        return self._get_obs_sequence(), reward, done, {}   # 返回 step() 的结果（符合 Gym 接口标准）：

    def reset(self):
        #  每当一个 episode 结束（例如碰撞、到达目标、超时），算法会自动调用 env.reset() 来开始下一轮。这段代码完成了以下初始化工作：   
        self.env.reset()  #重置底层仿真环境（irsim），清空状态、时间步等。
         #为障碍物 随机重新生成位置，避免每次都是同一个环境布局；
        self.env.random_obstacle_position(range_low= [2, 2.5, -3.14], range_high= [13, 11.5, 3.14], ids= [6,7,8,9,10,11,12,13,14,15], non_overlapping = True)  
        #清空与初始化状态变量：
        self.velocity = 0.0
        self.steering_angle = 0.0
        self.prev_distance = None
        self.step_count = 0
        self.goal_count = 0
        #设置机器人初始位置：
        x = np.random.uniform(2, 12)
        y = np.random.uniform(0.7, 1.3)
        theta = np.random.uniform(0, np.pi)
        self.env.robot.set_state([x,y,theta,0])
        #设置新的目标点
        goal = self.sample_goal()
        self.env.robot.set_goal(goal, init=False)
        #初始化观测序列：
        obs = self._get_obs()
        self.obs_history.clear()
        for _ in range(self.history_len):
            self.obs_history.append(obs)
        #返回 (5, obs_dim) 的观测序列，作为当前 episode 的初始输入。
        return self._get_obs_sequence()

    def close(self):   #关闭底层仿真器，释放资源；
        self.env.end()

class LaserGoalFeatureExtractor(BaseFeaturesExtractor):       #继承自 SB3 的 BaseFeaturesExtractor，是自定义特征提取器的标准方式。
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 128):
        super().__init__(observation_space, features_dim)

        self.laser_dim = 180    # 每帧激光观测维度
        self.goal_dim = 5       # 每帧目标相关信息（目标位置 + 速度 + 朝向等）
        self.seq_len = 5        # 时序长度（观测历史帧数）

##????????????????????????????????????????????????????????????????????
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

    def forward(self, observations: torch.Tensor) -> torch.Tensor:    #把环境观测输入，经过特征提取和融合，输出用于策略或价值估计的特征
        
       
        B = observations.size(0)     #取 batch 大小 B。

        # 将观测拆分为激光雷达序列和目标状态序列。
        laser_seq = observations[:, :, :self.laser_dim]  # [B, 5, 180]
        goal_seq = observations[:, :, self.laser_dim:]  # [B, 5, 5]

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
        action = F.relu(self.fc(pool))
        return action

    def forward_actor(self, features: torch.Tensor) -> torch.Tensor:
        return self.policy_net(features)

    def forward_critic(self, features: torch.Tensor) -> torch.Tensor:
        return self.value_net(features)


if __name__ == "__main__":
    # 创建环境
    env = IRSIMEnv('env/moving.yaml', display=True)

    # 模型保存目录
    log_dir = "./sequence_moving_models"     #创建一个目录 ./sequence_moving_models 用于保存模型、日志和中间结果。
    os.makedirs(log_dir, exist_ok=True)      #exist_ok=True 表示：如果目录已存在就不会报错。   

    #评估回调函数（EvalCallback）   训练过程中，定期（每 eval_freq 步）在环境上评估模型表现； 如果发现新的更优模型（例如 reward 更高），就会保存到 best_model_save_path 指定的目录；
    callback_save_best_model = EvalCallback(
        env,                          # 评估环境
        best_model_save_path=log_dir,  # 保存“当前最优模型”的路径
        log_path=log_dir,              # 评估指标日志保存路径
        eval_freq=4096,                # 每隔多少步评估一次
        deterministic=True,           # 使用确定性策略（适用于测试）
        render=False                  # 是否在评估时可视化渲染
    )   
    
    # 回调函数就是训练过程中的“自动监视器”，在关键节点自动执行某些任务（保存模型、评估性能、提前停止等）
    callback_list = CallbackList([callback_save_best_model])       #把多个功能整合到一起，只需把这个 callback_list 传给 .learn() 即可
    
    #策略网络参数 policy_kwargs     #这个配置会传入例如 PPO 或 SAC 的构造函数中，来控制神经网络结构和特征提取器。
    policy_kwargs = dict( 
        features_extractor_class=LaserGoalFeatureExtractor,    #自定义的特征提取器类，例如用于处理激光雷达+目标位置等组合输入
        features_extractor_kwargs=dict(features_dim=256),      #提取器的参数，这里指定输出特征维度为 256
        net_arch=dict(pi=[256,128,64], qf=[256,128,64])        #策略网络(pi 用于输出动作的隐藏层维度)和价值网络(qf 用于估计Q值的隐藏层维度)   3 层全连接隐藏层，每层的神经元个数依次为：256 128 64
    )

    # 初始化模型
    model = SAC(        #虽然显示的是多层感知机策略（MLP Policy），使用 Soft Actor-Critic 算法，是一种基于值函数的离策略方法，具有高样本效率和稳定性，适合连续动作空间任务，如机器人控制、无人驾驶等。
        policy="MlpPolicy",      #实际上用的是你自定义的 LaserGoalFeatureExtractor，虽然外层写着 MlpPolicy，但本质会加载你指定的网络结构
        env=env,
        policy_kwargs=policy_kwargs,    #用于传入自定义策略结构的配置参数，你之前定义的是：
        verbose=1,                     #控制控制台日志的输出等级； 0 表示静默，1 表示每步训练都会有摘要输出，2 是更详细的调试信息。                                                           
        learning_rate=1e-4,              # 学习率，控制策略和价值网络的梯度更新步长； 对 SAC 来说，通常设置在 3e-4 到 1e-4 都是合理的，你设置得比较保守
        tensorboard_log="./sac_laser_goal_tensorboard/"   #表示把训练过程中的日志输出到该目录；
    )


    # model = SAC.load("sequence_moving_models/akm_best_model", env=env)
    # model.learn(total_timesteps=1000000, callback=callback_list, progress_bar=True)     #  表示训练总步数为 100 万步；  callback_list 回调函数列表   会在训练时在控制台显示一个 tqdm 的进度条；
    model = SAC.load("sequence_moving_models/new_best_model", env=env)                    #  从路径 sequence_moving_models/new_best_model 加载一个训练好的模型；
    # model.learn(total_timesteps=1000000, callback=callback_list, progress_bar=True)
    # 评估
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=50)   #这行是用 SB3 自带的评估工具 evaluate_policy(...) 进行评估，n_eval_episodes=50：在 50 个独立 episode 上测试模型表现
    print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")

    #  整体流程如下：
    #  循环直到达到 total_timesteps:
    # - 执行动作
    # - 与环境交互获取反馈（obs, reward, done）
    # - 存储经验
    # - 达到一定间隔后，从经验中采样，进行策略网络与Q网络更新
    # - 每隔 eval_freq 步，进行一次回调（评估、保存等）
    # - 输出 tensorboard 日志和控制台进度
