import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from .network import ActorCritic
import time
import os

class Memory:
    """
    经验回放缓冲区
    """
    def __init__(self, capacity=1024):
        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.dones = []
        self.log_probs = []
        self.capacity = capacity  # 添加容量限制
        
    def add(self, state, action, reward, next_state, done, log_prob):
        """添加一条经验"""
        # 确保数据类型一致性
        if isinstance(action, np.ndarray) and len(action.shape) > 1:
            # 如果是多维数组，展平为一维
            action = action.flatten()
        
        # 处理log_prob，确保是标量或一维数组
        if isinstance(log_prob, np.ndarray):
            if log_prob.size > 1:  # 如果不是标量
                log_prob = float(log_prob.flatten()[0])  # 只取第一个元素并转为标量
        elif isinstance(log_prob, (list, tuple)):
            log_prob = float(log_prob[0] if log_prob else 0)  # 取第一个元素或0
            
        # 如果达到容量上限，移除最早的经验
        if len(self.states) >= self.capacity:
            self.states.pop(0)
            self.actions.pop(0)
            self.rewards.pop(0)
            self.next_states.pop(0)
            self.dones.pop(0)
            self.log_probs.pop(0)
            
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.next_states.append(next_state)
        self.dones.append(done)
        self.log_probs.append(log_prob)
        
    def clear(self):
        """清空缓冲区"""
        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.dones = []
        self.log_probs = []
        
    def get_batch(self):
        """获取整个批次的经验数据"""
        # 预处理log_probs，确保形状一致
        processed_log_probs = []
        for log_prob in self.log_probs:
            # 检查类型
            if isinstance(log_prob, (list, tuple)):
                # 尝试转换并展平
                processed_log_probs.append(np.array(log_prob).flatten()[0])
            elif isinstance(log_prob, np.ndarray):
                # 如果是数组，展平并取第一个元素（简化处理）
                processed_log_probs.append(log_prob.flatten()[0])
            else:
                # 否则直接添加
                processed_log_probs.append(log_prob)
                
        # 转换其他数据
        states_array = np.array(self.states)
        actions_array = np.array(self.actions)
        rewards_array = np.array(self.rewards).reshape(-1, 1)
        next_states_array = np.array(self.next_states)
        dones_array = np.array(self.dones).reshape(-1, 1)
        
        # 处理log_probs，将其转为统一形状的数组
        log_probs_array = np.array(processed_log_probs).reshape(-1, 1)
        
        return {
            'states': torch.FloatTensor(states_array),
            'actions': torch.FloatTensor(actions_array),
            'rewards': torch.FloatTensor(rewards_array),
            'next_states': torch.FloatTensor(next_states_array),
            'dones': torch.FloatTensor(dones_array),
            'log_probs': torch.FloatTensor(log_probs_array)
        }
        
    def size(self):
        """返回缓冲区大小"""
        return len(self.states)

class PPO:
    """
    近端策略优化 (Proximal Policy Optimization) 算法实现
    """
    def __init__(self, state_dim, action_dim, lr=3e-4, gamma=0.99, 
                 clip_ratio=0.2, entropy_coef=0.01, value_coef=0.5,
                 max_grad_norm=0.5, batch_size=64, update_epochs=10,
                 hidden_dim=128, memory_capacity=1024,
                 exploration_noise=0.3, exploration_min=0.01,
                 total_decay_steps=5000,
                 device='cuda' if torch.cuda.is_available() else 'cpu'):
        """
        初始化PPO算法
        
        参数:
            state_dim: 状态空间维度
            action_dim: 动作空间维度
            lr: 学习率
            gamma: 折扣因子
            clip_ratio: PPO剪裁比例
            entropy_coef: 熵正则化系数
            value_coef: 价值损失系数
            max_grad_norm: 梯度裁剪阈值
            batch_size: 批次大小
            update_epochs: 每批数据的更新轮数
            hidden_dim: 隐藏层维度
            memory_capacity: 经验回放缓冲区容量
            exploration_noise: 探索噪声的初始标准差
            exploration_min: 探索噪声的最小值
            total_decay_steps: 探索噪声从初始值衰减到最小值所需的总步数
            device: 计算设备
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.lr = lr
        self.gamma = gamma
        self.clip_ratio = clip_ratio
        self.entropy_coef = entropy_coef  # 默认0.01，可通过参数调整
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm
        self.batch_size = batch_size
        self.update_epochs = update_epochs
        self.device = device
        self.memory_capacity = memory_capacity
        
        # 探索参数
        self.exploration_noise = exploration_noise
        self.exploration_min = exploration_min
        self.total_decay_steps = total_decay_steps  # 添加总衰减步数参数
        self.total_steps = 0  # 记录总步数，用于探索衰减
        
        # 创建演员-评论家网络
        self.ac_net = ActorCritic(state_dim, action_dim, hidden_dim).to(device)
        
        # 创建优化器
        self.optimizer = optim.Adam(self.ac_net.parameters(), lr=lr)
        
        # 创建经验回放缓冲区
        self.memory = Memory(capacity=memory_capacity)
        
        # 训练信息
        self.train_info = {
            'actor_loss': [],
            'critic_loss': [],
            'entropy': [],
            'total_loss': []
        }
        
    def select_action(self, state, deterministic=False):
        """
        根据当前策略选择动作，并在探索阶段添加随机性
        
        参数:
            state: 当前状态
            deterministic: 是否使用确定性策略
            
        返回:
            action: 选择的动作
            log_prob: 动作的对数概率
        """
        try:
            # 转换为张量并移至指定设备
            state_tensor = torch.FloatTensor(state).to(self.device)
            
            # 使用演员网络选择动作
            with torch.no_grad():
                action, log_prob = self.ac_net.act(state_tensor, deterministic)
                
            # 将动作转换为numpy数组
            action_np = action.cpu().numpy()
            log_prob_np = log_prob.cpu().numpy()
            
            # 在训练模式下添加探索噪声
            if not deterministic:
                # 计算当前的探索噪声 - 使用线性衰减
                decay_progress = min(1.0, self.total_steps / self.total_decay_steps)
                current_noise = self.exploration_noise - decay_progress * (self.exploration_noise - self.exploration_min)
                
                # 添加随机噪声到动作中 - 增强随机性
                # 使用更多样化的噪声分布，混合高斯和均匀分布
                if np.random.random() < 0.8:  # 80%概率使用高斯噪声
                    noise = np.random.normal(0, current_noise, size=action_np.shape)
                else:  # 20%概率使用均匀分布噪声，产生更大的随机跳跃
                    noise = np.random.uniform(-current_noise*2, current_noise*2, size=action_np.shape)
                    
                # 有10%的几率添加额外的随机扰动，完全随机化某些动作分量
                # 先检查动作数组的形状，确保它有多个维度
                if np.random.random() < 0.1 and len(action_np.shape) > 1 and action_np.shape[1] > 0:
                    # 安全地获取动作维度
                    action_dim = action_np.shape[1]
                    random_idx = np.random.randint(0, action_dim)  # 随机选择一个动作分量
                    
                    # 只有当action_np至少是二维的，且第一维有数据时才执行此操作
                    if action_np.shape[0] > 0:
                        action_np[:, random_idx] = np.random.uniform(-1, 1, size=action_np.shape[0])
                
                action_np += noise
                
                # 记录步数
                self.total_steps += 1
            
            return action_np, log_prob_np
            
        except Exception as e:
            print(f"选择动作时出错: {e}")
            print(f"状态形状: {np.array(state).shape if isinstance(state, (list, np.ndarray)) else 'unknown'}")
            
            # 创建安全的默认动作和log_prob
            # 确定无人机数量
            if hasattr(self, 'action_dim'):
                total_action_dim = self.action_dim
                # 假设每个无人机有4个控制输入
                num_drones = total_action_dim // 4
                action_per_drone = 4
            else:
                num_drones = 3  # 默认值
                action_per_drone = 4
                
            # 创建默认的悬停动作 [thrust=9.8, roll=0, pitch=0, yaw=0]
            default_actions = np.zeros((num_drones, action_per_drone))
            # 设置推力为9.8以平衡重力
            default_actions[:, 0] = 9.8
            
            # 默认的对数概率
            default_log_prob = np.zeros((num_drones, 1))
            
            return default_actions, default_log_prob
    
    def decay_exploration(self):
        """
        手动衰减探索噪声 - 使用线性衰减
        """
        # 每次调用减少初始噪声的一小部分
        decay_step = (self.exploration_noise - self.exploration_min) / 100
        self.exploration_noise = max(self.exploration_noise - decay_step, self.exploration_min)
        return self.exploration_noise
        
    def store_transition(self, state, action, reward, next_state, done, log_prob):
        """
        存储经验到缓冲区
        
        参数:
            state: 当前状态
            action: 执行的动作
            reward: 获得的奖励
            next_state: 下一个状态
            done: 是否结束
            log_prob: 动作的对数概率
        """
        # 确保数据类型和形状一致性
        if isinstance(state, np.ndarray) and len(state.shape) == 1:
            state = state.copy()  # 避免引用问题
        
        if isinstance(next_state, np.ndarray) and len(next_state.shape) == 1:
            next_state = next_state.copy()
            
        # 预处理log_prob，确保形状统一且简单（标量值）
        if isinstance(log_prob, np.ndarray):
            if log_prob.size > 0:
                # 取第一个元素作为代表
                log_prob = float(log_prob.flatten()[0])
            else:
                # 空数组，设为0
                log_prob = 0.0
        elif isinstance(log_prob, (list, tuple)):
            # 列表或元组，取第一个元素
            log_prob = float(log_prob[0] if log_prob else 0.0)
        elif isinstance(log_prob, torch.Tensor):
            # 如果是张量，转为numpy然后取标量
            log_prob = float(log_prob.detach().cpu().numpy().flatten()[0])
            
        # 委托给Memory类的add方法处理
        self.memory.add(state, action, reward, next_state, done, log_prob)
        
    def compute_returns(self, rewards, dones, next_values):
        """
        计算每个时间步的回报值（使用广义优势估计 - GAE）
        
        参数:
            rewards: 奖励序列
            dones: 完成标志序列
            next_values: 下一状态的价值估计
            
        返回:
            returns: 每个时间步的估计回报
        """
        # 确保所有输入都是一维数组
        if len(rewards.shape) > 1:
            rewards = rewards.squeeze()
        if len(dones.shape) > 1:
            dones = dones.squeeze()
        if len(next_values.shape) > 1:
            next_values = next_values.squeeze()
        
        # 确保所有数组长度一致
        min_length = min(len(rewards), len(dones), len(next_values))
        rewards = rewards[:min_length]
        dones = dones[:min_length]
        next_values = next_values[:min_length]
        
        returns = []
        gae = 0
        next_value = next_values[-1]
        
        # 从后向前计算GAE
        for i in reversed(range(len(rewards))):
            delta = rewards[i] + self.gamma * next_value * (1 - dones[i]) - next_values[i]
            gae = delta + self.gamma * (1 - dones[i]) * gae
            next_value = next_values[i]
            returns.insert(0, gae + next_values[i])
            
        return np.array(returns).reshape(-1, 1)  # 确保返回的是2D数组
    
    def update(self):
        """
        使用收集的经验更新策略
        
        返回:
            train_info: 训练过程的统计信息
        """
        # 如果缓冲区为空，不进行更新
        if self.memory.size() == 0:
            return {}
        
        try:
            # 获取所有经验数据
            batch = self.memory.get_batch()
            states = batch['states'].to(self.device)
            actions = batch['actions'].to(self.device)
            rewards = batch['rewards'].to(self.device)
            next_states = batch['next_states'].to(self.device)
            dones = batch['dones'].to(self.device)
            old_log_probs = batch['log_probs'].to(self.device)
            
            # 计算下一状态的价值估计
            with torch.no_grad():
                next_values = self.ac_net.critic(next_states)
                
            # 计算每个时间步的回报
            current_values = self.ac_net.critic(states)
            
            # 修改为使用变量存储计算的回报值而不是直接创建张量
            computed_returns = self.compute_returns(
                rewards.cpu().numpy(), 
                dones.cpu().numpy(), 
                current_values.detach().cpu().numpy()
            )
            
            # 计算好的returns已经是2D数组，直接转换为张量
            returns = torch.FloatTensor(computed_returns).to(self.device)
            
            # 初始化统计信息
            total_actor_loss = 0
            total_critic_loss = 0
            total_entropy = 0
            total_loss = 0
            
            # 多轮更新
            for _ in range(self.update_epochs):
                # 计算每个状态-动作对的价值和对数概率
                values, log_probs, entropy = self.ac_net.evaluate(states, actions)
                
                # 计算优势
                advantages = returns - values.detach()
                
                # 标准化优势
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
                
                # 计算策略比例
                ratio = torch.exp(log_probs - old_log_probs)
                
                # 计算PPO剪裁目标
                surr1 = ratio * advantages
                surr2 = torch.clamp(ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio) * advantages
                
                # 策略损失 (负号表示梯度上升)
                actor_loss = -torch.min(surr1, surr2).mean()
                
                # 价值损失
                critic_loss = nn.MSELoss()(values, returns)
                
                # 熵损失 (鼓励探索)
                entropy_loss = -entropy.mean()
                
                # 总损失
                loss = actor_loss + self.value_coef * critic_loss + self.entropy_coef * entropy_loss
                
                # 执行梯度下降
                self.optimizer.zero_grad()
                loss.backward()
                
                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(self.ac_net.parameters(), self.max_grad_norm)
                
                self.optimizer.step()
                
                # 累计统计信息
                total_actor_loss += actor_loss.item()
                total_critic_loss += critic_loss.item()
                total_entropy += entropy.mean().item()
                total_loss += loss.item()
            
            # 更新训练信息
            train_info = {
                'actor_loss': total_actor_loss / self.update_epochs,
                'critic_loss': total_critic_loss / self.update_epochs,
                'entropy': total_entropy / self.update_epochs,
                'total_loss': total_loss / self.update_epochs
            }
            
            # 记录训练信息
            for k, v in train_info.items():
                self.train_info[k].append(v)
                
            # 清空缓冲区
            self.memory.clear()
            
            return train_info
            
        except Exception as e:
            print(f"更新过程中发生错误: {e}")
            print("清空缓冲区并继续...")
            self.memory.clear()
            return {}
    
    def save(self, path):
        """
        保存模型
        """
        # 确保目录存在
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # 保存模型参数
        torch.save({
            'model_state_dict': self.ac_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_info': self.train_info
        }, path)
        
        print(f"模型已保存到 {path}")
    
    def load(self, path):
        """
        加载模型
        """
        # 检查文件是否存在
        if not os.path.exists(path):
            print(f"错误：文件 {path} 不存在！")
            return False
        
        # 加载模型参数
        checkpoint = torch.load(path, map_location=self.device)
        self.ac_net.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.train_info = checkpoint['train_info']
        
        print(f"模型已从 {path} 加载")
        return True

def preprocess_state(state_dict):
    """
    将环境状态预处理为适合网络输入的格式
    
    参数:
        state_dict: 环境返回的状态字典
        
    返回:
        state_vector: 预处理后的状态向量
    """
    # 从字典中提取有用信息并展平为向量
    state_vector = []
    
    # 获取共享的目标点 - 确保即使只有一个目标点也能正确处理
    target_positions = state_dict.get('target_positions', [])
    shared_target = target_positions[0] if target_positions else [0, 0, 0]
    
    # 提取无人机信息
    for i, drone in enumerate(state_dict['drones']):
        # 基本状态信息
        drone_pos = drone['position']
        drone_vel = drone['velocity']
        drone_att = drone['attitude']
        
        # 位置
        state_vector.extend(drone_pos)
        # 速度
        state_vector.extend(drone_vel)
        # 姿态
        state_vector.extend(drone_att)
        
        # 添加目标点相关特征（使用共享目标点）
        target_pos = shared_target
        
        # 目标点绝对位置
        state_vector.extend(target_pos)
        
        # 计算到目标点的相对位置向量
        rel_x = target_pos[0] - drone_pos[0]
        rel_y = target_pos[1] - drone_pos[1]
        rel_z = target_pos[2] - drone_pos[2]
        state_vector.extend([rel_x, rel_y, rel_z])
        
        # 计算到目标点的欧氏距离
        distance_to_target = (rel_x**2 + rel_y**2 + rel_z**2)**0.5
        state_vector.append(distance_to_target)
        
        # 计算到目标点的水平角度（偏航角）- 弧度
        target_yaw = np.arctan2(rel_y, rel_x)
        
        # 计算到目标点的垂直角度（俯仰角）- 弧度
        horizontal_distance = (rel_x**2 + rel_y**2)**0.5
        target_pitch = np.arctan2(rel_z, horizontal_distance)
        
        # 添加角度差（无人机当前朝向与目标方向之间的差异）
        yaw_diff = target_yaw - drone_att[2]
        # 归一化到 [-π, π] 范围
        yaw_diff = (yaw_diff + np.pi) % (2 * np.pi) - np.pi
        
        # 添加角度特征
        state_vector.extend([target_yaw, target_pitch, yaw_diff])
        
        # 添加与所有障碍物的相对信息
        obstacles_info = []
        
        for enemy in state_dict['enemies']:
            enemy_pos = enemy['position']
            enemy_radius = enemy['radius']
            
            # 计算到障碍物的相对位置向量
            rel_x = enemy_pos[0] - drone_pos[0]
            rel_y = enemy_pos[1] - drone_pos[1]
            rel_z = enemy_pos[2] - drone_pos[2]
            
            # 计算到障碍物的欧氏距离
            distance = (rel_x**2 + rel_y**2 + rel_z**2)**0.5
            
            # 计算障碍物相对于无人机的方向角度（水平和垂直）
            enemy_yaw = np.arctan2(rel_y, rel_x)
            
            horizontal_distance = (rel_x**2 + rel_y**2)**0.5
            enemy_pitch = np.arctan2(rel_z, horizontal_distance)
            
            # 计算考虑障碍物半径后的实际距离（无人机表面到障碍物表面的距离）
            # 假设无人机半径为0.3
            drone_radius = 0.3
            surface_distance = max(0, distance - enemy_radius - drone_radius)
            
            # 将每个障碍物的相关信息存储到一个列表中
            # 使用障碍物相对位置、距离和角度
            obstacle_info = [
                rel_x, rel_y, rel_z,  # 相对位置
                distance,             # 中心点间距离
                surface_distance,     # 表面间距离
                enemy_yaw, enemy_pitch  # 方向角度
            ]
            obstacles_info.append(obstacle_info)
        
        # 按照距离排序障碍物信息，只关注最近的N个障碍物（比如3个）
        obstacles_info.sort(key=lambda x: x[3])  # 按距离排序
        nearest_obstacles = obstacles_info[:3]  # 取最近的3个障碍物
        
        # 将最近的障碍物信息添加到状态向量
        for obs_info in nearest_obstacles:
            state_vector.extend(obs_info)
        
        # 如果障碍物少于3个，用零向量填充
        for _ in range(3 - len(nearest_obstacles)):
            state_vector.extend([0, 0, 0, 999, 999, 0, 0])  # 使用大距离值表示无障碍物
    
    return np.array(state_vector, dtype=np.float32)

def postprocess_action(action, num_drones):
    """
    将网络输出的动作转换为环境接受的格式
    
    参数:
        action: 网络输出的动作
        num_drones: 无人机数量
        
    返回:
        env_actions: 适合环境的动作格式
    """
    # 确保动作是numpy数组
    if isinstance(action, torch.Tensor):
        action = action.cpu().numpy()
    
    # 重塑动作以适应环境 (每个无人机4个控制输入)
    action_dim = 4  # [thrust, roll, pitch, yaw]
    
    # 处理不同形状的动作数组
    if action.shape == (num_drones, action_dim):
        # 已经是正确形状，直接使用
        pass
    elif len(action.shape) == 1:
        # 一维数组，重塑为(num_drones, action_dim)
        action = action.reshape(num_drones, action_dim)
    elif action.shape == (1, num_drones * action_dim):
        # 单批次但扁平的动作，重塑为(num_drones, action_dim)
        action = action.reshape(num_drones, action_dim)
    elif len(action.shape) == 2 and action.shape[0] == 1:
        # 形如(1, X)，检查X是否可以重塑为(num_drones, action_dim)
        if action.shape[1] == num_drones * action_dim:
            action = action.reshape(num_drones, action_dim)
    else:
        print(f"警告: 动作形状不符合预期: {action.shape}, 尝试调整...")
        # 尝试重塑，但可能会失败
        try:
            action = action.reshape(num_drones, action_dim)
        except:
            print(f"错误: 无法将形状为{action.shape}的动作重塑为({num_drones}, {action_dim})")
            # 如果无法重塑，创建一个零动作数组（悬停）
            action = np.zeros((num_drones, action_dim))
            action[:, 0] = 9.8  # 设置默认推力以平衡重力
    
    # 转换为环境期望的列表格式
    env_actions = []
    for i in range(num_drones):
        drone_action = action[i].tolist()
        
        # 对推力进行处理 (确保大于0)
        drone_action[0] = max(0, drone_action[0])
        
        env_actions.append(drone_action)
    
    return env_actions
