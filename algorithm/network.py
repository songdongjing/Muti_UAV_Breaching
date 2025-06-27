import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Actor(nn.Module):
    """
    演员网络 (策略网络)：输出动作分布
    """
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        """
        初始化演员网络
        
        参数:
            state_dim: 状态空间维度
            action_dim: 动作空间维度 (输出的动作数量)
            hidden_dim: 隐藏层维度
        """
        super(Actor, self).__init__()
        
        # 增加网络层深度和宽度
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)  # 添加第三层
        
        # 添加层归一化，帮助训练稳定性
        self.layer_norm1 = nn.LayerNorm(hidden_dim)
        self.layer_norm2 = nn.LayerNorm(hidden_dim)
        
        # 动作均值输出层
        self.mean_layer = nn.Linear(hidden_dim, action_dim)
        
        # 动作标准差 (可学习的参数)
        self.log_std = nn.Parameter(torch.zeros(action_dim))
        
        # 初始化参数，使用正交初始化
        nn.init.orthogonal_(self.fc1.weight, gain=1.0)
        nn.init.orthogonal_(self.fc2.weight, gain=1.0)
        nn.init.orthogonal_(self.fc3.weight, gain=1.0)
        nn.init.orthogonal_(self.mean_layer.weight, gain=0.01)
        
    def forward(self, state):
        """
        前向传播
        
        参数:
            state: 输入状态
            
        返回:
            动作均值和标准差
        """
        # 使用多种激活函数组合，增加网络表达能力
        x = F.leaky_relu(self.fc1(state))  # 使用LeakyReLU
        x = self.layer_norm1(x)  # 层归一化
        
        x = F.relu(self.fc2(x))  # 使用ReLU
        x = self.layer_norm2(x)
        
        x = F.elu(self.fc3(x))  # 使用ELU
        
        # 计算动作均值，使用tanh限制范围在[-1,1]
        action_mean = torch.tanh(self.mean_layer(x))
        
        # 计算动作标准差
        action_std = torch.exp(self.log_std)
        
        return action_mean, action_std
    
    def get_action(self, state, deterministic=False):
        """
        根据状态获取动作
        
        参数:
            state: 输入状态
            deterministic: 是否使用确定性策略 (不采样)
            
        返回:
            action: 选择的动作
            log_prob: 动作的对数概率
        """
        try:
            # 确保状态是tensor
            if isinstance(state, np.ndarray):
                state = torch.FloatTensor(state)
                
            # 确保状态有批次维度
            if len(state.shape) == 1:
                state = state.unsqueeze(0)

            # 获取动作分布参数
            mean, std = self.forward(state)
            
            # 创建正态分布
            dist = torch.distributions.Normal(mean, std)
            
            # 选择动作
            if deterministic:
                action = mean
            else:
                action = dist.sample()
                
            # 计算对数概率
            log_prob = dist.log_prob(action).sum(dim=-1, keepdim=True)
            
            return action, log_prob
        
        except Exception as e:
            print(f"获取动作时出错: {e}")
            # 创建默认的安全值
            action_dim = self.mean_layer.out_features
            batch_size = 1 if not hasattr(state, 'shape') else state.shape[0]
            device = next(self.parameters()).device
            
            # 返回零动作和对数概率
            default_action = torch.zeros((batch_size, action_dim), device=device)
            default_log_prob = torch.zeros((batch_size, 1), device=device)
            
            return default_action, default_log_prob
    
    def evaluate_action(self, state, action):
        """
        评估给定状态-动作对的对数概率和熵
        
        参数:
            state: 输入状态
            action: 待评估的动作
            
        返回:
            log_prob: 动作的对数概率
            entropy: 策略的熵
        """
        # 获取动作分布参数
        mean, std = self.forward(state)
        
        # 创建正态分布
        dist = torch.distributions.Normal(mean, std)
        
        # 计算对数概率
        log_prob = dist.log_prob(action).sum(dim=-1, keepdim=True)
        
        # 计算熵
        entropy = dist.entropy().sum(dim=-1, keepdim=True)
        
        return log_prob, entropy
    
class Critic(nn.Module):
    """
    评论家网络 (价值网络)：估计状态价值
    """
    def __init__(self, state_dim, hidden_dim=128):
        """
        初始化评论家网络
        
        参数:
            state_dim: 状态空间维度
            hidden_dim: 隐藏层维度
        """
        super(Critic, self).__init__()
        
        # 增加网络层深度和宽度
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)  # 添加第三层
        
        # 添加层归一化
        self.layer_norm1 = nn.LayerNorm(hidden_dim)
        self.layer_norm2 = nn.LayerNorm(hidden_dim)
        
        self.value_layer = nn.Linear(hidden_dim, 1)
        
        # 初始化参数
        nn.init.orthogonal_(self.fc1.weight, gain=1.0)
        nn.init.orthogonal_(self.fc2.weight, gain=1.0)
        nn.init.orthogonal_(self.fc3.weight, gain=1.0)
        nn.init.orthogonal_(self.value_layer.weight, gain=1.0)
        
    def forward(self, state):
        """
        前向传播
        
        参数:
            state: 输入状态
            
        返回:
            value: 状态价值估计
        """
        # 使用多种激活函数
        x = F.leaky_relu(self.fc1(state))
        x = self.layer_norm1(x)
        
        x = F.relu(self.fc2(x))
        x = self.layer_norm2(x)
        
        x = F.elu(self.fc3(x))
        
        value = self.value_layer(x)
        
        return value

class ActorCritic(nn.Module):
    """
    演员-评论家网络组合
    """
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        """
        初始化演员-评论家网络
        
        参数:
            state_dim: 状态空间维度
            action_dim: 动作空间维度
            hidden_dim: 隐藏层维度
        """
        super(ActorCritic, self).__init__()
        
        # 创建演员和评论家网络
        self.actor = Actor(state_dim, action_dim, hidden_dim)
        self.critic = Critic(state_dim, hidden_dim)
        
    def act(self, state, deterministic=False):
        """
        根据状态选择动作
        
        参数:
            state: 输入状态
            deterministic: 是否使用确定性策略
            
        返回:
            action: 选择的动作
            log_prob: 动作的对数概率
        """
        try:
            # 确保状态维度正确
            if len(state.shape) == 1:
                state = state.unsqueeze(0)
                
            return self.actor.get_action(state, deterministic)
        
        except Exception as e:
            print(f"选择动作时出错: {e}")
            # 获取动作维度
            action_dim = self.actor.mean_layer.out_features
            batch_size = 1
            device = next(self.parameters()).device
            
            # 返回零动作和对数概率
            default_action = torch.zeros((batch_size, action_dim), device=device)
            default_log_prob = torch.zeros((batch_size, 1), device=device)
            
            return default_action, default_log_prob
    
    def evaluate(self, state, action):
        """
        评估状态-动作对
        
        参数:
            state: 输入状态
            action: 待评估的动作
            
        返回:
            value: 状态价值
            log_prob: 动作的对数概率
            entropy: 策略的熵
        """
        value = self.critic(state)
        log_prob, entropy = self.actor.evaluate_action(state, action)
        
        return value, log_prob, entropy
