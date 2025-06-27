"""
用于将TensorBoard集成到训练过程中
此模块提供了一个包装器，可以轻松地将TensorBoard记录功能添加到现有训练代码中
"""
import os
import sys
import numpy as np
import torch
import time
from functools import wraps

# 将当前目录添加到Python模块搜索路径
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from data_tool.tensorboard_logger import TensorboardLogger

class TensorboardTrainingMonitor:
    """
    训练过程监控器，将TensorBoard集成到训练过程中
    """
    def __init__(self, log_dir='./logs/tensorboard', exp_name=None):
        """
        初始化训练监控器
        
        参数:
            log_dir: 日志保存目录
            exp_name: 实验名称，如果提供，将被添加到日志目录中
        """
        # 创建实验目录
        if exp_name:
            # 添加时间戳，确保目录唯一
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            log_dir = os.path.join(log_dir, f"{exp_name}_{timestamp}")
        
        # 创建TensorBoard日志记录器
        self.logger = TensorboardLogger(log_dir)
        
        # 训练状态跟踪
        self.episode = 0
        self.global_step = 0
        self.best_reward = -float('inf')
        self.start_time = time.time()
        
        # 指标历史记录
        self.episode_rewards = []
        self.moving_avg_rewards = []
        self.window_size = 100  # 移动平均窗口大小
        
        # 动作分布历史记录（与train_ppo.py中的结构保持一致）
        self.action_mean_history = []
        self.action_var_history = []
        self.episode_action_history = []
        self.episode_numbers = []
        
        # 标志
        self.initialized_action_history = False
    
    def initialize_action_history(self, num_drones):
        """
        初始化动作历史记录数据结构
        
        参数:
            num_drones: 无人机数量
        """
        if not self.initialized_action_history:
            self.action_mean_history = [[[] for _ in range(4)] for _ in range(num_drones)]  # [无人机][动作分量][历史]
            self.action_var_history = [[[] for _ in range(4)] for _ in range(num_drones)]   # [无人机][动作分量][历史]
            self.episode_action_history = [[] for _ in range(num_drones)]  # 临时存储每个回合的动作
            self.initialized_action_history = True
    
    def start_episode(self, episode):
        """
        开始一个新的训练回合
        
        参数:
            episode: 回合编号
        """
        self.episode = episode
        
        # 清空当前回合的动作历史
        if self.initialized_action_history:
            for i in range(len(self.episode_action_history)):
                self.episode_action_history[i] = []
        
        # 记录回合开始信息
        self.logger.log_scalar('训练/当前回合', episode, self.global_step)
        self.episode_start_time = time.time()
    
    def log_step(self, state_dict, action, rewards, next_state_dict, done, info=None):
        """
        记录单个训练步骤
        
        参数:
            state_dict: 当前状态
            action: 执行的动作
            rewards: 获得的奖励
            next_state_dict: 下一个状态
            done: 是否结束
            info: 额外信息
        """
        self.global_step += 1
        
        # 记录动作
        if isinstance(action, np.ndarray):
            # 尝试记录每个无人机的动作
            if len(action.shape) > 1 and action.shape[0] > 1:
                num_drones = action.shape[0]
                
                # 如果需要，初始化动作历史记录数据结构
                if not self.initialized_action_history:
                    self.initialize_action_history(num_drones)
                
                # 记录每个无人机的动作
                self.logger.log_drone_actions(action, self.global_step)
                
                # 存储动作到历史记录
                for i in range(num_drones):
                    if i < len(self.episode_action_history):
                        if action.shape[1] > 0:
                            self.episode_action_history[i].append(action[i].copy())
        
        # 记录奖励
        self.logger.log_rewards(rewards, self.global_step)
        
        # 记录状态信息（如果可用）
        if isinstance(state_dict, dict) and 'drones' in state_dict:
            for i, drone in enumerate(state_dict['drones']):
                if 'position' in drone:
                    pos = drone['position']
                    self.logger.log_scalar(f'状态/无人机_{i+1}/x', pos[0], self.global_step)
                    self.logger.log_scalar(f'状态/无人机_{i+1}/y', pos[1], self.global_step)
                    self.logger.log_scalar(f'状态/无人机_{i+1}/z', pos[2], self.global_step)
        
        # 记录额外信息
        if info:
            for key, value in info.items():
                if isinstance(value, (int, float)):
                    self.logger.log_scalar(f'信息/{key}', value, self.global_step)
    
    def end_episode(self, episode_rewards, episode_steps):
        """
        结束当前训练回合
        
        参数:
            episode_rewards: 回合总奖励
            episode_steps: 回合总步数
        """
        # 记录回合持续时间
        episode_duration = time.time() - self.episode_start_time
        self.logger.log_scalar('训练/回合持续时间', episode_duration, self.global_step)
        
        # 记录总步数
        self.logger.log_scalar('训练/总步数', self.global_step, self.episode)
        
        # 记录平均奖励
        if isinstance(episode_rewards, (list, np.ndarray)):
            mean_reward = np.mean(episode_rewards)
        else:
            mean_reward = episode_rewards
            
        self.logger.log_scalar('训练/回合奖励', mean_reward, self.episode)
        self.episode_rewards.append(mean_reward)
        
        # 更新并记录移动平均奖励
        if len(self.episode_rewards) >= self.window_size:
            moving_avg = np.mean(self.episode_rewards[-self.window_size:])
            self.moving_avg_rewards.append(moving_avg)
            self.logger.log_scalar('训练/移动平均奖励', moving_avg, self.episode)
        elif len(self.episode_rewards) > 1:
            # 使用所有可用数据计算移动平均
            moving_avg = np.mean(self.episode_rewards)
            self.moving_avg_rewards.append(moving_avg)
            self.logger.log_scalar('训练/移动平均奖励', moving_avg, self.episode)
        
        # 更新最佳奖励
        if mean_reward > self.best_reward:
            self.best_reward = mean_reward
            self.logger.log_scalar('训练/最佳奖励', self.best_reward, self.episode)
        
        # 计算每个无人机在本回合的动作均值和方差
        if self.initialized_action_history:
            for i in range(len(self.episode_action_history)):
                try:
                    if self.episode_action_history[i] and len(self.episode_action_history[i]) > 0:
                        # 将列表转换为numpy数组以便计算统计量
                        drone_actions = np.array(self.episode_action_history[i])
                        
                        # 确保数据形状正确
                        if len(drone_actions.shape) >= 2 and drone_actions.shape[1] >= 4:
                            # 计算每个动作维度的均值和方差
                            for action_dim in range(4):  # 4个动作分量
                                # 提取该动作分量的所有值
                                dim_actions = drone_actions[:, action_dim]
                                # 计算均值和方差
                                mean_value = float(np.mean(dim_actions))
                                var_value = float(np.var(dim_actions))
                                
                                # 记录到TensorBoard
                                self.logger.log_scalar(f'动作分布/无人机_{i+1}/均值_{action_dim}', mean_value, self.episode)
                                self.logger.log_scalar(f'动作分布/无人机_{i+1}/方差_{action_dim}', var_value, self.episode)
                                
                                # 存储到历史记录
                                self.action_mean_history[i][action_dim].append(mean_value)
                                self.action_var_history[i][action_dim].append(var_value)
                                
                except Exception as e:
                    print(f"计算动作统计量出错: {e}")
                    continue
            
            # 记录当前回合编号
            self.episode_numbers.append(self.episode)
            
            # 每10个回合生成并记录动作分布图表
            if self.episode % 10 == 0 and self.episode > 0:
                self.logger.log_action_distributions(
                    self.action_mean_history,
                    self.action_var_history,
                    self.episode_numbers,
                    self.episode
                )
        
        # 记录回合步数
        self.logger.log_scalar('训练/回合步数', episode_steps, self.episode)
    
    def log_update(self, update_info, step=None):
        """
        记录策略更新信息
        
        参数:
            update_info: 更新信息字典
            step: 更新步数，如果为None则使用全局步数
        """
        if step is None:
            step = self.global_step
            
        # 记录所有更新信息
        for key, value in update_info.items():
            self.logger.log_scalar(f'更新/{key}', value, step)
    
    def log_network_weights(self, model, step=None):
        """
        记录神经网络权重
        
        参数:
            model: 神经网络模型
            step: 更新步数，如果为None则使用全局步数
        """
        if step is None:
            step = self.global_step
            
        self.logger.log_network_weights(model, step)
    
    def save_training_summary(self):
        """保存训练总结"""
        # 计算训练总时长
        total_time = time.time() - self.start_time
        hours, remainder = divmod(total_time, 3600)
        minutes, seconds = divmod(remainder, 60)
        
        # 记录总结信息
        self.logger.log_scalar('总结/总训练时长(小时)', hours + minutes/60, self.episode)
        self.logger.log_scalar('总结/总训练步数', self.global_step, self.episode)
        self.logger.log_scalar('总结/总训练回合', self.episode, self.episode)
        self.logger.log_scalar('总结/最佳奖励', self.best_reward, self.episode)
        
        if len(self.episode_rewards) > 0:
            self.logger.log_scalar('总结/平均奖励', np.mean(self.episode_rewards), self.episode)
            self.logger.log_scalar('总结/奖励标准差', np.std(self.episode_rewards), self.episode)
            
        if len(self.moving_avg_rewards) > 0:
            self.logger.log_scalar('总结/最终移动平均奖励', self.moving_avg_rewards[-1], self.episode)
        
        print(f"训练总结:")
        print(f"- 总训练时长: {int(hours)}小时 {int(minutes)}分钟 {int(seconds)}秒")
        print(f"- 总训练步数: {self.global_step}")
        print(f"- 总训练回合: {self.episode}")
        print(f"- 最佳奖励: {self.best_reward:.2f}")
        if len(self.moving_avg_rewards) > 0:
            print(f"- 最终移动平均奖励: {self.moving_avg_rewards[-1]:.2f}")
    
    def close(self):
        """关闭监控器"""
        self.save_training_summary()
        self.logger.close()
        print("TensorBoard监控器已关闭")


def with_tensorboard(log_dir='./logs/tensorboard', exp_name=None):
    """
    装饰器，用于向训练函数添加TensorBoard支持
    
    参数:
        log_dir: 日志保存目录
        exp_name: 实验名称
    
    示例:
        @with_tensorboard(log_dir='./logs', exp_name='ppo_experiment')
        def train(args):
            # 训练代码...
    """
    def decorator(train_func):
        @wraps(train_func)
        def wrapper(*args, **kwargs):
            # 创建监控器
            monitor = TensorboardTrainingMonitor(log_dir, exp_name)
            
            # 将监控器添加到kwargs中
            kwargs['tb_monitor'] = monitor
            
            try:
                # 运行训练函数
                result = train_func(*args, **kwargs)
                return result
            finally:
                # 确保关闭监控器
                monitor.close()
                
        return wrapper
    
    return decorator


# 提供一个函数来获取监控器，可以在train_ppo.py中使用
def get_tensorboard_monitor(log_dir='./logs/tensorboard', exp_name=None):
    """
    获取一个TensorBoard训练监控器实例
    
    参数:
        log_dir: 日志保存目录
        exp_name: 实验名称
        
    返回:
        TensorboardTrainingMonitor实例
    """
    return TensorboardTrainingMonitor(log_dir, exp_name) 