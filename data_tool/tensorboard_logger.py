"""
用于使用TensorBoard进行数据分析的工具类
"""
import os
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import io
from PIL import Image
import torch

class TensorboardLogger:
    """
    TensorBoard日志记录器，用于可视化训练过程
    """
    def __init__(self, log_dir='./logs/tensorboard'):
        """
        初始化TensorBoard日志记录器
        
        参数:
            log_dir: 日志保存目录
        """
        # 确保日志目录存在
        os.makedirs(log_dir, exist_ok=True)
        
        # 创建SummaryWriter实例
        self.writer = SummaryWriter(log_dir)
        self.log_dir = log_dir
        print(f"TensorBoard日志将保存到: {os.path.abspath(log_dir)}")
        print(f"使用'tensorboard --logdir={log_dir}'命令启动TensorBoard")
    
    def log_scalar(self, tag, value, step):
        """
        记录标量值
        
        参数:
            tag: 数据标签
            value: 要记录的值
            step: 当前步数
        """
        self.writer.add_scalar(tag, value, step)
    
    def log_scalars(self, main_tag, tag_value_dict, step):
        """
        记录一组标量值
        
        参数:
            main_tag: 主标签
            tag_value_dict: 标签-值字典
            step: 当前步数
        """
        self.writer.add_scalars(main_tag, tag_value_dict, step)
    
    def log_histogram(self, tag, values, step):
        """
        记录直方图
        
        参数:
            tag: 数据标签
            values: 要记录的值数组
            step: 当前步数
        """
        if isinstance(values, list):
            values = np.array(values)
        self.writer.add_histogram(tag, values, step)
    
    def log_drone_actions(self, actions, step, drone_idx=None):
        """
        记录无人机动作
        
        参数:
            actions: 无人机动作数组，形状为[num_drones, action_dim]或[action_dim]
            step: 当前步数
            drone_idx: 无人机索引，如果为None则记录所有无人机
        """
        action_names = ['推力', '横滚', '俯仰', '偏航']
        
        # 如果是单个无人机的动作
        if len(actions.shape) == 1:
            actions = actions.reshape(1, -1)
        
        # 如果指定了无人机索引
        if drone_idx is not None:
            if 0 <= drone_idx < actions.shape[0]:
                # 记录指定无人机的动作
                for i, name in enumerate(action_names):
                    if i < actions.shape[1]:
                        self.writer.add_scalar(f'动作/无人机_{drone_idx+1}/{name}', 
                                             actions[drone_idx, i], step)
        else:
            # 记录所有无人机的动作
            for drone_idx in range(actions.shape[0]):
                for i, name in enumerate(action_names):
                    if i < actions.shape[1]:
                        self.writer.add_scalar(f'动作/无人机_{drone_idx+1}/{name}', 
                                             actions[drone_idx, i], step)
    
    def log_figure(self, tag, figure, step):
        """
        记录matplotlib图表
        
        参数:
            tag: 数据标签
            figure: matplotlib图表对象
            step: 当前步数
        """
        buf = io.BytesIO()
        figure.savefig(buf, format='png')
        buf.seek(0)
        image = Image.open(buf)
        image_tensor = torch.tensor(np.array(image).transpose(2, 0, 1))
        self.writer.add_image(tag, image_tensor, step)
    
    def log_state_value(self, state_dict, value, step):
        """
        记录状态及其价值
        
        参数:
            state_dict: 状态字典
            value: 状态价值
            step: 当前步数
        """
        # 记录状态价值
        self.writer.add_scalar('状态价值', value, step)
        
        # 提取一些关键状态信息记录
        if 'drones' in state_dict:
            for i, drone in enumerate(state_dict['drones']):
                if 'position' in drone:
                    pos = drone['position']
                    self.writer.add_scalar(f'位置/无人机_{i+1}/x', pos[0], step)
                    self.writer.add_scalar(f'位置/无人机_{i+1}/y', pos[1], step)
                    self.writer.add_scalar(f'位置/无人机_{i+1}/z', pos[2], step)
    
    def log_rewards(self, rewards, step):
        """
        记录奖励
        
        参数:
            rewards: 奖励列表或单个奖励值
            step: 当前步数
        """
        # 如果是奖励列表
        if isinstance(rewards, (list, np.ndarray)) and len(rewards) > 1:
            # 记录每个无人机的奖励
            for i, reward in enumerate(rewards):
                self.writer.add_scalar(f'奖励/无人机_{i+1}', reward, step)
            # 记录平均奖励
            self.writer.add_scalar('奖励/平均', np.mean(rewards), step)
        else:
            # 记录单个奖励值
            if isinstance(rewards, (list, np.ndarray)):
                reward = rewards[0] if len(rewards) > 0 else 0
            else:
                reward = rewards
            self.writer.add_scalar('奖励', reward, step)
    
    def log_action_distributions(self, action_mean_history, action_var_history, episode_numbers, step):
        """
        记录动作分布信息
        
        参数:
            action_mean_history: 动作均值历史记录 [num_drones][action_dim][history]
            action_var_history: 动作方差历史记录 [num_drones][action_dim][history]
            episode_numbers: 对应的回合编号
            step: 当前步数
        """
        action_names = ['推力', '横滚', '俯仰', '偏航']
        num_drones = len(action_mean_history)
        
        # 为每个无人机的每个动作分量创建图表
        for drone_idx in range(num_drones):
            for action_dim in range(len(action_names)):
                # 如果有历史数据
                if (len(action_mean_history[drone_idx][action_dim]) > 0 and 
                    len(action_var_history[drone_idx][action_dim]) > 0):
                    
                    # 创建图表
                    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
                    
                    # 绘制均值图
                    ax1.plot(episode_numbers[-len(action_mean_history[drone_idx][action_dim]):], 
                          action_mean_history[drone_idx][action_dim])
                    ax1.set_xlabel('回合')
                    ax1.set_ylabel('均值')
                    ax1.set_title(f'无人机 {drone_idx+1} - {action_names[action_dim]} 均值')
                    ax1.grid(True)
                    
                    # 绘制方差图
                    ax2.plot(episode_numbers[-len(action_var_history[drone_idx][action_dim]):], 
                          action_var_history[drone_idx][action_dim])
                    ax2.set_xlabel('回合')
                    ax2.set_ylabel('方差')
                    ax2.set_title(f'无人机 {drone_idx+1} - {action_names[action_dim]} 方差')
                    ax2.grid(True)
                    
                    # 添加总标题
                    fig.suptitle(f'无人机 {drone_idx+1} - {action_names[action_dim]} 动作分布')
                    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
                    
                    # 记录图表
                    self.log_figure(f'动作分布/无人机_{drone_idx+1}/{action_names[action_dim]}', fig, step)
                    
                    # 关闭图表以释放内存
                    plt.close(fig)
    
    def log_network_weights(self, model, step):
        """
        记录神经网络权重
        
        参数:
            model: PyTorch模型
            step: 当前步数
        """
        for name, param in model.named_parameters():
            self.writer.add_histogram(f'权重/{name}', param.data, step)
            if param.grad is not None:
                self.writer.add_histogram(f'梯度/{name}', param.grad, step)
    
    def log_ppo_info(self, info, step):
        """
        记录PPO训练信息
        
        参数:
            info: 包含训练信息的字典
            step: 当前步数
        """
        for key, value in info.items():
            self.writer.add_scalar(f'PPO/{key}', value, step)
    
    def close(self):
        """关闭SummaryWriter"""
        self.writer.close() 