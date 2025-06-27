"""
TensorBoard工具使用示例
"""
import os
import sys
import argparse
import numpy as np
import torch
import time

# 将当前目录添加到Python模块搜索路径
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from data_tool.tensorboard_logger import TensorboardLogger
from data_tool.tensorboard_integration import TensorboardTrainingMonitor, with_tensorboard

def basic_logger_example():
    """基本TensorBoard记录器使用示例"""
    print("=== 基本TensorBoard记录器使用示例 ===")
    
    # 创建记录器
    logger = TensorboardLogger('./logs/examples/basic')
    
    # 记录一些标量值
    for step in range(100):
        # 记录正弦和余弦波形
        logger.log_scalar('示例/正弦', np.sin(step * 0.1), step)
        logger.log_scalar('示例/余弦', np.cos(step * 0.1), step)
        logger.log_scalar('示例/指数', np.exp(step * 0.01), step)
        
    # 记录多组相关值
    for step in range(100):
        values = {
            '线性': step * 0.1,
            '二次': (step * 0.1) ** 2,
            '平方根': np.sqrt(step * 0.1),
        }
        logger.log_scalars('示例/多组值', values, step)
    
    # 记录直方图
    for step in range(10):
        # 生成一些随机数据
        data = np.random.normal(0, 1 + step * 0.1, 1000)
        logger.log_histogram('示例/直方图', data, step)
    
    # 关闭记录器
    logger.close()
    print("基本示例完成，使用 'tensorboard --logdir=./logs/examples/basic' 查看结果")

def monitor_example():
    """训练监控器使用示例"""
    print("=== 训练监控器使用示例 ===")
    
    # 创建监控器
    monitor = TensorboardTrainingMonitor('./logs/examples/monitor', 'test_experiment')
    
    # 模拟训练过程
    num_drones = 3
    action_dim = 4
    
    # 初始化动作历史
    monitor.initialize_action_history(num_drones)
    
    # 模拟10个回合的训练
    for episode in range(10):
        print(f"回合 {episode+1}/10")
        
        # 开始新回合
        monitor.start_episode(episode)
        
        # 模拟每回合的步骤
        episode_steps = np.random.randint(50, 150)  # 随机步数
        episode_rewards = np.zeros(num_drones)
        
        for step in range(episode_steps):
            # 创建模拟状态
            state_dict = {
                'drones': [
                    {'position': [episode + step * 0.01, i * 2.0, 1.5]} for i in range(num_drones)
                ]
            }
            
            # 创建模拟动作
            actions = np.random.uniform(-1, 1, size=(num_drones, action_dim))
            
            # 创建模拟奖励
            rewards = [np.sin(step * 0.1) + i * 0.1 for i in range(num_drones)]
            episode_rewards += rewards
            
            # 记录步骤
            monitor.log_step(state_dict, actions, rewards, state_dict, step == episode_steps-1)
            
            # 模拟策略更新
            if step > 0 and step % 10 == 0:
                update_info = {
                    'actor_loss': np.random.uniform(0.1, 0.5),
                    'critic_loss': np.random.uniform(0.2, 0.6),
                    'entropy': np.random.uniform(0.01, 0.1),
                    'total_loss': np.random.uniform(0.5, 1.0)
                }
                monitor.log_update(update_info)
            
            # 暂停一小段时间
            time.sleep(0.001)
        
        # 结束回合
        monitor.end_episode(episode_rewards, episode_steps)
    
    # 关闭监控器
    monitor.close()
    print("监控器示例完成，使用 'tensorboard --logdir=./logs/examples/monitor' 查看结果")

@with_tensorboard(log_dir='./logs/examples/decorated', exp_name='decorated_example')
def decorated_train_func(args, tb_monitor=None):
    """使用装饰器添加TensorBoard支持的训练函数示例"""
    print("=== 使用装饰器的训练函数示例 ===")
    
    num_drones = args.num_drones
    num_episodes = args.num_episodes
    
    # 初始化动作历史
    tb_monitor.initialize_action_history(num_drones)
    
    # 模拟训练
    for episode in range(num_episodes):
        print(f"回合 {episode+1}/{num_episodes}")
        
        # 开始新回合
        tb_monitor.start_episode(episode)
        
        # 模拟步骤
        episode_steps = np.random.randint(10, 30)
        episode_rewards = np.zeros(num_drones)
        
        for step in range(episode_steps):
            # 模拟状态和动作
            state_dict = {
                'drones': [
                    {'position': [episode + step * 0.01, i * 2.0, 1.5]} for i in range(num_drones)
                ]
            }
            
            actions = np.random.uniform(-1, 1, size=(num_drones, 4))
            rewards = [np.random.normal(episode * 0.1, 0.5) for _ in range(num_drones)]
            episode_rewards += rewards
            
            # 记录步骤
            tb_monitor.log_step(state_dict, actions, rewards, state_dict, step == episode_steps-1)
        
        # 结束回合
        tb_monitor.end_episode(episode_rewards, episode_steps)
    
    print("装饰器示例完成，使用 'tensorboard --logdir=./logs/examples/decorated' 查看结果")
    
    return "训练完成"

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='TensorBoard工具示例')
    parser.add_argument('--example', type=str, default='all', 
                        choices=['basic', 'monitor', 'decorated', 'all'],
                        help='要运行的示例')
    parser.add_argument('--num_drones', type=int, default=3, help='无人机数量')
    parser.add_argument('--num_episodes', type=int, default=5, help='训练回合数')
    
    return parser.parse_args()

def main():
    """主函数"""
    args = parse_args()
    
    # 确保日志目录存在
    os.makedirs('./logs/examples', exist_ok=True)
    
    # 运行选定的示例
    if args.example in ['basic', 'all']:
        basic_logger_example()
    
    if args.example in ['monitor', 'all']:
        monitor_example()
    
    if args.example in ['decorated', 'all']:
        decorated_train_func(args)
    
    print("\n所有示例完成!")
    print("使用以下命令查看结果:")
    print("tensorboard --logdir=./logs/examples")

if __name__ == "__main__":
    main() 