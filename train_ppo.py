"""
使用PPO算法训练多无人机系统
"""
import sys
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import time
from tqdm import tqdm
import argparse

# 将当前目录添加到Python模块搜索路径
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

# 导入各个模块
from env.tufang_env import env
from algorithm.ppo import PPO, preprocess_state, postprocess_action

def parse_args():
    """
    解析命令行参数
    """
    parser = argparse.ArgumentParser(description='训练多无人机避障突防系统')
    parser.add_argument('--num_drones', type=int, default=3, help='无人机数量')
    parser.add_argument('--num_episodes', type=int, default=10000, help='训练回合数')
    parser.add_argument('--max_steps', type=int, default=500, help='每个回合最大步数')
    parser.add_argument('--gamma', type=float, default=0.99, help='折扣因子')
    parser.add_argument('--lr', type=float, default=3e-4, help='学习率')
    parser.add_argument('--clip_ratio', type=float, default=0.2, help='PPO裁剪系数')
    parser.add_argument('--entropy_coef', type=float, default=0.03, help='熵正则化系数')
    parser.add_argument('--value_coef', type=float, default=0.5, help='价值损失系数')
    parser.add_argument('--max_grad_norm', type=float, default=0.5, help='梯度裁剪')
    parser.add_argument('--batch_size', type=int, default=64, help='批次大小')
    parser.add_argument('--update_epochs', type=int, default=10, help='每次更新的迭代次数')
    parser.add_argument('--hidden_dim', type=int, default=256, help='隐藏层维度')
    parser.add_argument('--memory_capacity', type=int, default=4096, help='经验回放缓冲区容量')
    parser.add_argument('--update_interval', type=int, default=2048, help='更新网络的频率')
    parser.add_argument('--render', action='store_true', help='是否渲染环境')
    parser.add_argument('--render_interval', type=int, default=5, help='渲染间隔')
    parser.add_argument('--render_episode_interval', type=int, default=100, help='渲染的回合间隔')
    parser.add_argument('--save_model', action='store_true', help='是否保存模型')
    parser.add_argument('--save_interval', type=int, default=500, help='保存模型的频率')
    parser.add_argument('--model_dir', type=str, default='./models', help='模型保存目录')
    parser.add_argument('--load_model', type=str, default='', help='加载模型路径')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--exploration_noise', type=float, default=0.4, help='初始探索噪声')
    parser.add_argument('--exploration_min', type=float, default=0.05, help='最小探索噪声')
    parser.add_argument('--random_episodes', type=int, default=100, help='完全随机动作的回合数')
    parser.add_argument('--total_decay_steps', type=int, default=100000, help='探索衰减的总步数')
    
    return parser.parse_args()

def train():
    """
    训练多无人机系统
    """
    # 解析命令行参数
    args = parse_args()
    args.render = False
    args.save_model = False
    
    # 设置随机种子
    if args.seed < 0:
        args.seed = int(time.time()) % 10000  # 使用时间作为随机种子
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    print(f"使用随机种子: {args.seed}")
    
    # 定义无人机初始位置
    drone_positions = []
    for i in range(args.num_drones):
        x = 0
        y = 3 * (i - args.num_drones // 2)
        z = 2
        drone_positions.append([x, y, z])
    
    # 定义单一目标点 - 所有无人机共用同一个目标点
    target_position = [10, 0, 2]  # 在X轴正方向10单位处
    
    # 定义障碍物配置 - 将障碍物放在起点和目标点之间
    enemy_configs = [
        {'position': [5, 0, 2], 'radius': 1.0, 'velocity': [0.0, 0.0, 0.0]},  # 正中间的主要障碍物
        {'position': [3, 2, 2], 'radius': 0.8, 'velocity': [0.0, 0.0, 0.0]},  # 右上方障碍物
        {'position': [3, -2, 2], 'radius': 0.8, 'velocity': [0.0, 0.0, 0.0]}, # 右下方障碍物
        {'position': [7, 2, 2], 'radius': 0.7, 'velocity': [0.0, 0.0, 0.0]},  # 右上方障碍物
        {'position': [7, -2, 2], 'radius': 0.7, 'velocity': [0.0, 0.0, 0.0]}  # 右下方障碍物
    ]
    
    # 创建环境
    environment = env(drone_positions=drone_positions, enemy_configs=enemy_configs)
    
    # 设置目标点 - 直接传递单一目标点，不再需要为每个无人机创建副本
    environment.set_target_positions(target_position)
    
    # 计算状态和动作维度
    state_dict = environment.reset()
    state_vector = preprocess_state(state_dict)
    state_dim = len(state_vector)
    action_dim = 4 * args.num_drones  # 每个无人机4个控制输入
    
    print(f"状态维度: {state_dim}, 动作维度: {action_dim}")
    
    # 创建PPO代理
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    agent = PPO(state_dim=state_dim,
               action_dim=action_dim,
               lr=args.lr,
               gamma=args.gamma,
               clip_ratio=args.clip_ratio,
               entropy_coef=args.entropy_coef,
               value_coef=args.value_coef,
               max_grad_norm=args.max_grad_norm,
               batch_size=args.batch_size,
               update_epochs=args.update_epochs,
               hidden_dim=args.hidden_dim,
               memory_capacity=args.memory_capacity,
               exploration_noise=args.exploration_noise,
               exploration_min=args.exploration_min,
               total_decay_steps=args.total_decay_steps,
               device=device)
    
    # 如果指定了模型路径，加载模型
    if args.load_model and os.path.exists(args.load_model):
        agent.load(args.load_model)
    
    # 确保模型保存目录存在（如果需要保存模型）
    if args.save_model:
        os.makedirs(args.model_dir, exist_ok=True)
    
    # 训练循环
    step_count = 0
    total_rewards_history = []
    # 记录损失函数历史
    loss_history = {
        'actor_loss': [],
        'critic_loss': [],
        'entropy': [],
        'total_loss': []
    }
    # 记录探索噪声
    exploration_noise_history = []
    
    # 记录无人机动作均值和方差的历史
    action_mean_history = [[[] for _ in range(4)] for _ in range(args.num_drones)]  # [无人机][动作分量][历史]
    action_var_history = [[[] for _ in range(4)] for _ in range(args.num_drones)]   # [无人机][动作分量][历史]
    episode_action_history = [[] for _ in range(args.num_drones)]  # 临时存储每个回合的动作
    episode_numbers = []  # 记录回合编号用于绘图
    action_names = ['推力', '横滚', '俯仰', '偏航']  # 动作分量名称

    # if args.render:
    #     fig = plt.figure(figsize=(10, 8))
    
    # 创建动作分布图表，但不立即显示
    action_dist_fig, axs = plt.subplots(2, 4, figsize=(16, 10))  # 2行4列：第一行是均值，第二行是方差
    action_dist_fig.suptitle('无人机动作分布统计', fontsize=16)
    plt.ion()  # 打开交互模式
    # 显示空的图表以创建窗口
    plt.show(block=False)
    
    for episode in range(args.num_episodes):
        # 每个回合使用不同的随机种子，增加多样性
        episode_seed = args.seed + episode * 997  # 使用质数作为乘数避免周期性
        np.random.seed(episode_seed)
        torch.manual_seed(episode_seed)
        
        # 重置环境
        state_dict = environment.reset()
        state = preprocess_state(state_dict)
        
        episode_rewards = [0] * args.num_drones
        done = False
        
        # 清空当前回合的动作历史
        for i in range(args.num_drones):
            episode_action_history[i] = []
        
        # 决定当前回合是否需要渲染
        render_this_episode = args.render and (episode % args.render_episode_interval == 0)
        
        # 如果这个回合需要渲染，初始化渲染器
        renderer = None
        if render_this_episode:
            plt.ion()  # 打开交互模式
            renderer = environment.render()  # 创建渲染器并进行初次渲染
            plt.pause(0.01)
        
        # 单回合循环
        for step in range(args.max_steps):
            step_count += 1
            
            # 选择动作 - 前几个回合可以完全随机选择动作，增加探索性
            if episode < args.random_episodes:
                # 完全随机动作 - 为每个无人机单独生成随机动作
                action = np.zeros((args.num_drones, 4))
                
                # 每个回合使用不同的随机策略，增加多样性
                random_strategy = episode % 5  # 5种不同的随机策略
                
                for drone_idx in range(args.num_drones):
                    if random_strategy == 0:
                        # 标准均匀随机
                        action[drone_idx] = np.random.uniform(-1, 1, size=4)
                    elif random_strategy == 1:
                        # 高斯随机
                        action[drone_idx] = np.random.normal(0, 0.5, size=4)
                        # 裁剪到[-1,1]范围
                        action[drone_idx] = np.clip(action[drone_idx], -1, 1)
                    elif random_strategy == 2:
                        # 偏向极端值的随机
                        for i in range(4):
                            if np.random.random() < 0.3:
                                # 30%概率选择接近极值的动作
                                action[drone_idx, i] = np.random.choice([-1, 1]) * np.random.uniform(0.7, 1.0)
                            else:
                                # 70%概率选择普通随机值
                                action[drone_idx, i] = np.random.uniform(-0.7, 0.7)
                    elif random_strategy == 3:
                        # 随机动作但保持推力在合理范围
                        action[drone_idx, 0] = np.random.uniform(0.5, 1.0)  # 推力
                        action[drone_idx, 1:] = np.random.uniform(-1, 1, size=3)  # 姿态控制
                    else:  # random_strategy == 4
                        # 混合分布
                        for i in range(4):
                            if np.random.random() < 0.5:
                                action[drone_idx, i] = np.random.uniform(-1, 1)
                            else:
                                action[drone_idx, i] = np.random.normal(0, 0.3)
                                action[drone_idx, i] = np.clip(action[drone_idx, i], -1, 1)
                
                # 为随机动作创建虚拟log_prob，使用标量而不是数组
                log_prob = 0.0  # 简化为标量
                
                if step == 0:
                    print(f"随机回合 {episode+1}/{args.random_episodes}, 动作形状: {action.shape}, 策略类型: {random_strategy}")
            else:
                # 常规PPO选择动作 - 在训练模式下自动添加探索噪声
                action, log_prob = agent.select_action(state, deterministic=False)
            
            # 记录每个无人机的动作到回合历史
            for i in range(args.num_drones):
                if isinstance(action, np.ndarray) and len(action.shape) > 1 and action.shape[0] >= args.num_drones:
                    episode_action_history[i].append(action[i].copy())
                elif isinstance(action, list) and len(action) >= args.num_drones:
                    # 处理列表形式的动作
                    episode_action_history[i].append(np.array(action[i]))
            
            # 记录当前探索噪声
            if episode % 100 == 0 and step == 0:
                # 计算当前的探索噪声 - 与PPO类中的逻辑保持一致
                decay_progress = min(1.0, agent.total_steps / agent.total_decay_steps)
                current_noise = agent.exploration_noise - decay_progress * (agent.exploration_noise - agent.exploration_min)
                
                exploration_noise_history.append(current_noise)
                print(f"当前探索噪声: {current_noise:.4f}, 探索步数: {agent.total_steps}/{agent.total_decay_steps}")
            
            # 转换为环境接受的格式
            env_action = postprocess_action(action, args.num_drones)
            
            # 执行动作
            next_state_dict, rewards, done, info = environment.step(env_action)
            next_state = preprocess_state(next_state_dict)
            
            # 累积奖励
            for i in range(args.num_drones):
                episode_rewards[i] += rewards[i]
            
            # 存储经验 - 注意随机回合的经验也存储，但仅用于适应PPO的更新机制
            agent.store_transition(state, action, np.mean(rewards), next_state, float(done), log_prob)
            
            # 更新状态
            state = next_state
            
            # 渲染环境
            if render_this_episode and step % args.render_interval == 0 and renderer is not None:
                renderer = environment.render()      
                plt.pause(0.01)  # 短暂暂停以刷新显示
            
            # 更新策略
            if step_count % args.update_interval == 0 or (episode == args.random_episodes and step == 0):
                # 如果刚好结束随机回合，立即更新一次策略
                train_info = agent.update()
                if train_info:
                    # 记录损失函数
                    for key in loss_history.keys():
                        if key in train_info:
                            loss_history[key].append(train_info[key])
                    print(f"回合 {episode}, 步数 {step}, 训练信息: {train_info}")
            
            # 如果结束，退出循环
            if done:
                break
        
        # 计算每个无人机在本回合的动作均值和方差
        if episode > 0:  # 对所有回合都收集数据
            for i in range(args.num_drones):
                try:
                    if episode_action_history[i] and len(episode_action_history[i]) > 0:  # 确保有数据
                        # 将列表转换为numpy数组以便计算统计量
                        drone_actions = np.array(episode_action_history[i])
                        
                        # 确保数据形状正确
                        if len(drone_actions.shape) >= 2 and drone_actions.shape[1] >= 4:
                            # 计算每个动作维度的均值和方差
                            for action_dim in range(4):  # 4个动作分量
                                # 提取该动作分量的所有值
                                dim_actions = drone_actions[:, action_dim]
                                # 计算均值和方差
                                action_mean_history[i][action_dim].append(float(np.mean(dim_actions)))
                                action_var_history[i][action_dim].append(float(np.var(dim_actions)))
                except Exception as e:
                    print(f"警告: 计算动作统计量时出错: {e}")
                    continue  # 跳过这个无人机
            
            # 记录当前回合编号（只记录一次）
            if episode == 1 or len(episode_numbers) == 0:
                episode_numbers = []
            episode_numbers.append(episode)
        
        # 记录回合奖励
        mean_reward = np.mean(episode_rewards)
        total_rewards_history.append(mean_reward)
        
        # 显示回合信息
        if episode % 10 == 0 or render_this_episode:
            print(f"回合 {episode}/{args.num_episodes}, 平均奖励: {mean_reward:.2f}, 步数: {step+1}")
            
            # 每10个回合展示动作分布统计图表
            if episode % 10 == 0 and episode > 10:  # 从第10个回合开始，每10个回合更新一次
                # 检查是否有足够的数据来绘制
                has_data = False
                for i in range(args.num_drones):
                    for dim in range(4):
                        if len(action_mean_history[i][dim]) > 0:
                            has_data = True
                            break
                    if has_data:
                        break
                        
                if has_data:
                    # 清空所有子图
                    for row in range(2):
                        for col in range(4):
                            axs[row, col].clear()
                
                # 为每个动作分量绘制均值图和方差图
                for action_dim in range(4):
                    # 绘制均值图 - 上面一行
                    for i in range(args.num_drones):
                        if len(action_mean_history[i][action_dim]) > 0:  # 确保有数据
                            axs[0, action_dim].plot(
                                episode_numbers[-len(action_mean_history[i][action_dim]):], 
                                action_mean_history[i][action_dim], 
                                label=f'无人机 {i+1}'
                            )
                    axs[0, action_dim].set_xlabel('回合')
                    axs[0, action_dim].set_ylabel('均值')
                    axs[0, action_dim].set_title(f'{action_names[action_dim]} - 均值')
                    axs[0, action_dim].legend()
                    axs[0, action_dim].grid(True)
                    
                    # 绘制方差图 - 下面一行
                    for i in range(args.num_drones):
                        if len(action_var_history[i][action_dim]) > 0:  # 确保有数据
                            axs[1, action_dim].plot(
                                episode_numbers[-len(action_var_history[i][action_dim]):], 
                                action_var_history[i][action_dim], 
                                label=f'无人机 {i+1}'
                            )
                    axs[1, action_dim].set_xlabel('回合')
                    axs[1, action_dim].set_ylabel('方差')
                    axs[1, action_dim].set_title(f'{action_names[action_dim]} - 方差')
                    axs[1, action_dim].legend()
                    axs[1, action_dim].grid(True)
                
                # 更新图表
                plt.tight_layout()
                action_dist_fig.canvas.draw_idle()
                plt.pause(0.1)
        
        # 每隔一定回合保存模型
        if args.save_model and episode > 0 and episode % args.save_interval == 0:
            save_path = os.path.join(args.model_dir, f'ppo_model_ep{episode}.pt')
            agent.save(save_path)
            print(f"模型已保存至: {save_path}")
        
        # 如果这个回合渲染了，在最后关闭渲染器释放资源
        if render_this_episode and renderer is not None:
            plt.ioff()
            # plt.close('all')  # 关闭所有图形窗口
    
    # 关闭环境
    environment.close()
    
    # 关闭动作分布图表
    plt.close(action_dist_fig)
    
    # 保存最终模型
    if args.save_model:
        final_model_path = os.path.join(args.model_dir, 'ppo_model_final.pt')
        agent.save(final_model_path)
        print(f"最终模型已保存至: {final_model_path}")
    
    # 绘制奖励曲线
    plt.figure(figsize=(10, 6))
    plt.plot(total_rewards_history)
    plt.xlabel('回合')
    plt.ylabel('平均奖励')
    plt.title('训练奖励曲线')
    plt.grid(True)
    plt.savefig(os.path.join(args.model_dir, 'reward_curve.png'))
    
    # 绘制探索噪声衰减曲线
    if exploration_noise_history:
        plt.figure(figsize=(10, 6))
        plt.plot(range(0, len(exploration_noise_history) * 100, 100), exploration_noise_history)
        plt.xlabel('回合')
        plt.ylabel('探索噪声大小')
        plt.title('探索噪声衰减曲线')
        plt.grid(True)
        plt.savefig(os.path.join(args.model_dir, 'exploration_decay.png'))
    
    # 绘制损失函数变化曲线
    if len(loss_history['actor_loss']) > 0:
        plt.figure(figsize=(12, 10))
        
        # 创建子图
        plt.subplot(2, 2, 1)
        plt.plot(loss_history['actor_loss'])
        plt.xlabel('更新次数')
        plt.ylabel('Actor Loss')
        plt.title('Actor Loss曲线')
        plt.grid(True)
        
        plt.subplot(2, 2, 2)
        plt.plot(loss_history['critic_loss'])
        plt.xlabel('更新次数')
        plt.ylabel('Critic Loss')
        plt.title('Critic Loss曲线')
        plt.grid(True)
        
        plt.subplot(2, 2, 3)
        plt.plot(loss_history['entropy'])
        plt.xlabel('更新次数')
        plt.ylabel('Entropy')
        plt.title('Entropy曲线')
        plt.grid(True)
        
        plt.subplot(2, 2, 4)
        plt.plot(loss_history['total_loss'])
        plt.xlabel('更新次数')
        plt.ylabel('Total Loss')
        plt.title('Total Loss曲线')
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(args.model_dir, 'loss_curves.png'))
        plt.show(block=True)  # 使用阻塞模式显示

def evaluate(model_path, num_episodes=5, render=True):
    """
    评估训练好的模型
    """
    # 解析命令行参数
    args = parse_args()
    
    # 设置随机种子
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # 定义无人机初始位置
    drone_positions = []
    for i in range(args.num_drones):
        x = 0
        y = 3 * (i - args.num_drones // 2)
        z = 2
        drone_positions.append([x, y, z])
    
    # 定义单一目标点 - 所有无人机共用同一个目标点
    target_position = [10, 0, 2]  # 在X轴正方向10单位处
    
    # 定义障碍物配置 - 将障碍物放在起点和目标点之间
    enemy_configs = [
        {'position': [5, 0, 2], 'radius': 1.0, 'velocity': [0.0, 0.0, 0.0]},  # 正中间的主要障碍物
        {'position': [3, 2, 2], 'radius': 0.8, 'velocity': [0.0, 0.0, 0.0]},  # 右上方障碍物
        {'position': [3, -2, 2], 'radius': 0.8, 'velocity': [0.0, 0.0, 0.0]}, # 右下方障碍物
        {'position': [7, 2, 2], 'radius': 0.7, 'velocity': [0.0, 0.0, 0.0]},  # 右上方障碍物
        {'position': [7, -2, 2], 'radius': 0.7, 'velocity': [0.0, 0.0, 0.0]}  # 右下方障碍物
    ]
    
    # 创建环境
    environment = env(drone_positions=drone_positions, enemy_configs=enemy_configs)
    
    # 设置目标点 - 直接传递单一目标点
    environment.set_target_positions(target_position)
    
    # 计算状态和动作维度
    state_dict = environment.reset()
    state_vector = preprocess_state(state_dict)
    state_dim = len(state_vector)
    action_dim = 4 * args.num_drones  # 每个无人机4个控制输入
    
    # 创建PPO代理
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    agent = PPO(
        state_dim=state_dim,
        action_dim=action_dim,
        total_decay_steps=args.total_decay_steps,
        device=device
    )
    
    # 加载模型
    if not agent.load(model_path):
        print(f"无法加载模型 {model_path}")
        return
    
    # 如果需要渲染，初始化渲染器
    if render:
        plt.ion()
        renderer = environment.render()
        plt.show(block=False)
    
    # 评估循环
    total_rewards = []
    
    for episode in range(num_episodes):
        # 重置环境
        state_dict = environment.reset()
        state = preprocess_state(state_dict)
        
        episode_rewards = [0] * args.num_drones
        done = False
        
        # 单回合循环
        for step in range(args.max_steps):
            # 选择动作 (确定性策略)
            action, _ = agent.select_action(state, deterministic=True)
            
            # 转换为环境接受的格式
            env_action = postprocess_action(action, args.num_drones)
            
            # 执行动作
            next_state_dict, rewards, done, info = environment.step(env_action)
            next_state = preprocess_state(next_state_dict)
            
            # 累积奖励
            for i in range(args.num_drones):
                episode_rewards[i] += rewards[i]
            
            # 更新状态
            state = next_state
            
            # 渲染环境
            if render:
                renderer = environment.render()
                plt.pause(0.01)
            
            # 如果结束，退出循环
            if done:
                break
        
        # 记录回合奖励
        mean_reward = np.mean(episode_rewards)
        total_rewards.append(mean_reward)
        
        # 打印回合信息
        print(f"评估回合 {episode+1}/{num_episodes}, 平均奖励: {mean_reward:.2f}, 步数: {step+1}")
        for i, reward in enumerate(episode_rewards):
            print(f"  无人机 {i+1} 奖励: {reward:.2f}")
    
    # 关闭环境
    environment.close()
    
    # 打印评估结果
    print(f"\n评估结果:")
    print(f"平均奖励: {np.mean(total_rewards):.2f} ± {np.std(total_rewards):.2f}")

if __name__ == "__main__":
    # 默认进行训练
    train()
    
    # 如需评估，取消以下注释并指定模型路径
    # evaluate("./models/ppo_model_final.pt") 