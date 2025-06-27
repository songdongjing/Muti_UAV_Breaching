"""
使用TensorBoard分析训练数据
"""
import os
import sys
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from glob import glob
import json

# 将当前目录添加到Python模块搜索路径
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from data_tool.tensorboard_logger import TensorboardLogger

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='使用TensorBoard分析训练数据')
    parser.add_argument('--model_dir', type=str, default='./models', help='模型和日志目录')
    parser.add_argument('--log_dir', type=str, default='./logs/tensorboard', help='TensorBoard日志保存目录')
    parser.add_argument('--data_paths', type=str, nargs='+', default=None, 
                        help='要分析的数据文件路径，可以指定多个，如 "./models/ppo_model_ep1000.pt ./models/ppo_model_ep2000.pt"')
    parser.add_argument('--mode', type=str, default='model', choices=['model', 'logs', 'all'], 
                        help='分析模式: model - 分析模型文件, logs - 分析日志文件, all - 两者都分析')
    
    return parser.parse_args()

def analyze_model(model_path, logger):
    """
    分析模型文件
    
    参数:
        model_path: 模型文件路径
        logger: TensorBoard日志记录器
    """
    print(f"分析模型文件: {model_path}")
    
    try:
        # 加载模型
        checkpoint = torch.load(model_path, map_location='cpu')
        
        # 获取模型名称和训练步数
        model_name = os.path.basename(model_path).split('.')[0]
        step = int(model_name.split('_')[-1][2:]) if 'ep' in model_name else 0
        
        # 检查checkpoint中包含的内容
        if 'train_info' in checkpoint:
            train_info = checkpoint['train_info']
            
            # 记录训练信息
            if 'actor_loss' in train_info and len(train_info['actor_loss']) > 0:
                # 创建曲线图
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
                
                # 绘制损失曲线
                ax1.plot(train_info['actor_loss'], label='Actor Loss')
                ax1.plot(train_info['critic_loss'], label='Critic Loss')
                ax1.plot(train_info['entropy'], label='Entropy')
                ax1.set_xlabel('更新次数')
                ax1.set_ylabel('损失值')
                ax1.set_title('损失函数曲线')
                ax1.legend()
                ax1.grid(True)
                
                # 绘制总损失曲线
                ax2.plot(train_info['total_loss'], label='Total Loss')
                ax2.set_xlabel('更新次数')
                ax2.set_ylabel('总损失值')
                ax2.set_title('总损失曲线')
                ax2.legend()
                ax2.grid(True)
                
                # 添加总标题
                fig.suptitle(f'模型 {model_name} 训练损失', fontsize=16)
                fig.tight_layout(rect=[0, 0.03, 1, 0.95])
                
                # 将图保存到TensorBoard
                logger.log_figure(f'模型分析/{model_name}/损失曲线', fig, step)
                plt.close(fig)
                
                # 记录最后的损失值
                last_update = len(train_info['actor_loss']) - 1
                if last_update >= 0:
                    last_losses = {
                        'actor_loss': train_info['actor_loss'][last_update],
                        'critic_loss': train_info['critic_loss'][last_update],
                        'entropy': train_info['entropy'][last_update],
                        'total_loss': train_info['total_loss'][last_update]
                    }
                    logger.log_scalars(f'模型分析/{model_name}/最终损失', last_losses, step)
            
        # 如果包含模型状态字典，记录网络参数统计信息
        if 'model_state_dict' in checkpoint:
            model_state = checkpoint['model_state_dict']
            
            for name, param in model_state.items():
                # 记录参数分布
                logger.writer.add_histogram(f'模型分析/{model_name}/参数/{name}', param, step)
                
                # 计算参数统计信息
                param_mean = torch.mean(param).item()
                param_std = torch.std(param).item()
                param_min = torch.min(param).item()
                param_max = torch.max(param).item()
                
                # 记录参数统计信息
                param_stats = {
                    'mean': param_mean,
                    'std': param_std,
                    'min': param_min,
                    'max': param_max
                }
                logger.log_scalars(f'模型分析/{model_name}/参数统计/{name}', param_stats, step)
        
        print(f"模型 {model_name} 分析完成")
        return True
        
    except Exception as e:
        print(f"分析模型出错: {e}")
        return False

def analyze_logs(log_dir, logger):
    """
    分析日志文件
    
    参数:
        log_dir: 日志目录
        logger: TensorBoard日志记录器
    """
    print(f"分析日志目录: {log_dir}")
    
    # 查找日志文件
    log_files = glob(os.path.join(log_dir, "*.log"))
    json_files = glob(os.path.join(log_dir, "*.json"))
    
    if not log_files and not json_files:
        print("未找到日志文件")
        return False
    
    # 处理JSON文件
    for json_file in json_files:
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
            
            file_name = os.path.basename(json_file).split('.')[0]
            
            # 提取训练指标
            if 'rewards' in data:
                rewards = data['rewards']
                # 记录奖励曲线
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.plot(rewards)
                ax.set_xlabel('回合')
                ax.set_ylabel('奖励')
                ax.set_title('训练奖励曲线')
                ax.grid(True)
                
                # 添加移动平均线
                if len(rewards) > 10:
                    window_size = min(100, len(rewards) // 10)
                    moving_avg = np.convolve(rewards, np.ones(window_size)/window_size, mode='valid')
                    ax.plot(range(window_size-1, len(rewards)), moving_avg, 'r-', label=f'移动平均 (窗口大小={window_size})')
                    ax.legend()
                
                logger.log_figure(f'日志分析/{file_name}/奖励曲线', fig, 0)
                plt.close(fig)
            
            # 提取其他指标
            for key, values in data.items():
                if isinstance(values, list) and key != 'rewards':
                    try:
                        # 绘制其他指标曲线
                        fig, ax = plt.subplots(figsize=(10, 6))
                        ax.plot(values)
                        ax.set_xlabel('步数')
                        ax.set_ylabel(key)
                        ax.set_title(f'{key} 曲线')
                        ax.grid(True)
                        
                        logger.log_figure(f'日志分析/{file_name}/{key}', fig, 0)
                        plt.close(fig)
                    except Exception as e:
                        print(f"绘制 {key} 曲线出错: {e}")
            
            print(f"分析日志文件 {json_file} 完成")
            
        except Exception as e:
            print(f"处理JSON文件 {json_file} 出错: {e}")
    
    # 处理普通日志文件
    for log_file in log_files:
        try:
            # 读取日志文件
            with open(log_file, 'r') as f:
                lines = f.readlines()
            
            file_name = os.path.basename(log_file).split('.')[0]
            
            # 提取常见训练指标
            rewards = []
            losses = []
            
            for line in lines:
                if '奖励' in line:
                    try:
                        reward = float(line.split(':')[-1].strip())
                        rewards.append(reward)
                    except:
                        pass
                    
                if '损失' in line or 'loss' in line.lower():
                    try:
                        loss = float(line.split(':')[-1].strip())
                        losses.append(loss)
                    except:
                        pass
            
            # 如果有奖励数据，则绘制奖励曲线
            if rewards:
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.plot(rewards)
                ax.set_xlabel('记录点')
                ax.set_ylabel('奖励')
                ax.set_title('奖励曲线')
                ax.grid(True)
                
                logger.log_figure(f'日志分析/{file_name}/奖励曲线', fig, 0)
                plt.close(fig)
            
            # 如果有损失数据，则绘制损失曲线
            if losses:
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.plot(losses)
                ax.set_xlabel('记录点')
                ax.set_ylabel('损失')
                ax.set_title('损失曲线')
                ax.grid(True)
                
                logger.log_figure(f'日志分析/{file_name}/损失曲线', fig, 0)
                plt.close(fig)
            
            print(f"分析日志文件 {log_file} 完成")
            
        except Exception as e:
            print(f"处理日志文件 {log_file} 出错: {e}")
    
    return True

def main():
    """主函数"""
    args = parse_args()
    
    # 创建TensorBoard日志记录器
    logger = TensorboardLogger(args.log_dir)
    
    # 根据指定的模式进行分析
    if args.mode in ['model', 'all']:
        # 分析模型文件
        if args.data_paths:
            model_files = [path for path in args.data_paths if path.endswith('.pt')]
        else:
            # 如果没有指定，则搜索模型目录下的所有模型文件
            model_files = glob(os.path.join(args.model_dir, '*.pt'))
        
        if model_files:
            print(f"找到 {len(model_files)} 个模型文件")
            for model_path in model_files:
                analyze_model(model_path, logger)
        else:
            print("未找到模型文件")
    
    if args.mode in ['logs', 'all']:
        # 分析日志文件
        if args.data_paths:
            log_dirs = [path for path in args.data_paths if os.path.isdir(path)]
        else:
            # 如果没有指定，则使用模型目录
            log_dirs = [args.model_dir]
        
        for log_dir in log_dirs:
            analyze_logs(log_dir, logger)
    
    # 关闭日志记录器
    logger.close()
    print(f"分析完成。使用命令 'tensorboard --logdir={args.log_dir}' 查看结果")

if __name__ == "__main__":
    main() 