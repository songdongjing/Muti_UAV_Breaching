"""
多无人机仿真系统主程序
"""
import sys
import os
import numpy as np
import matplotlib.pyplot as plt

# 将当前目录添加到Python模块搜索路径
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

# 导入各个模块
from env.tufang_env import env

# 主函数
if __name__ == "__main__":
    # 定义无人机初始位置
    drone_positions = [
        [0, 0, 2],       # 第一架无人机位置
        [0, 3, 2],       # 第二架无人机位置
        [0, -3, 2]       # 第三架无人机位置
    ]
    
    # 定义障碍物配置
    enemy_configs = [
        {'position': [8, 8, 5], 'radius': 1.0, 'velocity': [0.0, 0.0, 0.0]},
        {'position': [-8, -8, 5], 'radius': 1.5, 'velocity': [0.0, 0.0, 0.0]},
        {'position': [0, 8, 6], 'radius': 0.8, 'velocity': [0.0, 0.0, 0.0]},
        {'position': [8, -4, 4], 'radius': 1.2, 'velocity': [0.0, 0.0, 0.0]},
        {'position': [-8, 6, 3], 'radius': 0.7, 'velocity': [0.0, 0.0, 0.0]}
    ]
    
    # 创建环境，指定无人机初始位置和障碍物配置
    environment = env(drone_positions=drone_positions,
                     enemy_configs=enemy_configs)
    
    # 定义一个简单的策略：让无人机沿着直线轨迹飞行
    def policy(state):
        # 为每个无人机创建控制输入 - 这里是简单的悬停控制
        actions = []
        
        # 为每个无人机设置基本控制
        for _ in range(len(state['drones'])):
            # 悬停控制: 推力补偿重力，其他控制为0
            actions.append([9.8, 0.0, 0.0, 0.0])
        
        return actions
    
    try:
        # 初始化环境
        print("初始化仿真环境...")
        state = environment.reset()
        
        # 开启matplotlib的交互模式
        plt.ion()
        
        # 渲染初始状态
        renderer = environment.render()
        plt.show(block=False)
        
        # 运行仿真
        print("开始运行仿真...")
        total_rewards = [0] * environment.num_drones
        max_steps = 300
        
        for step in range(max_steps):
            # 根据策略获取动作
            actions = policy(state)
            
            # 执行动作
            state, rewards, done, _ = environment.step(actions)
            
            # 累积奖励
            for i in range(environment.num_drones):
                total_rewards[i] += rewards[i]
            
            # 实时渲染
            renderer = environment.render()
            plt.pause(0.01)  # 短暂暂停以更新显示
            
            # 显示当前步骤
            if step % 10 == 0:
                print(f"步骤: {step}")
            
            # 检查是否结束
            if done:
                print("仿真结束!")
                break
        
        print(f"回合结束！总步数: {step+1}")
        for i, reward in enumerate(total_rewards):
            print(f"无人机 {i+1} 的总奖励: {reward:.2f}")
        
        # 关闭交互模式，但保持图形窗口打开
        plt.ioff()
        plt.show()  # 这会阻塞直到用户关闭窗口
        
    except Exception as e:
        print(f"发生错误: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # 关闭环境
        environment.close() 