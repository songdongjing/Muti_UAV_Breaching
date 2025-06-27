import numpy as np
import time
import sys
import os

# 添加项目根目录到Python模块搜索路径
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

# 导入其他模块
from model.aircraft import aircraft
from model.enemy import enemy
from render.matplotlib_render import Renderer
import matplotlib.pyplot as plt

class env:
    """环境类，管理整个仿真环境的运行"""
    
    def __init__(self, drone_positions, enemy_configs, boundary=None, dt=0.01):
        """
        初始化环境
        
        参数:
            drone_positions: 无人机初始位置列表，每个元素为 [x, y, z]
            enemy_configs: 障碍物配置列表，每个元素为 {'position': [x, y, z], 'radius': r, 'velocity': [vx, vy, vz]}
            boundary: 环境边界 [x_min, x_max, y_min, y_max, z_min, z_max]
            dt: 仿真时间步长
        """
        # 确保提供了必要的位置信息
        if drone_positions is None or len(drone_positions) == 0:
            raise ValueError("必须提供无人机初始位置列表")
        if enemy_configs is None or len(enemy_configs) == 0:
            raise ValueError("必须提供障碍物配置列表")
            
        # 设置环境边界
        if boundary is None:
            self.boundary = [-10, 10, -10, 10, 0, 10]
        else:
            self.boundary = boundary
            
        # 时间步长
        self.dt = dt
        
        # 初始化无人机集群
        self.drone_positions = drone_positions
        self.num_drones = len(drone_positions)
        self._create_drones()
        
        # 初始化障碍物
        self.enemy_configs = enemy_configs
        self.num_enemies = len(enemy_configs)
        self.enemies = []
        self._create_enemies()
        
        # 初始化目标位置
        self.target_positions = []
        self.reached_target = [False] * self.num_drones
        self._set_default_targets()
        
        # 初始化渲染器
        self.renderer = None  # 懒加载，只在需要渲染时创建
        
        # 运行时变量
        self.done = False  # 仿真是否结束
        self.time = 0.0    # 当前仿真时间
        self.steps = 0     # 当前步数
        self.collisions = [False] * self.num_drones  # 每个无人机是否发生碰撞
        self.max_steps = 1000   # 最大步数
        
        # 状态和动作空间
        self.state = self._get_state()  # 当前状态
    
    def _create_drones(self):
        """创建无人机集群"""
        # 清空现有无人机列表
        if not hasattr(self, 'drones'):
            self.drones = []
        else:
            self.drones.clear()
        
        # 确保drone_positions是深拷贝而不是引用
        position_copies = []
        for pos in self.drone_positions:
            position_copies.append(pos.copy() if isinstance(pos, list) else list(pos))
            
        # 使用提供的位置创建无人机
        for i in range(self.num_drones):
            pos = position_copies[i]
            # 确保位置是独立拷贝，不是引用
            position = pos.copy()
            # 打印创建的无人机位置
            drone = aircraft(position=pos)
            self.drones.append(drone)
        
    def _create_enemies(self):
        """创建障碍物"""
        self.enemies = []
        
        # 使用提供的障碍物配置
        for i in range(self.num_enemies):
            config = self.enemy_configs[i]
            
            # 提取配置参数
            pos = config.get('position', [0, 0, 0])
            radius = config.get('radius', 1.0)
            vel = config.get('velocity', [0, 0, 0])
            
            # 创建障碍物并设置边界
            obs = enemy(position=pos, radius=radius, velocity=vel)
            obs.set_boundary(
                self.boundary[0], self.boundary[1],
                self.boundary[2], self.boundary[3],
                self.boundary[4], self.boundary[5]
            )
            
            self.enemies.append(obs)
    
    def _get_state(self):
        """获取环境状态"""
        # 状态包括所有无人机的位置、速度，姿态，目标点，以及所有障碍物的位置、半径、速度
        state = {
            'drones': [],
            'enemies': [],
            'target_positions': self.target_positions.copy()  # 只包含一个目标点
        }
        
        # 添加所有无人机的状态
        for drone in self.drones:
            state['drones'].append({
                'position': drone.get_position(),
                'velocity': drone.get_velocity(),
                'attitude': drone.get_attitude()
            })
        
        # 添加所有障碍物的状态
        for enemy in self.enemies:
            state['enemies'].append({
                'position': enemy.get_position(),
                'radius': enemy.get_radius(),
                'velocity': enemy.velocity
            })
            
        return state
    
    def _check_collisions(self):
        """检查所有无人机是否与障碍物碰撞或相互碰撞"""
        # 初始化碰撞状态
        collisions = [False] * self.num_drones
        
        # 检查每个无人机
        for i, drone in enumerate(self.drones):
            drone_pos = drone.get_position()
            
            # 检查与障碍物的碰撞
            for enemy in self.enemies:
                if enemy.check_collision(drone_pos):
                    collisions[i] = True
                    break
            
            # 如果已经检测到碰撞，继续检查下一个无人机
            if collisions[i]:
                continue
                
            # 检查与其他无人机的碰撞（简化为点对点距离检查）
            for j, other_drone in enumerate(self.drones):
                if i == j:  # 跳过自身
                    continue
                    
                other_pos = other_drone.get_position()
                # 假设无人机半径为0.3
                drone_radius = 0.3
                # 计算两架无人机之间的距离
                distance = np.linalg.norm(np.array(drone_pos) - np.array(other_pos))
                
                # 如果距离小于两倍半径，认为发生碰撞
                if distance < 2 * drone_radius:
                    collisions[i] = True
                    break
                    
        return collisions
    
    def _check_boundaries(self):
        
        """检查所有无人机是否超出边界"""
        out_of_boundaries = [False] * self.num_drones
        
        for i, drone in enumerate(self.drones):
            pos = drone.get_position()
            
            if (pos[0] < self.boundary[0] or pos[0] > self.boundary[1] or
                pos[1] < self.boundary[2] or pos[1] > self.boundary[3] or
                pos[2] < self.boundary[4] or pos[2] > self.boundary[5]):
                out_of_boundaries[i] = True
                
        return out_of_boundaries
    
    def reward_model(self):
        """
        计算奖励模型 - 基于到达目标点并避障
        
        返回:
            rewards: 所有无人机的奖励值列表
        """
        # 检查碰撞
        self.collisions = self._check_collisions()
        
        # 检查边界
        out_of_boundaries = self._check_boundaries()
        
        # 检查是否结束 (如果任何一个无人机碰撞或出界，或者达到最大步数)
        if any(self.collisions) or any(out_of_boundaries) or self.steps >= self.max_steps:
            self.done = True
        else:
            # 检查是否所有无人机都到达目标点
            all_reached = True
            for i in range(self.num_drones):
                if not self.reached_target[i]:
                    all_reached = False
                    break
                    
            if all_reached:
                self.done = True
                print("所有无人机到达目标点，任务完成！")
            else:
                self.done = False
        
        # 计算每个无人机的奖励
        rewards = []
        for i in range(self.num_drones):
            reward = self._compute_reward(self.collisions[i], out_of_boundaries[i], i)
            rewards.append(reward)
            
        return rewards
        
    def _compute_reward(self, collision, out_of_boundary, drone_index):
        """
        计算单个无人机的奖励
        
        参数:
            collision: 是否发生碰撞
            out_of_boundary: 是否超出边界
            drone_index: 无人机索引
        """
        if collision:
            return -100  # 碰撞惩罚
        elif out_of_boundary:
            return -50   # 出界惩罚
        
        # 获取无人机当前位置
        drone_pos = self.drones[drone_index].get_position()
        
        # 计算到目标点的距离 - 所有无人机使用同一个目标点
        target_pos = self.target_positions[0]  # 使用唯一的目标点
        distance = ((drone_pos[0] - target_pos[0])**2 + 
                   (drone_pos[1] - target_pos[1])**2 + 
                   (drone_pos[2] - target_pos[2])**2)**0.5
        
        # 判断是否到达目标点
        if distance < 0.5:  # 距离目标点小于0.5个单位视为到达
            self.reached_target[drone_index] = True
            # 提供一个大的稀疏奖励，鼓励无人机到达目标点
            # 如果是首次到达，给予额外的大奖励
            if not hasattr(self, 'target_reached_before'):
                self.target_reached_before = [False] * self.num_drones
                
            if not self.target_reached_before[drone_index]:
                self.target_reached_before[drone_index] = True
                return 200.0  # 首次到达目标点的巨大奖励
            else:
                return 50.0  # 后续到达的标准奖励
        else:
            self.reached_target[drone_index] = False
            
            # 添加距离反比奖励
            # 计算距离奖励：距离越近，奖励越大，使用双曲函数形式
            distance_reward = 5.0 / (1.0 + distance)
            
            # 避障奖励（实际上是惩罚）
            obstacle_reward = self._compute_obstacle_reward(drone_pos)
            
            # 返回奖励，包含距离奖励和避障惩罚
            return distance_reward + obstacle_reward
            
    def _compute_obstacle_reward(self, drone_pos):
        """
        计算避障奖励
        
        参数:
            drone_pos: 无人机位置 [x, y, z]
            
        返回:
            obstacle_reward: 避障奖励，只有惩罚没有奖励
        """
        min_distance = float('inf')
        
        # 计算无人机到所有障碍物的最小距离
        for enemy in self.enemies:
            enemy_pos = enemy.get_position()
            enemy_radius = enemy.get_radius()
            
            # 计算无人机到障碍物中心的距离
            distance = ((drone_pos[0] - enemy_pos[0])**2 + 
                       (drone_pos[1] - enemy_pos[1])**2 + 
                       (drone_pos[2] - enemy_pos[2])**2)**0.5
            
            # 无人机到障碍物表面的距离
            surface_distance = max(0.0, distance - enemy_radius)
            
            min_distance = min(min_distance, surface_distance)
        
        # 如果距离小于安全距离，则给予惩罚，否则无奖励
        safety_distance = 1.5
        if min_distance < safety_distance:
            # 接近障碍物惩罚，越接近惩罚越大
            obstacle_reward = -5.0 * (safety_distance - min_distance) / safety_distance
        else:
            # 超过安全距离则无奖励也无惩罚
            obstacle_reward = 0.0
            
        return obstacle_reward
    
    def update(self, controls=None):
        """
        更新环境状态 - 只负责坐标更新
        
        参数:
            controls: 所有无人机的控制输入列表，每个元素为 [thrust, roll_cmd, pitch_cmd, yaw_cmd]
                     如果为None，则使用默认控制
        
        返回:
            state: 更新后的环境状态
        """
        # 默认控制输入
        if controls is None:
            # 为每个无人机创建默认控制输入（悬停）
            controls = [[9.8, 0.0, 0.0, 0.0] for _ in range(self.num_drones)]
        
        # 确保控制输入数量与无人机数量一致
        if len(controls) != self.num_drones:
            raise ValueError(f"控制输入数量({len(controls)})与无人机数量({self.num_drones})不匹配")
        
        # 更新每个无人机的状态 - 只负责坐标更新
        for i, drone in enumerate(self.drones):
            # 设置无人机控制
            drone.set_control(*controls[i])
            
            # 更新无人机状态
            drone.update(self.dt)
        
        # 更新障碍物状态
        for enemy in self.enemies:
            enemy.update(self.dt)
        
        # 更新步数和时间
        self.steps += 1
        self.time += self.dt
        
        # 更新状态
        self.state = self._get_state()
        
        # 初始化done为False，将在reward_model中更新
        self.done = False
    
    def step(self, actions):
        """
        执行一步动作
        
        参数:
            actions: 所有无人机的动作列表
            
        返回:
            与update方法相同
        """
        self.update(actions)
        rewards = self.reward_model()
        done = self.done
        info = {}
        return self.state, rewards, done, info
    
    def render(self):
        """渲染当前环境状态"""
        # 懒加载渲染器
        if self.renderer is None:
            self.renderer = Renderer(self.boundary)
        
        # 准备渲染数据
        # 收集所有无人机的位置
        drone_positions = [drone.get_position() for drone in self.drones]
        
        # 准备障碍物数据
        enemies_data = []
        for enemy in self.enemies:
            enemies_data.append((enemy.get_position(), enemy.get_radius(), 'red'))
        
        # 收集所有无人机的轨迹
        trajectories = [drone.get_trajectory() for drone in self.drones]
        
        # 更新渲染，包括目标点和轨迹
        self.renderer.update(drone_positions, enemies_data, self.target_positions, trajectories)
        
        return self.renderer
    
    def close(self):
        """关闭环境，释放资源"""
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None
    
    def _set_default_targets(self):
        """设置默认的目标位置"""
        # 只设置一个共享的目标点
        target_point = [8.0, 0.0, 2.0]  # 默认目标点在X轴前方8个单位
        
        # 所有无人机共享同一个目标点
        self.target_positions = [target_point.copy()]
        
        # 设置到达标志 - 仍然每个无人机一个标志
        self.reached_target = [False] * self.num_drones
    
    def set_target_positions(self, target_positions):
        """
        设置目标位置 - 修改为只接受一个目标点
        
        参数:
            target_positions: 目标位置 [x, y, z] 或目标位置的列表
        """
        # 如果传入的是列表的列表，取第一个元素
        if isinstance(target_positions, list) and len(target_positions) > 0:
            if isinstance(target_positions[0], list):
                target_point = target_positions[0]
            else:
                # 如果只是一个坐标列表，直接使用
                target_point = target_positions
        else:
            # 默认目标点
            target_point = [8.0, 0.0, 2.0]
            
        # 存储为单元素列表，便于后续处理
        self.target_positions = [target_point.copy()]
        
        # 每个无人机的到达标志仍然保留
        self.reached_target = [False] * self.num_drones
        
    def reset(self, drone_positions=None, enemy_configs=None):
        """
        重置环境状态
        
        参数:
            drone_positions: 无人机初始位置列表，每个元素为 [x, y, z]
                           如果为None，则使用构造函数提供的位置
            enemy_configs: 障碍物配置列表，每个元素为 {'position': [x, y, z], 'radius': r, 'velocity': [vx, vy, vz]}
                         如果为None，则使用构造函数提供的配置
                         
        返回:
            state: 重置后的环境状态
        """
        # 更新无人机和障碍物配置（如果提供）
        if drone_positions is not None:
            self.drone_positions = drone_positions
            self.num_drones = len(drone_positions)
        
        if enemy_configs is not None:
            self.enemy_configs = enemy_configs
            self.num_enemies = len(enemy_configs)

        
        # 重置无人机
        # 确保清空现有无人机列表
        self.drones = []
        self._create_drones()

        
        # 重置障碍物
        self.enemies = []
        self._create_enemies()
        
        # 重置目标位置
        self._set_default_targets()
        
        # 重置运行时变量
        self.done = False
        self.time = 0.0
        self.steps = 0
        self.collisions = [False] * self.num_drones
        
        # 重置首次到达标志
        self.target_reached_before = [False] * self.num_drones
        
        # 更新状态
        self.state = self._get_state()
        
        return self.state
    
    def run_episode(self, policy_func=None, max_steps=1000, render=True):
        """
        运行一个完整的回合
        
        参数:
            policy_func: 策略函数，接受状态返回动作列表
            max_steps: 最大步数
            render: 是否渲染
            
        返回:
            total_rewards: 所有无人机的累积奖励列表
            steps: 实际步数
        """
        # 重置环境
        state = self.reset()
        self.max_steps = max_steps
        total_rewards = [0] * self.num_drones
        
        # 如果需要渲染，创建渲染器但不调用show()
        if render:
            # 在首次渲染时初始化图形
            renderer = self.render()
            # 使用非阻塞模式显示图形
            plt.ion()  # 打开交互模式
            plt.show(block=False)  # 非阻塞显示
        
        # 运行回合
        for step in range(max_steps):
            # 根据策略获取动作
            if policy_func is not None:
                actions = policy_func(self.state)
            else:
                # 默认动作：所有无人机悬停
                actions = [[9.8, 0.0, 0.0, 0.0] for _ in range(self.num_drones)]
            
            # 执行动作
            state, rewards, done, _ = self.step(actions)
            
            # 累积奖励
            for i in range(self.num_drones):
                total_rewards[i] += rewards[i]
            
            # 渲染，包括轨迹和目标点
            if render:
                self.render()
                plt.pause(self.dt)  # 使用plt.pause代替time.sleep，同时刷新图形
            
            # 检查是否结束
            if done:
                break
        
        # 关闭交互模式，但保持图形窗口打开
        if render:
            plt.ioff()
        
        return total_rewards, self.steps


# 示例使用
if __name__ == "__main__":
    # 定义无人机初始位置
    drone_positions = [
        [0, 0, 2],       # 第一架无人机位置
        [0, 3, 2],       # 第二架无人机位置
        [0, -3, 2]       # 第三架无人机位置
    ]
    
    # 定义障碍物配置 - 将障碍物放在远离直线路径的位置
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
    
    # 定义目标点
    target_points = [
        [8, 0, 2],    # 第一架无人机目标点：沿X轴正方向直线飞行
        [8, 3, 2],    # 第二架无人机目标点：沿X轴正方向平行飞行
        [8, -3, 2]    # 第三架无人机目标点：沿X轴正方向平行飞行
    ]
    
    # 设置环境中的目标点
    environment.set_target_positions(target_points)
    
    # 定义一个简单的策略：让无人机沿着直线轨迹飞行
    def straight_line_policy(state):
        # 获取当前时间
        t = environment.time
        
        # 为每个无人机创建控制输入
        actions = []
        
        for i, drone_state in enumerate(state['drones']):
            # 获取目标点
            target_x, target_y, target_z = environment.target_positions[i]
            
            # 当前位置
            current_x, current_y, current_z = drone_state['position']
            
            # 计算到目标的方向向量
            dx = target_x - current_x
            dy = target_y - current_y
            dz = target_z - current_z
            
            # 计算距离
            distance = (dx**2 + dy**2 + dz**2)**0.5
            
            # 根据距离调整推力和姿态控制
            if distance > 0.1:  # 如果距离目标还有一段距离
                # 计算姿态控制（简化模型）
                # 这里使用比例控制，让无人机朝向目标
                roll_cmd = -0.2 * dy  # y方向偏差影响横滚
                pitch_cmd = 0.2 * dx  # x方向偏差影响俯仰
                yaw_cmd = 0.0  # 保持偏航不变
                
                # 使用固定推力
                thrust = 9.8  # 悬停推力
                
                # 如果需要上升或下降，调整推力
                if dz > 0.1:
                    thrust += 0.5  # 需要上升
                elif dz < -0.1:
                    thrust -= 0.5  # 需要下降
            else:
                # 已经接近目标，悬停
                thrust = 9.8
                roll_cmd = 0.0
                pitch_cmd = 0.0
                yaw_cmd = 0.0
            
            # 添加到动作列表
            actions.append([thrust, roll_cmd, pitch_cmd, yaw_cmd])
        
        return actions
    
    # 运行一个回合，使用直线飞行策略
    rewards, steps = environment.run_episode(policy_func=straight_line_policy, max_steps=500, render=True)
    
    print(f"回合结束，总步数: {steps}")
    print(f"累积奖励: {rewards}")
    
    # 保持图形窗口打开直到用户关闭
    plt.show()