import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import time
import matplotlib

# 设置matplotlib支持中文显示
matplotlib.rcParams['font.family'] = ['SimHei']  # 使用黑体
matplotlib.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

class Renderer:
    """三维场景渲染器，用于渲染无人机和障碍物"""
    
    def __init__(self, boundary=None):
        """
        初始化渲染器
        
        参数:
            boundary: 渲染边界，格式为 [x_min, x_max, y_min, y_max, z_min, z_max]
                     默认为 [-10, 10, -10, 10, 0, 10]
        """
        # 设置渲染边界
        if boundary is None:
            self.boundary = [-10, 10, -10, 10, 0, 10]
        else:
            self.boundary = boundary
            
        # 创建图形和轴
        self.fig = plt.figure(figsize=(10, 8))
        self.ax = self.fig.add_subplot(111, projection='3d')
        
        # 设置轴标签
        self.ax.set_xlabel('X轴')
        self.ax.set_ylabel('Y轴')
        self.ax.set_zlabel('Z轴')
        
        # 设置坐标轴范围
        self.ax.set_xlim(self.boundary[0], self.boundary[1])
        self.ax.set_ylim(self.boundary[2], self.boundary[3])
        self.ax.set_zlim(self.boundary[4], self.boundary[5])
        
        # 设置标题
        self.ax.set_title('四旋翼无人机与障碍物三维渲染')
        
        # 存储渲染对象
        self.aircraft_plots = []
        self.enemy_plots = []
        self.target_plots = []
        self.trajectory_plots = []
        
        # 视角设置
        self.ax.view_init(elev=30, azim=45)  # 设置初始视角
        
    def render_aircrafts(self, positions):
        """
        渲染多架无人机
        
        参数:
            positions: 无人机位置列表，每个元素为 [x, y, z]
        """
        # 获取位置数据的坐标
        x_coords = [pos[0] for pos in positions]
        y_coords = [pos[1] for pos in positions]
        z_coords = [pos[2] for pos in positions]
        
        # 为不同的无人机定义不同的颜色
        colors = ['blue', 'green', 'orange', 'purple', 'cyan', 'magenta', 'yellow']
        
        # 清除之前的无人机绘图
        self.aircraft_plots = []
        
        # 为每架无人机单独绘制，使用不同颜色和标签
        for i, (x, y, z) in enumerate(zip(x_coords, y_coords, z_coords)):
            color = colors[i % len(colors)]  # 循环使用颜色列表
            plot = self.ax.scatter(
                [x], [y], [z],
                color=color, s=100, marker='o', label=f'无人机{i+1}'
            )
            self.aircraft_plots.append(plot)
            
            # 添加标签
            self.ax.text(x, y, z + 0.5, f'{i+1}', color=color, fontsize=12)
        
        # 添加图例
        self.ax.legend(loc='upper right')
    
    def render_targets(self, targets):
        """
        渲染目标点
        
        参数:
            targets: 目标点位置列表，每个元素为 [x, y, z]
        """
        # 清除之前的目标点绘图
        self.target_plots = []
        
        # 如果目标点列表为空，直接返回
        if not targets:
            return
        
        # 检查是否只有一个共享目标点
        is_shared_target = len(targets) == 1
        
        # 为不同的无人机定义不同的颜色（与无人机颜色对应）
        colors = ['blue', 'green', 'orange', 'purple', 'cyan', 'magenta', 'yellow']
        
        # 渲染每个目标点
        for i, target in enumerate(targets):
            x, y, z = target
            
            # 共享目标点使用红色且更大的五角星标记
            if is_shared_target:
                color = 'red'  # 共享目标点使用红色
                size = 200     # 更大的标记尺寸
                label = '共享目标'
            else:
                color = colors[i % len(colors)]  # 使用与对应无人机相同的颜色
                size = 150
                label = f'目标{i+1}'
            
            # 渲染目标点为五角星
            plot = self.ax.scatter(
                [x], [y], [z],
                color=color, s=size, marker='*', alpha=0.8, 
                label=label
            )
            self.target_plots.append(plot)
            
            # 添加标签
            self.ax.text(x, y, z + 0.5, label, color=color, fontsize=10)
    
    def render_trajectories(self, trajectories):
        """
        渲染无人机轨迹
        
        参数:
            trajectories: 轨迹列表，每个元素为一个无人机的轨迹点列表
        """
        # 清除之前的轨迹绘图
        self.trajectory_plots = []
        
        # 为不同的无人机定义不同的颜色（与无人机颜色对应）
        colors = ['blue', 'green', 'orange', 'purple', 'cyan', 'magenta', 'yellow']
        
        # 渲染每个无人机的轨迹
        for i, trajectory in enumerate(trajectories):
            # 如果轨迹为空，跳过
            if not trajectory:
                continue
                
            # 提取轨迹点的坐标
            x_coords = [point[0] for point in trajectory]
            y_coords = [point[1] for point in trajectory]
            z_coords = [point[2] for point in trajectory]
            
            color = colors[i % len(colors)]  # 使用与对应无人机相同的颜色
            
            # 渲染轨迹为线段
            plot = self.ax.plot(
                x_coords, y_coords, z_coords,
                color=color, linestyle='-', linewidth=1, alpha=0.5
            )
            self.trajectory_plots.append(plot)
    
    def render_enemy(self, enemies):
        """
        渲染障碍物
        
        参数:
            enemies: 障碍物列表，每个元素为 (position, radius, color) 元组
                    position 是 [x, y, z] 坐标，radius 是球体半径，color 是颜色
                    color 若未提供，则默认为'red'
        """
        # 不再尝试移除旧的，直接渲染新的
        self.enemy_plots = []
        
        # 渲染每个障碍物
        for enemy_data in enemies:
            # 支持旧格式 (position, radius) 和新格式 (position, radius, color)
            if len(enemy_data) >= 3:
                position, radius, color = enemy_data
            else:
                position, radius = enemy_data
                color = 'red'  # 默认颜色
            
            # 创建球体网格
            u = np.linspace(0, 2 * np.pi, 20)
            v = np.linspace(0, np.pi, 20)
            x = position[0] + radius * np.outer(np.cos(u), np.sin(v))
            y = position[1] + radius * np.outer(np.sin(u), np.sin(v))
            z = position[2] + radius * np.outer(np.ones(np.size(u)), np.cos(v))
            
            # 渲染障碍物为指定颜色的球体
            plot = self.ax.plot_surface(
                x, y, z, color=color, alpha=0.5, linewidth=0
            )
            self.enemy_plots.append(plot)
    
    def update(self, aircraft_positions, enemies_data, target_positions=None, trajectories=None):
        """
        更新渲染
        
        参数:
            aircraft_positions: 无人机位置列表，每个元素为 [x, y, z]
            enemies_data: 障碍物数据列表，每个元素为 (position, radius, color) 元组
            target_positions: 目标点位置列表，每个元素为 [x, y, z]
            trajectories: 轨迹列表，每个元素为一个无人机的轨迹点列表
        """
        # 清除图形
        self.ax.clear()
        
        # 重新设置坐标轴
        self.ax.set_xlabel('X轴')
        self.ax.set_ylabel('Y轴')
        self.ax.set_zlabel('Z轴')
        self.ax.set_xlim(self.boundary[0], self.boundary[1])
        self.ax.set_ylim(self.boundary[2], self.boundary[3])
        self.ax.set_zlim(self.boundary[4], self.boundary[5])
        self.ax.set_title('四旋翼无人机与障碍物三维渲染')
        
        # 渲染无人机
        self.render_aircrafts(aircraft_positions)
        
        # 渲染障碍物
        self.render_enemy(enemies_data)
        
        # 渲染目标点（如果提供）
        if target_positions is not None:
            self.render_targets(target_positions)
            
        # 渲染轨迹（如果提供）
        if trajectories is not None:
            self.render_trajectories(trajectories)
        
        # 添加网格
        self.ax.grid(True)
        
    def show(self):
        """显示渲染窗口"""
        plt.show()
        
    def close(self):
        """关闭渲染窗口"""
        plt.close(self.fig)
        
    def animate(self, update_func, frames=100, interval=50):
        """
        创建动画
        
        参数:
            update_func: 更新函数，接受一个帧号参数，返回 (aircraft_positions, enemies_data, target_positions, trajectories)
            frames: 总帧数
            interval: 帧间隔，单位为毫秒
        """
        def animation_update(frame):
            result = update_func(frame)
            # 检查返回的结果元组长度
            if len(result) == 2:
                # 兼容旧版本的update_func，只返回无人机位置和障碍物数据
                aircraft_positions, enemies_data = result
                self.update(aircraft_positions, enemies_data)
            elif len(result) == 4:
                # 新版本的update_func，返回无人机位置、障碍物数据、目标点和轨迹
                aircraft_positions, enemies_data, target_positions, trajectories = result
                self.update(aircraft_positions, enemies_data, target_positions, trajectories)
            # 不再返回self.aircraft_plot，这样不会尝试使用blit
            return []
        
        # 创建动画，禁用blit
        ani = FuncAnimation(
            self.fig, animation_update, frames=frames,
            interval=interval, blit=False
        )
        return ani


# 简单的使用示例
if __name__ == "__main__":
    from model.aircraft import aircraft
    from model.enemy import enemy
    import time
    
    try:
        # 创建多架无人机
        drones = [
            aircraft(position=[0, 0, 0]),
            aircraft(position=[2, 0, 0]),
            aircraft(position=[-2, 0, 0])
        ]
        
        # 创建障碍物
        obstacles = [
            enemy(position=[5, 5, 5], radius=1.0, velocity=[0.1, 0.1, 0.1]),
            enemy(position=[-5, -5, 5], radius=1.5, velocity=[-0.1, -0.1, 0.1])
        ]
        
        # 创建渲染器
        renderer = Renderer()
        
        # 简单的更新函数
        def update_simulation(frame):
            # 更新无人机位置
            t = frame * 0.1
            
            # 所有无人机沿不同轨迹飞行
            drone_positions = [
                [5 * np.cos(t), 5 * np.sin(t), 2 + np.sin(t)],
                [4 * np.cos(t + np.pi/3), 4 * np.sin(t + np.pi/3), 3 + np.sin(t)],
                [3 * np.cos(t + 2*np.pi/3), 3 * np.sin(t + 2*np.pi/3), 4 + np.sin(t)]
            ]
            
            # 更新无人机位置
            for i, drone in enumerate(drones):
                drone.position = drone_positions[i]
            
            # 更新障碍物位置
            for obs in obstacles:
                obs.update(0.1)
            
            # 返回更新后的位置数据
            enemies_data = [(obs.position, obs.radius, 'red') for obs in obstacles]
            
            return drone_positions, enemies_data
        
        # 创建动画
        ani = renderer.animate(update_simulation, frames=200, interval=50)
        
        # 显示渲染窗口
        renderer.show()
    except Exception as e:
        print(f"发生错误: {e}")
        import traceback
        traceback.print_exc()
