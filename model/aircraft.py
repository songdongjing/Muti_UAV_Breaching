class aircraft:
    ## 飞机类
    def __init__(self, position=[0, 0, 0], velocity=[0, 0, 0], attitude=[0, 0, 0]):
        """
        初始化四旋翼无人机
        
        参数:
            position: 三维位置坐标 [x, y, z]
            velocity: 三维速度向量 [vx, vy, vz]
            attitude: 姿态角 [roll, pitch, yaw]
        """
        # 深拷贝输入参数，避免引用问题
        self.position = position.copy() if isinstance(position, list) else list(position)  # 位置 [x, y, z]
        self.velocity = velocity.copy() if isinstance(velocity, list) else list(velocity)  # 速度 [vx, vy, vz]
        self.attitude = attitude.copy() if isinstance(attitude, list) else list(attitude)  # 姿态角 [roll, pitch, yaw]
        
        self.acceleration = [0, 0, 0]  # 加速度 [ax, ay, az]
        self.angular_velocity = [0, 0, 0]  # 角速度 [p, q, r]
        
        # 无人机物理参数
        self.mass = 1.0  # 质量 kg
        self.size = [0.5, 0.5, 0.2]  # 尺寸 [长, 宽, 高] m
        
        # 轨迹记录
        self.trajectory = [self.position.copy()]  # 初始位置也记录在轨迹中
        self.max_trajectory_length = 1000  # 最大轨迹点数，防止内存占用过大
    
    def update(self, dt=0.01):
        """
        更新无人机状态（运动学更新）
        
        参数:
            dt: 时间步长，默认0.01秒
        """
        # 位置更新 (p = p + v*dt + 0.5*a*dt^2)
        for i in range(3):
            self.position[i] += self.velocity[i] * dt + 0.5 * self.acceleration[i] * dt * dt
        
        # 速度更新 (v = v + a*dt)
        for i in range(3):
            self.velocity[i] += self.acceleration[i] * dt
            
        # 姿态更新 (简化版，直接积分角速度)
        for i in range(3):
            self.attitude[i] += self.angular_velocity[i] * dt
            
        # 标准化偏航角到 [0, 2π]
        self.attitude[2] = self.attitude[2] % (2 * 3.14159)
        
        # 记录轨迹
        self.add_trajectory_point(self.position.copy())
    
    def add_trajectory_point(self, point):
        """添加轨迹点，并限制轨迹长度"""
        self.trajectory.append(point)
        # 如果轨迹点数超过最大限制，移除最早的点
        if len(self.trajectory) > self.max_trajectory_length:
            self.trajectory.pop(0)
    
    def get_trajectory(self):
        """获取无人机轨迹"""
        return self.trajectory
    
    def clear_trajectory(self):
        """清空轨迹记录"""
        self.trajectory = [self.position.copy()]
    
    def set_control(self, thrust, roll_cmd, pitch_cmd, yaw_cmd):
        """
        设置控制输入
        
        参数:
            thrust: 总推力
            roll_cmd: 横滚控制
            pitch_cmd: 俯仰控制
            yaw_cmd: 偏航控制
        """
        # 简化的控制模型
        g = 9.8  # 重力加速度
        
        # 加速度计算 (简化模型)
        self.acceleration[0] = thrust * (self.attitude[1])  # pitch影响x方向加速度
        self.acceleration[1] = thrust * (-self.attitude[0])  # roll影响y方向加速度
        self.acceleration[2] = thrust - g  # 垂直方向加速度
        
        # 角速度计算 (简化模型)
        self.angular_velocity[0] = roll_cmd  # 横滚角速度
        self.angular_velocity[1] = pitch_cmd  # 俯仰角速度
        self.angular_velocity[2] = yaw_cmd  # 偏航角速度
    
    def get_position(self):
        """返回无人机当前位置"""
        return self.position
        
    def get_velocity(self):
        """返回无人机当前速度"""
        return self.velocity
        
    def get_attitude(self):
        """返回无人机当前姿态"""
        return self.attitude
