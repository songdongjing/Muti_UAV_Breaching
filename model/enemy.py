class enemy:
    ## 敌人类
    def __init__(self, position=[0, 0, 0], radius=1.0, velocity=[0, 0, 0]):
        """
        初始化移动的球形障碍物
        
        参数:
            position: 球心三维位置坐标 [x, y, z]
            radius: 球体半径
            velocity: 三维速度向量 [vx, vy, vz]
        """
        self.position = position  # 球心位置 [x, y, z]
        self.radius = radius  # 球体半径
        self.velocity = velocity  # 速度 [vx, vy, vz]
        
        # 运动边界
        self.boundary = {
            'x_min': -10, 'x_max': 10,
            'y_min': -10, 'y_max': 10,
            'z_min': 0, 'z_max': 10
        }
        
        # 外观属性
        self.color = [255, 0, 0]  # 默认为红色 RGB
        
    def update(self, dt=0.01):
        """
        更新障碍物状态（位置更新）
        
        参数:
            dt: 时间步长，默认0.01秒
        """
        # 位置更新
        for i in range(3):
            self.position[i] += self.velocity[i] * dt
        
        # 边界检查和反弹
        self._check_boundary()
        
    def _check_boundary(self):
        """检查边界并处理碰撞反弹"""
        # X轴边界检查
        if self.position[0] - self.radius < self.boundary['x_min']:
            self.position[0] = self.boundary['x_min'] + self.radius
            self.velocity[0] = -self.velocity[0]  # 反向
        elif self.position[0] + self.radius > self.boundary['x_max']:
            self.position[0] = self.boundary['x_max'] - self.radius
            self.velocity[0] = -self.velocity[0]  # 反向
            
        # Y轴边界检查
        if self.position[1] - self.radius < self.boundary['y_min']:
            self.position[1] = self.boundary['y_min'] + self.radius
            self.velocity[1] = -self.velocity[1]  # 反向
        elif self.position[1] + self.radius > self.boundary['y_max']:
            self.position[1] = self.boundary['y_max'] - self.radius
            self.velocity[1] = -self.velocity[1]  # 反向
            
        # Z轴边界检查
        if self.position[2] - self.radius < self.boundary['z_min']:
            self.position[2] = self.boundary['z_min'] + self.radius
            self.velocity[2] = -self.velocity[2]  # 反向
        elif self.position[2] + self.radius > self.boundary['z_max']:
            self.position[2] = self.boundary['z_max'] - self.radius
            self.velocity[2] = -self.velocity[2]  # 反向
    
    def set_velocity(self, velocity):
        """设置障碍物速度"""
        self.velocity = velocity
        
    def set_position(self, position):
        """设置障碍物位置"""
        self.position = position
        
    def set_boundary(self, x_min, x_max, y_min, y_max, z_min, z_max):
        """设置运动边界"""
        self.boundary = {
            'x_min': x_min, 'x_max': x_max,
            'y_min': y_min, 'y_max': y_max,
            'z_min': z_min, 'z_max': z_max
        }
        
    def get_position(self):
        """返回障碍物位置"""
        return self.position
        
    def get_radius(self):
        """返回障碍物半径"""
        return self.radius
        
    def check_collision(self, point):
        """
        检查点是否与障碍物发生碰撞
        
        参数:
            point: 要检查的点 [x, y, z]
        返回:
            bool: 是否碰撞
        """
        # 计算点到球心的距离
        distance_squared = sum((point[i] - self.position[i])**2 for i in range(3))
        
        # 如果距离小于等于半径，则发生碰撞
        return distance_squared <= self.radius**2