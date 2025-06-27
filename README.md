# 突防系统

## 四旋翼无人机运动学方程

本系统使用简化的四旋翼无人机运动学模型。以下是无人机的主要运动学方程：

### 状态变量

四旋翼无人机的状态由以下变量表示：

- 位置：$\mathbf{p} = [x, y, z]^T$
- 速度：$\mathbf{v} = [v_x, v_y, v_z]^T$
- 姿态角：$\mathbf{\eta} = [\phi, \theta, \psi]^T$（横滚角、俯仰角、偏航角）
- 加速度：$\mathbf{a} = [a_x, a_y, a_z]^T$
- 角速度：$\mathbf{\omega} = [p, q, r]^T$

### 位置更新方程

位置更新采用二阶欧拉积分方法：

$$\mathbf{p}(t+\Delta t) = \mathbf{p}(t) + \mathbf{v}(t) \cdot \Delta t + \frac{1}{2} \mathbf{a}(t) \cdot \Delta t^2$$

具体对每个坐标轴：

$$x(t+\Delta t) = x(t) + v_x(t) \cdot \Delta t + \frac{1}{2} a_x(t) \cdot \Delta t^2$$

$$y(t+\Delta t) = y(t) + v_y(t) \cdot \Delta t + \frac{1}{2} a_y(t) \cdot \Delta t^2$$

$$z(t+\Delta t) = z(t) + v_z(t) \cdot \Delta t + \frac{1}{2} a_z(t) \cdot \Delta t^2$$

### 速度更新方程

速度更新采用一阶欧拉积分：

$$\mathbf{v}(t+\Delta t) = \mathbf{v}(t) + \mathbf{a}(t) \cdot \Delta t$$

具体对每个坐标轴：

$$v_x(t+\Delta t) = v_x(t) + a_x(t) \cdot \Delta t$$

$$v_y(t+\Delta t) = v_y(t) + a_y(t) \cdot \Delta t$$

$$v_z(t+\Delta t) = v_z(t) + a_z(t) \cdot \Delta t$$

### 姿态更新方程

姿态角更新采用简化的一阶欧拉积分（对小角度变化有效）：

$$\mathbf{\eta}(t+\Delta t) = \mathbf{\eta}(t) + \mathbf{\omega}(t) \cdot \Delta t$$

具体对每个角度：

$$\phi(t+\Delta t) = \phi(t) + p(t) \cdot \Delta t$$

$$\theta(t+\Delta t) = \theta(t) + q(t) \cdot \Delta t$$

$$\psi(t+\Delta t) = \psi(t) + r(t) \cdot \Delta t \mod 2\pi$$

### 加速度计算

在简化模型中，加速度与姿态角和总推力的关系为：

$$a_x = T \cdot \theta$$

$$a_y = T \cdot (-\phi)$$

$$a_z = T - g$$

其中：
- $T$ 是归一化总推力
- $g$ 是重力加速度 (9.8 m/s²)
- $\phi$ 是横滚角
- $\theta$ 是俯仰角

### 控制输入

无人机接受以下控制输入：

- 总推力 $T$
- 横滚控制 $u_{\phi}$
- 俯仰控制 $u_{\theta}$
- 偏航控制 $u_{\psi}$

这些控制输入直接影响角速度和垂直加速度：

$$p = u_{\phi}$$

$$q = u_{\theta}$$

$$r = u_{\psi}$$

## 注意事项

这个模型是一个简化的四旋翼无人机运动学模型：

1. 忽略了空气阻力和其他气动效应
2. 使用小角度近似，假设姿态角较小
3. 没有考虑姿态角之间的耦合效应
4. 控制输入直接映射到角速度，忽略了实际系统中的动力学延迟

实际应用中，更精确的模型可能需要考虑上述因素，尤其是在高速飞行或大角度机动时。 