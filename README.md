# 多无人机避障突防系统

这个项目实现了一个基于强化学习的多无人机协同避障突防系统。系统使用PPO (Proximal Policy Optimization) 算法训练多个无人机在复杂环境中协同避开障碍物，到达指定目标点。

## 项目特点

- 多无人机协同控制：支持多架无人机的同时控制与协同训练
- 动态障碍物避障：能够识别并绕过环境中的静态及动态障碍物
- 自适应路径规划：无人机能够自主规划从起点到目标点的路径
- 实时可视化：使用matplotlib进行实时渲染，直观展示无人机飞行轨迹
- 高效训练：基于PPO算法的高效训练框架，支持GPU加速

## 安装与环境配置

### 依赖库
```bash
pip install numpy torch matplotlib tqdm
```

### 文件结构
```
突防_final/
  ├── algorithm/          # 算法实现
  │   ├── network.py      # 神经网络模型
  │   └── ppo.py          # PPO算法实现
  ├── data_tool/          # 数据分析工具
  ├── env/                # 环境模拟
  │   └── tufang_env.py   # 突防环境定义
  ├── model/              # 物理模型
  │   ├── aircraft.py     # 无人机模型
  │   └── enemy.py        # 障碍物模型
  ├── render/             # 渲染模块
  │   └── matplotlib_render.py  # 图形渲染
  ├── models/             # 保存的模型
  ├── main.py             # 简单仿真示例
  ├── train_ppo.py        # PPO训练脚本
  └── runner.py           # 运行脚本
```

## 使用方法

### 1. 运行简单仿真

运行`main.py`可以查看无人机在默认环境中的基本行为：

```bash
python main.py
```

这将使用预设的简单策略（悬停控制）来控制无人机。

### 2. 训练模型

使用PPO训练多无人机系统：

```bash
python train_ppo.py --num_drones 3 --num_episodes 10000 --render
```

主要参数说明：
- `--num_drones`：无人机数量（默认3）
- `--num_episodes`：训练回合数（默认10000）
- `--max_steps`：每个回合最大步数（默认500）
- `--render`：是否渲染环境
- `--save_model`：是否保存训练模型
- `--model_dir`：模型保存目录（默认'./models'）

更多参数请查看`train_ppo.py`文件中的`parse_args`函数。

### 3. 评估训练模型

训练结束后，可以评估训练好的模型：

```bash
# 取消train_ppo.py最后的注释来运行评估
# evaluate("./models/ppo_model_final.pt")
```

或修改`train_ppo.py`中的main部分，直接调用evaluate函数。

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

## 训练结果可视化

训练过程中会自动生成以下可视化结果，保存在`models`目录下：

1. 奖励曲线（reward_curve.png）：展示训练过程中平均奖励的变化
2. 损失函数曲线（loss_curves.png）：显示Actor、Critic、Entropy和总损失的变化
3. 探索衰减曲线（exploration_decay.png）：展示探索噪声的衰减过程

此外，训练过程中还会实时显示无人机动作的均值和方差曲线。 