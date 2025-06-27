# TensorBoard 数据分析工具

本工具集提供了使用 TensorBoard 进行训练数据可视化和分析的功能。

## 安装依赖

首先需要安装必要的依赖：

```bash
pip install torch tensorboard pillow matplotlib numpy
```

## 工具介绍

该工具集包含以下几个主要组件：

### 1. TensorboardLogger

一个基础的 TensorBoard 记录器，提供了各种数据记录方法：

- 记录标量值
- 记录多组相关标量值
- 记录直方图数据
- 记录图像和图表
- 记录无人机动作分布
- 记录神经网络权重

### 2. TensorboardTrainingMonitor

一个训练过程监控器，可以集成到训练循环中，自动记录：

- 训练回合信息
- 每步动作、奖励和状态
- 每个无人机的动作分布统计
- 训练过程更新信息
- 训练总结

### 3. with_tensorboard 装饰器

一个用于快速将 TensorBoard 支持添加到现有训练函数的装饰器。

### 4. analyze_training.py

一个独立的工具脚本，可以分析训练日志和模型文件，生成可视化报告。

## 使用方法

### 基本使用

导入 TensorboardLogger 并创建一个日志记录器：

```python
from data_tool.tensorboard_logger import TensorboardLogger

# 创建记录器
logger = TensorboardLogger('./logs/my_experiment')

# 记录标量值
logger.log_scalar('train/loss', loss_value, step)
```

### 监控训练过程

使用 TensorboardTrainingMonitor 监控整个训练过程：

```python
from data_tool.tensorboard_integration import TensorboardTrainingMonitor

# 创建监控器
monitor = TensorboardTrainingMonitor('./logs/my_experiment')

# 开始回合
monitor.start_episode(episode)

# 记录训练步骤
monitor.log_step(state_dict, action, rewards, next_state_dict, done)

# 记录更新信息
monitor.log_update(update_info)

# 结束回合
monitor.end_episode(episode_rewards, episode_steps)

# 关闭监控器
monitor.close()
```

### 装饰器用法

使用装饰器快速添加 TensorBoard 支持：

```python
from data_tool.tensorboard_integration import with_tensorboard

@with_tensorboard(log_dir='./logs', exp_name='my_experiment')
def train(args, tb_monitor=None):
    # 训练代码
    # 可以使用 tb_monitor 记录数据
    pass
```

### 分析已有训练数据

使用分析工具分析之前的训练数据：

```bash
python -m data_tool.analyze_training --model_dir ./models --log_dir ./logs/analysis --mode all
```

## 在train_ppo.py中集成

要将TensorBoard记录功能集成到现有的`train_ppo.py`训练脚本中，可以：

1. 导入监控器：

```python
from data_tool.tensorboard_integration import get_tensorboard_monitor
```

2. 在训练函数开始时创建监控器：

```python
# 创建TensorBoard监控器
tb_monitor = get_tensorboard_monitor('./logs/tensorboard', f'ppo_drones{args.num_drones}')
```

3. 在训练循环中记录数据：

```python
# 开始回合
tb_monitor.start_episode(episode)

# 记录步骤
tb_monitor.log_step(state_dict, action, rewards, next_state_dict, done)

# 记录策略更新
tb_monitor.log_update(train_info)

# 结束回合
tb_monitor.end_episode(episode_rewards, step+1)
```

4. 最后关闭监控器：

```python
# 关闭监控器
tb_monitor.close()
```

## 查看结果

使用以下命令启动TensorBoard服务器：

```bash
tensorboard --logdir=./logs
```

然后在浏览器中访问 http://localhost:6006 查看训练数据可视化。

## 示例

运行示例脚本了解如何使用这些工具：

```bash
python -m data_tool.example_usage
``` 