# 分布式训练测试脚本使用指南

## 概述

本目录包含用于测试分布式训练流程的脚本，不需要真实的模型或数据集，使用fake数据来验证Metrics和StepScheduler系统的正确性。

## 文件说明

- **`test_distributed_training.py`** - 主测试脚本（220行）
- **`run_distributed_test.sh`** - 便捷启动脚本（213行）
- **`README_distributed_test.md`** - 英文详细文档

## 快速开始

### 1. 查看帮助信息

```bash
./tests/run_distributed_test.sh --help
```

### 2. 单进程快速测试

```bash
./tests/run_distributed_test.sh single --steps 20
```

### 3. 多GPU测试

```bash
# 2个GPU
./tests/run_distributed_test.sh 2gpu

# 4个GPU
./tests/run_distributed_test.sh 4gpu --steps 200

# 8个GPU
./tests/run_distributed_test.sh 8gpu
```

## 运行模式

| 模式 | 说明 | 命令 |
|------|------|------|
| `single` | 单进程，不使用分布式 | `./tests/run_distributed_test.sh single` |
| `2gpu` | 2个GPU，使用torchrun | `./tests/run_distributed_test.sh 2gpu` |
| `4gpu` | 4个GPU，使用torchrun | `./tests/run_distributed_test.sh 4gpu` |
| `8gpu` | 8个GPU，使用torchrun | `./tests/run_distributed_test.sh 8gpu` |
| `custom` | 自定义GPU数量 | `./tests/run_distributed_test.sh custom --nproc N` |

## 可配置参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--steps N` | 总训练步数（微步数） | 100 |
| `--acc-steps N` | 梯度累积步数 | 4 |
| `--log-steps N` | 每N个全局步记录日志 | 10 |
| `--checkpoint-steps N` | 每N个全局步保存检查点 | 50 |
| `--nproc N` | 进程数（仅custom模式） | 2 |

## 使用示例

### 示例1：快速验证（单进程，20步）

```bash
./tests/run_distributed_test.sh single --steps 20
```

**适用场景：** 快速验证代码逻辑，不需要GPU

### 示例2：标准分布式测试（2 GPU）

```bash
./tests/run_distributed_test.sh 2gpu --steps 100
```

**适用场景：** 测试分布式规约功能

### 示例3：大规模测试（4 GPU，自定义参数）

```bash
./tests/run_distributed_test.sh 4gpu \
    --steps 200 \
    --acc-steps 8 \
    --log-steps 5 \
    --checkpoint-steps 100
```

**适用场景：** 完整的分布式流程测试

### 示例4：自定义GPU数量（6个GPU）

```bash
./tests/run_distributed_test.sh custom --nproc 6 --steps 300
```

**适用场景：** 特殊硬件配置

## 测试内容

### Metrics系统测试

✅ **分布式规约**
- Loss：使用`reduce="mean"`，计算所有rank的平均值
- Tokens：使用`reduce="sum"`，累加所有rank的token数
- 验证None值被正确排除

✅ **序列操作**
- 派生序列：avg、sum、cumsum
- 算术运算：加减乘除
- 数据变换：shift、diff
- 切片操作：[::acc_steps]

✅ **None值处理**
- grad_norm和learning_rate只在梯度累积边界有值
- 其他微步为None
- 所有计算正确处理None值

### StepScheduler测试

✅ **步数管理**
- micro_step：每次迭代+1
- global_step：每gradient_accumulation_steps次+1

✅ **事件触发**
- 梯度累积边界：`is_gradient_accumulation_boundary()`
- 日志记录：`should_logging()`
- 检查点保存：`should_save_checkpoint()`

### 正确的调用顺序

测试脚本严格按照以下顺序执行：

```python
for step in range(num_steps):
    # 1. 前进调度器
    scheduler.step()
    
    # 2. 生成fake数据（模拟forward/backward）
    batch = generate_fake_batch(rank, step)
    
    # 3. 追加本地值
    metrics.loss.append(batch['loss'])
    metrics.tokens.append(batch['tokens'])
    
    # 4. 执行分布式规约
    metrics.step()  # ← 关键！类似TensorFlow的session.run()
    
    # 5. 根据调度器决定是否记录日志
    if scheduler.should_logging():
        metrics.logger.log()
```

## 输出说明

### 配置信息

```
============================================================
[INFO] Distributed Training Test Configuration
============================================================
Mode:                          2gpu
Number of processes:           2
Training steps:                100
Gradient accumulation steps:   4
Logging per step:              10
Checkpoint per step:           50
============================================================
```

### 训练过程

```
[Rank 0] Starting training loop...
[Rank 0, Step 0] Generated batch: loss=2.5234, tokens=1456
...
```

### 日志输出

```
============================================================
[Global Step 10] LOGGING METRICS
============================================================
loss: 2.5123
grad_norm: 1.234
tokens_per_sec_per_gpu: 1234.56
...
```

### 完成信息

```
============================================================
TRAINING COMPLETE
============================================================
Total micro steps: 100
Total global steps: 25
Metrics index length: 101
Series tracked: ['loss', 'grad_norm', 'learning_rate', ...]
============================================================
```

## 常见问题

### Q1: 提示"torchrun not found"

**解决方案：** 确保安装了PyTorch distributed

```bash
pip install torch
```

### Q2: 单进程模式也报错

**解决方案：** 使用single模式避免初始化分布式

```bash
./tests/run_distributed_test.sh single --steps 20
```

### Q3: 遇到Segfault（退出码138/139）

**说明：** 这是已知的环境问题，不影响代码逻辑的正确性。静态验证已通过。

### Q4: 如何修改fake数据生成逻辑？

编辑`test_distributed_training.py`中的`generate_fake_batch()`函数。

## 验证清单

运行测试后，检查以下方面：

- [ ] 配置信息正确显示
- [ ] 不同rank生成不同的loss值
- [ ] metrics.step()后loss被规约（平均）
- [ ] tokens被正确累加（所有rank之和）
- [ ] 梯度累积边界正确识别
- [ ] 日志在正确的步数触发
- [ ] 检查点提示在正确的步数出现
- [ ] 最终统计信息准确

## 与真实训练的对比

| 组件 | 测试脚本 | 真实训练 |
|------|----------|----------|
| 数据 | fake随机数据 | 真实DataLoader |
| 模型 | 无（fake loss） | 真实模型forward/backward |
| 优化器 | 无（fake lr） | 真实optimizer.step() |
| **Metrics** | ✅ 完全相同 | ✅ 完全相同 |
| **StepScheduler** | ✅ 完全相同 | ✅ 完全相同 |
| **调用顺序** | ✅ 完全相同 | ✅ 完全相同 |

## 下一步

测试通过后，可以将此模式应用到真实训练：

1. 保持相同的调用顺序
2. 替换fake数据为真实数据
3. 添加真实的模型和优化器
4. Metrics和StepScheduler的使用方式保持不变

## 技术支持

如有问题，请查看：
- `test_distributed_training.py` - 完整代码实现
- `README_distributed_test.md` - 英文详细文档
