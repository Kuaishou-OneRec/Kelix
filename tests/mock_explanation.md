# Mock 原理和使用方法详解

## 什么是 Mock？

Mock（模拟）是单元测试中的一个重要概念，用于**替换真实的外部依赖**，让测试可以：
1. **隔离测试**：不依赖外部系统（数据库、网络、文件系统等）
2. **控制行为**：模拟各种场景（成功、失败、异常等）
3. **提高速度**：避免真实的I/O操作
4. **可重复性**：每次运行结果一致

## 示例代码分析

让我们分析 `test_get_worker_info_with_worker` 这个测试：

```python
@patch('torch.utils.data.get_worker_info')
def test_get_worker_info_with_worker(self, mock_get_worker_info):
    """Test get_worker_info when worker context exists"""
    mock_worker_info = Mock()
    mock_worker_info.id = 2
    mock_worker_info.num_workers = 4
    mock_get_worker_info.return_value = mock_worker_info

    worker, num_workers = get_worker_info()
    assert worker == 2
    assert num_workers == 4
```

### 步骤分解

#### 1. `@patch('torch.utils.data.get_worker_info')` 装饰器

**作用**：临时替换 `torch.utils.data.get_worker_info` 函数

**原理**：
- `@patch` 会在测试函数执行前，将指定的对象替换为一个 Mock 对象
- 测试结束后，自动恢复原始对象
- 参数 `'torch.utils.data.get_worker_info'` 是**字符串路径**，指向要替换的对象

**为什么用字符串？**
```python
# 字符串路径：在导入时替换（推荐）
@patch('torch.utils.data.get_worker_info')  # ✅ 正确

# 直接对象：可能已经导入，替换无效
@patch(torch.utils.data.get_worker_info)    # ❌ 可能失败
```

#### 2. `mock_get_worker_info` 参数

**作用**：`@patch` 装饰器会自动将创建的 Mock 对象作为参数传入测试函数

**流程**：
```python
# 1. @patch 创建 Mock 对象
mock_get_worker_info = Mock()

# 2. 替换 torch.utils.data.get_worker_info
torch.utils.data.get_worker_info = mock_get_worker_info

# 3. 作为参数传入测试函数
def test_get_worker_info_with_worker(self, mock_get_worker_info):
    # mock_get_worker_info 就是上面创建的 Mock 对象
    pass
```

#### 3. 创建模拟的 worker_info 对象

```python
mock_worker_info = Mock()
mock_worker_info.id = 2
mock_worker_info.num_workers = 4
```

**说明**：
- `Mock()` 创建一个模拟对象
- 可以给 Mock 对象设置任意属性
- 这些属性会在被测试代码访问时返回设定的值

#### 4. 设置返回值

```python
mock_get_worker_info.return_value = mock_worker_info
```

**作用**：当 `torch.utils.data.get_worker_info()` 被调用时，返回 `mock_worker_info`

**实际执行流程**：
```python
# 在 get_worker_info() 函数内部：
def get_worker_info():
    worker_info = torch.utils.data.get_worker_info()  # 这里被 mock 替换了
    # 实际调用的是：mock_get_worker_info()
    # 返回：mock_worker_info
    
    if worker_info is not None:
        worker = worker_info.id      # 返回 2
        num_workers = worker_info.num_workers  # 返回 4
    return worker, num_workers
```

## Mock 的常用方法

### 1. `return_value` - 设置返回值

```python
mock_func = Mock()
mock_func.return_value = 42
assert mock_func() == 42  # 调用返回 42
```

### 2. `side_effect` - 设置副作用（异常、多个返回值等）

```python
# 抛出异常
mock_func.side_effect = ValueError("Error!")

# 多个返回值（每次调用返回不同值）
mock_func.side_effect = [1, 2, 3]
assert mock_func() == 1
assert mock_func() == 2
assert mock_func() == 3

# 函数（动态返回值）
mock_func.side_effect = lambda x: x * 2
assert mock_func(5) == 10
```

### 3. 属性访问

```python
mock_obj = Mock()
mock_obj.name = "test"
mock_obj.age = 25
assert mock_obj.name == "test"
assert mock_obj.age == 25
```

### 4. `assert_called_with()` - 验证调用

```python
mock_func = Mock()
mock_func(1, 2, 3)
mock_func.assert_called_with(1, 2, 3)  # 验证是否用这些参数调用过
```

## 其他 Mock 示例

### 示例 1：Mock 文件操作

```python
@patch('builtins.open', create=True)
def test_read_file(mock_open):
    # 模拟文件内容
    mock_open.return_value.__enter__.return_value.read.return_value = "file content"
    
    with open('test.txt') as f:
        content = f.read()
    
    assert content == "file content"
    mock_open.assert_called_with('test.txt')
```

### 示例 2：Mock 网络请求

```python
@patch('requests.get')
def test_api_call(mock_get):
    # 模拟 API 响应
    mock_response = Mock()
    mock_response.json.return_value = {'status': 'ok'}
    mock_response.status_code = 200
    mock_get.return_value = mock_response
    
    response = requests.get('https://api.example.com')
    assert response.json() == {'status': 'ok'}
    assert response.status_code == 200
```

### 示例 3：Mock 环境变量

```python
@patch.dict(os.environ, {'API_KEY': 'test-key'})
def test_with_env():
    assert os.environ['API_KEY'] == 'test-key'
```

### 示例 4：Mock 多个对象

```python
@patch('module.function1')
@patch('module.function2')
def test_multiple_mocks(mock_func2, mock_func1):
    # 注意：装饰器顺序是自下而上的，参数顺序相反
    mock_func1.return_value = 1
    mock_func2.return_value = 2
    # ...
```

## 实际应用场景

### 场景 1：测试分布式环境（你的代码）

```python
@patch('muse.data.datasets.base.get_data_parallel_rank')
@patch('muse.data.datasets.base.get_data_parallel_world_size')
def test_distributed_dataset(mock_world_size, mock_rank):
    # 模拟 rank=1, world_size=4 的分布式环境
    mock_rank.return_value = 1
    mock_world_size.return_value = 4
    
    dataset = DistributedDataset(...)
    assert dataset.rank == 1
    assert dataset.world_size == 4
```

### 场景 2：测试 HDFS 操作

```python
@patch('os.system')
def test_load_parquet_hdfs(mock_system):
    mock_system.return_value = 0  # 模拟命令成功
    
    # 测试 HDFS 文件加载
    parquet_file = load_parquet("hdfs://path/to/file.parquet")
    # ...
```

### 场景 3：测试异常处理

```python
@patch('muse.data.datasets.base.load_parquet')
def test_error_handling(mock_load_parquet):
    # 模拟文件加载失败
    mock_load_parquet.side_effect = IOError("File not found")
    
    reader = ParquetReader(["nonexistent.parquet"])
    samples = list(reader)
    assert len(samples) == 0  # 应该优雅处理错误
```

## 最佳实践

1. **使用字符串路径**：`@patch('module.function')` 而不是 `@patch(module.function)`
2. **最小化 Mock**：只 Mock 必要的外部依赖
3. **验证调用**：使用 `assert_called_with()` 验证函数被正确调用
4. **清理资源**：`@patch` 会自动清理，但复杂场景可能需要手动清理
5. **文档说明**：在测试中说明为什么需要 Mock

## 总结

Mock 的核心思想是：**在测试中，用可控的假对象替换不可控的真实对象**，从而：
- ✅ 让测试更快、更稳定
- ✅ 测试各种边界情况和异常
- ✅ 隔离测试，不依赖外部环境
- ✅ 提高测试覆盖率

在你的测试中，`@patch` 替换了 `torch.utils.data.get_worker_info`，让测试可以在没有真实 PyTorch DataLoader worker 环境的情况下，测试 `get_worker_info()` 函数的行为。

