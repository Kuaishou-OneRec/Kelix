"""
Mock 使用示例 - 演示 mock 的工作原理
"""
from unittest.mock import Mock, patch


# ========== 示例 1: 基本 Mock 使用 ==========
def demo_basic_mock():
    """演示基本的 Mock 使用"""
    print("=" * 50)
    print("示例 1: 基本 Mock 使用")
    print("=" * 50)
    
    # 创建一个 Mock 对象
    mock_func = Mock()
    
    # 设置返回值
    mock_func.return_value = 42
    
    # 调用 Mock 函数
    result = mock_func()
    print(f"mock_func() 返回: {result}")
    assert result == 42
    
    # 可以设置属性
    mock_obj = Mock()
    mock_obj.name = "Alice"
    mock_obj.age = 30
    print(f"mock_obj.name = {mock_obj.name}")
    print(f"mock_obj.age = {mock_obj.age}")


# ========== 示例 2: 模拟 get_worker_info 的场景 ==========
def demo_worker_info_mock():
    """演示如何模拟 get_worker_info"""
    print("\n" + "=" * 50)
    print("示例 2: 模拟 get_worker_info")
    print("=" * 50)
    
    # 模拟 torch.utils.data.get_worker_info 的行为
    def get_worker_info():
        """被测试的函数"""
        try:
            import torch.utils.data
            worker_info = torch.utils.data.get_worker_info()
            if worker_info is not None:
                worker = worker_info.id
                num_workers = worker_info.num_workers
                return worker, num_workers
        except:
            pass
        return 0, 1
    
    # 方式 1: 直接 Mock（不推荐，因为需要导入 torch）
    print("\n方式 1: 直接 Mock")
    try:
        import torch.utils.data
        
        # 创建模拟的 worker_info 对象
        mock_worker_info = Mock()
        mock_worker_info.id = 2
        mock_worker_info.num_workers = 4
        
        # 替换函数
        original_func = torch.utils.data.get_worker_info
        torch.utils.data.get_worker_info = Mock(return_value=mock_worker_info)
        
        # 测试
        worker, num_workers = get_worker_info()
        print(f"worker = {worker}, num_workers = {num_workers}")
        assert worker == 2
        assert num_workers == 4
        
        # 恢复原始函数
        torch.utils.data.get_worker_info = original_func
    except ImportError:
        print("torch 未安装，跳过此示例")
    
    # 方式 2: 使用 @patch 装饰器（推荐）
    print("\n方式 2: 使用 @patch 装饰器")
    
    @patch('torch.utils.data.get_worker_info')
    def test_with_patch(mock_get_worker_info):
        # mock_get_worker_info 是 @patch 自动创建的 Mock 对象
        mock_worker_info = Mock()
        mock_worker_info.id = 2
        mock_worker_info.num_workers = 4
        mock_get_worker_info.return_value = mock_worker_info
        
        # 调用被测试的函数
        worker, num_workers = get_worker_info()
        print(f"worker = {worker}, num_workers = {num_workers}")
        assert worker == 2
        assert num_workers == 4
        return worker, num_workers
    
    try:
        test_with_patch()
    except ImportError:
        print("torch 未安装，跳过此示例")


# ========== 示例 3: @patch 的工作原理 ==========
def demo_patch_mechanism():
    """演示 @patch 的工作原理"""
    print("\n" + "=" * 50)
    print("示例 3: @patch 的工作原理")
    print("=" * 50)
    
    # 模拟一个模块
    class MyModule:
        @staticmethod
        def get_value():
            return "original value"
    
    original_value = MyModule.get_value()
    print(f"原始值: {original_value}")
    
    # 使用 @patch 替换
    @patch.object(MyModule, 'get_value')
    def test_patched(mock_get_value):
        # 设置返回值
        mock_get_value.return_value = "mocked value"
        
        # 调用函数
        result = MyModule.get_value()
        print(f"Mock 后的值: {result}")
        assert result == "mocked value"
        
        # 验证调用
        mock_get_value.assert_called_once()
    
    test_patched()
    
    # 测试结束后，原始函数恢复
    restored_value = MyModule.get_value()
    print(f"恢复后的值: {restored_value}")
    assert restored_value == "original value"


# ========== 示例 4: side_effect 的使用 ==========
def demo_side_effect():
    """演示 side_effect 的使用"""
    print("\n" + "=" * 50)
    print("示例 4: side_effect 的使用")
    print("=" * 50)
    
    mock_func = Mock()
    
    # 场景 1: 抛出异常
    print("\n场景 1: 模拟异常")
    mock_func.side_effect = ValueError("Something went wrong")
    try:
        mock_func()
    except ValueError as e:
        print(f"捕获异常: {e}")
    
    # 场景 2: 多个返回值
    print("\n场景 2: 多个返回值")
    mock_func.side_effect = [1, 2, 3]
    print(f"第1次调用: {mock_func()}")
    print(f"第2次调用: {mock_func()}")
    print(f"第3次调用: {mock_func()}")
    
    # 场景 3: 动态返回值
    print("\n场景 3: 动态返回值（函数）")
    mock_func.side_effect = lambda x: x * 2
    print(f"mock_func(5) = {mock_func(5)}")
    print(f"mock_func(10) = {mock_func(10)}")


# ========== 示例 5: 验证函数调用 ==========
def demo_assert_called():
    """演示如何验证函数调用"""
    print("\n" + "=" * 50)
    print("示例 5: 验证函数调用")
    print("=" * 50)
    
    mock_func = Mock()
    
    # 调用函数
    mock_func("arg1", "arg2", keyword="value")
    
    # 验证调用
    print("验证函数被调用过:")
    mock_func.assert_called()
    
    print("验证调用参数:")
    mock_func.assert_called_with("arg1", "arg2", keyword="value")
    
    print("验证调用次数:")
    assert mock_func.call_count == 1
    print(f"调用次数: {mock_func.call_count}")


# ========== 主函数 ==========
if __name__ == "__main__":
    print("\n" + "=" * 50)
    print("Mock 使用示例演示")
    print("=" * 50)
    
    demo_basic_mock()
    demo_worker_info_mock()
    demo_patch_mechanism()
    demo_side_effect()
    demo_assert_called()
    
    print("\n" + "=" * 50)
    print("演示完成！")
    print("=" * 50)

