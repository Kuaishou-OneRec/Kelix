import torch
from typing import Any, Dict, List, Tuple, Union

def print_input_info(data: Any, prefix: str = "", max_str_len: int = 50, return_str: bool = False) -> Union[None, str]:
    """
    递归打印或返回输入数据的详细信息。Args:
        data: 要打印信息的数据，可以是任意类型
        prefix: 打印信息的前缀，用于显示层级结构
        max_str_len: 字符串类型数据显示的最大长度
        return_str: 如果为True，返回格式化的字符串而不是打印
    
    Returns:
        如果return_str为True，返回格式化的字符串；否则返回None
    
    Examples:
        >>> tensor = torch.randn(2, 3)
        >>> bool_tensor = torch.tensor([[True, False, True], [False, True, True]])
        >>> nested_data = {
        ...     "tensor": tensor,
        ...     "mask": bool_tensor,
        ...     "text": "Hello, world!"
        ... }
        >>> # 直接打印
        >>> print_input_info(nested_data)
        >>> # 返回字符串
        >>> result = print_input_info(nested_data, return_str=True)
        >>> print(result)
    """
    lines = []
    
    def add_line(text: str):
        if return_str:
            lines.append(text)
        else:
            print(text)
            
    if data is None:
        add_line(f"{prefix}None")
        return "\n".join(lines) if return_str else None
    if isinstance(data, torch.Tensor):
        base_info = f"{prefix}Tensor: shape={tuple(data.shape)}, dtype={data.dtype}, device={data.device}, data={data.flatten()[:4]}...{data.flatten()[-4:]}"
        
        if data.dtype == torch.bool:
            total_elements = data.numel()
            true_count = data.sum().item()
            false_count = total_elements - true_count
            true_ratio = true_count / total_elements * 100
            false_ratio = false_count / total_elements * 100
            add_line(base_info)
            add_line(f"{prefix}  True:  count={true_count:,d} ({true_ratio:.2f}%)")
            add_line(f"{prefix}  False: count={false_count:,d} ({false_ratio:.2f}%)")
        else:
            add_line(base_info)
    elif isinstance(data, str):
        display_str = data[:max_str_len] + "..." if len(data) > max_str_len else data
        add_line(f"{prefix}String: length={len(data)}, value='{display_str}'")
        
    elif isinstance(data, (list, tuple)):
        container_type = "List" if isinstance(data, list) else "Tuple"
        add_line(f"{prefix}{container_type}: length={len(data)}")
        for i, item in enumerate(data):
            add_line(f"{prefix}[{i}]:")
            sub_result = print_input_info(item, prefix + "  ", max_str_len, return_str=True)
            if return_str:
                lines.extend(sub_result.split('\n'))
            else:
                print(sub_result)
            
    elif isinstance(data, dict):
        add_line(f"{prefix}Dict: keys={len(data)}")
        for key, value in data.items():
            add_line(f"{prefix}'{key}':")
            sub_result = print_input_info(value, prefix + "  ", max_str_len, return_str=True)
            if return_str:
                lines.extend(sub_result.split('\n'))
            else:
                print(sub_result)
                
    elif isinstance(data, (int, float)):
        add_line(f"{prefix}{type(data).__name__}: {data}")
    else:
        add_line(f"{prefix}Other type ({type(data).__name__}): {str(data)}")
        
    return "\n".join(lines) if return_str else None



def debug_inputs(inputs: Any, name: str = "inputs", return_str: bool = False) -> Union[None, str]:
    """
    用于调试时打印或返回输入数据的包装函数。Args:
        inputs: 要打印信息的输入数据
        name: 输入数据的名称
        return_str: 如果为True，返回格式化的字符串而不是打印
    
    Returns:
        如果return_str为True，返回格式化的字符串；否则返回None
    """
    header = f"\n{'='*20} Debug {name} {'='*20}"
    footer = '='*50+ '\n'
    if return_str:
        content = print_input_info(inputs, return_str=True)
        return f"{header}\n{content}\n{footer}"
    else:
        print(header)
        print_input_info(inputs)
        print(footer)
        return None


# 测试代码
if __name__ == "__main__":
    test_data = {
        "attention_mask": torch.tensor([
            [True, True, True, False, False],
            [True, True, False, False, False]
        ]),
        "normal_tensor": torch.randn(2, 3),
        "nested": {
            "another_mask": torch.tensor([[True, False], [False, True]])
        }
    }
    
    # 测试直接打印
    print("Direct printing:")
    print_input_info(test_data)
    
    # 测试返回字符串
    print("\nPrinting returned string:")
    result_str = print_input_info(test_data, return_str=True)
    print(result_str)
    
    # 测试debug_inputs
    print("\nTesting debug_inputs with string return:")
    debug_str = debug_inputs(test_data, "test_data", return_str=True)
    print(debug_str)