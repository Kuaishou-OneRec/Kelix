import torch
from typing import Any, Dict, List, Tuple, Union
import math
import os
from dataclasses import is_dataclass, asdict
from typing import Any

def convert_dataclass_to_dict(obj: Any) -> Any:
    """
    将dataclass对象转换为字典，非dataclass对象返回原对象
    
    参数:
        obj: 任意类型的对象
        
    返回:
        如果obj是@dataclass装饰的类的实例，则返回对应的字典；否则返回obj本身
    """
    # 判断是否为dataclass实例（排除类本身，只处理实例）
    if is_dataclass(obj) and not isinstance(obj, type):
        return asdict(obj)
    return obj


def tensor_statistics(tensor, n=-1, **kwargs):
    """
    Statistics for a tensor of any shape, supporting 4 levels of statistics:
    full tensor, specified partial, magnitude-based partial, and 1/10 magnitude-based partial.
    
    Args:
        tensor: PyTorch tensor of any shape (may be empty)
        n: Controls the range of partial elements (default -1):
            - n=-1: statistics for the first half elements
            - n>0: statistics for the first n elements (ensure n <= total elements)
    
    Returns:
        Four strings: 
            - line1: full tensor statistics
            - line2: specified partial elements statistics
            - line3: magnitude-based partial elements statistics (1,10,100... elements)
            - line4: 1/10 magnitude-based partial elements statistics (mag_count//10 elements)
    
    Raises:
        ValueError: When n is invalid or exceeds total elements
    """
    # Flatten tensor for easy processing
    flattened = tensor.reshape(-1)
    total_elements = flattened.numel()
    
    # Handle empty tensor case
    if total_elements == 0:
        base = "mean: NaN, variance: NaN, max: NaN, min: NaN, non-zeros: 0"
        return (
            f"Full - {base}",
            f"Partial - {base}",
            f"Magnitude-based - {base}",
            f"1/10 Magnitude-based - {base}"
        )
    
    # --------------------------
    # Line2: Original partial stats (user-specified)
    # --------------------------
    if n == -1:
        part_count = (total_elements + 1) // 2
        part_tensor = flattened[:part_count]
        part_label = f"first half ({part_count} elements)"
    elif isinstance(n, int) and n > 0:
        if n > total_elements:
            raise ValueError(f"n={n} exceeds total elements ({total_elements})")
        part_count = n
        part_tensor = flattened[:n]
        part_label = f"first {n} elements"
    else:
        raise ValueError(f"n must be -1 or positive integer, got: {n}")
    
    # --------------------------
    # Line3: Magnitude-based stats (1,10,100... elements)
    # --------------------------
    if total_elements <= 1:
        mag_count = 0
        mag_label = "no elements (total <= 1)"
        mag_tensor = flattened[:0]
    else:
        log_val = math.log10(total_elements)
        k = int(log_val) - 1 if log_val.is_integer() else math.floor(log_val)
        mag_count = 10** k
        mag_count = min(mag_count, total_elements)  # Safeguard
        mag_tensor = flattened[:mag_count]
        mag_label = f"first {mag_count} elements (magnitude-based)"
    
    # --------------------------
    # Line4: 1/10 of magnitude-based stats (mag_count//10 elements)
    # --------------------------
    line4_count = mag_count // 10
    # Handle boundary: ensure count is valid and doesn't exceed total elements
    if line4_count <= 0:
        line4_label = "no elements (1/10 of magnitude-based <= 0)"
        line4_tensor = flattened[:0]  # Empty tensor
    else:
        line4_count = min(line4_count, total_elements)  # Avoid exceeding total
        line4_tensor = flattened[:line4_count]
        line4_label = f"first {line4_count} elements (1/10 of magnitude-based)"
    
    # --------------------------
    # Calculate all statistics
    # --------------------------
    def calc_stats(t):
        """Helper to calculate stats for a tensor slice"""
        if t.numel() == 0:
            return (float('nan'), float('nan'), float('nan'), float('nan'), 0)
        return (
            torch.mean(t.float()).item(),
            torch.var(t.float(), unbiased=False).item(),
            torch.max(t).item(),
            torch.min(t).item(),
            torch.count_nonzero(t).item()
        )
    
    # Full tensor
    full_mean, full_var, full_max, full_min, full_nonzero = calc_stats(flattened)
    # Line2 partial
    part_mean, part_var, part_max, part_min, part_nonzero = calc_stats(part_tensor)
    # Line3 magnitude-based
    mag_mean, mag_var, mag_max, mag_min, mag_nonzero = calc_stats(mag_tensor)
    # Line4 1/10 magnitude-based
    line4_mean, line4_var, line4_max, line4_min, line4_nonzero = calc_stats(line4_tensor)
    
    # --------------------------
    # Format output strings
    # --------------------------
    def format_line(label, mean, var, max_val, min_val, nonzero):
        return (f"{label} - mean: {mean:.6f}, variance: {var:.6f}, "
                f"max: {max_val:.6f}, min: {min_val:.6f}, non-zeros: {nonzero}")
    
    line1 = format_line("Full", full_mean, full_var, full_max, full_min, full_nonzero)
    line2 = format_line(part_label, part_mean, part_var, part_max, part_min, part_nonzero)
    line3 = format_line(mag_label, mag_mean, mag_var, mag_max, mag_min, mag_nonzero)
    line4 = format_line(line4_label, line4_mean, line4_var, line4_max, line4_min, line4_nonzero)
    
    return line1, line2, line3, line4
    
    



def print_input_info(data: Any, prefix: str = "", max_str_len: int = 50, return_str: bool = False, max_show: int = 4, save_path: Union[str, None] = None, **kargs) -> Union[None, str]:
    # return
    """
    递归打印或返回输入数据的详细信息，支持保存数据到指定路径（张量会detach到CPU）。
    
    新增功能：当save_path不为None时，将数据处理后（张量detach到CPU）用torch.save保存
    """
    data = convert_dataclass_to_dict(data)

    # 辅助函数：递归处理所有张量，detach并移到CPU
    def _detach_to_cpu(obj: Any) -> Any:
        if isinstance(obj, torch.Tensor):
            # 处理张量：detach脱离计算图，移到CPU
            return obj.detach().cpu()
        elif isinstance(obj, (list, tuple)):
            # 递归处理列表/元组元素
            return type(obj)(_detach_to_cpu(item) for item in obj)
        elif isinstance(obj, dict):
            # 递归处理字典值
            return {k: _detach_to_cpu(v) for k, v in obj.items()}
        elif hasattr(obj, '__dict__'):
            # 简单处理类对象（保存其字典属性）
            return {k: _detach_to_cpu(v) for k, v in obj.__dict__.items()}
        else:
            # 其他类型直接返回
            return obj

    # 当save_path不为None时，处理并保存数据
    if save_path is not None:
        try:
            # 处理数据：所有张量detach到CPU
            data_to_save = _detach_to_cpu(data)
            # 创建保存路径的父目录（如果不存在）
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            # 保存数据
            torch.save(data_to_save, save_path)
            print(f"save data to: {save_path}")  # 提示保存成功
        except Exception as e:
            print(f"save data failed: {str(e)}")  # 捕获保存异常

    lines = []
    try:
        data = dict(data)
    except:
        pass
    
    def add_line(text: str):
        if return_str:
            lines.append(text)
        else:
            print(text)
            
    if data is None:
        add_line(f"{prefix}None")
        return "\n".join(lines) if return_str else None
    if isinstance(data, torch.Tensor):
        base_info = f"{prefix}Tensor: shape={tuple(data.shape)}, dtype={data.dtype}, device={data.device}, data={data.flatten()[:max_show]}...{data.flatten()[-max_show:]}"
        
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
            for li, line in enumerate(tensor_statistics(data, **kargs)):
                add_line(f"{prefix}  stat{li}:  {line}")
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
        add_line(f"{prefix}Other type ({type(data).__name__}): {str(data)[:max_show]}...{str(data)[-max_show:]}")
        
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


def format_dict_or_list(obj, indent_level=0, indent_size=2):
    """
    格式化打印dict/list，用来替代json.dumps
    """
    def format_value(value, indent_level=0, indent_size=2):
        if isinstance(value, (dict, list)):
            return format_dict_or_list(value, indent_level, indent_size)
        elif isinstance(value, str):
            return f'"{value}"'
        else:
            return str(value)

    if isinstance(obj, dict):
        items = [f": {format_value(v, indent_level + 1)}" for k, v in obj.items()]
        keys = [f'"{k}"' for k in obj.keys()]
        formatted_items = ',\n'.join(f'{(" " * indent_size * (indent_level + 1))}{k}{v}' for k, v in zip(keys, items))
        return '{\n' + formatted_items + '\n' + (' ' * indent_size * indent_level) + '}'
    elif isinstance(obj, list):
        items = [format_value(item, indent_level + 1) for item in obj]
        formatted_items = ',\n'.join(' ' * indent_size * (indent_level + 1) + item for item in items)
        return '[\n' + formatted_items + '\n' + (' ' * indent_size * indent_level) + ']'
    else:
        return obj
    
    
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