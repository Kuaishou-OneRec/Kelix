from typing import Dict, Any, Union, Optional, Generator, \
    Iterable, Tuple, List

import torch
from torch import Tensor
import math
import os
import shutil
from pathlib import Path
import torch.distributed as dist

import contextlib

from torch.utils._foreach_utils import (
    _device_has_foreach_support,
    _group_tensors_by_device_and_dtype,
    _has_foreach_support,
)

@contextlib.contextmanager
def set_default_dtype(dtype: Union[str, torch.dtype]) -> Generator[None, None, None]:
    """
    Context manager to set torch's default dtype.

    Args:
        dtype (Union[str, torch.dtype]): The desired default dtype inside the context manager.
            Can be either a string ("bfloat16", "float16", "float32") or a torch.dtype.

    Returns:
        ContextManager: context manager for setting default dtype.

    Example:
        >>> with set_default_dtype(torch.bfloat16):
        >>>     x = torch.tensor([1, 2, 3])
        >>>     x.dtype
        torch.bfloat16
        
        >>> with set_default_dtype("bfloat16"):
        >>>     x = torch.tensor([1, 2, 3])
        >>>     x.dtype
        torch.bfloat16

    """
    # Convert string to torch.dtype if needed
    if isinstance(dtype, str):
        dtype_map = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }
        if dtype not in dtype_map:
            raise ValueError(
                f"Invalid dtype string: {dtype}. "
                f"Supported values: {list(dtype_map.keys())}"
            )
        dtype = dtype_map[dtype]
    
    old_dtype = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        yield
    finally:
        torch.set_default_dtype(old_dtype)

def clip_grad_by_value(model, clip_range=None):
    if clip_range is not None:
        torch.nn.utils.clip_grad_value_(model.parameters(), clip_range)

def get_total_norm(
    tensors: Union[torch.Tensor, Iterable[torch.Tensor]],
    norm_type: float = 2.0,
    error_if_nonfinite: bool = False,
    foreach: Optional[bool] = None,
) -> torch.Tensor:
    r"""Compute the norm of an iterable of tensors.

    The norm is computed over the norms of the individual tensors, as if the norms of
    the individual tensors were concatenated into a single vector.

    Args:
        tensors (Iterable[Tensor] or Tensor): an iterable of Tensors or a
            single Tensor that will be normalized
        norm_type (float): type of the used p-norm. Can be ``'inf'`` for
            infinity norm.
        error_if_nonfinite (bool): if True, an error is thrown if the total
            norm of :attr:`tensors` is ``nan``, ``inf``, or ``-inf``.
            Default: ``False``
        foreach (bool): use the faster foreach-based implementation.
            If ``None``, use the foreach implementation for CUDA and CPU native tensors and silently
            fall back to the slow implementation for other device types.
            Default: ``None``

    Returns:
        Total norm of the tensors (viewed as a single vector).
    """
    if isinstance(tensors, torch.Tensor):
        tensors = [tensors]
    else:
        tensors = list(tensors)
    norm_type = float(norm_type)
    if len(tensors) == 0:
        return torch.tensor(0.0)
    first_device = tensors[0].device
    grouped_tensors: Dict[
        Tuple[torch.device, torch.dtype], Tuple[List[List[Tensor]], List[int]]
    ] = _group_tensors_by_device_and_dtype(
        [tensors]  # type: ignore[list-item]
    )  # type: ignore[assignment]

    norms: List[Tensor] = []
    for (device, _), ([device_tensors], _) in grouped_tensors.items():
        if (foreach is None and _has_foreach_support(device_tensors, device)) or (
            foreach and _device_has_foreach_support(device)
        ):
            norms.extend(torch._foreach_norm(device_tensors, norm_type))
        elif foreach:
            raise RuntimeError(
                f"foreach=True was passed, but can't use the foreach API on {device.type} tensors"
            )
        else:
            norms.extend(
                [torch.linalg.vector_norm(g, norm_type) for g in device_tensors]
            )

    total_norm = torch.linalg.vector_norm(
        torch.stack([norm.to(first_device) for norm in norms]), norm_type
    )

    if error_if_nonfinite and torch.logical_or(total_norm.isnan(), total_norm.isinf()):
        raise RuntimeError(
            f"The total norm of order {norm_type} for gradients from "
            "`parameters` is non-finite, so it cannot be clipped. To disable "
            "this error and scale the gradients by the non-finite norm anyway, "
            "set `error_if_nonfinite=False`"
        )
    return total_norm

def get_global_grad_norm(model):
    grads = [
        param.grad.data for param in model.parameters() \
            if param.grad is not None]
    return get_total_norm(grads, norm_type=2.0)



def compute_fsdp_zero2_grad_norm(model, ignore_unused_parameters=True):
    """
    计算FSDP Zero-2模式下的梯度L2范数（grad norm）
    
    参数:
        model: FSDP包装后的模型
        ignore_unused_parameters: 是否忽略未计算梯度的参数
    返回:
        全局梯度L2范数
    """
    # 初始化全局平方和
    total_sq = torch.tensor(0.0, device=next(model.parameters()).device)
    
    # 遍历所有参数，累加本地梯度分片的平方和
    for param in model.parameters():
        if param.grad is None:
            if not ignore_unused_parameters:
                raise ValueError(f"参数 {param} 没有梯度，请检查是否被正确使用")
            continue
        
        # 从DTensor中获取本地分片（FSDP的梯度是DTensor类型）
        # 注意：DTensor的.local_tensor()方法返回当前进程持有的分片
        local_grad = param.grad.to_local()
        
        # 计算本地分片的平方和，并累加到total_sq
        total_sq += torch.sum(local_grad **2)
    
    # 跨进程聚合所有分片的平方和（使用all_reduce求和）
    # 注意：需要使用FSDP的通信组，或默认的全局通信组
    dist.all_reduce(total_sq, op=dist.ReduceOp.SUM, group=dist.group.WORLD)
    
    # 计算全局L2范数（平方和的平方根）
    grad_norm = torch.sqrt(total_sq).item()
    
    return grad_norm

class GradNormLogger:
    def __init__(self, log_dir="grad_norm_logs"):
        """
        初始化梯度范数记录器（会清空原有文件夹内容）
        :param log_dir: 日志文件夹路径
        """
        self.step = 0  # 记录当前是第几个step
        self.log_dir = log_dir
        self.rank = self._get_rank()  # 获取当前进程的rank
        
        # 若文件夹已存在则删除并重建（清空效果），否则直接创建
        if self._get_rank() == 0:
            if os.path.exists(self.log_dir):
                shutil.rmtree(self.log_dir)  # 递归删除文件夹及内部所有内容
            Path(self.log_dir).mkdir(parents=True, exist_ok=True)  # 重新创建文件夹
        
    def _get_rank(self):
        """
        自动获取当前进程的rank，处理非分布式环境情况
        在非分布式环境或未初始化分布式时返回0
        """
        try:
            if not hasattr(torch, 'distributed'):
                return 0
                
            if torch.distributed.is_initialized():
                return torch.distributed.get_rank()
            else:
                return 0
        except (RuntimeError, ImportError):
            return 0
        
    def __call__(self, model, step=None):
        """
        调用方法，计算并记录当前step的所有参数梯度范数
        :param model: PyTorch模型
        :param step: 可选参数，指定当前step，不指定则自动递增
        """
        if step is None:
            self.step += 1
            step = self.step
        
        # 为当前rank创建独立的日志文件
        log_file = os.path.join(self.log_dir, f"grad_norm_rank_{self.rank}.txt")
        
        # 第一次写入时添加表头
        write_header = not os.path.exists(log_file)
        
        with open(log_file, 'a') as f:
            if write_header:
                f.write("Step,Parameter Name,Gradient Norm,shape,local_shape\n")
                
            for name, param in model.named_parameters():
                if param.grad is not None:
                    grad_norm = math.sqrt(torch.sum(param.grad**2).item())
                    shape = param.data.shape
                    try:
                        local_shape = param.grad.to_local().shape
                    except:
                        local_shape = None
                    f.write(f"{step},{name},{grad_norm:.6f}. {shape}/{local_shape}\n")
                    if 'head' in name:
                        f.write('\n' + str(param) + '\n')
                else:
                    f.write(f"{step},{name},NaN,None,None\n")
