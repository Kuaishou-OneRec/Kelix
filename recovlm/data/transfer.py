import torch
import torch.distributed as dist
import os
import struct
from typing import List, Tuple, Dict

def serialize_tensor_group(tensors: List[torch.Tensor], names: List[str]) -> bytes:
    assert len(tensors) == len(names), "张量数量与名称数量必须匹配"
    
    # 原始序列化逻辑
    metadata = []
    data = bytearray()
    
    for tensor, name in zip(tensors, names):
        dtype = str(tensor.dtype)
        shape = tensor.shape
        tensor_data = tensor.numpy().tobytes()
        
        # 元数据格式：名称长度 + 名称 + dtype长度 + dtype + 维度数 + 维度列表 + 数据长度 + 数据
        metadata.append((name, dtype, shape, len(tensor_data)))
        data.extend(tensor_data)
    
    # 序列化元数据
    metadata_bytes = struct.pack(">I", len(metadata))  # 张量数量
    for name, dtype, shape, data_len in metadata:
        metadata_bytes += struct.pack(">I", len(name)) + name.encode()
        metadata_bytes += struct.pack(">I", len(dtype)) + dtype.encode()
        metadata_bytes += struct.pack(">I", len(shape)) + struct.pack(">" + "I"*len(shape), *shape)
        metadata_bytes += struct.pack(">Q", data_len)
    
    # 完整的组数据：组大小（8字节） + 元数据 + 张量数据
    group_bytes = struct.pack(">Q", len(metadata_bytes) + len(data)) + metadata_bytes + bytes(data)
    return group_bytes

def deserialize_tensor_group(buffer: bytes) -> Tuple[List[torch.Tensor], List[str]]:
    ptr = 0
    
    # 读取组大小（跳过，因为整个buffer就是一个完整的组）
    group_size = struct.unpack(">Q", buffer[ptr:ptr+8])[0]
    ptr += 8
    
    # 解析张量数量
    num_tensors = struct.unpack(">I", buffer[ptr:ptr+4])[0]
    ptr += 4
    
    tensors = []
    names = []
    
    for _ in range(num_tensors):
        # 解析名称
        name_len = struct.unpack(">I", buffer[ptr:ptr+4])[0]
        ptr += 4
        name = buffer[ptr:ptr+name_len].decode()
        ptr += name_len
        
        # 解析dtype
        dtype_len = struct.unpack(">I", buffer[ptr:ptr+4])[0]
        ptr += 4
        dtype = buffer[ptr:ptr+dtype_len].decode()
        ptr += dtype_len
        
        # 解析形状
        shape_len = struct.unpack(">I", buffer[ptr:ptr+4])[0]
        ptr += 4
        shape = struct.unpack(">" + "I"*shape_len, buffer[ptr:ptr+4*shape_len])
        ptr += 4*shape_len
        shape = tuple(shape)
        
        # 解析数据长度
        data_len = struct.unpack(">Q", buffer[ptr:ptr+8])[0]
        ptr += 8
        
        # 解析数据
        tensor_data = buffer[ptr:ptr+data_len]
        ptr += data_len
        
        # 转换为张量
        tensor = torch.frombuffer(tensor_data, dtype=getattr(torch, dtype)).reshape(shape)
        tensors.append(tensor)
        names.append(name)
    
    return tensors, names

def exchange_batch_data(transfer_scheme, batch_data, pivot="__ds__"):
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    # 构建发送数据：{target_rank: List[bytes]}，每组数据为序列化后的字节流
    send_data = {target: [] for target in range(world_size)}
    
    # 生成当前节点的发送数据(示例数据)
    for sender, receiver, count in transfer_scheme:
        if sender != rank:
            continue
        
        batch_list = batch_data[receiver]
        assert len(batch_list) == count
        for batch in batch_list:
            batch_tensors = []
            batch_names = []
            for sample in batch:
                assert pivot in sample
                names = [pivot] + [k for k in sample.keys() if k != pivot]
                tensors = [sample[k] for k in names]
                batch_names.extend(names)
                batch_tensors.extend(tensors)
            serialized = serialize_tensor_group(batch_tensors, batch_names)
            send_data[receiver].append(serialized)
    
    # 构建send_counts和recv_counts：每个节点发送/接收至其他节点的总字节数
    send_counts = [0] * world_size
    for target in send_data:
        send_counts[target] = sum(len(group) for group in send_data[target])
    
    send_counts_tensor = torch.tensor(send_counts, dtype=torch.int64)
    recv_counts_tensor = torch.zeros_like(send_counts_tensor)
    dist.all_to_all_single(recv_counts_tensor, send_counts_tensor)
    recv_counts = [recv_counts_tensor[i].item() for i in range(world_size)]
    
    send_counts = [send_counts[i] for i in range(world_size)]
    
    # 构建发送缓冲区：将所有发送数据按目标节点顺序连接
    send_buffer = []
    for target in range(world_size):
        send_buffer.extend(send_data[target])
    send_buffer = torch.tensor(b''.join(send_buffer), dtype=torch.uint8)
    
    # 构建接收缓冲区
    recv_buffer = torch.zeros(sum(recv_counts), dtype=torch.uint8)
    
    # 执行all_to_all操作
    dist.all_to_all_single(
        recv_buffer,
        send_buffer,
        send_counts=send_counts,
        recv_counts=recv_counts,
        async_op=False
    )
    
    # 解析接收数据
    received_groups = []
    ptr = 0
    while ptr < len(recv_buffer):
        try:
            # 读取组大小
            if ptr + 8 > len(recv_buffer):
                break  # 数据不足，退出
            
            group_size = struct.unpack(">Q", recv_buffer[ptr:ptr+8].numpy())[0]
            
            # 检查缓冲区是否包含完整的组
            if ptr + 8 + group_size > len(recv_buffer):
                print(f"数据不完整：期望大小 {ptr + 8 + group_size}，实际大小 {len(recv_buffer)}")
                break
            
            # 提取组数据
            group_data = recv_buffer[ptr:ptr + 8 + group_size].numpy().tobytes()
            
            # 反序列化
            tensors, names = deserialize_tensor_group(group_data)
            group = []
            sample = {}
            for name, t in zip(names, tensors):
                if name == pivot:
                    if len(sample) > 0:
                        group.append(sample)
                    sample = {}
                sample[name] = t
            if len(samlpe) > 0:
                group.append(sample)
   
            received_groups.append(group)
            
            # 更新指针
            ptr += 8 + group_size
            
        except Exception as e:
            print(f"解析数据失败: {e}")
            ptr += 8  # 跳过组大小字段，避免死循环
            break
    
    # 打印结果
    print(f"Rank {rank} 共接收 {len(received_groups)} 组数据")
    return received_groups


def convert_data_source(name):
    decode = name.encode("utf-8")
    return torch.tensor(list(decode), dtype=torch.uint8)
