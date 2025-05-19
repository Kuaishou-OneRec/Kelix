import torch
import torch.distributed as dist
import os
import struct
from typing import List, Tuple, Dict

def serialize_tensor_group(tensors: List[torch.Tensor], names: List[str], ds_names) -> bytes:
    assert len(tensors) == len(names), "张量数量与名称数量必须匹配"
    
    # 原始序列化逻辑
    ds_data = bytearray()
    ds_data.extend(struct.pack(">I", len(ds_names)))
    for ds in ds_names:
        encoded = ds.encode("utf-8")
        ds_data.extend(struct.pack(">I", len(encoded)))
        ds_data.extend(encoded)
    
    # 序列化元数据
    tensor_bytes = bytearray()
    tensor_bytes.extend(struct.pack(">I", len(tensors)))  # 张量数量
    
    for name, tensor in zip(names, tensors):
        dtype = str(tensor.dtype)
        shape = tensor.shape
        # 序列化名称
        name_bytes = name.encode('utf-8')
        tensor_bytes.extend(struct.pack(">I", len(name_bytes)))
        tensor_bytes.extend(name_bytes)
        
        # 序列化数据类型
        dtype_bytes = dtype.encode('utf-8')
        tensor_bytes.extend(struct.pack(">I", len(dtype_bytes)))
        tensor_bytes.extend(dtype_bytes)
        
        # 序列化形状
        tensor_bytes.extend(struct.pack(">I", len(shape)))  # 维度数量
        tensor_bytes.extend(struct.pack(f">{len(shape)}I", *shape))  # 各维度大小
        
        # 序列化数据长度
        tensor_data = tensor.numpy().tobytes()
        tensor_bytes.extend(struct.pack(">Q", len(tensor_data)))
        tensor_bytes.extend(tensor_data)
    
    # 组合元数据和张量数据（添加总长度前缀）
    total_size = len(ds_data) + len(tensor_bytes)
    return struct.pack(">Q", total_size) + bytes(ds_data) + bytes(tensor_bytes)


def deserialize_tensor_group(buffer: bytes) -> Tuple[List[torch.Tensor], List[str]]:
    ptr = 0
    if ptr + 8 > len(buffer):
        raise ValueError("Not enough buffer, cannot read buffer_size")
    total_size = struct.unpack(">Q", buffer[ptr:ptr+8])[0]
    ptr += 8
    
    # 验证缓冲区长度
    if len(buffer) != 8 + total_size:
        raise ValueError(f"Deserialized failed: Expect: {8 + total_size}, Got: {len(buffer)} bytes")
    num_ds = struct.unpack(">I", buffer[ptr:ptr+4])[0]
    ds_list = []
    ptr += 4
    for _ in range(num_ds):
        ds_len = struct.unpack(">I", buffer[ptr:ptr+4])[0]
        ptr += 4
        ds_name = buffer[ptr:ptr+ds_len].decode("utf-8")
        ptr += ds_len
        ds_list.append(ds_name)

    # 读取元数据
    num_tensors = struct.unpack(">I", buffer[ptr:ptr+4])[0]
    print(f"rank={dist.get_rank()}, num_tensors={num_tensors}")
    ptr += 4
    
    tensors = []
    names = []
    
    for _ in range(num_tensors):
        # 读取名称
        name_len = struct.unpack(">I", buffer[ptr:ptr+4])[0]
        print(f"rank={dist.get_rank()}, name_len={name_len}")
        ptr += 4
        name = buffer[ptr:ptr+name_len].decode('utf-8')
        print(f"rank={dist.get_rank()}, name={name}")
        ptr += name_len
        
        # 读取数据类型
        dtype_len = struct.unpack(">I", buffer[ptr:ptr+4])[0]
        ptr += 4
        dtype = buffer[ptr:ptr+dtype_len].decode('utf-8')
        print(f"rank={dist.get_rank()}, dtype={dtype}")
        ptr += dtype_len
        
        # 读取形状
        shape_len = struct.unpack(">I", buffer[ptr:ptr+4])[0]
        ptr += 4
        shape = struct.unpack(f">{shape_len}I", buffer[ptr:ptr+4*shape_len])
        ptr += 4 * shape_len
        shape = tuple(shape)
        print(f"rank={dist.get_rank()}, shape={shape}")
        
        # 读取数据长度
        data_len = struct.unpack(">Q", buffer[ptr:ptr+8])[0]
        ptr += 8
        
        # 读取张量数据
        tensor_data = buffer[ptr:ptr+data_len]
        ptr += data_len
        
        # 创建张量
        if data_len > 0:
            tensor = torch.frombuffer(tensor_data, dtype=eval(dtype)).reshape(shape)
        else:
            tensor = torch.empty(0, dtype=eval(dtype)).reshape(shape)
        print(f"rank={dist.get_rank()}, tensor={tensor}")
        tensors.append(tensor)
        names.append(name)
    
    # 验证指针是否到达缓冲区末尾
    if ptr != 8 + total_size:
        raise ValueError(f"Inconsistent data: current_ptr: {ptr}, total={8 + total_size}")
    
    return tensors, names, ds_list
    
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
        assert len(batch_list) == count, f"{sender}, {receiver}, {count}, {batch_list}"
        for batch in batch_list:
            batch_tensors = []
            batch_names = []
            ds_names = []
            for sample in batch:
                assert pivot in sample
                ds_name = sample.pop(pivot)
                assert "input_ids" in sample
                names = ["input_ids"] + [k for k in sample.keys() if k != "input_ids"]
                tensors = [sample[k] for k in names]
                ds_names.append(ds_name)
                batch_names.extend(names)
                batch_tensors.extend(tensors)
            serialized = serialize_tensor_group(batch_tensors, batch_names, ds_names)
            send_data[receiver].append(serialized)
    
    # 构建send_counts和recv_counts：每个节点发送/接收至其他节点的总字节数
    send_counts = [sum(len(group) for group in send_data.get(r, [])) for r in range(world_size)]
    
    send_counts_tensor = torch.tensor(send_counts, dtype=torch.int64)
    recv_counts_tensor = torch.zeros_like(send_counts_tensor)
    dist.all_to_all_single(recv_counts_tensor, send_counts_tensor)
    recv_counts = [recv_counts_tensor[i].item() for i in range(world_size)]
    
    # 构建发送缓冲区：将所有发送数据按目标节点顺序连接
    send_buffer = []
    for r in range(world_size):
        send_buffer.extend(send_data.get(r, []))
    buf = b''.join(send_buffer)
    send_buffer = torch.empty(0, dtype=torch.uint8) if len(buf) == 0 else torch.frombuffer(buf, dtype=torch.uint8)
    
    # 构建接收缓冲区
    recv_buffer = torch.zeros(sum(recv_counts), dtype=torch.uint8)
    recv_shape = recv_buffer.shape
    
    num_recv = sum([recv > 0 for recv in recv_counts])
    # 执行all_to_all操作
    dist.all_to_all_single(
        recv_buffer,
        send_buffer,
        recv_counts,
        send_counts,
        async_op=False
    )
    
    # 解析接收数据
    received_groups = []
    ptr = 0
    recv_buffer = recv_buffer.numpy().tobytes()
    print(f"rank={rank}, recv:{recv_shape}, {recv_counts}, send:{send_buffer.shape}, {send_counts}, recv_buf: {len(recv_buffer)}")
    while ptr < len(recv_buffer):
        try:
            # 读取组大小
            if ptr + 8 > len(recv_buffer):
                break  # 数据不足，退出
            
            group_size = struct.unpack(">Q", recv_buffer[ptr:ptr+8])[0]
            print(f"rank={rank}, group_size: {group_size}")
            
            # 检查缓冲区是否包含完整的组
            if ptr + 8 + group_size > len(recv_buffer):
                print(f"inconsistent data: expect_size={ptr + 8 + group_size}, got={len(recv_buffer)}")
                break
            
            # 提取组数据
            group_data = recv_buffer[ptr:ptr + 8 + group_size]
            
            # 反序列化
            tensors, names, ds_list = deserialize_tensor_group(group_data)
            print(f"rank={rank}, parsed_names: {names}")
            group = []
            sample = {}
            for name, t in zip(names, tensors):
                if name == "input_ids":
                    if sample:
                        group.append(sample)
                    sample = {name: t}
                else:
                    sample[name] = t

            if samlpe:
                group.append(sample)
            for s, ds in zip(group, ds_list):
                s[pivot] = ds
   
            received_groups.append(group)
            
            # 更新指针
            ptr += 8 + group_size
            
        except Exception as e:
            print(f"deserialized failed: rank={rank}, scheme={transfer_scheme}, {e}")
            ptr += 8
            break
    
    print(f"Rank {rank} received {len(received_groups)}")
    return received_groups


def convert_data_source(name):
    buf = name.encode("ascii")
    print(f"rank={dist.get_rank()}, raw_name={name}, encode_name: {buf}")
    return torch.frombuffer(buf, dtype=torch.uint8)
