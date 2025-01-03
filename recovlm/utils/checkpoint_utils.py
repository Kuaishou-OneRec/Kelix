from typing import Dict, List, Any

import os
import re
import torch
import torch.distributed as dist

from recovlm.utils.common import print_rank_0

def _dp_rank_mapping(source_dp_size, target_dp_size):
    """
    Helper function to get rank map for source config to target config.

    Note: this function temporarily only support expansion case.
    """
    dp_rank_mapping = {}
    multiple_coefficient = target_dp_size//source_dp_size
    for i in range(source_dp_size):
        dp_rank_mapping[i] = [i*multiple_coefficient+j for j in range(multiple_coefficient)]
    return dp_rank_mapping

def get_max_source_rank(source_path):
    if not os.path.isdir(source_path):
        return None
    max_source_rank = -1
    file_pattern = re.compile(r"_(\d+)_")

    for filename in os.listdir(source_path):
        match = file_pattern.search(filename)
        if match is not None:
            source_rank = int(match.group(1))
            max_source_rank = max(max_source_rank, source_rank)

    if max_source_rank == -1:
        return None

    return max_source_rank+1

def load_model_buffer(model_states: Dict[Any, Any], model: torch.nn.Module):
    module_state_dict = model_states['module']
    model_buffer_dict = dict(model.named_buffers())
    for name, value in module_state_dict.items():
        if name in model_buffer_dict:
            model_buffer_dict[name] = value

def reshard_params_using_offset(flat_tensors: List[torch.Tensor], 
                                offset_list: List[int],
                                source_dp_rank: int,
                                target_dp_rank: int,
                                target_model: torch.nn.Module,
                                source_dp_size: int, 
                                target_dp_size: int):
    """
    This function is mainly used to reshard parameters with respect to the offset list, and scatter the new parameter
    partitions to related ranks.

    Workflow:
        1. Model initialize with deepspeed.zero.Init context
        2. Get reshard parameter with respect to target size and offset.
        3. Assign the reshard parameters to the random initialized model.
    
    Note: 
        1. Currently, we just support llama-65B model and llama-13B model, the numel of all the parameters are 
        a multiple of 4096, so no zero-3 padding in this case.
        2. We support expansion/shrinkage case now.
    """
    if target_dp_size >= source_dp_size:
        # resource expansion
        assert target_dp_size%source_dp_size==0, 'temporarily we just support the situation that target size is multiple of source size.'
        assert len(flat_tensors)==1, 'The flat_tensors should only contain exactly one tensor in the expansion case.'
        tensor_to_process = flat_tensors[0]
        start = 0
        param_list = list(target_model.parameters())
        rank_mapping = _dp_rank_mapping(source_dp_size, target_dp_size)
        assert len(param_list)==len(offset_list), 'param_list should have same length as offset_list'
        for offset, param in zip(offset_list, param_list):
            original_tensor_shard = tensor_to_process.narrow(0, start, offset)
            local_offset = rank_mapping[source_dp_rank].index(target_dp_rank)
            target_partition_size = param.ds_tensor.numel()
            param.ds_tensor.data = original_tensor_shard.narrow(0,
                    target_partition_size*local_offset,
                    target_partition_size).to(param.ds_tensor.dtype)
            start += offset
        del tensor_to_process
    else:
        # resource shrink
        assert source_dp_size%target_dp_size ==0, 'temporarily we just support the situation that source size is multiple of target size.'
        assert len(flat_tensors[0])==1, 'The flat_tensors should only contain exactly one tensor in the shrink case.'
        start = 0
        param_list = list(target_model.parameters())
        assert len(param_list)==len(offset_list), 'param_list should have same length as offset_list'
        tensor_to_process_list = [tensor_list[0] for tensor_list in flat_tensors]
        for offset, param in zip(offset_list, param_list):
            original_tensor_shard_list = [tensor_to_process.narrow(0, start, offset) for tensor_to_process in tensor_to_process_list]
            concatenated_tensor = torch.cat(original_tensor_shard_list, dim=0)
            assert concatenated_tensor.numel() == param.ds_tensor.numel(), "Concatenated tensor size does not match param.ds_tensor size"
            param.ds_tensor.data = concatenated_tensor.to(param.ds_tensor.dtype)
            start += offset
        del tensor_to_process_list

def tensor_map_with_padding(source_world_size, target_world_size, real_numel):
    # Calculate partition size for source and target configurations
    source_padding_to_next_multiple = (source_world_size - real_numel % source_world_size) % source_world_size
    source_total_numel_with_padding = real_numel + source_padding_to_next_multiple
    source_partition_size = source_total_numel_with_padding // source_world_size
    assert source_total_numel_with_padding%source_world_size ==0
    target_padding_to_next_multiple = (target_world_size - real_numel % target_world_size) % target_world_size
    target_total_numel_with_padding = real_numel + target_padding_to_next_multiple
    target_partition_size = target_total_numel_with_padding // target_world_size  
    assert target_total_numel_with_padding%target_world_size ==0

    # Pre-initialize all target ranks with empty mappings
    target_to_source_file_mapping = {i: [] for i in range(target_world_size)}
    offset_mapping = {i: [] for i in range(target_world_size)}

    # Determine the mapping of source ranks to target ranks
    for source_rank in range(source_world_size):
        start_index = source_rank * source_partition_size
        end_index = (source_rank + 1) * source_partition_size

        # Calculate which target ranks this source rank will contribute to
        start_target_rank = start_index // target_partition_size
        end_target_rank = min((end_index - 1) // target_partition_size, target_world_size - 1)

        for target_rank in range(start_target_rank, end_target_rank + 1):
            target_to_source_file_mapping[target_rank].append(source_rank)

            # Calculate the start and end offset within the source rank for this target rank
            start_offset = max(start_index, target_rank * target_partition_size) - start_index
            end_offset = min(end_index, (target_rank + 1) * target_partition_size) - start_index - 1

            offset_mapping[target_rank].append([start_offset, end_offset])

    return target_to_source_file_mapping, offset_mapping , source_partition_size

def generate_flatten_offset_mapping(offset_mapping_list, target_to_source_file_mapping_list, param_length_list):
    flatten_offset_mapping_list = []

    # Initialize accumulated offsets for each source rank
    acc_offsets = defaultdict(int)

    for offset_mapping, target_to_source_mapping, param_length in zip(offset_mapping_list, target_to_source_file_mapping_list, param_length_list):
        flatten_offset_mapping = {}
        
        for rank, source_ranks_offsets in offset_mapping.items():
            source_ranks = target_to_source_mapping[rank]

            for i, source_rank in enumerate(source_ranks):
                start, end = source_ranks_offsets[i]

                # Update both start and end offsets with the accumulated offset of the corresponding source rank
                start += acc_offsets[source_rank]
                end += acc_offsets[source_rank]

                flatten_offset_mapping[rank] = flatten_offset_mapping.get(rank, []) + [[start, end]]
        
        # Update the accumulated offsets for each source rank
        for source_rank in range(len(acc_offsets)):
            acc_offsets[source_rank] += param_length

        flatten_offset_mapping_list.append(flatten_offset_mapping)

    return flatten_offset_mapping_list

def reshard_params_using_real_numel(optimizer_dict,
                                    target_to_source_file_mapping_list,
                                    offset_mapping_list,
                                    real_numel_list,
                                    local_rank, 
                                    target_model, 
                                    old_world_size, 
                                    world_size):
    param_list = list(target_model.parameters())
    #todo:Rethink about padding for more than one rank.
    assert len(param_list)==len(target_to_source_file_mapping_list), 'param_list should have same length as target_to_source_file_mapping_list'
    assert len(param_list)==len(offset_mapping_list), 'param_list should have same length as offset_mapping_list'
    assert len(param_list)==len(real_numel_list), 'param_list should have same length as real_numel_list'
    for file_mapping,offset_mapping, param, real_numel in zip(target_to_source_file_mapping_list,offset_mapping_list, param_list,real_numel_list):
        source_rank_list = file_mapping[local_rank]
        if source_rank_list:  # Check if source_rank_list is not empty
            # dist.barrier()
            # if dist.get_rank() == 0:
            #     st()
            # dist.barrier()
            print_rank_0(f'loading with={source_rank_list},param={param},offset_mapping[local_rank]={offset_mapping[local_rank]}', local_rank)
            optimizer_dict_mapping = [optimizer_dict[source_rank] for source_rank in source_rank_list]
            fp32_flat_groups_list = [[flat_tensor.cuda() for flat_tensor in optimizer_dict_single['optimizer_state_dict']['fp32_flat_groups']] for optimizer_dict_single in optimizer_dict_mapping]
            tensor_to_process_list = [tensor_list[0] for tensor_list in fp32_flat_groups_list]
            assert len(tensor_to_process_list)==len(offset_mapping[local_rank]), 'tensor_to_process_list should have same length as offset_mapping[local_rank]'
            original_tensor_shard_list = [tensor_to_process.narrow(0, start, end-start+1) 
                                        for tensor_to_process, (start, end) in zip(tensor_to_process_list, offset_mapping[local_rank])]
            concatenated_tensor = torch.cat(original_tensor_shard_list, dim=0)
            #padding concatenated_tensor to the same size as the param.ds_tensor.numel()

        else:
            print("source_rank_list is empty")
            concatenated_tensor = torch.empty(param.ds_tensor.numel(), device=param.ds_tensor.device, dtype=param.ds_tensor.dtype)
        if concatenated_tensor.numel() < param.ds_tensor.numel():
            padding_size = param.ds_tensor.numel() - concatenated_tensor.numel()
            print_rank_0(f'padding with padding_size={padding_size}', local_rank)
            padding = torch.empty(padding_size, device=concatenated_tensor.device, dtype=concatenated_tensor.dtype)
            concatenated_tensor = torch.cat([concatenated_tensor, padding], dim=0)
        assert concatenated_tensor.numel() == param.ds_tensor.numel(), "Concatenated tensor size does not match param.ds_tensor size"
        param.ds_tensor.data = concatenated_tensor.to(param.ds_tensor.dtype)
        del optimizer_dict_mapping
        del fp32_flat_groups_list
        del tensor_to_process_list
        del original_tensor_shard_list

def reshard_optimizer_states_using_real_numel(optimizer_dict, 
                                              target_to_source_file_mapping_list,
                                              offset_mapping_list, 
                                              optimizer, 
                                              local_rank, 
                                              target_model, 
                                              old_world_size, 
                                              world_size,
                                              load_file_dict):

    param_list = list(target_model.parameters())
    assert len(param_list)==len(target_to_source_file_mapping_list), 'param_list should have same length as target_to_source_file_mapping_list'
    assert len(param_list)==len(offset_mapping_list), 'param_list should have same length as offset_mapping_list'
    target_optimizer_states = optimizer.state_dict()['optimizer_state_dict']['state']
    for param_id, state_dict in target_optimizer_states.items():
        for key, value in state_dict.items():
            if isinstance(value, torch.Tensor):
                target_tensor_list = []
                for file_mapping,offset_mapping, param in zip(target_to_source_file_mapping_list,offset_mapping_list, param_list):
                #Source:optimizer_dict
                #Target:state_dict[key].data
                    source_rank_list = file_mapping[local_rank]
                    print_rank_0(f'key={key},loading with={source_rank_list},param={param},offset_mapping[local_rank]={offset_mapping[local_rank]}', local_rank)
                    if source_rank_list:  
                        optimizer_dict_mapping = [optimizer_dict[source_rank] for source_rank in source_rank_list]
                        source_optimizer_states = [saved_optimizer['optimizer_state_dict']['optimizer_state_dict']['state'] for saved_optimizer in optimizer_dict_mapping]
                        tensor_to_process_list = [source_optimizer_states_single[param_id][key].clone().detach() for source_optimizer_states_single in source_optimizer_states]
                        assert len(tensor_to_process_list)==len(offset_mapping[local_rank]), 'tensor_to_process_list should have same length as offset_mapping[local_rank]'
                        original_tensor_shard_list = [tensor_to_process.narrow(0, start, end-start+1) 
                                                    for tensor_to_process, (start, end) in zip(tensor_to_process_list, offset_mapping[local_rank])]
                        concatenated_tensor = torch.cat(original_tensor_shard_list, dim=0)
                        #padding concatenated_tensor to the same size as the param.ds_tensor.numel()
                    else:
                        concatenated_tensor = torch.empty(param.ds_tensor.numel(), device=param.ds_tensor.device, dtype=param.ds_tensor.dtype)
                    if concatenated_tensor.numel() < param.ds_tensor.numel():
                        padding_size = param.ds_tensor.numel() - concatenated_tensor.numel()
                        padding = torch.empty(padding_size, device=concatenated_tensor.device, dtype=concatenated_tensor.dtype)
                        concatenated_tensor = torch.cat([concatenated_tensor, padding], dim=0)
                    target_tensor_list.append(concatenated_tensor)
                target_flat_tensor = torch.cat(target_tensor_list, 0)
                state_dict[key].data = target_flat_tensor.to(state_dict[key].device)
                del tensor_to_process_list
                del target_tensor_list

            else:
                state_dict[key] = value

def reshard_optimizer_states(saved_optimizer_states: Dict[Any, Any],
                             offset_list: List[int],
                             optimizer: torch.optim.Optimizer,
                             source_dp_rank: int,
                             target_dp_rank: int,
                             source_dp_size: int,
                             target_dp_size: int):

    if target_dp_size >= source_dp_size:
        # resource expansion
        assert target_dp_size%source_dp_size==0, 'temporarily we just support the situation that target size is multiple of source size.'
        target_optimizer_states = optimizer.state_dict()['optimizer_state_dict']['state']
        source_optimizer_states = saved_optimizer_states['optimizer_state_dict']['optimizer_state_dict']['state']
        def _reshard_tensor_expand(offset_list: List[int],
                            source_flat_tensor: torch.Tensor,
                            source_dp_rank: int,
                            target_dp_rank: int,
                            source_dp_size: int,
                            target_dp_size: int):
            rank_mapping = _dp_rank_mapping(source_dp_size, target_dp_size)
            local_offset = rank_mapping[source_dp_rank].index(target_dp_rank)
            target_tensor_list = []
            start = 0
            multiple_coefficient = target_dp_size//source_dp_size
            for offset in offset_list:
                # for llama-65B or llama-13B, there is no padding for zero3, so we could get target_size from offset.
                target_size = offset//multiple_coefficient
                target_tensor_list.append(source_flat_tensor.narrow(0, start+target_size*local_offset, target_size))
                start += offset
            target_flat_tensor = torch.cat(target_tensor_list, 0)
            return target_flat_tensor
        
        for param_id, state_dict in target_optimizer_states.items():
            for key, value in state_dict.items():
                if isinstance(value, torch.Tensor):
                    tensor_to_process = source_optimizer_states[param_id][key].clone().detach()
                    reshard_tensor = _reshard_tensor_expand(offset_list, tensor_to_process, source_dp_rank, target_dp_rank, source_dp_size, target_dp_size)
                    # print(state_dict[key].device, reshard_tensor.device)
                    state_dict[key].data = reshard_tensor.to(state_dict[key].device)
                    del tensor_to_process
                else:
                    state_dict[key] = value
    else:
        # resource shrink
        assert source_dp_size%target_dp_size == 0, 'temporarily we just support the situation that source size is multiple of target size.'
        target_optimizer_states = optimizer.state_dict()['optimizer_state_dict']['state']
        source_optimizer_states = [saved_optimizer['optimizer_state_dict']['optimizer_state_dict']['state'] for saved_optimizer in saved_optimizer_states]
        def _reshard_tensor_shrink(offset_list: List[int], 
                            source_flat_tensor: torch.Tensor):
            target_tensor_list = []
            start = 0
            for offset in offset_list:
                # for llama-65B or llama-13B, there is no padding for zero3, so we could get target_size from offset.
                for source_flat in source_flat_tensor:
                    target_tensor_list.append(source_flat.narrow(0, start, offset))
                start += offset
            target_flat_tensor = torch.cat(target_tensor_list, 0)
            return target_flat_tensor
        for param_id, state_dict in target_optimizer_states.items():
            for key, value in state_dict.items():
                if isinstance(value, torch.Tensor):
                    tensor_to_process_list = [source_optimizer_states_single[param_id][key].clone().detach() for source_optimizer_states_single in source_optimizer_states]
                    reshard_tensor = _reshard_tensor_shrink(offset_list, tensor_to_process_list)
                    state_dict[key].data = reshard_tensor.to(state_dict[key].device)
                    del tensor_to_process_list
                else:
                    state_dict[key] = value

def load_state_dict_general(args, model, dataloader, source_path):
    old_world_size = get_max_source_rank(source_path)
    if old_world_size is None:
        print_rank_0(f'dont exist checkpoint in path: {source_path}', args.global_rank)
        return None
    world_size = dist.get_world_size()
    local_rank = dist.get_rank()
    print_rank_0("old_world_size " + str(old_world_size),local_rank)
    print_rank_0("world_size   " + str(world_size),local_rank)
    # TODO: compatible to zero-2
    model_state_dict = torch.load(
        f'{source_path}/zero_pp_rank_0_mp_rank_00_model_states.pt',
        map_location=torch.device('cpu')
    )
    load_model_buffer(model_state_dict, model)
    target_to_source_file_mapping_list = []
    offset_mapping_list = []
    param_length_list = []
    flatten_offset_mapping_list = []
    load_file_dict = []
    if old_world_size % world_size == 0 or world_size % old_world_size == 0:
        print_rank_0("old_world_size and world_size are in an integer multiple relationship.",local_rank)
        if (world_size >= old_world_size):
            print_rank_0("expand/same the rank",local_rank)
            world_size_times = world_size//old_world_size
            source_rank = local_rank//world_size_times
            optimizer_dict = torch.load(f'{source_path}/bf16_zero_pp_rank_{source_rank}_mp_rank_00_optim_states.pt', map_location=torch.device('cpu'))
            fp32_flat_groups = optimizer_dict['optimizer_state_dict']['fp32_flat_groups']
            fp32_flat_groups = [flat_tensor.cuda() for flat_tensor in fp32_flat_groups]
            offset_list = optimizer_dict['optimizer_state_dict']['parameter_offset'][0]
            reshard_params_using_offset([fp32_flat_groups[0]], offset_list,
                    local_rank//world_size_times, local_rank, model, old_world_size, world_size)
        else:
            print_rank_0("shrinking the rank",local_rank)
            world_size_times = old_world_size // world_size
            rank_mapping = _dp_rank_mapping(world_size,old_world_size)
            source_rank_list = rank_mapping[local_rank]
            #In this case, optimizer_dict and model_state_dict are list.
            optimizer_dict = [torch.load(f'{source_path}/bf16_zero_pp_rank_{source_rank}_mp_rank_00_optim_states.pt', map_location=torch.device('cpu')) for source_rank in source_rank_list]
            fp32_flat_groups_list = [[flat_tensor.cuda() for flat_tensor in optimizer_dict_single['optimizer_state_dict']['fp32_flat_groups']] for optimizer_dict_single in optimizer_dict]
            offset_list = optimizer_dict[0]['optimizer_state_dict']['parameter_offset'][0]
            reshard_params_using_offset(fp32_flat_groups_list, offset_list,local_rank//world_size_times, local_rank, model, old_world_size, world_size)
    else:
        print_rank_0("old_world_size and world_size are NOT in an integer multiple relationship.",local_rank)
        real_numel_list = model_state_dict['parameter_real_numel'][0]
        # Calculate the mapping of source ranks to target ranks
        for real_numel in real_numel_list:
            target_to_source_file_mapping, offset_mapping,source_partition_size = tensor_map_with_padding(old_world_size, world_size, real_numel)
            target_to_source_file_mapping_list.append(target_to_source_file_mapping)
            offset_mapping_list.append(offset_mapping)
            param_length_list.append(source_partition_size)
            
        flatten_offset_mapping_list = generate_flatten_offset_mapping(offset_mapping_list,target_to_source_file_mapping_list,param_length_list)
        load_file_dict = defaultdict(set)
        for target_to_source_file_mapping in target_to_source_file_mapping_list:
            for key, value in target_to_source_file_mapping.items():
                load_file_dict[key].update(value)
        load_file_dict = dict(load_file_dict)
        # Load the checkpoint files
        optimizer_dict = {source_rank: torch.load(f'{source_path}/bf16_zero_pp_rank_{source_rank}_mp_rank_00_optim_states.pt', map_location=torch.device('cpu')) 
                    for source_rank in load_file_dict[local_rank]}
        #reshard weights for the target_model
        reshard_params_using_real_numel(optimizer_dict,target_to_source_file_mapping_list,flatten_offset_mapping_list,real_numel_list,local_rank, model, old_world_size, world_size)
    
    #for dataloader
    dataloader_path = f"{source_path}/dataloader_0.pt"
    if dataloader is not None and hasattr(dataloader, 'load_state_dict'):
        dataloader_state = torch.load(dataloader_path)
        dataloader.load_state_dict(dataloader_state)
        print_rank_0(f'load dataloader from path: {dataloader_path}', args.global_rank)
        if hasattr(dataloader, '_skip_indices') and dataloader._skip_indices:
            print_rank_0(f"Filtering samples with offsets: {', '.join(str(offset) for offset in dataloader._skip_indices)}", args.global_rank)
    
    return optimizer_dict, model_state_dict,target_to_source_file_mapping_list,flatten_offset_mapping_list,load_file_dict

def load_optim_state_dict_general(model,
                                  optimizer,
                                  optimizer_dict,
                                  load_model_state_dict,
                                  source_path,
                                  target_to_source_file_mapping_list,
                                  offset_mapping_list,load_file_dict):
    print_rank_0(f"before reshard: {model.optimizer.state_dict()['optimizer_state_dict']}", dist.get_rank())
    old_world_size = get_max_source_rank(source_path)
    if old_world_size is None:
        print_rank_0(f'dont exist checkpoint in path: {source_path}', dist.get_rank())
        return
    world_size = dist.get_world_size()
    local_rank = dist.get_rank()
    if old_world_size % world_size == 0 or world_size % old_world_size == 0:
        if (world_size >= old_world_size):
            print_rank_0("expand/same the rank", dist.get_rank())
            world_size_times = world_size//old_world_size
            source_rank = local_rank//world_size_times
            print_rank_0(f"loading optimizer: {optimizer_dict}", dist.get_rank())
            offset_list = optimizer_dict['optimizer_state_dict']['parameter_offset'][0]
            reshard_optimizer_states(
                optimizer_dict, offset_list, optimizer, source_rank,
                dist.get_rank(), old_world_size, world_size
            )
        else:
            print_rank_0("shrinking the rank",dist.get_rank())
            world_size_times = old_world_size // world_size
            rank_mapping = _dp_rank_mapping(world_size,old_world_size)
            source_rank_list = rank_mapping[local_rank]
            offset_list = optimizer_dict[0]['optimizer_state_dict']['parameter_offset'][0]
            reshard_optimizer_states(
                optimizer_dict, offset_list, optimizer, source_rank_list,
                dist.get_rank(), old_world_size, world_size
            )
            optimizer_dict = optimizer_dict[0]
    else:
        print_rank_0("Non-integer rank change", dist.get_rank())
        reshard_optimizer_states_using_real_numel(
            optimizer_dict, target_to_source_file_mapping_list,
            offset_mapping_list, optimizer,
            local_rank, model,old_world_size,
            world_size,load_file_dict
        )
        optimizer_dict = next(iter(optimizer_dict.values()))   
              
    model.loss_scaler = optimizer_dict['optimizer_state_dict']['loss_scaler']
    print_rank_0(f'loaded loss_scaler',local_rank)
    model.dynamic_loss_scale = optimizer_dict['optimizer_state_dict']['dynamic_loss_scale']
    print_rank_0(f'loaded dynamic_loss_scale',local_rank)
    model.overflow = optimizer_dict['optimizer_state_dict']['overflow']
    print_rank_0(f'loaded overflow',local_rank)
    if model.lr_scheduler is not None:    
        model.lr_scheduler.load_state_dict(load_model_state_dict['lr_scheduler'])
        print_rank_0(f'loaded lr_scheduler',local_rank)
    if model.random_ltd_enabled() and model.random_ltd_scheduler is not None and 'random_ltd' in load_model_state_dict:
        model.random_ltd_scheduler.load_state_dict(load_model_state_dict['random_ltd'])
        print_rank_0(f'loaded random_ltd_scheduler',local_rank)
    model.global_steps = load_model_state_dict['global_steps']
    print_rank_0(f'loaded global_steps',local_rank)
    model.global_samples = load_model_state_dict['global_samples']
    print_rank_0(f'loaded global_samples',local_rank)  

def load_optim_state_dict(model, optimizer, optimizer_dict, load_model_state_dict,source_path):
    print_rank_0(f"before reshard: {model.optimizer.state_dict()['optimizer_state_dict']}", dist.get_rank())
    old_world_size = get_max_source_rank(source_path)
    if old_world_size is None:
        print_rank_0(f'dont exist checkpoint in path: {source_path}', dist.get_rank())
        return
    world_size = dist.get_world_size()
    local_rank = dist.get_rank()

    if (world_size >= old_world_size):
        print_rank_0("expand/same the rank", dist.get_rank())
        world_size_times = world_size // old_world_size
        source_rank = local_rank // world_size_times
        print_rank_0(f"loading optimizer: {optimizer_dict}", dist.get_rank())
        offset_list = optimizer_dict['optimizer_state_dict']['parameter_offset'][0]
        reshard_optimizer_states(
            optimizer_dict, offset_list, optimizer, source_rank,
            dist.get_rank(), old_world_size, world_size
        )
    else:
        print_rank_0("shrinking the rank", dist.get_rank())
        world_size_times = old_world_size // world_size
        rank_mapping = _dp_rank_mapping(world_size,old_world_size)
        source_rank_list = rank_mapping[local_rank]
        offset_list = optimizer_dict[0]['optimizer_state_dict']['parameter_offset'][0]
        reshard_optimizer_states(
            optimizer_dict, offset_list, optimizer, source_rank_list,
            dist.get_rank(), old_world_size, world_size
        )
        optimizer_dict = optimizer_dict[0]
        load_model_state_dict = load_model_state_dict[0]

    model.loss_scaler = optimizer_dict['optimizer_state_dict']['loss_scaler']
    print_rank_0(f'loaded loss_scaler', local_rank)
    model.dynamic_loss_scale = optimizer_dict['optimizer_state_dict']['dynamic_loss_scale']
    print_rank_0(f'loaded dynamic_loss_scale', local_rank)
    model.overflow = optimizer_dict['optimizer_state_dict']['overflow']
    print_rank_0(f'loaded overflow', local_rank)


    if model.lr_scheduler is not None:
        model.lr_scheduler.load_state_dict(load_model_state_dict['lr_scheduler'])
        print_rank_0(f'loaded lr_scheduler',local_rank)
    if model.random_ltd_enabled() and model.random_ltd_scheduler is not None and 'random_ltd' in load_model_state_dict:
        model.random_ltd_scheduler.load_state_dict(load_model_state_dict['random_ltd'])
        print_rank_0(f'loaded random_ltd_scheduler',local_rank)
    model.global_steps = load_model_state_dict['global_steps']
    print_rank_0(f'loaded global_steps',local_rank)
    model.global_samples = load_model_state_dict['global_samples']
    print_rank_0(f'loaded global_samples',local_rank)

def load_model_from_dir(args, model, train_dataloader):
    source_path = ""
    if args.load_dir:
        with open(os.path.join(args.load_dir, 'latest'), 'r') as fd:
            latest_checkpoint_folder = fd.read()
        source_path = os.path.join(args.load_dir, latest_checkpoint_folder.rstrip('\n'))
        print_rank_0(f"init from {source_path}", args.global_rank)
    else:
        latest_file_in_output_dir = os.path.join(args.output_dir, 'latest') if args.output_dir else None
        if latest_file_in_output_dir and os.path.exists(latest_file_in_output_dir):
            with open(latest_file_in_output_dir, 'r') as fd:
                latest_checkpoint_folder = fd.read()
            source_path = os.path.join(args.output_dir, latest_checkpoint_folder.rstrip('\n'))
            print_rank_0(f"init from {source_path}", args.global_rank)
        else:
            print_rank_0(f"start from cold", args.global_rank)
            return None, None, None, None, None
    optimizer_dict, load_model_state_dict, target_to_source_file_mapping_list, offset_mapping_list, load_file_dict = \
        load_state_dict_general(args, model, train_dataloader, source_path)
    return optimizer_dict, load_model_state_dict, target_to_source_file_mapping_list, \
        offset_mapping_list,load_file_dict
