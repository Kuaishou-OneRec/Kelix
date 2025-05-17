import subprocess
import re

def get_cpu_topology():
    numa_info = subprocess.run(
        ["numactl", "--hardware"], 
        capture_output=True, 
        text=True
    ).stdout

    numa_map = {}
    for line in numa_info.split('\n'):
        if "node" in line and "cpus" in line:
            parts = re.split(r'\s+', line.strip())
            node = int(parts[1])
            cpus = list(map(int, parts[3:]))
            numa_map[node] = cpus

    logical_cores = []
    with open('/proc/cpuinfo') as f:
        cpu_data = f.read()
    for processor_block in cpu_data.split('\n\n'):
        if not processor_block:
            continue
        processor, physical_id, core_id = -1, -1, -1
        for line in processor_block.split('\n'):
            if 'processor' in line:
                processor = int(line.split(':')[1].strip())
            elif 'physical id' in line:
                physical_id = int(line.split(':')[1].strip())
            elif 'core id' in line:
                core_id = int(line.split(':')[1].strip())
        logical_cores.append((processor, physical_id, core_id))

    return numa_map, logical_cores 


def split_cpus_by_physical(numa_map, cores_list, local_world_size):
    num_numa = len(numa_map)
    numa_map = {}
    for numa_id in range(num_numa):
        cores = [core for core in cores_list if core[1] == numa_id]
        cores = sorted(cores, key=lambda x: x[2])
        numa_map[numa_id] = cores
    size_per_numa = local_world_size // num_numa
    result = {}
    for rank in range(local_world_size):
        numa_id = rank // size_per_numa
        num_cores_per_rank = len(numa_map[numa_id]) // size_per_numa
        begin = rank * num_cores_per_rank
        end = (rank + 1) * num_cores_per_rank
        infos = numa_map[numa_id][begin:end]
        result[rank] = [info[0] for info in infos]
    return result


def get_numa_bind_info(local_rank, local_world_size):
    numa_map, cpu_cores = get_cpu_topology()
    infos = split_cpus_by_physical(numa_map, cpu_cores, local_world_size)
    return infos[local_rank]


if __name__ == "__main__":
    numa_map, cpu_cores = get_cpu_topology()
    print(numa_map)
    res = split_cpus_by_physical(numa_map, cpu_cores, 8)
    print(res)
