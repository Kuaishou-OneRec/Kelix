from typing import Dict, Optional
import os
import subprocess
import threading
from mpi4py import MPI
import argparse
import torch

comm = MPI.COMM_WORLD

def get_num_devices():
  return torch.cuda.device_count()

def get_arguments():
  parser = argparse.ArgumentParser()

  parser.add_argument("--model-path", type=str, default=None,
                      help="The model path")
  parser.add_argument("--tp", type=int, default=8,
                      help="The tensor parallel size of model.")

  parser.add_argument("--port", type=int, default=30000,
                      help="The start port to launch server.")
  args, unknown_args = parser.parse_known_args()

  unknown_kwargs = {}
  i = 0
  while i < len(unknown_args):
      arg = unknown_args[i]
      if arg.startswith('--'):
          key = arg[2:]
          if i + 1 < len(unknown_args) and not unknown_args[i + 1].startswith('--'):
              value = unknown_args[i + 1]
              unknown_kwargs[key] = value
              i += 2
          else:
              unknown_kwargs[key] = None
              i += 1
      else:
          i += 1

  return args, unknown_args

def get_host():
  hosts = []
  if comm.Get_rank() == 0:
    with open("/etc/mpi/hostfile") as f:
      for line in f:
        host, _ = line.strip().split()
        hosts.append(host)
  else:
    hosts = None
  hosts = comm.bcast(hosts, root=0)
  return hosts[comm.Get_rank()]

def run_command_in_background(command, env):
    process = subprocess.Popen(
        command,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        shell=True,
        text=True,
        bufsize=1,
        universal_newlines=True,
    )

    def output_reader(stream, stream_type):
        for line in stream:
            print(f"[{stream_type}] {line.strip()}")

    stdout_thread = threading.Thread(
      target=output_reader, args=(process.stdout, "stdout"))
    stderr_thread = threading.Thread(
      target=output_reader, args=(process.stderr, "stderr"))

    stdout_thread.start()
    stderr_thread.start()

    return process

def init_server(args,
                extra_kwargs: Optional[Dict[str, str]] = None,
                port: int = 30000,
                cuda_visiable_devices: str = "0,1,2,3,4,5,6,7"):
  env = os.environ.copy()
  env["CUDA_VISIBLE_DEVICES"] = cuda_visiable_devices
  host = get_host()
  cmd_args = [
    "python3", "-m", "sglang.launch_server",
    f"--model-path {args.model_path}",
    f"--tp {args.tp}",
    f"--host {host}",
    f"--port {port}"
  ]
  if extra_kwargs:
    for key, value in extra_kwargs.items():
      cmd_args.append(f"--{key} {value}")
  cmd = " ".join(cmd_args)
  process = run_command_in_background(cmd, env=env)
  return process

def init_dist_server(args, host, port):
  # Load hosts file
  # parse master host
  comm.bcast(hosts, root=0)
  env = os.environ.copy()
  env["CUDA_VISIBLE_DEVICES"] = cuda_visiable_devices
  cmd_args = [
    "python3", "-m", "sglang.launch_server",
    "--model-path", args.model_path,
    "--tp", args.tp,
    "--host", args.host,
    "--port", port
  ]
  subprocess.run(cmd_args, shell=True, env=env)

def main():
  args, extra_kwargs = get_arguments()
  num_devices = get_num_devices()
  if args.tp <= num_devices:
    assert num_devices % args.tp == 0, \
      f"num_devices ({num_devices}) must be divisible by tp_size ({args.tp})" \
      f"when tp < num_devices"
    num_servers = num_devices // args.tp
    pool = []
    for idx in range(num_servers):
        cuda_visiable_devices = ",".join(
            [str(idx * args.tp + offset) for offset in range(args.tp)]
        )
        p = init_server(
            args, extra_kwargs=extra_kwargs, port=args.port + idx,
            cuda_visiable_devices=cuda_visiable_devices
        )
        pool.append(p)
    for p in pool:
      p.wait()
  else:
    assert num_devices % args.tp == 0, \
      f"tp_size ({args.tp}) must be divisible by num_devices ({num_devices})" \
      f"when tp > num_devices"

if __name__ == '__main__':
  main()
