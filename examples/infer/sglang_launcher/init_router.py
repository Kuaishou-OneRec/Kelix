from typing import Dict, Optional, List
import os
import subprocess
import threading
import argparse
import torch

def get_num_devices():
  return torch.cuda.device_count()

def get_arguments():
  parser = argparse.ArgumentParser()

  parser.add_argument("--tp", type=int, default=8,
                      help="The tensor parallel size of model.")

  parser.add_argument("--router_port", type=int, default=50000,
                      help="The start port to launch server.")

  parser.add_argument("--server_port", type=int, default=30000,
                      help="The start port to launch server.")

  args = parser.parse_args()

  return args

def get_hosts():
  hosts = []
  with open("/etc/mpi/hostfile") as f:
    for line in f:
      host, _ = line.strip().split()
      hosts.append(host)
  return hosts

def run_command_in_background(command):
    process = subprocess.Popen(
        command,
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

def init_router(port, urls: List[str] = None):
  cmd_args = [
    "python3", "-m", "sglang_router.launch_router", f"--port {port}", "--worker-urls",
  ]
  if urls:
    cmd_args.extend(urls)
  cmd = " ".join(cmd_args)
  print(cmd)
  process = run_command_in_background(cmd)
  return process

def main():
  args = get_arguments()
  num_devices = get_num_devices()
  if args.tp <= num_devices:
    assert num_devices % args.tp == 0, \
      f"num_devices ({num_devices}) must be divisible by tp_size ({args.tp})" \
      f"when tp < num_devices"
    num_servers = num_devices // args.tp
    urls = []
    hosts = get_hosts()
    for host in hosts:
        for idx in range(num_servers):
            urls.append(f"http://{host}:{args.server_port + idx}")
    print(urls)
    p = init_router(port=args.router_port, urls=urls)
    p.wait()
  else:
    assert args.tp % num_devices == 0, \
      f"tp_size ({args.tp}) must be divisible by num_devices ({num_devices})" \
      f"when tp > num_devices"
    nnodes = args.tp // num_devices
    urls = []
    hosts = get_hosts()
    for idx, host in enumerate(hosts):
      if idx % nnodes == 0:
        urls.append(f"http://{host}:{args.server_port}")
    print(urls)
    p = init_router(port=args.router_port, urls=urls)
    p.wait()

if __name__ == '__main__':
  main()
