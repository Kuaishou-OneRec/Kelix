#!/bin/bash

# 设置错误处理
set -euo pipefail

# 定义日志文件位置
LOG_DIR="./tests/spforward_sf"
mkdir -p "$LOG_DIR"

# 定义要执行的命令数组
commands=(
    "bash ./tests/spforward_sf/test4.sh > $LOG_DIR/test4.out 2>&1"
    "bash ./tests/spforward_sf/test1.sh > $LOG_DIR/test1.out 2>&1"
    "bash ./tests/spforward_sf/test14.sh > $LOG_DIR/test14.out 2>&1"
)

# 存储所有后台进程的PID
pids=()

# 并发执行命令
echo "开始并发执行测试脚本..."
for cmd in "${commands[@]}"; do
    echo "执行: $cmd"
    eval "$cmd" &
    pids+=($!)
done

# 等待所有后台进程完成
echo "等待所有测试脚本执行完成..."
wait "${pids[@]}"

# 检查所有命令是否成功执行
for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
        echo "错误: 测试脚本执行失败，PID: $pid"
        exit 1
    fi
done

echo "所有测试脚本执行完成，开始运行比较脚本"
python3 tests/spforward_sf/compare.py