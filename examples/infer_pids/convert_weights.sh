#!/bin/bash
set -e

# 获取当前脚本所在目录的父目录的父目录的绝对路径
#PARENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

# 将父目录添加到PYTHONPATH
export PYTHONPATH=$PWD:$PYTHONPATH

# 检查输入参数
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <checkpoint_dir>"
    echo "Example: $0 /path/to/deepspeed_checkpoint"
    exit 1
fi

CHECKPOINT_DIR=$1
OUTPUT_DIR="${CHECKPOINT_DIR}/hf"

echo "Converting DeepSpeed checkpoint to PyTorch model..."
echo "Input checkpoint: $CHECKPOINT_DIR"
echo "Output directory: $OUTPUT_DIR"

# 创建输出目录
mkdir -p "$OUTPUT_DIR"

# 调用转换脚本
python recovlm/utils/convert_deepspeed_checkpoint.py \
    --input_dir "$CHECKPOINT_DIR" \
    --output_dir "$OUTPUT_DIR"

echo "Checkpoint conversion completed successfully!"
echo "PyTorch model saved to: $OUTPUT_DIR/pytorch_model.bin"
