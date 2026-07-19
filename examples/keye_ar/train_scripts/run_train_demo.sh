#!/bin/bash
# Kelix-LLM (AR) training demo.
#
# Distributed training (OpenMPI / mpirun) of the Kelix AR model
# (Kelix-Tok + Kelix-LLM) via recipes/ar/train_ar.py.
#
# Configure by setting the environment variables below (all overridable):
#   MODEL_DIR        Kelix AR checkpoint dir (HF repo id or local path)
#   DATASET_CONFIG   JSON dataset config path (its `sources` field points to
#                    your data index)
#   OUTPUT_DIR       Where checkpoints/logs are written
#   HOSTFILE         MPI hostfile (defaults to a single-node "hostfile")
#   NPROC_PER_NODE   GPU slots per host (default: 8)
#
# Usage:
#   bash examples/keye_ar/train_scripts/run_train_demo.sh
#   MODEL_DIR=/path/to/ckpt DATASET_CONFIG=/path/to/cfg.json \
#     bash examples/keye_ar/train_scripts/run_train_demo.sh
set -euo pipefail

# --- Repo root on PYTHONPATH so `muse` / `recipes` import cleanly ---
REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH}"
cd "${REPO_ROOT}"

# --- Config (env-overridable) ---
MODEL_DIR="${MODEL_DIR:-OpenOneRec/Kelix-SFT}"
DATASET_CONFIG="${DATASET_CONFIG:-examples/keye_ar/train_scripts/run_train_demo.json}"
OUTPUT_DIR="${OUTPUT_DIR:-./output/keye_ar_train_demo}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"

# Default single-node hostfile if none provided.
HOSTFILE="${HOSTFILE:-hostfile}"
if [ ! -f "${HOSTFILE}" ]; then
    echo "localhost slots=${NPROC_PER_NODE}" > "${HOSTFILE}"
fi

mkdir -p "${OUTPUT_DIR}"

comment="keye_ar_train"
git_hash="$(git rev-parse --short HEAD 2>/dev/null || echo unknown)"

echo "Output: ${OUTPUT_DIR}"
{
    echo "$(date '+%Y-%m-%d %H:%M:%S')"
    echo "script: $0"
    echo "commit_id: ${git_hash}"
    echo "========================="
} >> "${OUTPUT_DIR}/task_info.log"

# Pick the primary network interface for NCCL/MPI (best-effort).
TCP_NIC="${TCP_NIC:-$(ifconfig | grep -B1 " $(hostname -i) " | grep -o '^\w*' || true)}"
export LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"

MASTER_ADDR="${MASTER_ADDR:-$(hostname -I | awk '{print $1}')}"
MASTER_PORT="${MASTER_PORT:-8499}"

# Clean NCCL env (no IB assumptions; works on single-node multi-GPU).
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export TOKENIZERS_PARALLELISM=false

mpirun --allow-run-as-root \
    -hostfile "${HOSTFILE}" \
    -mca btl self,tcp -mca pml ob1 \
    -mca btl_tcp_if_include "${TCP_NIC}" \
    -mca oob_tcp_if_include "${TCP_NIC}" \
    -x OMPI_MCA_btl=self,tcp \
    -x OMPI_MCA_pml=ob1 \
    -x OMPI_MCA_btl_tcp_if_include="${TCP_NIC}" \
    -x OMPI_MCA_oob_tcp_if_include="${TCP_NIC}" \
    -x NCCL_DEBUG="${NCCL_DEBUG}" \
    -x NCCL_SOCKET_IFNAME="${TCP_NIC}" \
    -x PYTHONIOENCODING=utf-8 \
    -x LD_LIBRARY_PATH="${LD_LIBRARY_PATH}" \
    -x PATH \
    -x PYTHONPATH="${PYTHONPATH}" \
    -x MASTER_ADDR="${MASTER_ADDR}" \
    -x MASTER_PORT="${MASTER_PORT}" \
    -x TOKENIZERS_PARALLELISM=false \
    bash -c "python3 recipes/ar/train_ar.py \
            --model-dir ${MODEL_DIR} \
            --model-name KeyeARModel \
            --output-dir ${OUTPUT_DIR} \
            --dataset-config ${DATASET_CONFIG} \
            --learning-rate 1e-4 \
            --weight-decay 0.0 \
            --beta1 0.9 \
            --beta2 0.95 \
            --model-dtype bfloat16 \
            --chuncked-loss-compute-size 1024 \
            --warmup-steps 1000 \
            --lr-scheduler cosine \
            --min-lr 1e-6 \
            --freeze-params 'visual_tokenizer.' \
            --logging-per-step 20 \
            --max-steps 2500000 \
            --save-checkpoint-per-step 1000 \
            --seed 19260817 \
            --max-length 1200 \
            --enable-gradient-checkpointing \
            --comment '${comment}' \
            --commit-id ${git_hash}"
