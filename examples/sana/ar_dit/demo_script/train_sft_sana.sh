#!/bin/bash
# Kelix-DiT training demo.
#
# Distributed training (OpenMPI / mpirun) of the Kelix-DiT de-tokenizer via
# recipes/sana/train_sana_ar_dit.py. Uses a frozen Kelix AR model to produce
# condition embeddings and trains the DiT with flow matching + DC-AE VAE.
#
# Configure by setting the environment variables below (all overridable):
#   MODEL_DIR        Kelix-DiT checkpoint dir (HF repo id or local path)
#   KEYE_AR_DIR      Frozen Kelix AR model dir (condition provider)
#   VAE_DIR          Frozen DC-AE VAE dir (HF repo id or local path)
#   DATASET_CONFIG   JSON dataset config path (its `sources` field points to
#                    your data index)
#   OUTPUT_DIR       Where checkpoints/logs are written
#   HOSTFILE         MPI hostfile (defaults to a single-node "hostfile")
#   NPROC_PER_NODE   GPU slots per host (default: 8)
#
# Usage:
#   bash examples/sana/ar_dit/demo_script/train_sft_sana.sh
#   MODEL_DIR=/path/to/dit KEYE_AR_DIR=/path/to/sft VAE_DIR=/path/to/vae \
#     DATASET_CONFIG=/path/to/cfg.json \
#     bash examples/sana/ar_dit/demo_script/train_sft_sana.sh
set -euo pipefail

# --- Repo root on PYTHONPATH so `muse` / `recipes` import cleanly ---
REPO_ROOT="$(cd "$(dirname "$0")/../../../.." && pwd)"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH}"
cd "${REPO_ROOT}"

# --- Config (env-overridable) ---
MODEL_DIR="${MODEL_DIR:-OpenOneRec/Kelix-DiT}"
KEYE_AR_DIR="${KEYE_AR_DIR:-OpenOneRec/Kelix-SFT}"
VAE_DIR="${VAE_DIR:-Efficient-Large-Model/SANA1.5_1.6B_1024px_diffusers/vae}"
DATASET_CONFIG="${DATASET_CONFIG:-examples/sana/ar_dit/demo_script/train_sft_sana.json}"
OUTPUT_DIR="${OUTPUT_DIR:-./output/sft_sana_train_demo}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"

# Default single-node hostfile if none provided.
HOSTFILE="${HOSTFILE:-hostfile}"
if [ ! -f "${HOSTFILE}" ]; then
    echo "localhost slots=${NPROC_PER_NODE}" > "${HOSTFILE}"
fi

mkdir -p "${OUTPUT_DIR}"

comment="sana_t2i_dit_train"
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
    bash -c "python3 recipes/sana/train_sana_ar_dit.py \
            --keye-ar-dir ${KEYE_AR_DIR} \
            --model-dir ${MODEL_DIR} \
            --vae-dir ${VAE_DIR} \
            --max-condition-length 720 \
            --output-dir ${OUTPUT_DIR} \
            --allow-random-init-params 'diffusion_connector.0.weight,diffusion_connector.0.bias,diffusion_connector.2.weight,diffusion_connector.2.bias,diffusion_connector.3.weight' \
            --skip-load-params 'y_embedder.y_embedding' \
            --dataset-config ${DATASET_CONFIG} \
            --resolution-budgets '1024:24' \
            --learning-rate 1e-4 \
            --min-lr 1e-4 \
            --num-decay-steps 10000 \
            --weight-decay 0.0 \
            --image-size 1024 \
            --beta1 0.9 \
            --beta2 0.95 \
            --batch-size 24 \
            --lr-scheduler-type cosine_v2 \
            --num-warmup-steps 100 \
            --num-training-steps 300000 \
            --model-config-overrides model_max_length=720 \
            --condition-on-special-tokens \
            --save-checkpoint-per-step 500 \
            --logging-per-step 20 \
            --clip-range 20 \
            --fp32-weight \
            --fp32-reduce \
            --seed 1917 \
            --global-step 0 \
            --enable-gradient-checkpointing \
            --prefetch-params-in-forward \
            --comment '${comment}' \
            --commit-id ${git_hash}"
