#!/usr/bin/env bash
# Run the inference/visualization demo for Keye AR + DiT
# Usage: modify the defaults below or call with env overrides, then run:
#   bash examples/sana/ar_dit/inference/run_infer_visualize_reconstruction.sh
# Example:
#   MODEL_DIR=/path/to/model VAEDIR=/path/to/vae KEYE_AR_DIR=/path/to/keye bash $0
# For DCP checkpoint:
#   MODEL_DIR=/path/to/dcp/checkpoint DCP_CKPT_DIR=/path/to/source/dir DCP_TAG=global_step8000 bash $0

sed 's/=1/=8/g' /etc/mpi/hostfile > /etc/mpi/hostfile_seq
set -euo pipefail

# Ensure project root is on PYTHONPATH so 'recipes' and 'muse' imports resolve
export PYTHONPATH="${PYTHONPATH:-.}"

# ---- Defaults (edit or override via env) ----
MODEL_DIR="/llm_reco_ssd/zhouyang12/models/muse/Sana_1600M_1024px/"
VAE_DIR="/llm_reco_ssd/zhouyang12/models/SANA1.5_1.6B_1024px_diffusers/vae/"
KEYE_AR_DIR="/mmu_mllm_hdd_2/zhouyang12/output/Keye/vqar_11.9/v2_stage3_1e-4_max1280/./step23000/global_step23000/muse_converted"
DATASET_CONFIG="examples/sana/ar_dit/run_ar_dit_lzx_4096_v2_1024im_multiscale.json"
PARQUET_PATH="/mmu_mllm_hdd_2/lingzhixin/recovlm_data/muse_v2/vis/vis_data1225.parquet"
OUTPUT_DIR="./vis_output"
RESULTS_DIR="./results"
NUM_IMAGES=1
DEVICE="cuda"           # set to "cuda" if running on GPU
DTYPE="bfloat16"        # float32 for CPU runs; bfloat16/float16 for GPU runs
NUM_SAMPLING_STEPS=20
FLOW_SHIFT=3.0
CFG_SCALE=1.0
MAX_CONDITION_LENGTH=2560
IMAGE_SIZE=1024
SEED=42
INITIALIZE_DIST=true  # initialize a local single-process dist group (set to true only if needed)
RANK=0
WORLD_SIZE=1
MODEL_CONFIG_OVERRIDES="caption_channels=4096 model_max_length=1280 y_norm_scale_factor=1 use_cross_attn_rope=True"  # Model config overrides, e.g., "caption_channels=4096 model_max_length=324"
DCP_CKPT_DIR="/mmu_mllm_hdd_2/lingzhixin/output/MuseV2/sana/ar_dit/exp11_run_ar_dit_multiscale_1280tokens_attnrope_128u"      # Source directory for DCP checkpoint conversion
DCP_TAG="global_step9000"             # Tag for DCP checkpoint (e.g., global_step8000)
TEACHER_FORCING=0



# Allow overrides from environment
MODEL_DIR=${MODEL_DIR:-$MODEL_DIR}
VAE_DIR=${VAE_DIR:-$VAE_DIR}
KEYE_AR_DIR=${KEYE_AR_DIR:-$KEYE_AR_DIR}
DATASET_CONFIG=${DATASET_CONFIG:-$DATASET_CONFIG}
PARQUET_PATH=${PARQUET_PATH:-$PARQUET_PATH}
OUTPUT_DIR=${OUTPUT_DIR:-$OUTPUT_DIR}
RESULTS_DIR=${RESULTS_DIR:-$RESULTS_DIR}
NUM_IMAGES=${NUM_IMAGES:-$NUM_IMAGES}
DEVICE=${DEVICE:-$DEVICE}
DTYPE=${DTYPE:-$DTYPE}
NUM_SAMPLING_STEPS=${NUM_SAMPLING_STEPS:-$NUM_SAMPLING_STEPS}
FLOW_SHIFT=${FLOW_SHIFT:-$FLOW_SHIFT}
CFG_SCALE=${CFG_SCALE:-$CFG_SCALE}
MAX_CONDITION_LENGTH=${MAX_CONDITION_LENGTH:-$MAX_CONDITION_LENGTH}
IMAGE_SIZE=${IMAGE_SIZE:-$IMAGE_SIZE}
SEED=${SEED:-$SEED}
INITIALIZE_DIST=${INITIALIZE_DIST:-$INITIALIZE_DIST}
RANK=${RANK:-$RANK}
WORLD_SIZE=${WORLD_SIZE:-$WORLD_SIZE}
MODEL_CONFIG_OVERRIDES=${MODEL_CONFIG_OVERRIDES:-$MODEL_CONFIG_OVERRIDES}
DCP_CKPT_DIR=${DCP_CKPT_DIR:-$DCP_CKPT_DIR}
DCP_TAG=${DCP_TAG:-$DCP_TAG}

# ---- Prepare flags ----
INIT_DIST_FLAG=""
if [ "$INITIALIZE_DIST" = true ] || [ "$INITIALIZE_DIST" = "True" ]; then
  INIT_DIST_FLAG="--initialize-dist --rank ${RANK} --world-size ${WORLD_SIZE}"
fi

# Prepare model config overrides flag
MODEL_CONFIG_OVERRIDES_FLAG=""
if [ -n "$MODEL_CONFIG_OVERRIDES" ]; then
  MODEL_CONFIG_OVERRIDES_FLAG="--model-config-overrides $MODEL_CONFIG_OVERRIDES"
fi

# Prepare DCP flags
DCP_FLAGS=""
if [ -n "$DCP_TAG" ]; then
  DCP_FLAGS="--dcp-tag $DCP_TAG"
  if [ -n "$DCP_CKPT_DIR" ]; then
    DCP_FLAGS="$DCP_FLAGS --dcp-ckpt-dir $DCP_CKPT_DIR"
  fi
fi



mkdir -p "${OUTPUT_DIR}"
mkdir -p "${RESULTS_DIR}"


export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

export PYTHONPATH=$PWD:$PYTHONPATH
             
source set_env.sh

hostfile=/etc/mpi/hostfile_seq
Port=$(cat /etc/ssh/ssh_config | grep 'Port' | cut -d'"' -f2)
np=$(cat $hostfile | cut -d'=' -f2 | awk '{sum += $0} END {print sum}')
TCP_NIC=$(ifconfig | grep -B1 " "$(hostname -i)" " | grep -o "^\w*")


MASTER_ADDR=$MY_NODE_IP
MASTER_PORT=8499

PYTHONPATH=. \
nohup mpirun --allow-run-as-root \
        -hostfile $hostfile \
        -mca btl self,tcp -mca pml ob1 \
        -mca plm_rsh_num_concurrent 600 \
        -mca routed_radix 600 \
        -mca btl_tcp_if_include $TCP_NIC \
        -mca oob_tcp_if_include $TCP_NIC \
        -mca btl_openib_allow_ib false \
        -mca opal_set_max_sys_limits 1 \
        -x OMPI_MCA_btl=self,tcp \
        -x OMPI_MCA_pml=ob1 \
        -x OMPI_MCA_btl_tcp_if_include=$TCP_NIC \
        -x OMPI_MCA_oob_tcp_if_include=$TCP_NIC \
        -x OMPI_MCA_btl_openib_allow_ib=false \
        -x NCCL_IB_DISABLE=0 \
        -x NCCL_IB_GID_INDEX=3 \
        -x NCCL_SOCKET_IFNAME=$TCP_NIC \
        -x NCCL_IB_HCA=mlx5 \
        -x NCCL_DEBUG=WARN \
        -x NCCL_IB_QPS_PER_CONNECTION=4 \
        -x NCCL_NET_OVERHEAD=1000 \
        -x NCCL_IB_TIMEOUT=20 \
        -x LD_PRELOAD=$LD_PRELOAD \
        -x http_proxy="" \
        -x https_proxy="" \
        -x HOROVOD_MPI_THREADS_DISABLE=1 \
        -x MPI_THREAD_SINGLE=1 \
        -x NO_COLOR=1 \
        -x TERM=dumb \
        -x COLORTERM=0 \
        -x PYTHONIOENCODING=utf-8 \
        -x LD_LIBRARY_PATH=$LIBRARY_PATH \
        -x PATH \
        -x PYTHONPATH=$PYTHONPATH \
        -x JAVA_HOME=$JAVA_HOME \
        -x HIVE_HOME=$HIVE_HOME \
        -x CLASSPATH=$CLASSPATH \
        -x HADOOP_USER_NAME=$HADOOP_USER_NAME \
        -x HADOOP_HOME=$HADOOP_HOME \
        -x SPARK_HOME=$SPARK_HOME \
        -x KWS_SERVICE_REGION=$KWS_SERVICE_REGION \
        -x KWS_SERVICE_DC=$KWS_SERVICE_DC \
        -x KWS_SERVICE_CATALOG=$KWS_SERVICE_CATALOG \
        -x KWS_SERVICE_NAME=$KWS_SERVICE_NAME \
        -x KWS_SERVICE_AZ=$KWS_SERVICE_AZ \
        -x KWS_SERVICE_PAZ=$KWS_SERVICE_PAZ \
        -x KWS_SERVICE_STAGE=$KWS_SERVICE_STAGE \
        -x MASTER_ADDR=$MASTER_ADDR \
        -x MASTER_PORT=$MASTER_PORT \
        -x LD_PRELOAD=$LD_PRELOAD \
        -x KAI_FLAG_FILE \
        -x KML_ID \
        -x HADOOP_USER_NAME=$HADOOP_USER_NAME \
        -x TOKENIZERS_PARALLELISM=false \
        -x http_proxy=\
        -x https_proxy=\
        with_nccl_local_env \
        bash -c "python inference/keye_ar_sana/mpi_infer_visualize_reconstruction.py \
      --model-dir "${MODEL_DIR}" \
      --vae-dir "${VAE_DIR}" \
      --keye-ar-dir "${KEYE_AR_DIR}" \
      --dataset-config "${DATASET_CONFIG}" \
      --parquet-path "${PARQUET_PATH}" \
      --output-dir "${OUTPUT_DIR}" \
      --num-images ${NUM_IMAGES} \
      --device ${DEVICE} \
      --dtype ${DTYPE} \
      --num-sampling-steps ${NUM_SAMPLING_STEPS} \
      --flow-shift ${FLOW_SHIFT} \
      --cfg-scale ${CFG_SCALE} \
      --max-condition-length ${MAX_CONDITION_LENGTH} \
      --image-size ${IMAGE_SIZE} \
      --seed ${SEED} \
      --results-dir "${RESULTS_DIR}" \
      --teacher-forcing ${TEACHER_FORCING} \
      ${INIT_DIST_FLAG} \
      ${MODEL_CONFIG_OVERRIDES_FLAG} \
      ${DCP_FLAGS}" > $OUTPUT_DIR/stdout.log 2>$OUTPUT_DIR/stderr.log &
