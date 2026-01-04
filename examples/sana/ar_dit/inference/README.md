# Inference and Evaluation Scripts

This directory contains three main scripts for running inference and evaluation on DCP checkpoints:

## 1. `mpi_infer_custom.sh` - Core Inference Script

**Purpose**: Run inference on GenEval dataset using MPI for distributed computing.

### Key Parameters (with defaults):
```bash
# Model paths
MODEL_DIR="/llm_reco_ssd/zhouyang12/models/muse/Sana_1600M_1024px/"
VAE_DIR="/llm_reco_ssd/zhouyang12/models/SANA1.5_1.6B_1024px_diffusers/vae/"
KEYE_AR_DIR="/mmu_mllm_hdd_2/zhouyang12/output/Keye/vqar_11.7/run_8b_vis_stage3.29_1e-4/step18000/global_step18000/muse_converted"

# Dataset configuration
DATASET_CONFIG="examples/sana/ar_dit/inference/run_ar_dit_lzx_4096_v2_1024im_multiscale_inf.json"
PARQUET_PATH="/mmu_mllm_hdd_2/lingzhixin/recovlm_data/muse_v2/vis/vis_data1225.parquet"

# DCP checkpoint parameters (set for auto-use)
DCP_CKPT_DIR="${DCP_CKPT_DIR:-}"
DCP_TAG="${DCP_TAG:-}"

# Output directory logic
if [ -n "$DCP_CKPT_DIR" ] && [ -n "$DCP_TAG" ]; then
    OUTPUT_DIR="${OUTPUT_DIR:-$DCP_CKPT_DIR/$DCP_TAG/inference/GenEval/outputs}"
else
    OUTPUT_DIR="${OUTPUT_DIR:-./vis_output}"
fi
```

### To run on different datasets:
- **Change `DATASET_CONFIG`**: Point to your custom dataset configuration file
- **Change `PARQUET_PATH`**: Update to your dataset's parquet file path

### Usage Examples:
```bash
# With DCP checkpoint
DCP_CKPT_DIR=/path/to/checkpoint DCP_TAG=global_step10000 bash mpi_infer_custom.sh

# With custom model
MODEL_DIR=/custom/model VAE_DIR=/custom/vae KEYE_AR_DIR=/custom/keye bash mpi_infer_custom.sh

# With custom dataset
DATASET_CONFIG=/custom/dataset.json PARQUET_PATH=/custom/data.parquet bash mpi_infer_custom.sh
```

---

## 2. `infer_and_eval.sh` - Single Checkpoint Inference + Evaluation

**Purpose**: Run a complete inference → evaluation pipeline for a single checkpoint.

### Key Parameters:
```bash
DCP_CKPT_DIR=/mmu_mllm_hdd_2/lingzhixin/output/MuseV2/sana/ar_dit/exp18_ar_dit_multiscale_324tokens_2e-5/
DCP_TAG=global_step10000
```

### Workflow:
1. **Inference**: Calls `mpi_infer_custom.sh` with DCP parameters
2. **Evaluation**: Runs ULMEvalKit evaluation on inference results
3. **Output**: Results saved in `${DCP_CKPT_DIR}/${DCP_TAG}/inference/GenEval/outputs/`

### Usage:
```bash
# For specific checkpoint
DCP_CKPT_DIR=/path/to/checkpoint DCP_TAG=global_step12345 bash infer_and_eval.sh
```

---

## 3. `run_auto_monitor.sh` - Automatic Monitoring Script

**Purpose**: Automatically monitor a DCP checkpoint directory and run inference + evaluation on new checkpoints.

### Key Parameters (set in the script):
```bash
DCP_CKPT_DIR="/mmu_mllm_hdd_2/lingzhixin/output/MuseV2/sana/ar_dit/exp18_ar_dit_multiscale_324tokens_2e-5"
MONITOR_INTERVAL=30
MODEL_TAG="BLIP3OTransformersSFT"
TB_LOG_NAME="auto_eval"
DATASET_CONFIG="examples/sana/ar_dit/inference/run_ar_dit_lzx_4096_v2_1024im_multiscale_inf.json"
KEYE_AR_DIR="/mmu_mllm_hdd_2/zhouyang12/output/Keye/vqar_11.7/run_8b_vis_stage3.29_1e-4/step18000/global_step18000/muse_converted"
INFERENCE_SCRIPT="examples/sana/ar_dit/inference/mpi_infer_custom.sh"
```

### Workflow:
1. **Monitors** the specified `DCP_CKPT_DIR` every `MONITOR_INTERVAL` seconds
2. **Detects** new `global_step*` directories with `.metadata` files
3. **Processes** checkpoints in descending step order (largest first)
4. **Runs** complete pipeline: Inference → Evaluation → Score Collection
5. **Logs** all activities to `${DCP_CKPT_DIR}/auto_monitor.log`

### Usage:
```bash
# Basic usage (with defaults)
bash run_auto_monitor.sh

# Custom parameters
DCP_CKPT_DIR=/custom/checkpoint MODEL_TAG=CustomModel bash run_auto_monitor.sh
```

---

## 🔧 Key Configuration Parameters for Adaptation

### 1. **Model Paths** (essential for different models):
- `--model-dir`: Path to the main model directory
- `--vae-dir`: Path to VAE model directory  
- `--keye-ar-dir`: Path to Keye AR model directory

### 2. **DCP Checkpoint Parameters** (for training checkpoints):
- `--dcp-ckpt-dir`: Directory containing DCP checkpoints
- `--dcp-tag`: Specific checkpoint tag (e.g., `global_step10000`)

### 3. **Dataset Parameters** (for different datasets):
- `--dataset-config`: Dataset configuration JSON file
- `--parquet-path`: Path to dataset parquet file

### 4. **Output Parameters**:
- `--output-dir`: Custom output directory
- `--results-dir`: Results storage directory
- `--tb-log-name`: TensorBoard log name prefix

---

## ?? Quick Start Guide

### For a single checkpoint:
```bash
# Modify these for your setup
export DCP_CKPT_DIR="/your/checkpoint/path"
export DCP_TAG="global_step10000"
export DATASET_CONFIG="/your/dataset.json"
export KEYE_AR_DIR="/your/keye/path"

# Run inference and evaluation
bash infer_and_eval.sh
```

### For automatic monitoring:
```bash
# Edit run_auto_monitor.sh and modify these variables:
DCP_CKPT_DIR="/your/checkpoint/path"           # ⬅️ REQUIRED
KEYE_AR_DIR="/your/keye/path"                  # ⬅️ REQUIRED  
DATASET_CONFIG="/your/dataset.json"            # ⬅️ OPTIONAL (change dataset)

# Start monitoring
bash run_auto_monitor.sh
```

### For custom inference script:
```bash
# Modify run_auto_monitor.sh
INFERENCE_SCRIPT="/custom/inference/script.sh"  # ⬅️ Use custom inference script
```

---

## 📝 Notes

- **Checkpoint Detection**: Scripts look for `global_step*` directories with `.metadata` files
- **Processing Order**: Processes larger step numbers first
- **Error Handling**: Single step failure doesn't interrupt other steps
- **Logging**: All operations logged to `${DCP_CKPT_DIR}/auto_monitor.log`
- **Environment**: Requires proper MPI and CUDA setup
- **Dependencies**: Requires ULMEvalKit for evaluation

---
## Troubleshooting

If you encounter issues:
1. Check that all model paths are correct and accessible
2. Verify dataset configuration files exist and are valid
3. Ensure the inference script has execution permissions
4. Check the log file for detailed error messages: `tail -f ${DCP_CKPT_DIR}/auto_monitor.log`