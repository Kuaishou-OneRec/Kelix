# Auto Inference and Evaluation Script

## 概述

`auto_infer_and_eval.sh` 是一个自动监控脚本，用于实时监控 DCP 检查点目录并自动执行推理、评估和分数收集流程。

## 功能特性

- 🔍 **实时监控**: 每5分钟检查一次新的 global_step 目录
- 📁 **文件检测**: 通过 `.metadata` 文件判断 checkpoint 是否就绪
- 🔄 **完整流程**: 自动执行推理 → 评估 → 分数收集
- 📊 **详细日志**: 所有操作都有时间戳和详细日志记录
- ❌ **错误处理**: 完善的错误处理和重试机制

## 使用方法

```bash
# 基本用法
bash examples/sana/ar_dit/inference/auto_infer_and_eval.sh DCP_CHECKPOINT_DIR

# 示例
bash examples/sana/ar_dit/inference/auto_infer_and_eval.sh /mmu_mllm_hdd_2/lingzhixin/output/MuseV2/sana/ar_dit/exp18_ar_dit_multiscale_324tokens_2e-5/
```

## 工作流程

1. **监控阶段**: 扫描 DCP_CHECKPOINT_DIR 下的 global_step* 目录
2. **就绪检测**: 检查是否存在 `.metadata` 文件
3. **推理阶段**: 调用 `mpi_run_infer_visualize_reconstruction_notf_324.sh`
4. **评估阶段**: 在 ULMEvalKit 中运行评估
5. **分数收集**: 使用 `collect_eval_scores` 收集结果

## 日志文件

- 主日志: `/tmp/auto_infer_eval_logs/auto_infer_eval.log`
- 推理日志: `$DCP_CHECKPOINT_DIR/global_stepXXXXX/inference/GenEval/outputs/inference_*.log`
- 评估日志: `$DCP_CHECKPOINT_DIR/global_stepXXXXX/inference/GenEval/outputs/ulmeval/aggresults/eval_*.out`

## 依赖项

- `mpi_run_infer_visualize_reconstruction_notf_324.sh` - 推理脚本
- `/llm_reco/lingzhixin/dit_eval_lzx/ULMEvalKit` - 评估工具包
- `recipes/sana/inference_ar2image.py` - 分数收集脚本

## 配置参数

可在脚本开头修改：
- `MONITOR_INTERVAL`: 监控间隔（秒）
- `MODEL_TAG`: 模型标签
- `TB_LOG_NAME`: TensorBoard 日志名称

## 注意事项

1. 确保所有依赖的目录和脚本存在
2. 监控会排除已存在的 global_step 目录
3. 每个步骤都有独立的错误处理
4. 日志文件会自动轮转和清理