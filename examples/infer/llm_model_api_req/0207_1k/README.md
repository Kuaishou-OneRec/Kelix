# 短视频内容理解评估工具

这是一个用于评估短视频内容的自动化工具，通过调用大语言模型API来分析视频内容并判断用户满意度。

## 功能说明

该工具主要实现以下功能：
1. 读取视频的关键帧图像
2. 处理视频相关的文本信息（标题、OCR内容、ASR语音识别内容、用户评论等）
3. 调用大语言模型API（支持Qwen、Doubao、Ark等）进行内容分析
4. 输出视频内容分析结果和用户满意度评估

## 配置文件说明

配置文件示例 (config_qwen_with_cmt_cot.yaml):

```yaml
client_type: "qwen"              # 选择使用的模型类型：qwen/doubao/ark
folder_path: "/path/to/data"     # 数据根目录路径
image_folder: "photo_folder/"    # 视频关键帧图片所在文件夹
excel_file_path: "data.xlsx"     # 包含视频信息的Excel文件路径
model_name: "qwen-vl-max"        # 使用的模型名称
result_file: "result.txt"        # 结果输出文件
prompt_pre: "你是一个短视频平台的内容理解专家..." # 模型提示词前缀
prompt_end: "请在回答的最后给出最终的满意度判断..." # 模型提示词后缀
enable_cmt: true                 # 是否启用评论分析
is_debug: true                   # 是否开启调试模式（仅处理一个样本）
```

## 使用方法

1. 准备数据：
   - 将视频关键帧图片放入指定文件夹
   - 准备包含视频信息的Excel文件（需包含photo_id、caption、ocr、asr、user_comment等字段）

2. 配置yaml文件：
   - 根据实际需求修改配置参数
   - 确保文件路径正确

3. 运行脚本：
```bash
python req_llm_model.py config.yaml api_key
```

## 注意事项

1. 运行前请确保已安装所需的Python依赖包
2. API密钥需要单独提供，不要将其包含在配置文件中
3. 调试模式下只会处理第一个样本，用于快速验证配置是否正确
4. 图片会自动调整大小以适应模型要求
5. 支持多线程并行处理，提高效率

## 输出结果

程序将根据配置文件中指定的result_file路径输出分析结果，包含每个视频的photo_id和对应的内容分析结果。
