# Visual Large Language Model Training for Recommandation

## Overview
This project is designed to train VLMs from scratch.

## Features
- **Data Handling**: The project includes scripts to handle and preprocess data, ensuring compatibility with the training pipeline.
- **Training Pipeline**: A comprehensive training pipeline that supports fine-tuning on custom datasets.
- **Evaluation**: Tools to evaluate the performance of the trained models on various benchmarks.

## Getting Started

### Prerequisites
- Python 3.10.12 or higher
- PyTorch 2.5.1 or higher
- CUDA 11.8 or higher (for GPU support)

### Installation
1. Clone the repository:
    ```bash
    git clone https://git.corp.kuaishou.com/recogpt/recovlm.git
    cd RecoVLM
    ```

2. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

### Data Preparation

#### Stage2 & SFT Data Format

参考文档：[WebDataset File Format Specification](https://docs.google.com/document/d/18OdLjruFNX74ILmgrdiCI9J1fQZuhzzRBCHV9URWto0/edit?tab=t.0)

##### 约定
1. 如果样本里面有图片一起打包到 .tar 文件里面，如果有视频则存储视频对应的 ceph 路径。
2. json 里面应该包含数据来源 __key__, source 字段

##### Examples
1. 纯文本
```json
000000000.json 
{
    "__key__": 000000000, 
    "messages": [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Tell me who you are."},
        {"role": "assistant", "content": "I am a large language model named Qwen..."}
    ],
    "source": "XXX"
}
```
2. 文本 + 图片
```json
000000000.0.jpg
000000000.json
{
    "__key__": 000000000, 
     "messages": [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": "0.jpg"},
                {"type": "text", "text": "Output all text in the image"},
            ],
        },
        {"role": "assistant", "content": "The text in the image is balabala..."},
    ],
    "source": "XXXOCR"
}
```
3. 文本 + 视频, max_pixels, start_ts, end_ts, fps 留空后会采用配置的默认值
```json
000000000.json
{
    "__key__": 000000000, 
    "messages": [
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": "file:///path/to/video1.mp4",
                    "max_pixels": 360 * 420,
                    "fps": 1.0,
                    "start_ts": 0,
                    "end_ts": 8,
                },
                {"type": "text", "text": "Describe this video."},
            ]
        },
        {"role": "assistant", "content": "The video describe ..."},
    ],
    "source": "kwai_video"
}
```

### Training

### Evaluation

## Troubleshooting

## References
- [LLaVA-Pretrain Dataset](https://huggingface.co/datasets/liuhaotian/LLaVA-Pretrain)
- [COCO train2017 Dataset](http://images.cocodataset.org/zips/train2017.zip)
- [GQA Images](https://downloads.cs.stanford.edu/nlp/data/gqa/images.zip)