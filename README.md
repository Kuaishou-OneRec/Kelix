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
    cd multimodal-large-model
    ```

2. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

### Data Preparation
1. Download the datasets:
    ```bash
    huggingface-cli download --repo-type dataset liuhaotian/LLaVA-Pretrain --local-dir ./playground/data/LLaVA-Pretrain
    ```

2. Ensure the data directory structure is correct. If there are missing files, handle them as follows:
    ```python
    import json
    import os

    with open('./playground/data/llava_v1_5_mix665k.json') as f:
        samples = json.load(f)
    for sample in samples:
        if 'image' not in sample:
            continue
        img_path = os.path.join('./playground/data', sample['image'])
        if not os.path.exists(img_path):
            img_path_wo_ext = os.path.splitext(img_path)[0]
            for ext in ['.png', '.gif']:
                real_path = img_path_wo_ext + ext
                if os.path.exists(real_path):
                    os.replace(real_path, img_path)
                    break
    ```

### Training
1. Modify the `finetune.sh` script to set the `data_path` parameter and other optional parameters like learning rate:
    ```bash
    bash scripts/finetune.sh
    ```

2. Start the training process:
    ```bash
    python train.py --config configs/train_config.json
    ```

### Evaluation
1. Evaluate the trained model:
    ```bash
    python evaluate.py --model_path path/to/your/model
    ```

## Troubleshooting
- Ensure that the data directory structure is correct by checking the missing file count:
    ```python
    import json
    import os

    with open('./playground/data/llava_v1_5_mix665k.json') as f:
        samples = json.load(f)
    missing_cnt = 0
    for sample in samples:
        if 'image' not in sample:
            continue
        img_path = os.path.join('./playground/data', sample['image'])
        if not os.path.exists(img_path):
            missing_cnt += 1
    print(missing_cnt)
    ```

- Check the [official tutorial](https://github.com/haotian-liu/LLaVA?tab=readme-ov-file#train) for additional guidance.

## References
- [LLaVA-Pretrain Dataset](https://huggingface.co/datasets/liuhaotian/LLaVA-Pretrain)
- [COCO train2017 Dataset](http://images.cocodataset.org/zips/train2017.zip)
- [GQA Images](https://downloads.cs.stanford.edu/nlp/data/gqa/images.zip)