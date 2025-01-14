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

#### Parquet Data Format

由于 webdataset 无法很好的利用 spark & mapreduce 的大数据处理能力，后续在做数据过滤和处理时更困难。因此使用 parquet 格式能够有更好的数据处理工具以及利用 hdfs 大规模的存储，减少 ssd ceph 存储的压力。

##### 格式约定

| Field name | Field dtype | comment |
| ---------- | ----------- | ------- |
| images     | string      | json string, 实际是 map<string, string> 内容是 image key -> image bytes base64 |
| videos     | string      | json string, 实际是 list<string> 内容是 messages 或者 segments 里面所有的 video path|
| source     | string      | 数据来源 |
| messages   | string      | json string, chat 格式的数据，参考 [chat 数据](#chat-数据)
| segments   | string      | json string, pretrain 格式的数据，参考 [pretrain 数据](#pretrain-数据)
| metadata   | string      | json string, map<string, string> 其他 meta 信息
| uuid       | string      | 样本 uuid，用来唯一标识一条样本 |

##### 数据路径 
- Stage1 数据 **viewfs://hadoop-lt-cluster/home/reco_wl/mpi/luoxinchen/recovlm_dataset_stage1/{dataset_name}**
- Stage2 数据 **viewfs://hadoop-lt-cluster/home/reco_wl/mpi/luoxinchen/recovlm_dataset_stage2/{dataset_name}**

#### WebDataset Data Format

参考文档：[WebDataset File Format Specification](https://docs.google.com/document/d/18OdLjruFNX74ILmgrdiCI9J1fQZuhzzRBCHV9URWto0/edit?tab=t.0)

制作 webdataset 可以参考脚本 tools/downloader/main.py, 可以利用 mpi4py 同时拉起多个进程加速数据处理

##### 如何生成 index.json
```shell
pip install -e /llm_reco_ssd/luoxinchen/repos/webdataset/ --upgrade
cd /PATH/TO/WEBDATASET/
widsindex create *.tar --output index.json --process-num 256
```

##### 约定
1. 如果样本里面有图片一起打包到 .tar 文件里面，如果有视频则存储视频对应的 ceph 路径。
2. json 里面应该包含数据来源 __key__, source 字段

##### Examples

###### pretrain 数据
```json
000000000.0.jpg
000000000.1.jpg
000000000.2.jpg
000000000.0.mp4
000000000.json
{
    "__key__": 000000000, 
    "segments": [
        {"type": "image", "image": "0.jpg"},
        {"type": "text", "text": "blablabla..."},
        {"type": "image", "image": "1.jpg"},
        {"type": "image", "image": "2.jpg"},
        {"type": "text", "text": "blablabla..."},
        {"type": "image", "image": "0.mp4"},
        {"type": "text", "text": "blablabla..."}
    ],
    "source": "XXX"
}
```
###### chat 数据
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
3. 文本 + 视频, 
   1. video  参数:
      - **video**: required，video 路径
      - **video_start**: optional, 起始时间戳, 
      - **video_end**: optional, 结束时间戳
      - **nframes**: optional, 表示从start-end等宽采样多少帧（nframes，fps 2选一必选）
      - **fps**: optional, 当nframes没设置的话，根据fps来计算要采样的帧（可以不给）
      - **min_frames**: optional, 最少采样帧数（nframes不填时，fps计算依赖，有默认值）
      - **max_frames**: optional, 最大采样帧数（nframes不填时，fps计算依赖，有默认值）
      - **max_pixels**: optional
      - **min_pixels**: optional 
      - **total_pixels**: optional
      - **resized_height**: optional
      - **resized_width**: optional
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
                    "video_start": 0,
                    "video_end":
                },
                {"type": "text", "text": "Describe this video."},
            ]
        },
        {"role": "assistant", "content": "The video describe ..."},
    ],
    "source": "kwai_video"
}
```

对于已经处理成图片的视频，video字段使用image list填写（需要保证图片顺序），例如：

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
                    "video": [
                        {"type": "image", "image": "0.jpg"},
                        {"type": "image", "image": "1.jpg"},
                        {"type": "image", "image": "2.jpg"}
                    ]
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

#### 约定

1. 模型路径

所有模型都放在`/llm_reco_ssd/luoxinchen/output/RecoVLM/`下，路径格式需要符合`/llm_reco_ssd/luoxinchen/output/RecoVLM/project/version`

- `project`代表一组实验，`version`区分具体的版本，例如：`/llm_reco_ssd/luoxinchen/output/RecoVLM/Qwen2-VL-7B-stage2/0.0.4/`

- `project`命名规范：{Model-Family}-{Model-Size}-{stage}，比如`Qwen2-VL-7B-stage2`中，`Qwen2-VL`就是model family，`7B`就是模型大小，`stage-2`表示二阶段预训练。

- `version`规范：版本命名符合`major.minor.patch`的形式，需要正式发版的模型增加minor或major。

- 目前stage约定如下：一阶段预训练：`stage1`，二阶段预训练：`stage2`，sft版本：`sft`，dpo版本：`dpo`，rl版本：`rl`

2. 实验记录

所有实验提交都要有commit链接，参考`examples/vlm/run_pretrain_stage1_7B.sh`，使用这个脚本会自动添加格式化的commit message，例如：

```text
email=zhouyang12@kuaishou.com,time=20250105 18:55:38,script=./examples/vlm/run_pretrain_stage2_7B.sh,node=10,comment=测试stage2，打开LLM训练，使用the_cauldron，修复lr_decay再跑,output=/llm_reco_ssd/luoxinchen/output/RecoVLM/Qwen2-VL-7B-stage2/0.0.4
```

3. 版本记录

暂时手动管理版本，将比较重要的实验登记在https://docs.corp.kuaishou.com/d/home/fcACueI9SxQVJoUayiA1dVNtE

### Evaluation

## Troubleshooting

## References
- [LLaVA-Pretrain Dataset](https://huggingface.co/datasets/liuhaotian/LLaVA-Pretrain)
- [COCO train2017 Dataset](http://images.cocodataset.org/zips/train2017.zip)
- [GQA Images](https://downloads.cs.stanford.edu/nlp/data/gqa/images.zip)