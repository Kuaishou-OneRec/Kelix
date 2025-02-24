# 数据解析工具说明

## 简介

本工具用于将图片数据以及文本标注信息转换成统一的 Parquet 格式，支持数据集自动分割（训练集 90% / 测试集 10%）和分布式存储。工具不仅支持多数据源（Excel 文件或 HDFS），还支持从多个图片目录提取图片数据，同时支持多线程并行处理以及 HDFS 数据上传。

## 主要特性

- **多数据源支持**  
  可同时处理 Excel 文件和 HDFS 数据源，根据配置选择数据获取方式。

- **多图片目录输入**  
  支持输入多个图片存放目录，多个路径之间用分号分隔。

- **自动分割数据集**  
  数据自动分割成训练集（90%）和测试集（10%）。

- **多线程并行处理**  
  利用线程池加速数据处理，提升大数据集处理的效率。

- **分片存储和 HDFS 上传**  
  训练集数据支持分片存储，同时可以配置自动上传结果到 HDFS。

- **自动生成数据集索引**  
  输出包含所有训练集分片文件路径的索引文件，便于后续数据加载与管理。

## 环境依赖

- Python 3.x
- OpenCV
- NumPy
- Pandas
- PyArrow
- PySpark
- PyYAML

请确保在运行脚本前已经安装以上依赖，可以使用 `pip install opencv-python numpy pandas pyarrow pyspark pyyaml` 进行安装。

## 使用方法

1. **配置文件**  
   编写 YAML 格式的配置文件（例如，本示例中的 `wenjuan_data_parse_config_0210_10w.yaml`）定义所有所需的参数。  
   
2. **运行脚本**  
   通过命令行执行脚本，指定配置文件路径：
   ```bash
   python parse_data_to_parquet.py config.yaml
   ```
   其中 `config.yaml` 为上一步定义好的配置文件路径。

## 配置参数说明

配置文件中定义的参数主要分为三大类：

### 1. 必需参数

- **image_folder**  
  图片目录路径，多个路径用分号分隔。  
  _示例_: `/path1/images;/path2/images`

- **data_source**  
  数据源类型，取值为 `'excel'` 或 `'hdfs'`。  
  当选择 `'excel'` 时，需要配置 `excel_file_path`；选择 `'hdfs'` 时，需要配置 `hdfs_path`。

- **parquet_path**  
  Parquet 文件输出路径，生成的训练集 (分片文件) 与测试集文件将以此为基础命名后存储。

- **txt_file_path**  
  CoT（Chain of Thought）结果文件路径，用于文本数据的额外处理。

### 2. 数据源相关参数

- **excel_file_path**  
  Excel 文件路径（当 `data_source` 为 `'excel'` 时必需），用于读取文本标注信息。

- **hdfs_path**  
  HDFS 数据路径（当 `data_source` 为 `'hdfs'` 时必需），用于读取数据源文件。

### 3. 可选参数

- **num_shards**  
  训练集分片数量（默认值：1）。  
  注意：当数据量不足时，实际生成的分片数可能小于配置值。

- **hdfs_output_path**  
  HDFS 输出路径，若配置该参数，则生成的 Parquet 分片和其他文件将自动上传至 HDFS。

- **index_file**  
  索引文件路径（默认值：`dataset_index.json`）。  
  该文件包含所有训练集分片（上传到 HDFS 后）的文件路径索引。

- **num_workers**  
  并行处理线程数（默认值：1），建议大数据集使用多线程提高处理效率。

- **source_name**  
  数据源标识（默认值：`wenjuan_photo_0207_1k`），用于生成数据存储中的 source 字段。

- **batch_size**  
  批处理大小（默认值：1000），可根据数据量情况调整。

- **resize_config**  
  图片调整大小配置，包含：  
  - `max_height`: 最大高度（默认640）  
  - `max_width`: 最大宽度（默认640）  
  如果 `resize_image` 设置为 `false`，则跳过图片大小调整过程。

## 示例配置文件

以下是 `wenjuan_data_parse_config_0210_10w.yaml` 中的示例配置内容：
```yaml
image_folder: "/llm_reco_ssd/huqigen/dataset/wenjuan_sft/photo_0210_11w/"
excel_file_path: "/llm_reco_ssd/huqigen/0207_1k.xlsx"
hdfs_path: "viewfs://hadoop-lt-cluster/home/reco_kaiworks/dw/reco_kaiworks.db/reco_llm_wenjuan_comment_sample_0210_tmp"
parquet_path: "/llm_reco_ssd/huqigen/dataset/wenjuan_sft/photo_0210_11w/photo_0210_11w_sft_data.parquet"
txt_file_path: "/llm_reco_ssd/huqigen/recovlm/examples/infer/llm_model_api_req/0207_1k/deepseek_0207_1k_response_with_cmt_cot.txt"
hdfs_output_path: "viewfs://hadoop-lt-cluster/home/reco_wl/mpi/huqigen/recovlm_dataset/wenjuan_sft/0210_11w/"
data_source: "hdfs"
index_file: "/llm_reco_ssd/huqigen/dataset/wenjuan_sft/photo_0210_10w/recovlm_dataset_wenjuan_0210_10w.json"
num_shards: 2048
resize_image: false
num_workers: 40
source_name: "wenjuan_photo_0210_10w"
```
