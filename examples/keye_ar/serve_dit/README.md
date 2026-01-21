# serve_dit：本地 DiT/AR2Image HTTP 服务 + Client Demo

这套脚本用来把 `tests/models/keye_ar/demo_local_infer_visualize_reconstruction.py` 以 **HTTP 服务**的方式跑起来：

- 服务端提供 `POST /generate`
- 客户端发送 `prompt`（可选指定输出路径 `output_path`）
- 服务端返回实际写入的图片路径 `output_path`

> 说明：当前实现使用 Python 标准库 `http.server`，不依赖 FastAPI/Flask。

---

## 目录结构

- `config_local_ar2image.json`：服务端配置（映射到 `LocalAR2ImageConfig`）
- `run_server.sh`：启动服务端
- `demo_client.sh`：最小 client demo（curl 调用）

---

## 1. 启动服务端

### 1.1 使用默认配置启动

```bash
bash examples/keye_ar/serve_dit/run_server.sh
```

默认会加载：

- `examples/keye_ar/serve_dit/config_local_ar2image.json`

并设置：

- `PYTHONPATH=$ROOT_DIR:$PYTHONPATH`

### 1.2 指定配置文件启动

```bash
bash examples/keye_ar/serve_dit/run_server.sh /abs/path/to/config.json
```

### 1.3 CUDA_VISIBLE_DEVICES

`run_server.sh` 会把 `--cuda-visible-devices` 传给 Python。

- 若你已经在环境里设置了 `CUDA_VISIBLE_DEVICES`，脚本会用该值；
- 否则默认使用 `1`。

例如：

```bash
CUDA_VISIBLE_DEVICES=0 bash examples/keye_ar/serve_dit/run_server.sh
```

---

## 2. Client 调用

### 2.1 最简调用（不指定输出路径）

```bash
bash examples/keye_ar/serve_dit/demo_client.sh "a black cat."
```

服务端会把图片写入到：

- `service_output_dir` 目录下（由 config 控制）

并返回：

```json
{"output_path": ".../gen_YYYYmmdd_HHMMSS_xxxxxxxx.jpg"}
```

### 2.2 指定输出路径（服务端写入）

```bash
bash examples/keye_ar/serve_dit/demo_client.sh "a black cat." "/tmp/a_black_cat.jpg"
```

### 2.3 指定服务端地址（HOST/PORT 环境变量）

```bash
HOST=127.0.0.1 PORT=18080 bash examples/keye_ar/serve_dit/demo_client.sh "a black cat."
```

---

## 3. HTTP API 说明

### Endpoint

- `POST /generate`

### Request JSON

| 字段 | 类型 | 必填 | 说明 |
|---|---|---:|---|
| `prompt` | string | 是 | 文本描述 |
| `output_path` | string | 否 | 让服务端把图片写入到该路径；不传则自动命名写到 `service_output_dir` |

示例：

```bash
curl -X POST http://127.0.0.1:18080/generate \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"a black cat."}'
```

### Response JSON

成功：

```json
{ "output_path": "/path/to/generated.jpg" }
```

失败：

```json
{ "error": "..." }
```

---

## 4. config_local_ar2image.json 配置项

该 JSON 会被 `LocalAR2ImageConfig(**json_dict)` 直接构造，所以字段名必须与 dataclass 一致。

### 4.1 服务相关

- `enable_service`：是否启服务（必须为 `true` 才会跑 HTTP）
- `service_host`：监听地址（默认 `0.0.0.0`）
- `service_port`：端口（默认 `18080`）
- `service_output_dir`：不指定 `output_path` 时，服务端生成图片的默认落盘目录

### 4.2 模型相关（常规模式）

- `model_dir`：DiT 模型目录（通常是 `converted/`）
- `model_config_overrides`：数组，形如 `"key=value"`，会传入 `parse_config_overrides()`
- `vae_dir`：VAE 目录
- `keye_ar_dir`：Keye AR tokenizer/processor 目录

### 4.3 DCP 模式（可选）

当你要用 `dcp_ckpt_dir + dcp_tag` 自动转换并加载时：

- `dcp_ckpt_dir`：DCP checkpoint 目录
- `dcp_tag`：例如 `global_step4800`
- `source_model_dir`：作为 `dcp_to_torch_convert(source_dir=...)` 的源模型目录（必须有）

最终加载的模型目录为：

- `${dcp_ckpt_dir}/${dcp_tag}/converted`

> 注意：若该目录不存在，服务端会在启动时执行一次 `dcp_to_torch_convert()`。

### 4.4 采样/推理相关

- `device`：`cuda`/`cpu`
- `dtype`：`bfloat16`/`float16`/`float32`
- `image_size`
- `seed`
- `cfg_scale`
- `num_sampling_steps`
- `flow_shift`
- `max_condition_length`
- `linspace_sigmas`
- `condition_on_special_tokens`

---

## 5. 常见问题

### 5.1 client 返回 `{"error": "'NoneType' object has no attribute 'input_ids'"}`

这通常意味着 `VisReconstructionLoader()` 返回了 None 或者返回对象不包含 `input_ids`。

当前服务端已加入了较多 debug 输出（prompt/parquet_path/dataset 等），请直接查看 server 的 stdout，定位是：

- `parquet_path` 是否可读
- `dataset_config` 是否匹配
- `given_samples` 是否能被 loader 正常解析

你也可以把服务端的 debug 日志贴出来继续排查。
