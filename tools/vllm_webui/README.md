当前部署方案：open-webui部署在开发机上，开放8888端口。模型通过vllm挂载在GPU实例上并开放内网端口，每个模型挂载到一个vllm实例上，并开发对应的端口，open-webui通过vllm api远程调用推理请求。

## 镜像

* open-webui:py11_ffmpeg
* vllm:py11_webui_vllm_flashattn

## vlm

```shell
cd tools/vllm_webui
bash run_vllm.sh ckpt_path deploy_path model_tag gpu_id port
# demo
# bash run_vllm.sh /llm_reco_ssd/luoxinchen/output2/RecoVLM/Qwen2-VL-7B-stage1/0.0.39/global_step306000 /llm_reco_ssd/zhangzixing/vllm_infer/model stg1_0.0.39_s306k 0 8000

```

目前每个gpu绑定一个模型，在启动的时候需要手动指定模型的gpu和端口，gpu和端口不能重复，否则可能会启动失败，token默认密码token-123456，可以curl访问url测试服务可用性

```shell
curl http://ip:port/v1/models -H "Authorization: Bearer token-123456" | jq

```

指令将返回：


```json
{
  "object": "list",
  "data": [
    {
      "id": "/llm_reco_ssd/zangdunju/vllm/models/test/",
      "object": "model",
      "created": 1737345127,
      "owned_by": "vllm",
      "root": "/llm_reco_ssd/zangdunju/vllm/models/test/",
      "parent": null,
      "max_model_len": 32768,
      "permission": [
        {
          "id": "modelperm-2cebe3d878c54948b8acbb744eb744f8",
          "object": "model_permission",
          "created": 1737345127,
          "allow_create_engine": false,
          "allow_sampling": true,
          "allow_logprobs": true,
          "allow_search_indices": false,
          "allow_view": true,
          "allow_fine_tuning": false,
          "organization": "*",
          "group": null,
          "is_blocking": false
        }
      ]
    }
  ]
}
```


## webui

```shell
open-webui serve --port 8888
```

DEMO：https://kml-dtmachine-18465-prod.kmlhb2az1l3-2.corp.kuaishou.com/

**添加模型**：左下角管理员面板 -> 上方tab的设置 -> 外部连接

url填写vllm对应的url（注意链接以v1结尾），密钥为vllm启动的token（默认：token-123456），进行模型配置，verfy connection OK后将模型保存。点击"模型"选项，此时应能看到部署的模型tag

**参数配置**：模型的推理参数会导致infer结果飞掉，建议可修改默认配置

点击左下角头像 -> 设置 -> 通用 -> 高级参数

Temperature->0.7

topP->0.9



