# Python demo

from __future__ import annotations

from typing import Any, Dict, Optional

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import requests


class ImageGenDemo:
    """图像生成 demo 客户端（HTTP POST /generate）。

    说明：
    - 这是一个 demo 脚本，不做复杂的命令行参数解析。
    - 默认只传 prompt / output_path（保持最简用法）。
    - 如需传递更多参数（LLM 生成参数 + DiT 采样参数），请使用 `demo_with_all_args()`。

    支持的额外参数：
    1) LLM 生成参数（透传到服务端 `tokenize_images(**generation_args)`）
       - generation_args: dict，例如 {"top_k": 2, "top_p": 0.9}

    2) DiT 采样参数（透传到服务端覆盖采样配置）
       - cfg_scale: float
       - num_sampling_steps: int
       - flow_shift: float
       - linspace_sigmas: bool
    """

    def __init__(self, host: str = "10.48.48.106", port: str = "18080") -> None:
        self.host = host
        self.port = port
        self.base_url = f"http://{host}:{port}"

    def generate(self, prompt: str, output_path: Optional[str] = None) -> Dict[str, Any]:
        """最简生成请求：仅传 prompt / output_path。"""

        url = f"{self.base_url}/generate"
        payload: Dict[str, Any] = {"prompt": prompt}
        if output_path:
            payload["output_path"] = output_path

        resp = requests.post(url, headers={"Content-Type": "application/json"}, json=payload)
        resp.raise_for_status()
        return resp.json()

    def generate_with_all_args(
        self,
        prompt: str,
        output_path: Optional[str] = None,
        gpu_id: Optional[int] = None,
        generation_args: Optional[Dict[str, Any]] = None,
        cfg_scale: Optional[float] = None,
        num_sampling_steps: Optional[int] = None,
        flow_shift: Optional[float] = None,
        linspace_sigmas: Optional[bool] = None,
        timeout_s: int = 600,
    ) -> Dict[str, Any]:
        """带全部可选参数的请求。"""

        url = f"{self.base_url}/generate"
        payload: Dict[str, Any] = {"prompt": prompt}

        if output_path:
            payload["output_path"] = output_path
        if gpu_id is not None:
            payload["gpu_id"] = int(gpu_id)

        if generation_args is not None:
            payload["generation_args"] = generation_args

        if cfg_scale is not None:
            payload["cfg_scale"] = float(cfg_scale)
        if num_sampling_steps is not None:
            payload["num_sampling_steps"] = int(num_sampling_steps)
        if flow_shift is not None:
            payload["flow_shift"] = float(flow_shift)
        if linspace_sigmas is not None:
            payload["linspace_sigmas"] = bool(linspace_sigmas)

        resp = requests.post(
            url,
            headers={"Content-Type": "application/json"},
            json=payload,
            timeout=timeout_s,
        )
        resp.raise_for_status()
        return resp.json()

    def show_image(self, image_path: str, prompt: str = ""):
        img = mpimg.imread(image_path)
        plt.figure(figsize=(8, 6))
        plt.imshow(img)
        plt.axis("off")
        if prompt:
            title = prompt[:50] + "..." if len(prompt) > 50 else prompt
            plt.title(f"生成结果: {title}", fontsize=12, pad=10)
        plt.tight_layout()
        plt.show()
        return img

    def run_demo(self, prompt: str, output_path: Optional[str] = None):
        print(f"正在生成: {prompt}")
        result = self.generate(prompt, output_path)
        out_path = result.get("output_path")
        print(f"生成完成，图像路径: {out_path}")
        return self.show_image(out_path, prompt)


def demo() -> None:
    """保持原来的最简 demo：不传递任何额外参数。"""

    demo_client = ImageGenDemo()
    demo_client.run_demo(
        "Generate an image base on the description: A fascinate story book titled by 'Harry Potter' and authored by 'Ziming Li'.",
        "vis_output_local/service_outputs/tmp.jpg",
    )


def demo_with_all_args() -> None:
    """使用示例：同时传 LLM 生成参数 + DiT 采样参数。"""

    demo_client = ImageGenDemo()

    prompt = "Generate an image base on the description: A fascinate story book titled by 'Harry Potter' and authored by 'Ziming Li'."

    # LLM generation args：透传到 tokenize_images
    generation_args = {
        "top_k": 2,
        # "top_p": 0.9,
    }

    # DiT sampling overrides
    cfg_scale = 1.2
    num_sampling_steps = 30
    flow_shift = 2.5
    linspace_sigmas = True

    result = demo_client.generate_with_all_args(
        prompt=prompt,
        output_path="vis_output_local/service_outputs/tmp_all_args.jpg",
        gpu_id=None,
        generation_args=generation_args,
        cfg_scale=cfg_scale,
        num_sampling_steps=num_sampling_steps,
        flow_shift=flow_shift,
        linspace_sigmas=linspace_sigmas,
    )

    out_path = result.get("output_path")
    print(f"生成完成（all args），图像路径: {out_path}")
    print(f"服务端返回: {result}")
    if out_path:
        demo_client.show_image(out_path, prompt)


if __name__ == "__main__":
    # 默认保持最简 demo
    demo()
    # 如需测试全参数 demo，取消下一行注释
    demo_with_all_args()
