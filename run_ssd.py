"""
SSD (Speculative Speculative Decoding) inference demo.

Model pairs (pick one based on your VRAM):

  A100 / H100 (40+ GB):
    Target : Qwen/Qwen3-8B        (~8 GB fp16, or ~4 GB fp8)
    Draft  : Qwen/Qwen3-1.7B      (~1.7 GB fp16, or ~0.9 GB fp8)

  RTX 3090/4090 (24 GB):
    Target : Qwen/Qwen3-4B        (~4 GB fp8)
    Draft  : Qwen/Qwen3-0.6B      (~0.6 GB fp8)

  RTX 4050 Laptop (6 GB):
    Target : Qwen/Qwen3-1.7B      (~1.7 GB fp8)
    Draft  : Qwen/Qwen3-0.6B      (~0.6 GB fp8)

Usage:
  python run_ssd.py                         # default: A100 pair
  TARGET=Qwen/Qwen3-1.7B DRAFT=Qwen/Qwen3-0.6B python run_ssd.py
"""

import os
from vllm import LLM, SamplingParams

TARGET_MODEL = os.environ.get("TARGET", "Qwen/Qwen3-8B")
DRAFT_MODEL  = os.environ.get("DRAFT",  "Qwen/Qwen3-1.7B")

prompts = [
    "The future of AI is",
    "Speculative decoding improves throughput by",
    "The key difference between transformers and RNNs is",
    "Explain gradient descent in simple terms:",
]

sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=128)

print(f"Target : {TARGET_MODEL}")
print(f"Draft  : {DRAFT_MODEL}")
print(f"SSD    : draft_async=True, K=5, fan_out=3")
print()

llm = LLM(
    model=TARGET_MODEL,
    quantization="fp8",
    dtype="half",
    gpu_memory_utilization=0.85,
    tensor_parallel_size=1,
    speculative_config={
        "model": DRAFT_MODEL,
        "num_speculative_tokens": 5,
        "method": "draft_model",
        "draft_async": True,         # activates SSD speculation cache
        "async_fan_out": 3,
        "jit_speculate": True,
        "speculative_model_quantization": "fp8",
    },
)

outputs = llm.generate(prompts, sampling_params)

for output in outputs:
    print(f"Prompt:    {output.prompt!r}")
    print(f"Generated: {output.outputs[0].text!r}")
    print()
