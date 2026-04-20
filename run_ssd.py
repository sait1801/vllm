"""
SSD (Speculative Speculative Decoding) inference demo.

Models chosen to fit in 6 GB VRAM with FP8 quantization:
  Target : Qwen/Qwen3-1.7B  (~1.7 GB FP8)
  Draft  : Qwen/Qwen3-0.6B  (~0.6 GB FP8)
  KV + overhead             (~2-3 GB)
  Total                     <5 GB  ✓

Run from the vllm-ref directory (your modified vLLM with SSD):
  cd /mnt/c/Users/saitb/ssd-vllm/vllm-ref
  python ../run_ssd.py
"""

from vllm import LLM, SamplingParams

prompts = [
    "The future of AI is",
    "Speculative decoding improves throughput by",
    "The capital of France is",
]

sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=64)

llm = LLM(
    model="Qwen/Qwen3-1.7B",
    quantization="fp8",              # dynamic FP8 on RTX 4050 (Ada Lovelace)
    dtype="half",
    gpu_memory_utilization=0.80,     # leave 20% headroom for draft + KV
    tensor_parallel_size=1,
    speculative_config={
        "model": "Qwen/Qwen3-0.6B",
        "num_speculative_tokens": 5,
        "method": "draft_model",
        "draft_async": True,         # activates SSD: async draft + speculation cache
        "async_fan_out": 3,          # K+1 = 6 glue positions, each fans out 3
        "jit_speculate": True,       # fall back to JIT decode on cache miss
        "speculative_model_quantization": "fp8",
    },
)

outputs = llm.generate(prompts, sampling_params)

for output in outputs:
    print(f"Prompt:    {output.prompt!r}")
    print(f"Generated: {output.outputs[0].text!r}")
    print()
