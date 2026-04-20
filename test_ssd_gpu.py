"""
End-to-end GPU smoke test for SSD (Speculative Speculative Decoding).

Tests the full target<->draft async loop using real HuggingFace models:
  - Target  : gpt2 (137M params, ~550 MB fp32 / ~275 MB fp16)
  - Draft   : gpt2 (same small model — kept tiny to fit 6 GB RTX 4050)

Both processes run on cuda:0 (single-GPU mode for testing).  In production
SSD the draft process would run on a *separate* GPU; this test validates
correctness on a single device by serialising access.

Usage:
    python test_ssd_gpu.py
"""

import os
import sys
import time
import tempfile
import multiprocessing as mp
from pathlib import Path

VLLM_REF = Path(__file__).parent / "vllm-ref"
sys.path.insert(0, str(VLLM_REF))

import torch
import types
import logging

# ---------------------------------------------------------------------------
# Minimal vllm stubs so our SSD modules import without a full vllm install.
# ---------------------------------------------------------------------------

def _ensure_stubs():
    import logging
    real_vllm_dir = str(VLLM_REF / "vllm")
    if "vllm" not in sys.modules:
        mod = types.ModuleType("vllm")
        mod.__path__ = [real_vllm_dir]
        mod.__package__ = "vllm"
        sys.modules["vllm"] = mod
    if "vllm.logger" not in sys.modules:
        mod = types.ModuleType("vllm.logger")
        mod.__path__ = []
        mod.__package__ = "vllm.logger"
        mod.init_logger = lambda name: logging.getLogger(name)
        sys.modules["vllm.logger"] = mod
    if "vllm.config" not in sys.modules:
        mod = types.ModuleType("vllm.config")
        mod.__path__ = []
        mod.__package__ = "vllm.config"
        mod.VllmConfig = object
        sys.modules["vllm.config"] = mod

_ensure_stubs()

BANNER = "=" * 60

def banner(title):
    print(f"\n{BANNER}")
    print(f"  {title}")
    print(BANNER)


# ---------------------------------------------------------------------------
# Subprocess init helper (mirrors test_ssd.py)
# ---------------------------------------------------------------------------

def _subprocess_init():
    import sys, types, logging
    sys.path.insert(0, str(VLLM_REF))
    real_vllm_dir = str(VLLM_REF / "vllm")
    if "vllm" not in sys.modules:
        mod = types.ModuleType("vllm")
        mod.__path__ = [real_vllm_dir]
        mod.__package__ = "vllm"
        sys.modules["vllm"] = mod
    if "vllm.logger" not in sys.modules:
        mod = types.ModuleType("vllm.logger")
        mod.__path__ = []
        mod.__package__ = "vllm.logger"
        mod.init_logger = lambda name: logging.getLogger(name)
        sys.modules["vllm.logger"] = mod
    if "vllm.config" not in sys.modules:
        mod = types.ModuleType("vllm.config")
        mod.__path__ = []
        mod.__package__ = "vllm.config"
        mod.VllmConfig = object
        sys.modules["vllm.config"] = mod


# ---------------------------------------------------------------------------
# Helpers shared between target and draft processes
# ---------------------------------------------------------------------------

# distilgpt2 = 82M params, ~330 MB fp16 — well under the 1 GB VRAM budget.
# Both processes (target mock + draft) run on cuda:0 for single-GPU testing.
DRAFT_MODEL = "distilgpt2"
TARGET_MODEL = "distilgpt2"
K = 3          # speculative tokens
F = 2          # fan-out
STEPS = 5      # number of speculation rounds
B = 2          # batch size
MAX_SEQ_LEN = 64
DTYPE = torch.float16


def _make_spec_cache_module():
    """Return SpeculationCache class from our SSD package."""
    import importlib
    mod = importlib.import_module(
        "vllm.v1.worker.gpu.spec_decode.ssd.speculation_cache"
    )
    return mod.SpeculationCache


def _make_nccl_mod():
    import importlib
    return importlib.import_module(
        "vllm.v1.worker.gpu.spec_decode.ssd.nccl_comm"
    )


# ---------------------------------------------------------------------------
# Draft worker process
# ---------------------------------------------------------------------------

def _draft_process(store_file: str, result_q: mp.Queue) -> None:
    """
    Simulates AsyncDraftWorker.draft_loop() using a real HuggingFace GPT-2
    as the draft model.  Implements the full SSD algorithm:
      - CMD_PREFILL  : populate per-sequence KV caches
      - CMD_SPEC_REQUEST : lookup cache, JIT-speculate misses, send response,
                           then run glue+tree decode to repopulate cache
    """
    _subprocess_init()
    import torch
    import torch.distributed as dist
    from transformers import AutoModelForCausalLM

    store = dist.FileStore(store_file, 2)
    dist.init_process_group(backend="gloo", store=store, rank=1, world_size=2)

    device = torch.device("cuda:0")
    SpeculationCache = _make_spec_cache_module()
    nccl = _make_nccl_mod()

    # Load draft model
    print("  [draft] Loading draft model ...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        DRAFT_MODEL, dtype=DTYPE
    ).to(device)
    model.eval()
    vocab_size = model.config.vocab_size
    fan_out_list = [F] * (K + 1)   # [F, F, F, F] — K+1 depths
    MQ_LEN = sum(fan_out_list)       # F*(K+1)
    print(f"  [draft] Model loaded. vocab={vocab_size} MQ_LEN={MQ_LEN}", flush=True)

    cache = SpeculationCache(
        device=device, speculate_k=K, vocab_size=vocab_size, dtype=DTYPE
    )
    seq_kv_caches: dict = {}
    fork_depth_kv: dict = {}

    CMD_SPEC_REQUEST = 0
    CMD_PREFILL = 1
    CMD_EXIT = 2

    def run_k_steps(input_tok, past_kv, k, temp=1.0):
        """Run k autoregressive steps; return (token_list, logit_list)."""
        toks, logs = [], []
        inp = torch.tensor([[input_tok]], dtype=torch.long, device=device)
        with torch.inference_mode():
            for _ in range(k):
                out = model(input_ids=inp, past_key_values=past_kv, use_cache=True)
                past_kv = out.past_key_values
                lk = out.logits[0, -1, :].float()
                if temp <= 0.0 or temp == 1.0:
                    nt = int(lk.argmax().item())
                else:
                    nt = int(torch.multinomial(torch.softmax(lk / temp, dim=-1).unsqueeze(0), 1).item())
                toks.append(nt)
                logs.append(lk.to(DTYPE))
                inp = torch.tensor([[nt]], dtype=torch.long, device=device)
        return toks, logs, past_kv

    steps_served = 0

    while True:
        cmd_buf = torch.zeros(1, dtype=torch.int64)
        dist.recv(cmd_buf, src=0)
        cmd = int(cmd_buf[0].item())

        if cmd == CMD_EXIT:
            break

        elif cmd == CMD_PREFILL:
            meta = torch.zeros(5, dtype=torch.int64)
            dist.recv(meta, src=0)
            total_tokens, _B, max_blocks, _eagle, _act_dim = (int(x) for x in meta.tolist())
            fused_len = total_tokens + _B + _B * max_blocks
            fused = torch.zeros(fused_len, dtype=torch.int64)
            dist.recv(fused, src=0)
            off = 0
            inp_ids = fused[off:off+total_tokens].to(torch.long); off += total_tokens
            num_toks = fused[off:off+_B]; off += _B
            # Run prefill per sequence
            seq_offset = 0
            with torch.inference_mode():
                for b in range(_B):
                    n = int(num_toks[b].item())
                    ctx = inp_ids[seq_offset:seq_offset+n].unsqueeze(0).to(device)
                    out = model(input_ids=ctx, use_cache=True)
                    seq_kv_caches[b] = out.past_key_values
                    seq_offset += n
            print(f"  [draft] Prefill done for B={_B}", flush=True)

        elif cmd == CMD_SPEC_REQUEST:
            meta = torch.zeros(3, dtype=torch.int64)
            dist.recv(meta, src=0)
            recv_B, recv_K, recv_F = (int(x) for x in meta.tolist())
            max_blocks = 16
            fused_len = 3 * recv_B + recv_B + recv_B * max_blocks + recv_B
            fused = torch.zeros(fused_len, dtype=torch.int64)
            dist.recv(fused, src=0)
            off = 0
            cache_keys = fused[off:off+3*recv_B].view(recv_B, 3); off += 3*recv_B
            num_tokens = fused[off:off+recv_B]; off += recv_B
            temps = fused[off+recv_B*max_blocks:off+recv_B*max_blocks+recv_B].to(torch.int32).view(torch.float32)

            # Lookup cache
            hits, out_toks, out_logs, _ = cache.lookup(cache_keys.to(device))

            # JIT speculate for misses
            with torch.inference_mode():
                for b in range(recv_B):
                    if int(hits[b].item()):
                        continue
                    seq_id = int(cache_keys[b, 0].item())
                    rec_tok = int(cache_keys[b, 2].item())
                    temp = float(temps[b].item()) if temps.numel() > b else 1.0
                    past_kv = seq_kv_caches.get(seq_id)
                    toks, logs, _ = run_k_steps(rec_tok, past_kv, K, temp)
                    out_toks[b] = torch.tensor(toks, dtype=torch.int64, device=device)
                    out_logs[b] = torch.stack(logs, dim=0)

            # Send response
            resp = torch.cat([hits.reshape(-1), out_toks.reshape(-1).to(torch.int64)])
            dist.send(resp.cpu(), dst=0)
            dist.send(out_logs[:, :K, :].contiguous().cpu(), dst=0)

            # Build next cache: glue decode -> fork -> tree decode
            cache.reset()
            fork_depth_kv.clear()
            rec_toks = cache_keys[:, 2].to(device)
            glue_ids = torch.cat([rec_toks.unsqueeze(1), out_toks], dim=1).view(-1)

            forked = torch.zeros(recv_B, MQ_LEN, dtype=torch.int64, device=device)
            with torch.inference_mode():
                for b in range(recv_B):
                    seq_id = b
                    cur_past = seq_kv_caches.get(seq_id)
                    start = b * (K + 1)
                    glue_toks_b = glue_ids[start:start+K+1]
                    mq_pos = 0
                    for depth, f_d in enumerate(fan_out_list):
                        tok_in = glue_toks_b[depth].reshape(1, 1).to(torch.long).to(device)
                        out = model(input_ids=tok_in, past_key_values=cur_past, use_cache=True)
                        cur_past = out.past_key_values
                        logit_d = out.logits[0, -1, :]
                        top_f = logit_d.topk(min(f_d, logit_d.shape[0]), dim=-1).indices
                        forked[b, mq_pos:mq_pos+f_d] = top_f[:f_d]
                        mq_pos += f_d
                        fork_depth_kv[(b, depth)] = cur_past

            # Tree decode
            N = recv_B * MQ_LEN
            spec_toks = torch.zeros((N, K), dtype=torch.int64, device=device)
            spec_logs = torch.zeros((N, K, vocab_size), dtype=DTYPE, device=device)
            with torch.inference_mode():
                n = 0
                for b in range(recv_B):
                    mq_pos = 0
                    for depth, f_d in enumerate(fan_out_list):
                        base_kv = fork_depth_kv.get((b, depth))
                        for j in range(f_d):
                            seed = int(forked[b, mq_pos].item())
                            mq_pos += 1
                            toks, logs, _ = run_k_steps(seed, base_kv, K)
                            spec_toks[n] = torch.tensor(toks, dtype=torch.int64, device=device)
                            spec_logs[n] = torch.stack(logs, dim=0)
                            n += 1

            # Build cache keys and populate
            depth_idx = torch.zeros(recv_B, MQ_LEN, dtype=torch.int64, device=device)
            pos = 0
            for depth, f_d in enumerate(fan_out_list):
                depth_idx[:, pos:pos+f_d] = depth
                pos += f_d
            seq_ids_exp = cache_keys[:, 0].unsqueeze(1).expand(recv_B, MQ_LEN).reshape(-1).to(device)
            new_keys = torch.stack([seq_ids_exp, depth_idx.reshape(-1), forked.reshape(-1)], dim=1)
            cache.populate(new_keys, spec_toks, spec_logs)

            steps_served += 1

    dist.destroy_process_group()
    result_q.put(("draft_done", {"steps_served": steps_served}))


# ---------------------------------------------------------------------------
# Target (proposer) process
# ---------------------------------------------------------------------------

def _target_process(store_file: str, result_q: mp.Queue) -> None:
    """
    Simulates the target engine calling AsyncSSDProposer.prefill() and
    AsyncSSDProposer.propose() for STEPS rounds.  Uses the gloo backend
    (CPU tensors) so no second CUDA device is required.
    """
    try:
        _subprocess_init()
        import torch
        import torch.distributed as dist

        print("  [target] init dist ...", flush=True)
        store = dist.FileStore(store_file, 2)
        dist.init_process_group(backend="gloo", store=store, rank=0, world_size=2)
        print("  [target] dist ready", flush=True)

        B_local = B
        vocab = 50257  # GPT-2 vocab
        max_blocks = 16

        CMD_SPEC_REQUEST = 0
        CMD_PREFILL = 1
        CMD_EXIT = 2

        # --- Prefill round ---
        total_tokens = B_local * 10   # 10 tokens per sequence
        input_ids = torch.randint(0, vocab, (total_tokens,), dtype=torch.int64)
        num_toks = torch.tensor([10] * B_local, dtype=torch.int64)
        block_table = torch.zeros(B_local, max_blocks, dtype=torch.int64)

        dist.send(torch.tensor([CMD_PREFILL], dtype=torch.int64), dst=1)
        dist.send(torch.tensor([total_tokens, B_local, max_blocks, 0, 0], dtype=torch.int64), dst=1)
        dist.send(torch.cat([input_ids, num_toks, block_table.reshape(-1)]), dst=1)
        print("  [target] prefill sent", flush=True)

        # --- STEPS speculation rounds ---
        all_results = []
        for step in range(STEPS):
            # Build cache keys: (seq_id, k_index, recovery_token)
            cache_keys = torch.tensor(
                [[b, step, (step * B_local + b) % vocab] for b in range(B_local)],
                dtype=torch.int64,
            ).reshape(B_local, 3)

            # Send CMD_SPEC_REQUEST + meta + fused payload
            dist.send(torch.tensor([CMD_SPEC_REQUEST], dtype=torch.int64), dst=1)
            dist.send(torch.tensor([B_local, K, F], dtype=torch.int64), dst=1)
            temps_int = torch.ones(B_local, dtype=torch.float32).view(torch.int32).to(torch.int64)
            fused_payload = torch.cat([
                cache_keys.reshape(-1),
                torch.tensor([10 + step] * B_local, dtype=torch.int64),
                torch.zeros(B_local * max_blocks, dtype=torch.int64),
                temps_int,
            ])
            dist.send(fused_payload, dst=1)

            # Receive response: fused int64 + logits float16
            resp = torch.zeros(B_local + B_local * K, dtype=torch.int64)
            dist.recv(resp, src=1)
            cache_hits = resp[:B_local]
            draft_tokens = resp[B_local:].view(B_local, K)

            # Match draft's DTYPE=float16 send
            logits_q = torch.zeros(B_local, K, vocab, dtype=torch.float16)
            dist.recv(logits_q, src=1)

            all_results.append({
                "step": step,
                "cache_hits": cache_hits.tolist(),
                "draft_tokens": draft_tokens.tolist(),
                "logits_q_shape": list(logits_q.shape),
            })
            print(
                f"  [target] step {step}: hits={cache_hits.tolist()} "
                f"draft_tokens={draft_tokens.tolist()}",
                flush=True,
            )

        # Send CMD_EXIT
        dist.send(torch.tensor([CMD_EXIT], dtype=torch.int64), dst=1)
        dist.destroy_process_group()
        result_q.put(("target_done", all_results))

    except Exception as exc:
        import traceback
        print(f"  [target] EXCEPTION: {exc}", flush=True)
        traceback.print_exc()
        raise


# ---------------------------------------------------------------------------
# Test driver
# ---------------------------------------------------------------------------

def test_ssd_gpu_e2e():
    banner("SSD GPU End-to-End Test (GPT-2 draft, GPT-2 target mock)")

    if not torch.cuda.is_available():
        print("  SKIP: no CUDA device available.")
        return

    print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print(f"  Config: K={K} F={F} B={B} steps={STEPS}")

    store_file = os.path.join(tempfile.gettempdir(), f"ssd_gpu_test_{os.getpid()}.store")
    if os.path.exists(store_file):
        os.remove(store_file)

    result_q = mp.Queue()
    procs = [
        mp.Process(target=_target_process, args=(store_file, result_q), daemon=True),
        mp.Process(target=_draft_process,  args=(store_file, result_q), daemon=True),
    ]
    for p in procs:
        p.start()

    results = {}
    deadline = time.time() + 300  # 5-minute timeout (model download may be needed)
    while len(results) < 2 and time.time() < deadline:
        try:
            tag, data = result_q.get(timeout=10)
            results[tag] = data
            print(f"  Received '{tag}'", flush=True)
        except Exception:
            # Check if processes died
            for p in procs:
                if not p.is_alive() and p.exitcode != 0:
                    print(f"  Process exited with code {p.exitcode}", flush=True)
            pass

    for p in procs:
        p.join(timeout=15)
        if p.is_alive():
            p.terminate()

    assert "target_done" in results, "Target process did not complete"
    assert "draft_done"  in results, "Draft process did not complete"

    steps = results["target_done"]
    draft_info = results["draft_done"]

    print(f"\n  Completed {len(steps)}/{STEPS} steps.")
    for s in steps:
        hits = s["cache_hits"]
        dt   = s["draft_tokens"]
        lq_shape = s["logits_q_shape"]
        print(f"    step {s['step']}: hits={hits} draft_tokens={dt} logits_q={lq_shape}")
        # Validate shapes
        assert len(dt) == B, f"Expected B={B} rows"
        assert all(len(row) == K for row in dt), f"Expected K={K} tokens per row"
        assert lq_shape == [B, K, 50257], f"Unexpected logits_q shape {lq_shape}"
        assert all(0 <= t < 50257 for row in dt for t in row), "Token out of vocab"

    assert draft_info["steps_served"] == STEPS, (
        f"Draft served {draft_info['steps_served']} steps, expected {STEPS}"
    )

    print(f"\n  Draft served {draft_info['steps_served']} steps.")
    print("  All token IDs in vocab range, shapes correct.")
    print("  PASS")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)

    try:
        test_ssd_gpu_e2e()
    except Exception as e:
        import traceback
        print(f"\nFAILED: {e}")
        traceback.print_exc()
        sys.exit(1)
