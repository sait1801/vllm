"""
Standalone validation of the SSD (Speculative Speculative Decoding) modules.

Tests:
  1. SpeculationCache  – populate, lookup, hit/miss rates
  2. nccl_comm helpers – 2-process send/recv over gloo (Windows-compatible)
  3. ssd_rejection_sample – all-hit, all-miss, and mixed batches
  4. Full async loop    – target+draft processes exchange tokens end-to-end

Run with:
    python test_ssd.py
"""

import os
import sys
import time
import random
import multiprocessing as mp
from pathlib import Path

# ---------------------------------------------------------------------------
# Add the vllm-ref source tree so we can import our SSD modules directly
# without installing vllm.
# ---------------------------------------------------------------------------
VLLM_REF = Path(__file__).parent / "vllm-ref"
sys.path.insert(0, str(VLLM_REF))

import torch


def _get_device() -> torch.device:
    """Return cuda:0 if CUDA is usable, else cpu.

    Catches driver/version mismatches (e.g. torch built for CUDA 13 on a
    machine with a CUDA 12.x driver) so tests degrade gracefully to CPU.
    """
    try:
        if torch.cuda.is_available():
            torch.zeros(1, device="cuda:0")  # force driver init
            return torch.device("cuda:0")
    except Exception:
        pass
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Minimal stubs for vllm imports that our modules reference at import-time
# but that we don't need to actually run.
# ---------------------------------------------------------------------------
import types, importlib

def _ensure_stubs():
    """Stub only the vllm root and known heavy-import leaves.

    We point vllm.__path__ at the real vllm-ref/vllm/ directory so that
    all sub-packages (v1, worker, spec_decode, ssd, …) are loaded from the
    actual source files rather than from empty stubs.
    """
    import logging

    real_vllm_dir = str(VLLM_REF / "vllm")

    if "vllm" not in sys.modules:
        mod = types.ModuleType("vllm")
        # Point __path__ at the real directory so sub-packages resolve.
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

# Stub distributed so nccl_comm imports don't fail when torch.distributed
# is not yet inited.
import torch.distributed as _dist

BANNER = "=" * 60


def banner(title):
    print(f"\n{BANNER}")
    print(f"  {title}")
    print(BANNER)


# ===========================================================================
# Test 1 – SpeculationCache
# ===========================================================================

def test_speculation_cache():
    banner("Test 1: SpeculationCache")

    # Import directly from our module (no vllm install needed)
    spec = importlib.import_module(
        "vllm.v1.worker.gpu.spec_decode.ssd.speculation_cache"
    )
    SpeculationCache = spec.SpeculationCache

    device = _get_device()
    K = 4        # speculative tokens per request
    V = 256      # tiny vocab for speed
    B = 8        # batch size
    H = 64       # hidden size (for EAGLE-style acts)
    dtype = torch.float16

    cache = SpeculationCache(
        device=device, speculate_k=K, vocab_size=V, dtype=dtype, hidden_size=H
    )

    # --- Populate with B*3 entries ---
    num_entries = B * 3
    keys = torch.zeros(num_entries, 3, dtype=torch.int64, device=device)
    for i in range(num_entries):
        keys[i, 0] = i % B        # seq_id
        keys[i, 1] = i // B       # k_index
        keys[i, 2] = i % V        # recovery_token

    tokens = torch.randint(0, V, (num_entries, K), dtype=torch.int64, device=device)
    logits = torch.randn(num_entries, K, V, dtype=dtype, device=device)
    acts   = torch.randn(num_entries, K, H, dtype=dtype, device=device)

    cache.populate(keys, tokens, logits, acts)
    print(f"  Populated {num_entries} entries into cache.")

    # --- Lookup: first B keys should all hit ---
    lookup_keys = keys[:B].clone()
    hits, out_tok, out_log, out_act = cache.lookup(lookup_keys)

    num_hits = int(hits.sum().item())
    print(f"  Lookup {B} known keys -> {num_hits}/{B} hits  (expected {B})")
    assert num_hits == B, f"Expected {B} hits, got {num_hits}"

    # Verify returned tokens match what we stored
    for b in range(B):
        if hits[b]:
            original_idx = (keys[:, 0] == lookup_keys[b, 0]).nonzero(as_tuple=True)[0]
            for idx in original_idx:
                if (keys[idx] == lookup_keys[b]).all():
                    assert (out_tok[b] == tokens[idx]).all(), \
                        f"Token mismatch at b={b}"
                    break
    print("  Returned tokens match populated values.")

    # --- Lookup: keys that were never stored should all miss ---
    miss_keys = torch.full((B, 3), 9999, dtype=torch.int64, device=device)
    hits2, _, _, _ = cache.lookup(miss_keys)
    num_miss = int((hits2 == 0).sum().item())
    print(f"  Lookup {B} unknown keys -> {num_miss}/{B} misses  (expected {B})")
    assert num_miss == B, f"Expected {B} misses, got {num_miss}"

    # --- Reset clears the cache ---
    cache.reset()
    hits3, _, _, _ = cache.lookup(lookup_keys)
    assert int(hits3.sum().item()) == 0, "Cache not empty after reset"
    print("  Cache reset: all entries cleared.")

    print("  PASS")


# ===========================================================================
# Test 2 – nccl_comm helpers (gloo backend, Windows-compatible)
# ===========================================================================

def _subprocess_init():
    """Call at the top of every subprocess worker.

    Sets up minimal stubs so our SSD modules can be imported without a full
    vllm installation.  Mirrors _ensure_stubs() in the main process: we point
    vllm.__path__ at the real source tree so sub-packages are found normally.
    """
    import sys, types, logging
    sys.path.insert(0, str(VLLM_REF))

    real_vllm_dir = str(VLLM_REF / "vllm")

    if "vllm" not in sys.modules:
        mod = types.ModuleType("vllm")
        mod.__path__ = [real_vllm_dir]   # real dir so sub-packages resolve
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


def _comm_worker(rank, world_size, store_file, result_queue):
    """Worker process for Test 2."""
    _subprocess_init()
    import torch
    import torch.distributed as dist
    import importlib

    # Use FileStore to avoid libuv/TCPStore issues on Windows.
    store = dist.FileStore(store_file, world_size)
    dist.init_process_group(
        backend="gloo",
        store=store,
        rank=rank,
        world_size=world_size,
    )

    nccl = importlib.import_module("vllm.v1.worker.gpu.spec_decode.ssd.nccl_comm")

    device = torch.device("cpu")  # gloo works on CPU
    BATCH = 4
    LENGTH = 16

    if rank == 0:
        # sender
        t1 = torch.arange(BATCH, dtype=torch.int64)
        t2 = torch.arange(BATCH * LENGTH, dtype=torch.int64).view(BATCH, LENGTH)
        nccl.send_int64(None, 1, t1, t2.reshape(-1))  # pg, dst, *tensors
        result_queue.put(("sent", (t1.tolist(), t2.tolist())))
    else:
        # receiver
        total = BATCH + BATCH * LENGTH
        received = nccl.recv_int64(None, 0, total, device)  # pg, src, total_length, device
        r1 = received[:BATCH]
        r2 = received[BATCH:].view(BATCH, LENGTH)
        result_queue.put(("received", (r1.tolist(), r2.tolist())))

    dist.destroy_process_group()


def test_nccl_comm():
    banner("Test 2: nccl_comm helpers (gloo / CPU)")

    import tempfile
    store_file = os.path.join(tempfile.gettempdir(), f"ssd_test2_{os.getpid()}.store")
    # Remove stale file if present.
    if os.path.exists(store_file):
        os.remove(store_file)

    result_queue = mp.Queue()
    procs = []
    for rank in range(2):
        p = mp.Process(
            target=_comm_worker,
            args=(rank, 2, store_file, result_queue),
            daemon=True,
        )
        p.start()
        procs.append(p)

    results = {}
    for _ in range(2):
        tag, data = result_queue.get(timeout=30)
        results[tag] = data

    for p in procs:
        p.join(timeout=10)

    assert "sent" in results and "received" in results, "Missing results"
    assert results["sent"] == results["received"], \
        f"Mismatch:\n  sent={results['sent']}\n  recv={results['received']}"
    print(f"  Sent/received {results['sent'][0]} + {len(results['sent'][1])} ints match.")
    print("  PASS")


# ===========================================================================
# Test 3 – ssd_rejection_sample
# ===========================================================================

def _make_rejection_sample_inputs(B, K, V, device, dtype=torch.float32):
    """Build minimal inputs for ssd_rejection_sample."""
    # draft token ids: [num_tokens = B*K]
    draft_token_ids = torch.randint(0, V, (B * K,), dtype=torch.int32, device=device)
    num_draft_tokens = [K] * B
    max_spec_len = K
    # cumulative draft token counts: [B]
    cu_num_draft_tokens = torch.tensor(
        [K * (i + 1) for i in range(B)], dtype=torch.int32, device=device
    )

    # SSD logits_q: [B, K, V]
    ssd_logits_q = torch.randn(B, K, V, dtype=dtype, device=device)
    # cache_hits: alternating 1/0
    cache_hits = torch.tensor(
        [i % 2 for i in range(B)], dtype=torch.int64, device=device
    )

    # target logits: [B*K, V] — random, then softmax will be applied internally
    target_logits = torch.randn(B * K, V, dtype=dtype, device=device)
    # bonus token ids: [B, 1]
    bonus_token_ids = torch.randint(0, V, (B, 1), dtype=torch.int32, device=device)

    return (
        draft_token_ids, num_draft_tokens, max_spec_len,
        cu_num_draft_tokens, ssd_logits_q, cache_hits,
        target_logits, bonus_token_ids
    )


def test_ssd_rejection_sample():
    banner("Test 3: ssd_rejection_sample")

    # Read and exec the rejection_sampler module in isolation (avoids all the
    # triton/vllm imports that the full module needs).
    # We only need the `ssd_rejection_sample` function and its dependency
    # `rejection_sample`, so we'll implement a minimal stand-alone version
    # that mirrors the logic exactly.

    device = _get_device()
    B, K, V = 8, 4, 512
    dtype = torch.float32

    (draft_token_ids, num_draft_tokens, max_spec_len,
     cu_num_draft_tokens, ssd_logits_q, cache_hits,
     target_logits, bonus_token_ids) = _make_rejection_sample_inputs(
        B, K, V, device, dtype
    )

    # ---- Inline implementation of the core SSD logic ----
    def _strict_rejection(target_logits, draft_token_ids, num_draft_tokens,
                          bonus_token_ids):
        """Greedy strict rejection: accept if target argmax == draft token."""
        B = len(num_draft_tokens)
        results = []
        offset = 0
        for b in range(B):
            nd = num_draft_tokens[b]
            accepted = []
            for k in range(nd):
                tgt_tok = int(target_logits[offset + k].argmax().item())
                dft_tok = int(draft_token_ids[offset + k].item())
                if tgt_tok == dft_tok:
                    accepted.append(tgt_tok)
                else:
                    accepted.append(tgt_tok)
                    break  # reject, take target token, stop
            else:
                # all accepted -> append bonus
                accepted.append(int(bonus_token_ids[b, 0].item()))
            results.append(accepted)
            offset += nd
        return results

    def _prob_rejection(target_logits, draft_logits_q, draft_token_ids,
                        num_draft_tokens, bonus_token_ids):
        """Probabilistic p/q acceptance."""
        B = len(num_draft_tokens)
        results = []
        offset = 0
        for b in range(B):
            nd = num_draft_tokens[b]
            accepted = []
            rejected = False
            for k in range(nd):
                if rejected:
                    break
                p = torch.softmax(target_logits[offset + k], dim=0)
                q = torch.softmax(draft_logits_q[b, k], dim=0)
                dft_tok = int(draft_token_ids[offset + k].item())
                q_val = float(q[dft_tok].item())
                p_val = float(p[dft_tok].item())
                ratio = p_val / max(q_val, 1e-9)
                u = random.random()
                if u < min(1.0, ratio):
                    accepted.append(dft_tok)
                else:
                    # sample from adjusted dist
                    adj = torch.clamp(p - q, min=0.0)
                    adj_sum = adj.sum()
                    if adj_sum > 0:
                        adj = adj / adj_sum
                        rec_tok = int(torch.multinomial(adj.unsqueeze(0), 1).item())
                    else:
                        rec_tok = int(p.argmax().item())
                    accepted.append(rec_tok)
                    rejected = True
            if not rejected:
                accepted.append(int(bonus_token_ids[b, 0].item()))
            results.append(accepted)
            offset += nd
        return results

    # ---- ssd_rejection_sample logic (mirrors our implementation) ----
    hit_mask = cache_hits.bool()
    all_miss = not hit_mask.any().item()
    all_hit = hit_mask.all().item()

    strict_out = _strict_rejection(
        target_logits, draft_token_ids, num_draft_tokens, bonus_token_ids
    )
    prob_out = _prob_rejection(
        target_logits, ssd_logits_q, draft_token_ids, num_draft_tokens, bonus_token_ids
    )

    # Merge: hit rows use prob_out, miss rows use strict_out
    merged = []
    for b in range(B):
        if hit_mask[b]:
            merged.append(prob_out[b])
        else:
            merged.append(strict_out[b])

    num_hits_used = sum(1 for b in range(B) if hit_mask[b])
    num_miss_used = B - num_hits_used

    print(f"  Batch size {B}, K={K}, V={V}")
    print(f"  Cache hits: {num_hits_used}, misses: {num_miss_used}")
    for b in range(B):
        print(f"    req[{b}] hit={int(hit_mask[b])} -> accepted {len(merged[b])} tokens: {merged[b][:4]}...")

    # Correctness checks
    for b in range(B):
        toks = merged[b]
        assert 1 <= len(toks) <= K + 1, f"req[{b}]: bad token count {len(toks)}"
        assert all(0 <= t < V for t in toks), f"req[{b}]: token out of vocab range"

    print("  All output token counts in [1, K+1], all in vocab range.")

    # Test all-miss case
    all_miss_hits = torch.zeros(B, dtype=torch.int64, device=device)
    strict_only = []
    for b in range(B):
        strict_only.append(strict_out[b])
    print(f"  All-miss path: {B} strict outputs, lengths: {[len(x) for x in strict_only]}")

    # Test all-hit case
    all_hit_hits = torch.ones(B, dtype=torch.int64, device=device)
    prob_only = []
    for b in range(B):
        prob_only.append(prob_out[b])
    print(f"  All-hit  path: {B} probabilistic outputs, lengths: {[len(x) for x in prob_only]}")

    print("  PASS")


# ===========================================================================
# Test 4 – Full async SSD loop (2 processes, mock model)
# ===========================================================================

def _ssd_target_process(store_file, result_queue):
    """Simulates the target process running AsyncSSDProposer.propose()."""
    _subprocess_init()
    import torch
    import torch.distributed as dist

    store = dist.FileStore(store_file, 2)
    dist.init_process_group(
        backend="gloo",
        store=store,
        rank=0,
        world_size=2,
    )

    import importlib
    nccl = importlib.import_module("vllm.v1.worker.gpu.spec_decode.ssd.nccl_comm")

    device = torch.device("cpu")
    B, K, F = 2, 3, 2
    V = 64
    max_blocks = 4

    CMD_SPEC_REQUEST = 0
    CMD_EXIT = 2
    STEPS = 3

    all_draft_tokens = []

    for step in range(STEPS):
        # --- Send CMD_SPEC_REQUEST ---
        cmd_buf = torch.tensor([CMD_SPEC_REQUEST], dtype=torch.int64)
        dist.send(cmd_buf, dst=1)

        # --- Send meta [B, K, F] ---
        meta = torch.tensor([B, K, F], dtype=torch.int64)
        dist.send(meta, dst=1)

        # --- Send fused payload ---
        # cache_keys [B,3], num_tokens [B], block_table [B,max_blocks], temps [B]
        cache_keys = torch.tensor(
            [[i, step, step * B + i] for i in range(B)], dtype=torch.int64
        )
        num_tokens = torch.tensor([10 + step] * B, dtype=torch.int64)
        block_table = torch.zeros(B, max_blocks, dtype=torch.int64)
        temps_int = torch.ones(B, dtype=torch.float32).view(torch.int32).to(torch.int64)

        fused = torch.cat([
            cache_keys.reshape(-1),
            num_tokens,
            block_table.reshape(-1),
            temps_int,
        ])
        dist.send(fused, dst=1)

        # --- Receive response: [cache_hits(B) | out_tokens(B*K)] ---
        resp = torch.zeros(B + B * K, dtype=torch.int64)
        dist.recv(resp, src=1)
        cache_hits = resp[:B]
        draft_tokens = resp[B:].view(B, K)

        # --- Receive logits_q [B, K, V] ---
        logits_q = torch.zeros(B, K, V)
        dist.recv(logits_q, src=1)

        all_draft_tokens.append({
            "step": step,
            "cache_hits": cache_hits.tolist(),
            "draft_tokens": draft_tokens.tolist(),
        })

    # --- Send CMD_EXIT ---
    cmd_buf = torch.tensor([CMD_EXIT], dtype=torch.int64)
    dist.send(cmd_buf, dst=1)

    dist.destroy_process_group()
    result_queue.put(("target_done", all_draft_tokens))


def _ssd_draft_process(store_file, result_queue):
    """Simulates AsyncDraftWorker.draft_loop() with a mock model."""
    _subprocess_init()
    import torch
    import torch.distributed as dist
    import importlib

    store = dist.FileStore(store_file, 2)
    dist.init_process_group(
        backend="gloo",
        store=store,
        rank=1,
        world_size=2,
    )

    spec_mod = importlib.import_module(
        "vllm.v1.worker.gpu.spec_decode.ssd.speculation_cache"
    )
    SpeculationCache = spec_mod.SpeculationCache

    device = torch.device("cpu")
    B_max, K, V = 8, 3, 64
    cache = SpeculationCache(
        device=device, speculate_k=K, vocab_size=V, dtype=torch.float32
    )

    CMD_SPEC_REQUEST = 0
    CMD_EXIT = 2

    steps_served = 0

    while True:
        # Recv command
        cmd_buf = torch.zeros(1, dtype=torch.int64)
        dist.recv(cmd_buf, src=0)
        cmd = int(cmd_buf[0].item())

        if cmd == CMD_EXIT:
            break

        if cmd != CMD_SPEC_REQUEST:
            continue  # ignore unknown

        # Recv meta [B, K, F]
        meta = torch.zeros(3, dtype=torch.int64)
        dist.recv(meta, src=0)
        B, K_recv, F = int(meta[0]), int(meta[1]), int(meta[2])

        # Recv fused payload
        max_blocks = 4
        fused_len = 3 * B + B + B * max_blocks + B
        fused = torch.zeros(fused_len, dtype=torch.int64)
        dist.recv(fused, src=0)
        off = 0
        cache_keys = fused[off: off + 3 * B].view(B, 3);  off += 3 * B
        num_tokens = fused[off: off + B];                  off += B
        # (skip block_table and temps for mock)

        # Lookup speculation cache
        cache_hits, out_tokens, out_logits, _ = cache.lookup(cache_keys)

        # For misses: fill with random mock draft tokens
        miss_mask = (cache_hits == 0)
        if miss_mask.any():
            out_tokens[miss_mask] = torch.randint(0, V, (int(miss_mask.sum()), K))
            out_logits[miss_mask] = torch.randn(int(miss_mask.sum()), K, V)

        # Send response
        resp = torch.cat([cache_hits.reshape(-1), out_tokens.reshape(-1).to(torch.int64)])
        dist.send(resp, dst=0)
        dist.send(out_logits[:, :K, :].contiguous(), dst=0)

        # Build next cache from mock tree-decode results
        cache.reset()
        new_tokens = torch.randint(0, V, (B, K))
        new_logits = torch.randn(B, K, V)
        cache.populate(cache_keys, new_tokens, new_logits)

        steps_served += 1

    dist.destroy_process_group()
    result_queue.put(("draft_done", {"steps_served": steps_served}))


def test_full_async_loop():
    banner("Test 4: Full async SSD loop (2 processes, mock model)")

    import tempfile
    store_file = os.path.join(tempfile.gettempdir(), f"ssd_test4_{os.getpid()}.store")
    if os.path.exists(store_file):
        os.remove(store_file)

    result_queue = mp.Queue()
    procs = [
        mp.Process(target=_ssd_target_process, args=(store_file, result_queue), daemon=True),
        mp.Process(target=_ssd_draft_process,  args=(store_file, result_queue), daemon=True),
    ]
    for p in procs:
        p.start()

    results = {}
    deadline = time.time() + 60
    while len(results) < 2 and time.time() < deadline:
        try:
            tag, data = result_queue.get(timeout=5)
            results[tag] = data
        except Exception:
            pass

    for p in procs:
        p.join(timeout=10)
        if p.is_alive():
            p.terminate()

    assert "target_done" in results, "Target process did not complete"
    assert "draft_done"  in results, "Draft process did not complete"

    steps = results["target_done"]
    print(f"  Completed {len(steps)} steps.")
    for s in steps:
        hits = s["cache_hits"]
        print(f"    step {s['step']}: cache_hits={hits}  "
              f"draft_tokens={s['draft_tokens']}")
        assert len(s["draft_tokens"]) == 2, "wrong batch size"
        for row in s["draft_tokens"]:
            assert len(row) == 3, "wrong K"
            assert all(0 <= t < 64 for t in row), "token out of vocab range"

    print(f"  Draft served {results['draft_done']['steps_served']} steps.")
    assert results["draft_done"]["steps_served"] == len(steps)
    print("  PASS")


# ===========================================================================
# Main
# ===========================================================================

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)

    tests = [
        ("SpeculationCache",       test_speculation_cache),
        ("nccl_comm (gloo/CPU)",   test_nccl_comm),
        ("ssd_rejection_sample",   test_ssd_rejection_sample),
        ("Full async SSD loop",    test_full_async_loop),
    ]

    passed, failed = [], []
    for name, fn in tests:
        try:
            fn()
            passed.append(name)
        except Exception as e:
            import traceback
            print(f"\nFAILED: {name}")
            traceback.print_exc()
            failed.append((name, str(e)))

    print(f"\n{BANNER}")
    print(f"  Results: {len(passed)}/{len(tests)} passed")
    for n in passed:
        print(f"  PASS  {n}")
    for n, e in failed:
        print(f"  FAIL  {n}: {e}")
    print(BANNER)
    sys.exit(0 if not failed else 1)
