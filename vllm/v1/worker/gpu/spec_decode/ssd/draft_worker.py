# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Async draft worker for SSD (Speculative Speculative Decoding).

This module runs in a *separate process* on a dedicated GPU (the last rank).
It implements the three-command protocol used by the SSD reference
implementation:

  cmd = 0  ->  SPEC_REQUEST  (serve cache + build next cache)
  cmd = 1  ->  PREFILL       (run draft prefill)
  cmd = 2  ->  EXIT          (clean up and quit)

Unlike the SSD reference (which subclasses its own ModelRunner), this worker
*wraps* a HuggingFace AutoModelForCausalLM so that the draft model can be
loaded and executed without the full vLLM model-runner infrastructure.

A follow-up PR will migrate ``_run_prefill``, ``_jit_speculate``,
``_fork_tokens`` and ``_decode_tree`` to use vLLM's ``GPUModelRunner`` with
paged attention and CUDA graphs for production-level throughput.

Architecture
------------
                    +-----------------------+
  target rank 0 ->  |  nccl send (cmd+payload)  |
                    +----------+------------+
                               |  NCCL private 2-rank group
                    +----------v------------+
                    |    AsyncDraftWorker   |
                    |  (runs on last rank)  |
                    |                       |
                    |  SpeculationCache     |
                    |  HF draft model       |
                    |  _seq_kv_caches       |
                    +-----------------------+
"""

from __future__ import annotations

import dataclasses
import os
import time
from multiprocessing import Queue
from typing import TYPE_CHECKING, Any

import torch
import torch.distributed as dist

from vllm.logger import init_logger
from vllm.v1.worker.gpu.spec_decode.ssd.nccl_comm import (
    _pg_recv,
    _pg_send,
    recv_cmd,
    recv_int64,
)
from vllm.v1.worker.gpu.spec_decode.ssd.speculation_cache import SpeculationCache

if TYPE_CHECKING:
    from vllm.config import VllmConfig

logger = init_logger(__name__)


@dataclasses.dataclass
class DraftWorkerBootstrap:
    """Minimal, pickle-safe config passed to the draft worker subprocess.

    Passing the full ``VllmConfig`` to the subprocess via ``mp.Process``
    with the ``spawn`` start method triggers PyTorch's CUDA tensor
    serialisation and can OOM the GPU (or fail if CUDA tensors embedded in
    vllm_config exceed available device memory).  This dataclass contains
    only plain Python values extracted before spawning.
    """
    # Model
    draft_model: str
    draft_dtype: str               # e.g. "float16" or "bfloat16"
    draft_quantization: str | None # e.g. "fp8" or None
    # Speculative-decode params
    num_speculative_tokens: int
    async_fan_out: int
    jit_speculate: bool
    fan_out_list: list[int] | None
    fan_out_list_miss: list[int] | None
    spec_method: str               # "draft_model", "eagle", …
    # KV cache sizing
    block_size: int
    num_gpu_blocks_override: int | None
    # CUDA device for the draft worker (index, not dist-rank)
    cuda_device_index: int

# ---------------------------------------------------------------------------
# Command constants (shared with AsyncSSDProposer)
# ---------------------------------------------------------------------------
CMD_SPEC_REQUEST = 0
CMD_PREFILL = 1
CMD_EXIT = 2


class AsyncDraftWorker:
    """Runs the draft model on a dedicated GPU and exposes a command loop.

    This class is instantiated *inside* a child process spawned by the engine.
    It blocks in :meth:`draft_loop` until it receives ``CMD_EXIT``.

    Model execution currently uses ``transformers.AutoModelForCausalLM`` with
    its built-in ``past_key_values`` KV cache.  A future PR will upgrade to
    vLLM's ``GPUModelRunner`` (paged attention, CUDA graphs, quantisation).

    Parameters
    ----------
    vllm_config:
        The full VllmConfig.  ``vllm_config.speculative_config`` must have
        ``draft_async=True``.
    rank:
        The GPU device index assigned to this worker.
    init_q:
        A ``multiprocessing.Queue`` used to send the number of allocated KV
        cache blocks back to the parent process.
    async_pg:
        The NCCL process group shared with the target.  Pass ``None`` to use
        the default (private 2-rank) distributed group.
    """

    def __init__(
        self,
        bootstrap: DraftWorkerBootstrap,
        init_q: Queue,
        async_pg: dist.ProcessGroup,
    ) -> None:
        self.bootstrap = bootstrap
        self.async_pg = async_pg
        # Use the explicit CUDA device index from bootstrap (not dist-rank).
        self.device = torch.device(f"cuda:{bootstrap.cuda_device_index}")

        self.K: int = bootstrap.num_speculative_tokens
        self.async_fan_out: int = bootstrap.async_fan_out
        self.jit_speculate_enabled: bool = bootstrap.jit_speculate
        self.fan_out_list: list[int] = bootstrap.fan_out_list or (
            [self.async_fan_out] * (self.K + 1)
        )
        self.fan_out_list_miss: list[int] = bootstrap.fan_out_list_miss or (
            [1] * (self.K + 1)
        )
        # MQ_LEN = total number of forked token candidates per sequence per step
        self.MQ_LEN: int = sum(self.fan_out_list)

        # Dtype for logits buffers.
        self.dtype: torch.dtype = getattr(torch, bootstrap.draft_dtype, torch.float16)

        # ------------------------------------------------------------------
        # Load the draft model using HuggingFace transformers.
        # ------------------------------------------------------------------
        self._model, self.draft_hf_config = self._load_draft_model()
        self.vocab_size: int = self.draft_hf_config.vocab_size
        self.use_eagle: bool = bootstrap.spec_method in ("eagle", "eagle3")
        self.hidden_size: int | None = (
            getattr(self.draft_hf_config, "hidden_size", None)
            if self.use_eagle else None
        )

        # ------------------------------------------------------------------
        # KV cache management.
        # ------------------------------------------------------------------
        self.block_size, self.num_kvcache_blocks = self._init_kv_cache()
        init_q.put(self.num_kvcache_blocks)
        init_q.close()

        # Per-sequence HF past_key_values cache.
        # Key: int seq_id  Value: HF past_key_values tuple (or None)
        self._seq_kv_caches: dict[int, Any] = {}
        # Per-(batch_idx, depth) KV cache populated during _fork_tokens
        # and consumed by _decode_tree in the same spec-request step.
        self._fork_depth_kv_caches: dict[tuple[int, int], Any] = {}

        # ------------------------------------------------------------------
        # Speculation cache.
        # ------------------------------------------------------------------
        self.cache = SpeculationCache(
            device=self.device,
            speculate_k=self.K,
            vocab_size=self.vocab_size,
            dtype=self.dtype,
            hidden_size=self.hidden_size,
        )

        # Pre-allocated command receive buffer.
        self._cmd_buf = torch.zeros(1, dtype=torch.int64, device=self.device)

        # Per-step timing.
        self._step_times: list[float] = []

        logger.info(
            "AsyncDraftWorker ready on rank %d (K=%d, F=%d, MQ_LEN=%d, "
            "kv_blocks=%d).",
            rank, self.K, self.async_fan_out, self.MQ_LEN,
            self.num_kvcache_blocks,
        )

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def draft_loop(self) -> None:
        """Block until CMD_EXIT is received, processing requests in between."""
        while True:
            cmd, self._cmd_buf = recv_cmd(
                self.async_pg, src=0, device=self.device, _buf=self._cmd_buf
            )

            if cmd == CMD_PREFILL:
                self._handle_prefill()

            elif cmd == CMD_SPEC_REQUEST:
                t0 = time.perf_counter()
                self._handle_spec_request()
                self._step_times.append(time.perf_counter() - t0)

            elif cmd == CMD_EXIT:
                if self._step_times:
                    avg_ms = sum(self._step_times) * 1000 / len(self._step_times)
                    logger.info(
                        "AsyncDraftWorker exiting. avg_step=%.2f ms over %d steps.",
                        avg_ms, len(self._step_times),
                    )
                break

            else:
                raise RuntimeError(f"AsyncDraftWorker: unknown command {cmd}")

    # ------------------------------------------------------------------
    # Command handlers
    # ------------------------------------------------------------------

    def _handle_prefill(self) -> None:
        """Receive a prefill payload and run the draft model in prefill mode."""
        meta_buf = torch.zeros(5, dtype=torch.int64, device=self.device)
        _pg_recv(self.async_pg, meta_buf, 0)
        total_tokens, B, max_blocks, use_eagle_flag, eagle_act_dim = (
            meta_buf.tolist()
        )

        fused_len = total_tokens + B + B * max_blocks
        fused = recv_int64(
            self.async_pg, src=0, total_length=fused_len, device=self.device
        )
        off = 0
        input_ids = fused[off : off + total_tokens]
        off += total_tokens
        num_tokens = fused[off : off + B]
        off += B
        block_table = (
            fused[off : off + B * max_blocks].view(B, max_blocks).to(torch.int32)
        )

        eagle_acts: torch.Tensor | None = None
        if use_eagle_flag:
            eagle_acts = torch.zeros(
                (total_tokens, int(eagle_act_dim)),
                dtype=self.dtype,
                device=self.device,
            )
            _pg_recv(self.async_pg, eagle_acts, 0)

        self._run_prefill(input_ids, num_tokens, block_table, eagle_acts)

    def _handle_spec_request(self) -> None:
        """Serve a speculation request from the target.

        1. Receive: meta [B, K, F] + fused payload.
        2. Lookup speculation cache (or JIT speculate on miss).
        3. Send response (cache_hits ++ out_tokens, out_logits) back.
        4. Reset cache, run glue decode + tree decode, populate cache.
        """
        # --- Receive request ---
        meta = torch.zeros(3, dtype=torch.int64, device=self.device)
        _pg_recv(self.async_pg, meta, 0)
        B, K, F = (int(x) for x in meta.tolist())

        max_blocks = self.vllm_config.cache_config.num_gpu_blocks_override or 512
        fused_len = 3 * B + B + B * max_blocks + B
        fused = recv_int64(
            self.async_pg, src=0, total_length=fused_len, device=self.device
        )
        off = 0
        cache_keys = fused[off : off + 3 * B].view(B, 3)
        off += 3 * B
        num_tokens = fused[off : off + B].to(torch.int64)
        off += B
        draft_block_tables = (
            fused[off : off + B * max_blocks].view(B, max_blocks).to(torch.int32)
        )
        off += B * max_blocks
        temperatures = fused[off : off + B].to(torch.int32).view(torch.float32)

        target_recovery_acts: torch.Tensor | None = None
        if self.use_eagle:
            act_dim = 3 * (self.hidden_size or 1)
            target_recovery_acts = torch.zeros(
                (B, act_dim), dtype=self.dtype, device=self.device
            )
            _pg_recv(self.async_pg, target_recovery_acts, 0)

        # --- Lookup / JIT speculate ---
        cache_hits, out_tokens, out_logits, _out_acts = self.cache.lookup(cache_keys)

        if self.jit_speculate_enabled and not cache_hits.all():
            self._jit_speculate(
                cache_keys, num_tokens, out_logits, out_tokens, temperatures,
                draft_block_tables, target_recovery_acts, cache_hits,
            )

        # --- Send response ---
        fused_response = torch.cat(
            [cache_hits.reshape(-1), out_tokens.reshape(-1).to(torch.int64)]
        )
        _pg_send(self.async_pg, fused_response, 0)
        _pg_send(self.async_pg, out_logits[:, : self.K, :].contiguous(), 0)

        # --- Build next cache (glue decode -> fork -> tree decode -> populate) ---
        self.cache.reset()
        self._fork_depth_kv_caches.clear()

        rec_tokens = cache_keys[:, 2]  # [B]
        glue_input_ids = torch.cat(
            [rec_tokens.unsqueeze(1), out_tokens], dim=1
        ).view(-1)  # [B*(K+1)]

        forked_tokens = self._fork_tokens(
            glue_input_ids, out_logits, cache_hits, rec_tokens, B
        )  # [B, MQ_LEN]

        tree_tokens, tree_logits, tree_acts = self._decode_tree(
            forked_tokens, num_tokens, draft_block_tables, B,
            cache_hits, target_recovery_acts,
        )

        new_keys = self._build_cache_keys(
            cache_keys, forked_tokens, cache_hits, B
        )  # [B*MQ_LEN, 3]

        self.cache.populate(new_keys, tree_tokens, tree_logits, tree_acts)

    # ------------------------------------------------------------------
    # Model execution helpers
    # ------------------------------------------------------------------

    def _run_prefill(
        self,
        input_ids: torch.Tensor,   # [total_tokens] int64
        num_tokens: torch.Tensor,  # [B] int64
        block_table: torch.Tensor, # [B, max_blocks] int32
        eagle_acts: torch.Tensor | None,
    ) -> None:
        """Run the draft model in prefill mode to populate per-sequence KV caches.

        For each sequence in the batch, runs a full forward pass over its
        prompt tokens and stores the resulting ``past_key_values`` for use
        in subsequent spec-request steps.
        """
        B = int(num_tokens.shape[0])
        offset = 0
        with torch.inference_mode():
            for b in range(B):
                n = int(num_tokens[b].item())
                ctx_ids = (
                    input_ids[offset : offset + n]
                    .to(torch.long)
                    .unsqueeze(0)
                    .to(self.device)
                )  # [1, n]
                out = self._model(input_ids=ctx_ids, use_cache=True)
                # Store by batch index; _jit_speculate and _fork_tokens look up
                # the same index since a new batch arrives with each prefill.
                self._seq_kv_caches[b] = out.past_key_values
                offset += n

        logger.debug("_run_prefill: B=%d total_tokens=%d", B, int(input_ids.shape[0]))

    def _jit_speculate(
        self,
        request_keys: torch.Tensor,         # [B, 3]
        num_tokens: torch.Tensor,            # [B]
        out_logits: torch.Tensor,            # [B, K, V] — written in place for misses
        out_tokens: torch.Tensor,            # [B, K] — written in place for misses
        temperatures: torch.Tensor,          # [B] float32
        draft_block_tables: torch.Tensor,    # [B, max_blocks]
        target_recovery_acts: torch.Tensor | None,
        cache_hits: torch.Tensor,            # [B] int64
    ) -> None:
        """Run K autoregressive decode steps for cache-miss rows.

        Writes sampled tokens and logits directly into ``out_tokens`` and
        ``out_logits`` for every row where ``cache_hits[b] == 0``.
        """
        B = cache_hits.shape[0]
        device = self.device

        with torch.inference_mode():
            for b in range(B):
                if int(cache_hits[b].item()):
                    continue  # skip cache hits

                rec_tok = int(request_keys[b, 2].item())
                temp = float(temperatures[b].item())
                # KV cache is keyed by batch index (matches _run_prefill).
                past_kv = self._seq_kv_caches.get(b)

                input_ids = torch.tensor(
                    [[rec_tok]], dtype=torch.long, device=device
                )
                toks: list[int] = []
                logits_list: list[torch.Tensor] = []

                for _ in range(self.K):
                    out = self._model(
                        input_ids=input_ids,
                        past_key_values=past_kv,
                        use_cache=True,
                    )
                    past_kv = out.past_key_values
                    logit_k = out.logits[0, -1, :].float()  # [V]

                    if temp <= 0.0 or temp == 1.0:
                        next_tok = int(logit_k.argmax().item())
                    else:
                        prob = torch.softmax(logit_k / temp, dim=-1)
                        next_tok = int(torch.multinomial(prob.unsqueeze(0), 1).item())

                    toks.append(next_tok)
                    logits_list.append(logit_k.to(self.dtype))
                    input_ids = torch.tensor(
                        [[next_tok]], dtype=torch.long, device=device
                    )

                out_tokens[b] = torch.tensor(
                    toks, dtype=torch.int64, device=device
                )
                out_logits[b] = torch.stack(logits_list, dim=0)  # [K, V]

        logger.debug(
            "_jit_speculate: %d/%d misses",
            int((cache_hits == 0).sum().item()),
            B,
        )

    def _fork_tokens(
        self,
        glue_input_ids: torch.Tensor,  # [B*(K+1)]
        out_logits: torch.Tensor,       # [B, K, V]
        cache_hits: torch.Tensor,       # [B]
        rec_tokens: torch.Tensor,       # [B]
        B: int,
    ) -> torch.Tensor:
        """Run the glue decode and return forked top-F token candidates.

        The *glue decode* runs the draft model over the K+1-token sequence
        ``[rec_tok, spec_0, ..., spec_{K-1}]`` per sequence.  At each of
        K+1 positions the top ``fan_out_list[depth]`` tokens are extracted as
        candidate recovery tokens for the *next* speculation round.

        Side-effect: populates ``self._fork_depth_kv_caches[(b, depth)]`` with
        the running KV state at each depth so that ``_decode_tree`` can start
        tree-decode branches from the correct context.

        Returns
        -------
        forked_tokens : int64 [B, MQ_LEN]
        """
        device = self.device
        K = self.K
        forked = torch.zeros(B, self.MQ_LEN, dtype=torch.int64, device=device)

        with torch.inference_mode():
            for b in range(B):
                cur_past_kv = self._seq_kv_caches.get(b)

                # glue tokens for this batch item: [K+1]
                start = b * (K + 1)
                glue_toks = glue_input_ids[start : start + K + 1]

                mq_pos = 0
                for depth, f in enumerate(self.fan_out_list):
                    tok_in = (
                        glue_toks[depth]
                        .reshape(1, 1)
                        .to(torch.long)
                        .to(device)
                    )  # [1, 1]
                    out = self._model(
                        input_ids=tok_in,
                        past_key_values=cur_past_kv,
                        use_cache=True,
                    )
                    cur_past_kv = out.past_key_values
                    logit_d = out.logits[0, -1, :]  # [V]

                    # Top-f tokens at this glue position.
                    top_f = logit_d.topk(min(f, logit_d.shape[0]), dim=-1).indices
                    forked[b, mq_pos : mq_pos + f] = top_f[:f]
                    mq_pos += f

                    # Store KV cache AFTER consuming the depth-th glue token,
                    # so tree decode at this depth sees the correct context.
                    self._fork_depth_kv_caches[(b, depth)] = cur_past_kv

        return forked

    def _decode_tree(
        self,
        forked_tokens: torch.Tensor,       # [B, MQ_LEN]
        num_tokens: torch.Tensor,           # [B]
        draft_block_tables: torch.Tensor,   # [B, max_blocks]
        B: int,
        cache_hits: torch.Tensor,           # [B]
        target_recovery_acts: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """Run K autoregressive decode steps over every forked token candidate.

        For each of the ``B * MQ_LEN`` forked candidates, runs K greedy decode
        steps starting from the KV state captured at the corresponding tree
        depth in ``_fork_tokens``.

        Returns
        -------
        spec_tokens : int64 [B*MQ_LEN, K]
        spec_logits : dtype  [B*MQ_LEN, K, V]
        spec_acts   : None   (EAGLE hidden states, future work)
        """
        device = self.device
        N = B * self.MQ_LEN
        spec_tokens = torch.zeros((N, self.K), dtype=torch.int64, device=device)
        spec_logits = torch.zeros(
            (N, self.K, self.vocab_size), dtype=self.dtype, device=device
        )

        with torch.inference_mode():
            n = 0
            for b in range(B):
                mq_pos = 0
                for depth, f in enumerate(self.fan_out_list):
                    # KV cache at the glue decode position for this depth.
                    past_kv_for_depth = self._fork_depth_kv_caches.get((b, depth))

                    for j in range(f):
                        seed_tok = int(forked_tokens[b, mq_pos].item())
                        mq_pos += 1

                        input_ids = torch.tensor(
                            [[seed_tok]], dtype=torch.long, device=device
                        )
                        past_kv = past_kv_for_depth

                        for k in range(self.K):
                            out = self._model(
                                input_ids=input_ids,
                                past_key_values=past_kv,
                                use_cache=True,
                            )
                            past_kv = out.past_key_values
                            logit_k = out.logits[0, -1, :].float()  # [V]
                            next_tok = int(logit_k.argmax().item())

                            spec_tokens[n, k] = next_tok
                            spec_logits[n, k] = logit_k.to(self.dtype)
                            input_ids = torch.tensor(
                                [[next_tok]], dtype=torch.long, device=device
                            )

                        n += 1

        return spec_tokens, spec_logits, None

    def _build_cache_keys(
        self,
        request_keys: torch.Tensor,  # [B, 3]
        forked_tokens: torch.Tensor,  # [B, MQ_LEN]
        cache_hits: torch.Tensor,     # [B]
        B: int,
    ) -> torch.Tensor:
        """Build [B*MQ_LEN, 3] cache keys for the new tree-decode entries.

        Key structure: (seq_id, tree_depth_index, forked_recovery_token).
        """
        device = self.device
        seq_ids = request_keys[:, 0]  # [B]

        # Depth index for each of the MQ_LEN positions.
        depth_idx = torch.zeros(B, self.MQ_LEN, dtype=torch.int64, device=device)
        pos = 0
        for depth, f in enumerate(self.fan_out_list):
            depth_idx[:, pos : pos + f] = depth
            pos += f

        seq_ids_exp = seq_ids.unsqueeze(1).expand(B, self.MQ_LEN).reshape(-1)
        depth_exp = depth_idx.reshape(-1)
        forked_exp = forked_tokens.reshape(-1)

        return torch.stack([seq_ids_exp, depth_exp, forked_exp], dim=1)  # [B*MQ_LEN, 3]

    # ------------------------------------------------------------------
    # Model / KV-cache initialisation
    # ------------------------------------------------------------------

    def _load_draft_model(self):
        """Load the draft model using HuggingFace ``transformers``.

        Returns ``(model, hf_config)`` where ``model`` is an
        ``AutoModelForCausalLM`` placed on ``self.device``.

        A future PR will replace this with vLLM's ``GPUModelRunner`` to
        gain paged attention, CUDA graphs, and quantisation support.
        """
        from transformers import AutoConfig, AutoModelForCausalLM  # type: ignore

        model_name = self.bootstrap.draft_model
        logger.info("Loading draft model '%s' on %s ...", model_name, self.device)

        hf_config = AutoConfig.from_pretrained(
            model_name, trust_remote_code=True
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=self.dtype,
            trust_remote_code=True,
        ).to(self.device)
        model.eval()
        logger.info("Draft model loaded (%d params).", sum(p.numel() for p in model.parameters()))
        return model, hf_config

    def _init_kv_cache(self) -> tuple[int, int]:
        """Return (block_size, num_blocks) for the draft KV cache.

        With the HF-based execution backend the KV cache is managed by
        ``past_key_values`` internally.  We return a nominal block count so
        that the parent can size its block manager accordingly.
        """
        block_size = self.bootstrap.block_size

        gpu_mem = torch.cuda.get_device_properties(self.device).total_memory
        # Reserve 15 % of VRAM as an estimate for the HF rolling KV cache.
        usable = int(gpu_mem * 0.15)
        hf = self.draft_hf_config
        num_layers = getattr(hf, "num_hidden_layers", 32)
        num_kv_heads = getattr(hf, "num_key_value_heads", getattr(hf, "num_attention_heads", 8))
        head_dim = getattr(hf, "head_dim", 64)
        dtype_bytes = 2  # float16 / bfloat16
        bytes_per_block = (
            block_size * num_layers * num_kv_heads * head_dim * 2 * dtype_bytes
        )
        num_blocks = max(64, usable // bytes_per_block)
        logger.info(
            "Draft KV-cache estimate: block_size=%d num_blocks=%d",
            block_size, num_blocks,
        )
        return block_size, num_blocks


# ---------------------------------------------------------------------------
# Entry-point for torch.multiprocessing.Process
# ---------------------------------------------------------------------------

def _draft_worker_entrypoint(
    bootstrap: DraftWorkerBootstrap,
    init_q: Queue,
    dist_init_addr: str,
) -> None:
    """Called by ``multiprocessing.Process`` to start the draft process.

    Forms a private 2-rank NCCL group with the target process (rank 0).
    The draft's rank in this private group is always 1.

    Parameters
    ----------
    bootstrap:
        Minimal pickle-safe config for the draft worker.
    init_q:
        Queue used to send the KV-block count to the parent process.
    dist_init_addr:
        ``"host:port"`` rendezvous address.
    """
    # Build the same ProcessGroupNCCL the target builds (rank=0), but as rank=1.
    # Both sides must use PrefixStore("", ...) so that the NCCL unique-ID
    # rendezvous keys are identical.  Using dist.init_process_group here would
    # create a PrefixStore with a different internal prefix, causing NCCL to
    # exchange keys on different store entries and never establish the
    # communicator (manifesting as a segfault on the first P2P op).
    import datetime

    from torch.distributed import TCPStore
    from torch.distributed.distributed_c10d import PrefixStore, ProcessGroupNCCL

    _host, _port_str = dist_init_addr.rsplit(":", 1)
    _timeout = datetime.timedelta(seconds=300)
    _store = TCPStore(
        host_name=_host,
        port=int(_port_str),
        world_size=2,
        is_master=False,  # target (rank 0) is the TCPStore master
        timeout=_timeout,
    )
    async_pg = ProcessGroupNCCL(
        PrefixStore("", _store),
        rank=1,  # draft is always rank 1
        size=2,
        timeout=_timeout,
    )

    worker = AsyncDraftWorker(
        bootstrap=bootstrap,
        init_q=init_q,
        async_pg=async_pg,
    )
    worker.draft_loop()
