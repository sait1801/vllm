# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Target-side proposer for SSD (Speculative Speculative Decoding).

``AsyncSSDProposer`` runs on rank 0 (the target rank).  It communicates
with ``AsyncDraftWorker`` (running on the last GPU rank in a child process)
via a dedicated NCCL process group (``async_pg``).

Protocol (target always initiates):
  1. ``propose()``  — sends CMD_SPEC_REQUEST + payload, receives response.
  2. ``prefill()``  — sends CMD_PREFILL + payload (no response expected).
  3. ``shutdown()`` — sends CMD_EXIT.

The draft worker processes these commands in order and never initiates
communication on its own.
"""

from __future__ import annotations

import os
from multiprocessing import Queue
from typing import TYPE_CHECKING

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from vllm.logger import init_logger

if TYPE_CHECKING:
    from vllm.config import VllmConfig

logger = init_logger(__name__)

# Command constants — must match those in draft_worker.py
CMD_SPEC_REQUEST = 0
CMD_PREFILL = 1
CMD_EXIT = 2


class AsyncSSDProposer:
    """Target-side handle for SSD async draft communication.

    This object lives on the target process (rank 0).  It spawns the draft
    worker in a child process, exchanges tensors with it over NCCL, and
    exposes a ``propose()`` method whose signature is compatible with how
    ``GPUModelRunner`` calls spec-decode proposers.

    Parameters
    ----------
    vllm_config:
        Full ``VllmConfig``.  Must have ``speculative_config.draft_async=True``.
    device:
        The target CUDA device (rank 0).
    """

    def __init__(
        self,
        vllm_config: "VllmConfig",
        device: torch.device,
    ) -> None:
        self.vllm_config = vllm_config
        self.device = device

        spec_cfg = vllm_config.speculative_config
        assert spec_cfg is not None and spec_cfg.draft_async, (
            "AsyncSSDProposer requires speculative_config.draft_async=True"
        )
        self.spec_cfg = spec_cfg
        self.K: int = spec_cfg.num_speculative_tokens
        self.async_fan_out: int = spec_cfg.async_fan_out

        # Vocab size from draft model config.
        self.vocab_size: int = spec_cfg.draft_model_config.get_vocab_size()
        self.dtype: torch.dtype = vllm_config.model_config.dtype

        # EAGLE hidden state size (None if not EAGLE).
        self.use_eagle: bool = spec_cfg.method in ("eagle", "eagle3")
        self.hidden_size: int | None = (
            spec_cfg.draft_model_config.get_hidden_size() if self.use_eagle else None
        )
        self.act_dim: int = (
            3 * self.hidden_size if (self.use_eagle and self.hidden_size) else 0
        )

        # Number of GPU blocks override for block-table sizing.
        self._max_blocks: int = (
            vllm_config.cache_config.num_gpu_blocks_override or 512
        )

        # async_pg=None means "use the default process group" after
        # spawn_draft_worker() initialises a private 2-rank NCCL group.
        # _worker_ready tracks whether the draft process has been spawned.
        self.async_pg: dist.ProcessGroup | None = None
        self._worker_ready: bool = False
        self._draft_proc: mp.Process | None = None
        self._draft_num_kv_blocks: int = 0
        # Draft rank in the private 2-rank group is always 1.
        self._stored_draft_rank: int = 1

        # Pre-allocated send/recv buffers.
        self._cmd_buf = torch.zeros(1, dtype=torch.int64, device=device)
        self._meta_buf = torch.zeros(3, dtype=torch.int64, device=device)
        self._prefill_meta_buf = torch.zeros(5, dtype=torch.int64, device=device)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def load_model(self, target_model: object) -> None:
        """No-op: AsyncSSDProposer loads its model in a separate process.

        The draft model is loaded inside ``AsyncDraftWorker`` (child process)
        during :meth:`spawn_draft_worker`.  The target model reference is not
        needed on the proposer side.
        """

    def spawn_draft_worker(
        self,
        dist_init_addr: str,
        world_size: int,
        draft_rank: int,
    ) -> int:
        """Spawn the draft worker process and return its kv-block count.

        This must be called once before :meth:`propose` or :meth:`prefill`.

        Parameters
        ----------
        dist_init_addr:
            ``"host:port"`` used for ``torch.distributed.init_process_group``.
        world_size:
            Total number of distributed ranks (target ranks + 1 draft rank).
        draft_rank:
            The rank assigned to the draft worker (always ``world_size - 1``).

        Returns
        -------
        int
            Number of KV cache blocks allocated by the draft worker, for use
            by the target's block manager.
        """
        from vllm.v1.worker.gpu.spec_decode.ssd.draft_worker import (
            _draft_worker_entrypoint,
        )

        init_q: Queue = mp.Queue()

        self._draft_proc = mp.Process(
            target=_draft_worker_entrypoint,
            args=(
                draft_rank,
                self.vllm_config,
                init_q,
                dist_init_addr,
                world_size,
            ),
            daemon=True,
        )
        self._draft_proc.start()
        # In the private 2-rank dist group: target=0, draft=1 (always).
        # draft_rank here is the CUDA device index, NOT the dist group rank.
        self._stored_draft_rank = 1  # dist rank of draft in private group

        # The target and draft processes each independently call
        # dist.init_process_group with the SAME rendezvous address, forming a
        # private 2-rank group.  We cannot use dist.new_group here because that
        # requires every process in the existing default world group to call it
        # simultaneously.  Instead both sides bootstrap their own 2-rank group.
        #
        # NOTE: This requires PyTorch ≥ 2.0.  On older versions, only one call
        # to init_process_group per process is allowed; a workaround using
        # TCPStore + ProcessGroupNCCL directly is documented in the design doc.
        dist.init_process_group(
            backend="nccl",
            init_method=f"tcp://{dist_init_addr}",
            world_size=2,
            rank=0,  # target is always rank 0 in the SSD private group
        )
        # After init, the default group IS the 2-rank SSD group.
        # We keep async_pg=None to signal "use the default group" (dist.send
        # with group=None uses the default group).
        # async_pg stays None — group=None in dist.send/recv uses the
        # default group, which is now the private 2-rank SSD group.
        self.async_pg = None
        self._worker_ready = True  # draft process is live

        # Block until the draft worker reports its kv-block count.
        self._draft_num_kv_blocks = init_q.get(timeout=300)
        init_q.close()

        logger.info(
            "AsyncSSDProposer: draft worker (rank %d) ready. "
            "kv_blocks=%d.",
            draft_rank,
            self._draft_num_kv_blocks,
        )
        return self._draft_num_kv_blocks

    def shutdown(self) -> None:
        """Send CMD_EXIT to the draft worker and join the process."""
        if not self._worker_ready:
            return
        self._send_cmd(CMD_EXIT)
        if self._draft_proc is not None:
            self._draft_proc.join(timeout=30)
        self._worker_ready = False
        logger.info("AsyncSSDProposer: draft worker shut down.")

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def propose(
        self,
        seq_ids: torch.Tensor,            # [B] int64
        last_accepted_lens: torch.Tensor, # [B] int64
        recovery_token_ids: torch.Tensor, # [B] int64
        num_tokens: torch.Tensor,         # [B] int64 — number of context tokens
        draft_block_tables: torch.Tensor, # [B, max_blocks] int32
        temperatures: torch.Tensor,       # [B] float32
        target_recovery_acts: torch.Tensor | None = None,  # [B, act_dim] dtype
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Send a speculation request and receive draft proposals.

        Matches the call pattern in ``ssd/engine/speculator_async.py``
        (``_speculation_request``).

        Parameters
        ----------
        seq_ids:
            Sequence IDs for the current batch.
        last_accepted_lens:
            Number of tokens accepted so far for each sequence (``k_index``).
        recovery_token_ids:
            The recovery token (last target-accepted token) per sequence.
        num_tokens:
            Number of KV-cache tokens for each sequence (context length for
            draft block-table indexing).
        draft_block_tables:
            Block tables for the draft KV cache, one row per sequence.
        temperatures:
            Sampling temperatures, packed as float32 reinterpreted as int32
            for the fused int64 send.
        target_recovery_acts:
            EAGLE hidden states at the recovery position, if applicable.

        Returns
        -------
        draft_token_ids : int64 [B, K]
        logits_q        : dtype  [B, K, V]   — draft distribution (for p/q)
        cache_hits      : int64 [B]           — 1 = cache hit, 0 = miss
        """
        assert self._worker_ready, "Call spawn_draft_worker() first."

        B = seq_ids.shape[0]
        K = self.K
        F = self.async_fan_out
        max_blocks = self._max_blocks

        # --- Send CMD_SPEC_REQUEST ---
        self._send_cmd(CMD_SPEC_REQUEST)

        # --- Send meta: [B, K, F] ---
        self._meta_buf[0] = B
        self._meta_buf[1] = K
        self._meta_buf[2] = F
        dist.send(self._meta_buf, dst=self._draft_rank, group=self.async_pg)

        # --- Build cache keys: [B, 3] = (seq_id, k_index, recovery_token) ---
        cache_keys = torch.stack(
            [seq_ids, last_accepted_lens, recovery_token_ids], dim=1
        ).to(torch.int64)  # [B, 3]

        # --- Fuse payload into single int64 send ---
        # Layout: keys(3B) | num_tokens(B) | block_table(B*max_blocks) | temps(B)
        bt_padded = _pad_block_table(draft_block_tables, max_blocks)  # [B, max_blocks]
        temps_as_int = temperatures.view(torch.int32).to(torch.int64)

        fused = torch.cat([
            cache_keys.reshape(-1),         # 3*B
            num_tokens.reshape(-1),          # B
            bt_padded.to(torch.int64).reshape(-1),  # B*max_blocks
            temps_as_int.reshape(-1),        # B
        ])  # total: 3B + B + B*max_blocks + B = (5+max_blocks)*B
        dist.send(fused, dst=self._draft_rank, group=self.async_pg)

        # --- Optionally send EAGLE hidden states ---
        if self.use_eagle and target_recovery_acts is not None:
            dist.send(
                target_recovery_acts.contiguous(),
                dst=self._draft_rank,
                group=self.async_pg,
            )

        # --- Receive response ---
        # fused_response: [cache_hits(B) | out_tokens(B*K)] = B*(K+1) int64
        resp_len = B + B * K
        fused_resp = torch.empty(resp_len, dtype=torch.int64, device=self.device)
        dist.recv(fused_resp, src=self._draft_rank, group=self.async_pg)

        cache_hits = fused_resp[:B]                         # [B]
        draft_token_ids = fused_resp[B:].view(B, K)        # [B, K]

        # logits_q: [B, K, V] in model dtype
        logits_q = torch.empty(
            (B, K, self.vocab_size), dtype=self.dtype, device=self.device
        )
        dist.recv(logits_q, src=self._draft_rank, group=self.async_pg)

        return draft_token_ids, logits_q, cache_hits

    def prefill(
        self,
        input_ids: torch.Tensor,          # [total_tokens] int64
        num_tokens: torch.Tensor,          # [B] int64
        block_table: torch.Tensor,         # [B, max_blocks] int32
        eagle_acts: torch.Tensor | None = None,  # [total_tokens, act_dim] dtype
    ) -> None:
        """Notify the draft worker of a prefill event and send KV context.

        The draft worker will run a prefill forward pass to populate its KV
        cache for the new sequences.
        """
        assert self._worker_ready, "Call spawn_draft_worker() first."

        B = num_tokens.shape[0]
        total_tokens = int(input_ids.shape[0])
        max_blocks = self._max_blocks
        act_dim = self.act_dim if (self.use_eagle and eagle_acts is not None) else 0

        # --- Send CMD_PREFILL ---
        self._send_cmd(CMD_PREFILL)

        # --- Send meta: [total_tokens, B, max_blocks, use_eagle, eagle_act_dim] ---
        self._prefill_meta_buf[0] = total_tokens
        self._prefill_meta_buf[1] = B
        self._prefill_meta_buf[2] = max_blocks
        self._prefill_meta_buf[3] = int(self.use_eagle and eagle_acts is not None)
        self._prefill_meta_buf[4] = act_dim
        dist.send(
            self._prefill_meta_buf, dst=self._draft_rank, group=self.async_pg
        )

        # --- Fuse payload: input_ids(total_tokens) | num_tokens(B) | bt(B*max_blocks) ---
        bt_padded = _pad_block_table(block_table, max_blocks)
        fused = torch.cat([
            input_ids.to(torch.int64).reshape(-1),
            num_tokens.to(torch.int64).reshape(-1),
            bt_padded.to(torch.int64).reshape(-1),
        ])
        dist.send(fused, dst=self._draft_rank, group=self.async_pg)

        # --- Optionally send EAGLE hidden states ---
        if self.use_eagle and eagle_acts is not None:
            dist.send(eagle_acts.contiguous(), dst=self._draft_rank, group=self.async_pg)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @property
    def _draft_rank(self) -> int:
        """Rank of the draft worker in the SSD private 2-rank dist group.

        Always 1: in the private group target=0, draft=1.
        """
        return self._stored_draft_rank

    def _send_cmd(self, cmd: int) -> None:
        """Send a single int64 command scalar to the draft worker."""
        self._cmd_buf[0] = cmd
        dist.send(self._cmd_buf, dst=self._draft_rank, group=self.async_pg)

    @property
    def num_draft_kv_blocks(self) -> int:
        """KV blocks allocated by the draft worker, available after spawn."""
        return self._draft_num_kv_blocks


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _pad_block_table(
    block_table: torch.Tensor,  # [B, actual_max_blocks]
    target_cols: int,
) -> torch.Tensor:
    """Pad or truncate *block_table* to *target_cols* columns."""
    B, cols = block_table.shape
    if cols == target_cols:
        return block_table
    if cols < target_cols:
        pad = torch.zeros(
            (B, target_cols - cols),
            dtype=block_table.dtype,
            device=block_table.device,
        )
        return torch.cat([block_table, pad], dim=1)
    return block_table[:, :target_cols]
