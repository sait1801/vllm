# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tensor-backed speculation cache for SSD (Speculative Speculative Decoding).

The draft worker maintains a flat tensor store keyed by
``(seq_id, k_index, recovery_token_id)``.  Before the target model finishes
verifying step T the draft has already computed K-token continuations for
every likely recovery outcome and stored them here.  When the target sends its
actual outcome the draft looks up the cache, returns the pre-computed tokens
instantly (cache hit), and immediately starts building the cache for step T+1.

Layout
------
``keys``        : int64 [N, 3]   — (seq_id, k_idx, rec_token_id)
``tokens``      : int64 [N, K]   — K draft token ids per entry
``logits``      : dtype [N, K, V]— draft logit distributions (for p/q ratio)
``activations`` : dtype [N, K, H] or None — EAGLE hidden states
"""

from __future__ import annotations

import torch


class SpeculationCache:
    """Stores pre-computed draft continuations indexed by
    ``(seq_id, k_idx, recovery_token_id)``.

    The cache is populated *once* per draft step (after the tree decode) and
    consumed *once* per target step (at the start of the next speculation
    request).  It is always fully replaced — there is no LRU or partial update.
    """

    def __init__(
        self,
        device: torch.device,
        speculate_k: int,
        vocab_size: int,
        dtype: torch.dtype,
        hidden_size: int | None = None,
    ) -> None:
        self.device = device
        self.K = speculate_k
        self.V = vocab_size
        self.dtype = dtype
        self.hidden_size = hidden_size

        # Start empty; populated after first tree decode.
        self._keys: torch.Tensor = torch.zeros(
            (0, 3), dtype=torch.int64, device=device
        )
        self._tokens: torch.Tensor | None = None
        self._logits: torch.Tensor | None = None
        self._activations: torch.Tensor | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Clear the cache (called before populating the new tree.)"""
        self._keys = torch.zeros((0, 3), dtype=torch.int64, device=self.device)
        self._tokens = None
        self._logits = None
        self._activations = None

    def lookup(
        self,
        request_keys: torch.Tensor,  # [B, 3] int64
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """Look up *request_keys* in the cache.

        Returns
        -------
        cache_hits   : int64 [B]   — 1 if key found, 0 otherwise
        out_tokens   : int64 [B, K]
        out_logits   : dtype [B, K, V]
        out_acts     : dtype [B, K, H] or None
        """
        B = request_keys.shape[0]

        # Allocate output buffers with sane defaults for misses.
        out_tokens = torch.empty((B, self.K), dtype=torch.int64, device=self.device)
        out_logits = torch.empty(
            (B, self.K, self.V), dtype=self.dtype, device=self.device
        ).uniform_()  # random logits → in-vocab argmax on miss
        out_tokens.copy_(out_logits.argmax(dim=-1))

        out_acts: torch.Tensor | None = None
        if self.hidden_size is not None:
            out_acts = torch.zeros(
                (B, self.K, self.hidden_size), dtype=self.dtype, device=self.device
            )

        cache_hits = torch.zeros(B, dtype=torch.int64, device=self.device)

        if self._keys.numel() == 0 or self._tokens is None:
            return cache_hits, out_tokens, out_logits, out_acts

        # Vectorised key matching: [B, N, 3] → [B, N] → [B]
        eq = request_keys.unsqueeze(1) == self._keys.unsqueeze(0)  # [B, N, 3]
        match = eq.all(dim=2)  # [B, N]
        hit_mask = match.any(dim=1)  # [B]
        cache_hits = hit_mask.to(torch.int64)

        if hit_mask.any():
            # For each hit row pick the first matching cache entry.
            idx = match.float().argmax(dim=1).to(torch.int64)  # [B]
            sel = hit_mask
            out_tokens[sel] = self._tokens[idx[sel]]
            out_logits[sel] = self._logits[idx[sel]]  # type: ignore[index]
            if out_acts is not None and self._activations is not None:
                out_acts[sel] = self._activations[idx[sel]]

        return cache_hits, out_tokens, out_logits, out_acts

    def populate(
        self,
        keys: torch.Tensor,         # [N, 3] int64
        tokens: torch.Tensor,       # [N, K] int64
        logits: torch.Tensor,       # [N, K, V] dtype
        activations: torch.Tensor | None = None,  # [N, K, H] dtype
    ) -> None:
        """Replace the entire cache contents with the new tree decode results."""
        self._keys = keys
        self._tokens = tokens
        self._logits = logits
        self._activations = activations

    # ------------------------------------------------------------------
    # Properties (read-only views for the draft worker)
    # ------------------------------------------------------------------

    @property
    def size(self) -> int:
        return int(self._keys.shape[0])

    @property
    def is_empty(self) -> bool:
        return self._keys.numel() == 0
