# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fused int64 NCCL helpers for SSD async draft communication.

The target and draft processes exchange multiple small tensors each step.
Fusing them into a single contiguous int64 buffer reduces NCCL overhead from
N separate send/recv calls to one, matching the approach in the SSD reference
implementation (tanishqkumar/ssd).

Protocol (enforced by the target/draft loop):
  - Target ALWAYS sends first (cmd), draft ALWAYS recvs first.
  - All tensors are int64 on the same CUDA device as the process group.
"""

import torch
import torch.distributed as dist


def _pg_send(pg: dist.ProcessGroup, tensor: torch.Tensor, dst: int) -> None:
    """Send *tensor* to *dst* using the raw ProcessGroup API.

    Unlike ``dist.send``, this does NOT require the process group to be
    registered in PyTorch's global group map, which is required when the
    group is created via ``ProcessGroupNCCL(...)`` directly rather than via
    ``dist.new_group()``.
    """
    pg.send([tensor.contiguous()], dst, 0).wait()


def _pg_recv(pg: dist.ProcessGroup, tensor: torch.Tensor, src: int) -> None:
    """Receive into *tensor* from *src* using the raw ProcessGroup API."""
    pg.recv([tensor], src, 0).wait()


def send_int64(
    pg: dist.ProcessGroup,
    dst: int,
    *tensors: torch.Tensor,
) -> None:
    """Concatenate *tensors* (all int64, any shape) into one flat buffer and
    send it to *dst* in a single NCCL call.

    All tensors must already be int64 and on the same device.
    """
    flat_parts = [t.reshape(-1) for t in tensors]
    fused = torch.cat(flat_parts)
    _pg_send(pg, fused, dst)


def recv_int64(
    pg: dist.ProcessGroup,
    src: int,
    total_length: int,
    device: torch.device,
) -> torch.Tensor:
    """Receive a flat int64 tensor of exactly *total_length* elements from
    *src* and return it.  The caller is responsible for slicing / reshaping.
    """
    buf = torch.empty(total_length, dtype=torch.int64, device=device)
    _pg_recv(pg, buf, src)
    return buf


def send_cmd(
    pg: dist.ProcessGroup,
    dst: int,
    cmd: int,
    device: torch.device,
    _buf: torch.Tensor | None = None,
) -> torch.Tensor:
    """Send a single int64 command scalar to *dst*.

    Returns the (possibly reused) buffer tensor so callers can cache it and
    avoid re-allocating each step.
    """
    if _buf is None:
        _buf = torch.zeros(1, dtype=torch.int64, device=device)
    _buf[0] = cmd
    _pg_send(pg, _buf, dst)
    return _buf


def recv_cmd(
    pg: dist.ProcessGroup,
    src: int,
    device: torch.device,
    _buf: torch.Tensor | None = None,
) -> tuple[int, torch.Tensor]:
    """Receive a single int64 command scalar from *src*.

    Returns ``(cmd_value, buffer)`` so the buffer can be cached.
    """
    if _buf is None:
        _buf = torch.zeros(1, dtype=torch.int64, device=device)
    _pg_recv(pg, _buf, src)
    return int(_buf[0].item()), _buf
