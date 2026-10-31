"""Scratch storage for small asynchronous host → CUDA control packets."""

from __future__ import annotations

import torch


class HostReturnPacket:
    """Keep pinned source memory alive and immutable until its DMA completes.

    A packet is scratch, never checkpoint state. Each returned CUDA tensor owns
    its storage: subsequent uploads must not mutate earlier well systems.
    """

    def __init__(self):
        self._host = None
        self._event = None
        self._device = None

    def __getstate__(self):
        return {}

    def __setstate__(self, _state):
        self.__init__()

    def upload(self, array, reference):
        if not reference.is_cuda:
            return reference.new_tensor(array)
        with torch.cuda.device(reference.device):
            # Normal FIM assembly has already drained the preceding transfer
            # through its D2H packet. Also protect explicit cross-stream use.
            if self._event is not None and not self._event.query():
                self._event.synchronize()
            source = torch.as_tensor(array, dtype=reference.dtype, device="cpu")
            if (
                self._host is None
                or self._host.shape != source.shape
                or self._host.dtype != source.dtype
            ):
                self._host = torch.empty(
                    source.shape, dtype=source.dtype, device="cpu", pin_memory=True
                )
            if self._event is None or self._device != reference.device:
                self._event = torch.cuda.Event(enable_timing=False)
                self._device = reference.device
            self._host.copy_(source)
            result = self._host.to(reference.device, non_blocking=True)
            self._event.record(torch.cuda.current_stream(reference.device))
            return result
