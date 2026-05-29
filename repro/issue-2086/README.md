# Repro for #2086 — TCP WRITE completion signaled before the receiver applies the data

Standalone TransferEngine-only repro (no vLLM, no RDMA, single host) for the `dst != src` with `src` stable reported in [#2086](https://github.com/kvcache-ai/Mooncake/issues/2086).

## What it shows

`batch_transfer_sync_write` on the TCP transport returns success as soon as the *sender* finishes writing to the socket, before the *receiver* has applied the bytes to the destination. Under concurrent load the receiver's single `io_context` worker thread lags, so a read of the destination right after the call returns can see the apply tail still landing.

In `mooncake-transfer-engine/src/transport/tcp_transport/tcp_transport.cpp`:

- WRITEs are sent in `kDefaultBufferSize = 65536` (64 KiB) chunks.
- The sender (`ClientSession::writeBody`) marks the slice COMPLETED when its own `async_write` of the last chunk drains — `on_finalize_(COMPLETED)` → `slice->markSuccess()`.
- The receiver (`ServerSession::readBody`) reads chunks into the destination and loops for the next request without any application-level ACK.

`2424832 = 37 x 65536` exactly — 37 chunks, a wide apply window. A one-chunk transfer has essentially none, which is why small descriptors stay consistent and large ones do not.

## Method

Two `TransferEngine` instances in one process over TCP loopback, so the destination is read directly (same address space) the instant `sync_write` returns — no network read-back to mask the window. After each write returns `rc == 0`, its region is read back twice on the same thread while sibling writes are still in flight. A short quiesce then re-reads the mismatched regions: if they now match the source, the data was merely late (premature completion); if they stay wrong, it would be real corruption.

## Run

```bash
pip install torch numpy mooncake-transfer-engine   # CUDA 12.1+ build
python repro.py --device cuda --concurrency 16 --iters 20
```

`--device cpu` also works; the window is narrower without the receiver's per-chunk `cudaMemcpy`, so device-memory destinations surface it most readily.

## Observed (mooncake-transfer-engine==0.3.10.post2, L4, TCP loopback)

```
size=2424832 concurrency=16 device=cuda
  writes rc==0           : 320
  rc==0 but dst != src   : 156
  unstable across 2 reads: 146
  healed after quiesce   : 156
  still bad after quiesce: 0
  => CONFIRMED: WRITE completion signaled before receiver applied data.
```
