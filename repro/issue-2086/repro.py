"""In-process repro for Mooncake #2086 (TCP WRITE completion signaled early).

Both TransferEngine instances live in ONE process, so the initiator can read the
destination buffer DIRECTLY (same address space) the instant
`batch_transfer_sync_write` returns — no network read-back to mask the window.
The receiver applies bytes on its own io_context worker thread, so if completion
is signaled on the sender's socket-write (the hypothesis), a direct read right
after the call catches the destination mid-apply.

Hypothesis test (rc==0 means the write SAID it succeeded):
  - mismatch : rc==0 but dst != src right after the call   -> not fully applied
  - unstable : two back-to-back reads of dst differ          -> bytes still landing
  - control  : after a short quiesce, re-read the bad regions
       healed   -> CONFIRMED premature completion (data was just late)
       stillbad -> REFUTED; persistent corruption, not a timing issue

CPU buffers by default (the bug is transport-level; no GPU needed, no CUDA
stream-race in the observation). --device cuda widens the window via the
receiver's cudaMemcpy bounce.

Usage:
    python repro.py --concurrency 8 --iters 10
    python repro.py --device cuda --size 2424832 --concurrency 8
"""

import os

# Bound per-transfer wait so a stuck transfer can't hang for the 30s default.
os.environ.setdefault("MC_TRANSFER_TIMEOUT", "5")

import argparse
import concurrent.futures
import hashlib
import time

import torch
from mooncake.engine import TransferEngine

p = argparse.ArgumentParser()
p.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
p.add_argument("--size", type=int, default=2424832)  # from the #2086 report
p.add_argument("--concurrency", type=int, default=8)
p.add_argument("--iters", type=int, default=10)
p.add_argument("--quiesce-sec", type=float, default=0.5)
args = p.parse_args()

if args.device == "cuda":
    torch.cuda.set_device(0)


def h(t: torch.Tensor) -> str:
    return hashlib.blake2b(t.cpu().numpy().tobytes(), digest_size=16).hexdigest()


def make_engine(name: str) -> TransferEngine:
    e = TransferEngine()
    rc = e.initialize(name, "P2PHANDSHAKE", "tcp", "")
    assert rc == 0, f"initialize failed: {rc}"
    return e


# Port 0 -> P2PHANDSHAKE assigns a real rpc port, read back via get_rpc_port().
tgt = make_engine("127.0.0.1:0")
ini = make_engine("127.0.0.1:0")
tgt_name = f"127.0.0.1:{tgt.get_rpc_port()}"
print(f"target rpc port: {tgt.get_rpc_port()}", flush=True)

K, size = args.concurrency, args.size
dst = torch.zeros(K * size, dtype=torch.uint8, device=args.device)
src = torch.empty(K * size, dtype=torch.uint8, device=args.device)
for i in range(K):
    src[i * size:(i + 1) * size] = (i % 251) + 1  # nonzero, region-unique
assert tgt.register_memory(dst.data_ptr(), dst.numel()) == 0
assert ini.register_memory(src.data_ptr(), src.numel()) == 0
src_h = [h(src[i * size:(i + 1) * size]) for i in range(K)]


def write_and_check(i: int):
    """Write region i, then read it back IMMEDIATELY on this same thread —
    while sibling writes are still in flight saturating the receiver's single
    worker thread, so region i's apply tail may not have landed yet."""
    rc = ini.batch_transfer_sync_write(
        tgt_name, [src.data_ptr() + i * size], [dst.data_ptr() + i * size], [size]
    )
    if rc != 0:
        return (i, rc, None, None)
    h1 = h(dst[i * size:(i + 1) * size])
    h2 = h(dst[i * size:(i + 1) * size])
    return (i, rc, h1, h2)


pool = concurrent.futures.ThreadPoolExecutor(max_workers=K)
mismatch = unstable = healed = stillbad = ok_writes = failed_writes = 0

for it in range(args.iters):
    dst.zero_()
    if args.device == "cuda":
        torch.cuda.synchronize()

    results = [f.result() for f in [pool.submit(write_and_check, i) for i in range(K)]]

    bad = []
    for (i, rc, h1, h2) in results:
        if rc != 0:
            failed_writes += 1
            continue
        ok_writes += 1
        if h1 != src_h[i]:
            mismatch += 1
            bad.append(i)
        if h1 != h2:
            unstable += 1

    if bad:
        time.sleep(args.quiesce_sec)
        if args.device == "cuda":
            torch.cuda.synchronize()
        for i in bad:
            if h(dst[i * size:(i + 1) * size]) == src_h[i]:
                healed += 1
            else:
                stillbad += 1

    print(f"[iter {it}] ok={ok_writes} failed={failed_writes} mismatch={mismatch} "
          f"unstable={unstable} healed={healed} stillbad={stillbad}", flush=True)

print(f"\n=== SUMMARY size={size} concurrency={K} device={args.device} ===")
print(f"  writes rc==0           : {ok_writes}")
print(f"  writes rc!=0 (timeout) : {failed_writes}")
print(f"  rc==0 but dst != src   : {mismatch}")
print(f"  unstable across 2 reads: {unstable}")
print(f"  healed after quiesce   : {healed}")
print(f"  still bad after quiesce: {stillbad}")
if mismatch and healed and not stillbad:
    print("  => CONFIRMED: WRITE completion signaled before receiver applied data.")
elif stillbad:
    print("  => REFUTED: persistent corruption; not a completion-timing issue.")
elif mismatch == 0 and ok_writes:
    print("  => NOT OBSERVED at this size/concurrency; raise --size or --concurrency.")
else:
    print("  => INCONCLUSIVE (check write failures above).")

# The two engines' io_context worker threads don't join on interpreter exit and
# hang the process; the measurement is done, so force a clean exit.
import sys
sys.stdout.flush()
os._exit(0 if (mismatch == 0 or (healed and not stillbad)) else 2)
