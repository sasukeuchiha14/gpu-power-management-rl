"""
live_transformer_decode.py -- continuous version of pt_transformer_decode.py
for live A/B testing against common/deploy_live.py, same status-file
contract as live_resnet50_train.py / live_glmark2_render.py.

Why this workload specifically: live_resnet50_train.py is one large batched
forward+backward pass per step (dense, highly parallel). This instead runs
a small GPT-style decoder AUTOREGRESSIVELY -- one new token per forward
pass, many small sequential kernel launches instead of one big one -- the
same distinction pt_transformer_decode.py draws against pt_bert's
single-shot encoder pass. Real LLM inference looks like this, and DVFS/
power research treats it as its own category. Context is capped and reset
periodically so this runs indefinitely instead of pt_transformer_decode.py's
fixed 60-token generation.

    python common/benchmarks/live_transformer_decode.py --minutes 15
    python common/benchmarks/live_transformer_decode.py --minutes 15 --status-file /tmp/dvfs_workload_status.json
"""
import argparse
import gc
import json
import os
import signal
import time

import torch
import torch.nn as nn

DEFAULT_STATUS_FILE = "/tmp/dvfs_workload_status.json"
WINDOW = 20  # generated tokens averaged per throughput report
MAX_CONTEXT = 64
RESET_AFTER = 256  # tokens generated before resetting the sequence, bounding memory growth indefinitely


def write_status(path, step, ms_per_iter, iters_per_sec):
    """Atomic write (temp file + rename), same contract as
    live_resnet50_train.py.write_status -- deploy_live.py reads this file
    regardless of which workload produced it."""
    tmp = f"{path}.tmp"
    with open(tmp, "w") as f:
        json.dump({
            "timestamp": time.time(),
            "step": step,
            "ms_per_iter": ms_per_iter,
            "iters_per_sec": iters_per_sec,
        }, f)
    os.replace(tmp, path)


def run(minutes, status_file):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("CUDA is not available!")

    vocab, d_model, n_head, n_layers = 8000, 512, 8, 6
    layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=n_head, batch_first=True)
    decoder = nn.TransformerDecoder(layer, num_layers=n_layers).to(device)
    embed = nn.Embedding(vocab, d_model).to(device)
    lm_head = nn.Linear(d_model, vocab).to(device)
    memory = torch.randn(1, 16, d_model, device=device)  # fixed "encoder output" context

    def step_fn(seq):
        hidden = decoder(embed(seq), memory)
        return lm_head(hidden)

    seq = torch.randint(0, vocab, (1, 8), device=device)

    stop = {"flag": False}

    def handle_exit(signum, frame):
        stop["flag"] = True

    signal.signal(signal.SIGINT, handle_exit)
    signal.signal(signal.SIGTERM, handle_exit)

    # Warmup -- not timed, same reasoning as every offline benchmark.
    with torch.no_grad():
        for _ in range(3):
            step_fn(seq)
    torch.cuda.synchronize()

    print(f"[live_transformer_decode] running for {minutes} min, "
          f"reporting every {WINDOW} tokens -> {status_file}", flush=True)

    deadline = time.time() + minutes * 60.0
    step = 0
    tokens_since_reset = 0
    window_start = time.perf_counter()

    with torch.no_grad():
        while time.time() < deadline and not stop["flag"]:
            logits = step_fn(seq)
            next_tok = logits[:, -1:, :].argmax(dim=-1)
            seq = torch.cat([seq, next_tok], dim=1)
            if seq.shape[1] > MAX_CONTEXT:
                seq = seq[:, -MAX_CONTEXT:]
            step += 1
            tokens_since_reset += 1

            if tokens_since_reset >= RESET_AFTER:
                seq = torch.randint(0, vocab, (1, 8), device=device)
                tokens_since_reset = 0

            if step % WINDOW == 0:
                torch.cuda.synchronize()
                elapsed = time.perf_counter() - window_start
                ms_per_iter = 1000.0 * elapsed / WINDOW
                iters_per_sec = WINDOW / elapsed
                write_status(status_file, step, ms_per_iter, iters_per_sec)
                window_start = time.perf_counter()

    del decoder, embed, lm_head, memory, seq
    gc.collect()
    torch.cuda.empty_cache()
    print(f"[live_transformer_decode] stopped at step {step}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--minutes", type=float, default=15.0)
    ap.add_argument("--status-file", default=DEFAULT_STATUS_FILE)
    a = ap.parse_args()
    run(a.minutes, a.status_file)


if __name__ == "__main__":
    main()
