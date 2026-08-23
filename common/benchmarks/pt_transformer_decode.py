import torch
import torch.nn as nn
import time
import gc


def run():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type != 'cuda':
        raise RuntimeError("CUDA is not available!")

    gc.collect()
    torch.cuda.empty_cache()

    # Small GPT-style decoder run AUTOREGRESSIVELY -- one new token per
    # forward pass, feeding the growing sequence back in -- rather than
    # pt_bert's single-shot encoder pass over a fixed-size batch. Real LLM
    # inference actually looks like this: many small, sequential kernel
    # launches instead of one large batched one. Current DVFS/power research
    # specifically calls out LLM inference as its own category distinct from
    # the encoder-style NLP workloads already in this suite.
    vocab, d_model, n_head, n_layers = 8000, 512, 8, 6
    layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=n_head, batch_first=True)
    decoder = nn.TransformerDecoder(layer, num_layers=n_layers).to(device)
    embed = nn.Embedding(vocab, d_model).to(device)
    lm_head = nn.Linear(d_model, vocab).to(device)
    memory = torch.randn(1, 16, d_model, device=device)  # fixed "encoder output" context

    def step(seq):
        hidden = decoder(embed(seq), memory)
        return lm_head(hidden)

    tokens = torch.randint(0, vocab, (1, 8), device=device)

    # Warmup
    with torch.no_grad():
        for _ in range(3):
            step(tokens)

    # Benchmark: generate 60 tokens one at a time, capping context length
    torch.cuda.synchronize()
    start = time.time()
    with torch.no_grad():
        seq = tokens
        for _ in range(60):
            logits = step(seq)
            next_tok = logits[:, -1:, :].argmax(dim=-1)
            seq = torch.cat([seq, next_tok], dim=1)
            if seq.shape[1] > 64:
                seq = seq[:, -64:]
    torch.cuda.synchronize()
    duration = time.time() - start
    print(f"Duration: {duration:.4f}")

    del decoder, embed, lm_head, memory, tokens
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == '__main__':
    run()
