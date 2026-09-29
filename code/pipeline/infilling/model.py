"""Frozen LLaDA wrapper for infilling: logits at chosen positions only.

LLaDA's forward computes logits at every position, about 700 x 126k values
per sequence (177 MB in fp16). The sampler needs them only at the masked
positions and PLL only at the probed one. A hook stops the pass after the
last block, and the final norm and output layer run on those positions alone.
check() compares this against the model's own logits once at startup and
falls back to the full forward if they differ.

Sampling bans special tokens, the mask token, ids past the tokenizer, and
tokens that decode to text with a newline, since a repaired sentence must
stay one line of running text.
"""

from collections import Counter

import torch

from classifier.extract import Stop, attach, find_blocks, load_model

MASK_ID = 126336  # LLaDA's <|mdm_mask|>


def find_module(model, suffix):
    for name, module in model.named_modules():
        if name.endswith(suffix):
            return module
    return None


def banned_ids(tok, vocab):
    ids = set(tok.all_special_ids) | {MASK_ID} | set(range(len(tok), vocab))
    for i in range(min(len(tok), vocab)):
        if "\n" in tok.decode([i]):
            ids.add(i)
    mask = torch.zeros(vocab, dtype=torch.bool)
    mask[sorted(ids)] = True
    return mask


class Model:
    def __init__(self, name, quant, tok):
        self.net = load_model(name, quant)
        _, self.blocks = find_blocks(self.net)
        self.device = next(self.net.parameters()).device
        self.ln_f = find_module(self.net, "transformer.ln_f")
        self.ff_out = find_module(self.net, "transformer.ff_out")
        self.fast = self.ln_f is not None and self.ff_out is not None
        self.passes = Counter()
        self.check(tok)
        self.banned = banned_ids(tok, self.vocab).to(self.device)

    @torch.inference_mode()
    def hidden(self, ids, layer):
        """Output of block `layer` (1-based) for a [B, L] batch."""
        store = {}
        handles = attach(self.blocks, [layer], store)
        try:
            self.net(input_ids=ids)
        except Stop:
            pass
        finally:
            for handle in handles:
                handle.remove()
        return store[layer]

    @torch.inference_mode()
    def logits_at(self, ids, positions, tag):
        """Float logits [B, K, V] at `positions` [B, K] of the [B, L] batch `ids`."""
        self.passes[tag] += ids.shape[0]
        if self.fast:
            h = self.hidden(ids, len(self.blocks))
            h = h.gather(1, positions[..., None].expand(-1, -1, h.shape[-1]))
            return self.ff_out(self.ln_f(h)).float()
        logits = self.net(input_ids=ids).logits
        return logits.gather(1, positions[..., None].expand(-1, -1, logits.shape[-1])).float()

    @torch.inference_mode()
    def check(self, tok):
        text = "The fire killed three people and destroyed the factory."
        ids = torch.tensor([tok(text, add_special_tokens=False)["input_ids"]], device=self.device)
        ids[0, 2] = MASK_ID
        full = self.net(input_ids=ids).logits.float()
        self.vocab = full.shape[-1]
        if not self.fast:
            print("infilling: no ln_f/ff_out modules found, using full-vocabulary logits")
            return
        positions = torch.arange(ids.shape[1], device=self.device)[None]
        fast = self.logits_at(ids, positions, "check")
        error = ((fast - full).abs().max() / full.abs().max().clamp_min(1e-6)).item()
        if not error < 1e-2:
            print(f"infilling: position-restricted logits differ from the model's "
                  f"(relative error {error:.3g}), using full-vocabulary logits")
            self.fast = False
