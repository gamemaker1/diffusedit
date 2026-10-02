"""LLaDA infilling sampler with low-confidence remasking (Section 2.2).

Only the given positions are masked. Each step runs one forward pass,
predicts every still-masked position, and commits the predictions the model
is most confident about. The rest stay masked for the next step, where they
see the tokens just committed. K masks take min(T, K) steps, with K // n
commits per step and the remainder spread over the first steps, as in
LLaDA's reference generate().

Temperature 0 takes the arg-max token. Above 0, tokens are drawn with LLaDA's
Gumbel noise, exp(logits) / (-log u) ** temperature.
"""

import torch

from infilling.model import MASK_ID


def schedule(k, steps):
    """Commits per step for k masks."""
    if k == 0:
        return []
    n = min(steps, k)
    return [k // n + (i < k % n) for i in range(n)]


@torch.inference_mode()
def sample(model, ids, positions, steps, temperature=0.0, generator=None):
    """Fill `positions` of the 1-D sequence `ids`.

    Returns the filled sequence and, per position in `positions` order, the
    probability the model gave the token it committed there.
    """
    x = ids.clone()
    x[positions] = MASK_ID
    remaining = torch.arange(len(positions), device=ids.device)
    confidence = torch.zeros(len(positions), device=ids.device)
    for n in schedule(len(positions), steps):
        logits = model.logits_at(x[None], positions[remaining][None], "sample")[0]
        logits[:, model.banned] = float("-inf")
        probs = logits.softmax(-1)
        if temperature > 0:
            noise = torch.rand(logits.shape, generator=generator, device=logits.device, dtype=torch.float64)
            pred = (logits.double().exp() / (-noise.log()) ** temperature).argmax(-1)
        else:
            pred = logits.argmax(-1)
        conf = probs.gather(1, pred[:, None])[:, 0]
        top = conf.topk(n).indices
        x[positions[remaining[top]]] = pred[top]
        confidence[remaining[top]] = conf[top]
        keep = torch.ones(len(remaining), dtype=torch.bool, device=ids.device)
        keep[top] = False
        remaining = remaining[keep]
    return x, confidence
