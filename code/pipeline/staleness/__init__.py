"""Token-level staleness scorer (Eq. 3 / paper Eq. 7).

Label computation (stale_marks) and feature extraction both live in
classifier.inputs / classifier.extract now -- Vedant folded the token pass
into the same forward pass classifier.extract already runs for sentence
pooling, so there is nothing separate to run here. train.py reads that
output directly (tokens/layer{L}.npy, tokens/labels.npy, config.json's
"token_layer") and trains the sigmoid head.
"""
