# Technical Analysis

This page documents the key design decisions and the root causes of training challenges
identified during development.

---

## Why the model collapses when training on multiple Pokémon

### Context

Experiment 1 (`experiment_1_word_level_tokenizer.ipynb`) demonstrates that the model **can learn
to reproduce a single sprite** when trained on one Pokémon. However, scaling to the full dataset
(~1 000 Pokémon) causes the model to collapse and generate only the `~` token (empty background)
for any input.

---

## Design Decisions

### Sequence Format

```python
text_split = [
    ["%02d" % pos] + row.split()[1:]
    for pos, row in enumerate(text_split)
    if not all([char == "~" for char in row.split()])
]
```

The design swaps the first pixel of each row for a row number, keeping each row at exactly
64 tokens (1 row number + 63 pixels). This allows the full sprite to fit within the 4 096-token
context window. The first pixel carries little information for most Pokémon, making the trade-off
reasonable.

### Sprite Size and Context Window

Sprites are processed at 64×64 resolution (4 096 pixels) so that the entire sprite fits within
the context window. This is important because the model needs to see the complete sprite when
generating each new pixel — a smaller window would break the global coherence of the figure.

In practice, after filtering blank rows, sequences are ~2 000–3 000 tokens, leaving headroom
for row numbers and special tokens.

### Pixel-level Tokenization

The tokenizer operates at the pixel level: each token represents exactly one pixel.
The vocabulary therefore contains only as many tokens as there are possible pixel values
plus special tokens — much smaller than BPE vocabularies (~50k tokens in standard GPT-2).

| Approach | Vocabulary size | Ambiguity |
|----------|:-:|:-:|
| Pixel-level (this project) | 65 tokens | None |
| BPE (GPT-2 default) | ~50 000 tokens | Possible |

**Rejected alternative — Run-Length Encoding (RLE)**

RLE was considered to compress runs of consecutive `~` pixels within each row, which would
reduce sequence length and bring "interesting" tokens closer together.

It was rejected because:

- Sequences already fit within 4 096 tokens after filtering blank rows.
- The root cause of collapse is model size and lack of conditioning, not the proportion of `~`.
- The weighted loss (`ForCausalLMLossWeighed`) already compensates for `~` dominance.
- RLE adds complexity to the tokenization pipeline without addressing the main issues.

---

## Root Causes of Training Collapse

!!! warning "Main finding"
    The collapse is caused by a combination of model capacity and insufficient conditioning —
    not by tokenization choices.

### Cause 1 — Loss dominated by background tokens

Background tokens (`~`) make up the majority of each sprite sequence. Without loss weighting,
the model learns to predict `~` for everything and gets a low loss.

**Fix**: Down-weight `~` tokens by ×0.6 in the `ForCausalLMLossWeighed` loss function.

### Cause 2 — Conditioning not strong enough

When training on a single Pokémon, the identity embedding is sufficient. At scale, the model
needs to rely on type, generation, and evolution stage embeddings to differentiate outputs.

**Fix**: Add Gaussian noise (σ = 0.1) to conditioning vectors during training to prevent
the model from over-relying on identity and force it to learn from the metadata embeddings.

### Cause 3 — Row reconstruction during inference

The row number swapped into position 0 during tokenization must be correctly reversed during
inference to reconstruct the sprite. A bug here results in row-shifted outputs that look like
noise.

**Fix**: Ensure the inference pipeline correctly maps row-number tokens back to pixel values
during reconstruction.
