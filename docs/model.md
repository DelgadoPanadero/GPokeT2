# Model Details

## Dataset

The training data covers all sprites from every mainline **Gen 3** and **Gen 4** game:

| Generation | Game | Sprites |
|:----------:|------|--------:|
| Gen 3 | Pokémon Emerald | 1 600 |
| Gen 3 | Pokémon FireRed / LeafGreen | 312 |
| Gen 3 | Pokémon Ruby / Sapphire | 837 |
| Gen 4 | Pokémon Diamond / Pearl | 2 528 |
| Gen 4 | Pokémon Platinum | 2 556 |
| Gen 4 | Pokémon HeartGold / SoulSilver | 2 560 |
| **Total** | | **10 393** |

### Data Augmentation

Each sprite is augmented to produce **12 variants** before training:

| Technique | Variants | Description |
|-----------|:--------:|-------------|
| Horizontal flip | ×2 | Each sprite is mirrored left↔right at the ASCII level |
| Color shift | ×6 | All 5 non-identity permutations of the RGB channels |

These augmentations are independent and combined: 1 original sprite → 2 flip variants × 6 color
variants = **12 total samples** — giving a final training set of **~124 700 sequences**.

Data sourced from:

- [PokéAPI](https://pokeapi.co/) — Pokémon metadata (types, generations, evolution chains)
- [Veekun](https://veekun.com/) — original 64×64 PNG sprites

---

## Pixel → ASCII Encoding

Each 64×64 sprite is serialized as a sequence of ASCII characters before being fed to the model.
Each pixel is quantized to **4 levels per channel** (R, G, B ∈ {0, 1, 2, 3}) and packed into a
single character:

```python
char = chr(R * 16 + G * 4 + B + 59)  # 64 possible color chars
char = '~'                             # white / transparent pixel
```

This yields a vocabulary of **65 pixel tokens** (one per color + `~` for background), plus special
row-marker tokens (`[ROW_00]`…`[ROW_63]`) that delimit each row of 64 pixels.
A full sprite is therefore a sequence of 64 rows × 64 pixels = **4 096 tokens**.

| Original sprite | ASCII representation |
|:---------------:|:--------------------:|
| <img src="sprite_image.png" width="200"/> | <img src="sprite_ascii.png" width="200"/> |

---

## GPT-2 Architecture

| Parameter | Value |
|-----------|-------|
| Context length | 4 096 |
| Embedding dim | 512 |
| Layers | 12 |
| Attention heads | 8 |

---

## Conditioning Embeddings

Every token in the sequence receives a sum of learned embeddings that condition the generation:

| Embedding | Categories | Description |
|-----------|:----------:|-------------|
| Pokémon identity | up to *N* | Unique embedding per Pokémon; can be interpolated |
| Type 1 | 19 | Primary type (18 types + unknown) |
| Type 2 | 20 | Secondary type (18 types + none + unknown) |
| Generation | 10 | Game generation (Gen I–IX + margin) |
| Evolution stage | 4 | Basic / Stage 1 / Stage 2 / other |
| Has evolution | 2 | Whether the Pokémon can still evolve |
| Is shiny | 2 | Normal vs. shiny palette |
| Color shift | 6 | Which RGB permutation was applied (augmentation label) |
| Row position | 65 | Which row (0–63) the current token belongs to |
| Column position | 65 | Which column (0–63) within the row |

!!! note "Training regularization"
    During training, a small Gaussian noise (σ = 0.1) is added to the conditioning vector to
    improve robustness. Background tokens (`~`) are also down-weighted (×0.6) in the loss so
    the model focuses on learning colored pixels.

---

## Training

| | |
|---|---|
| **Platform** | [RunPod](https://www.runpod.io/) |
| **GPU** | NVIDIA RTX A4000 (16 GB VRAM) |
| **CUDA** | 12.4 |
| **Steps** | 5 505 |
| **Training time** | ~53 hours |
| **Cost** | ~$0.26 / hour · **~$10 total** |
| **Precision** | BF16 |
| **Optimizer** | AdamW with cosine LR scheduler |
| **Gradient checkpointing** | ✅ |

---

## Acknowledgements

Inspired by [matthewRayfield/pokemon-gpt-2](https://github.com/matthewRayfield/pokemon-gpt-2),
which first explored the idea of generating Pokémon sprites with GPT-2.
This project builds on that concept with a custom-trained model, richer metadata conditioning,
and a tokenizer designed specifically for sprite sequences.
