# Usage

## Installation

Install the required dependencies:

```bash
pip install transformers huggingface_hub opencv-python torch
```

## Generating a Sprite

```python
import cv2
import numpy as np
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM, PreTrainedTokenizerFast

# Load model
ckpt = snapshot_download("iamthinbaker/GPokeT2")
tokenizer = PreTrainedTokenizerFast.from_pretrained(ckpt)
model = AutoModelForCausalLM.from_pretrained(ckpt, trust_remote_code=True)

# Generate Pokémon
image = model.generate_sprite(
    tokenizer,
    type1="fire",
    type2="dragon",
    verbose=True,
)

# Save image
cv2.imwrite("pokemon.png", cv2.cvtColor(np.uint8(image), cv2.COLOR_RGB2BGR))
```

## Available Types

You can condition the model on any combination of the 18 Pokémon types:

| | | |
|---|---|---|
| ⬜ `normal` | 🥊 `fighting` | 🔮 `psychic` |
| 🔥 `fire` | ☠️ `poison` | 🐛 `bug` |
| 💧 `water` | 🌍 `ground` | 🪨 `rock` |
| ⚡ `electric` | 🌪️ `flying` | 👻 `ghost` |
| 🌿 `grass` | 🐉 `dragon` | 🌑 `dark` |
| 🧊 `ice` | ⚙️ `steel` | 🧚 `fairy` |

!!! tip "Single-type Pokémon"
    Leave `type2` unset (or set it to `None`) to generate a single-type Pokémon.

## Additional Parameters

The model accepts several conditioning parameters beyond type:

| Parameter | Description |
|-----------|-------------|
| `type1` | Primary Pokémon type |
| `type2` | Secondary type (optional) |
| `generation` | Game generation (1–4) |
| `evolution_stage` | Basic / Stage 1 / Stage 2 |
| `has_evolution` | Whether the Pokémon can still evolve |
| `is_shiny` | Generate a shiny palette variant |

## Example Gallery

Some Pokémon generated with different type combinations:

| | | | |
|:---:|:---:|:---:|:---:|
| <img src="https://raw.githubusercontent.com/DelgadoPanadero/GPokeT2/main/data/gld/pokemons/pokemon_dragon-fire_sh0_g3_ev1_he0_cfd3ze.png" width="96"/><br>`dragon / fire` | <img src="https://raw.githubusercontent.com/DelgadoPanadero/GPokeT2/main/data/gld/pokemons/pokemon_electric-flying_sh0_g4_ev1_he0.png" width="96"/><br>`electric / flying` | <img src="https://raw.githubusercontent.com/DelgadoPanadero/GPokeT2/main/data/gld/pokemons/pokemon_grass-ghost_sh0_g4_ev1_he1_1aklfk.png" width="96"/><br>`grass / ghost` | <img src="https://raw.githubusercontent.com/DelgadoPanadero/GPokeT2/main/data/gld/pokemons/pokemon_ice-psychic_sh0_g4_ev0_he0.png" width="96"/><br>`ice / psychic` |
| <img src="https://raw.githubusercontent.com/DelgadoPanadero/GPokeT2/main/data/gld/pokemons/pokemon_ghost-dragon_sh0_g4_ev0_he1.png" width="96"/><br>`ghost / dragon` | <img src="https://raw.githubusercontent.com/DelgadoPanadero/GPokeT2/main/data/gld/pokemons/pokemon_steel-dragon_sh0_g4_ev1_he0.png" width="96"/><br>`steel / dragon` | <img src="https://raw.githubusercontent.com/DelgadoPanadero/GPokeT2/main/data/gld/pokemons/pokemon_fairy-steel_sh0_g4_ev1_he1.png" width="96"/><br>`fairy / steel` | <img src="https://raw.githubusercontent.com/DelgadoPanadero/GPokeT2/main/data/gld/pokemons/pokemon_water-bug_sh0_g3_ev1_he1_jez4il.png" width="96"/><br>`water / bug` |
