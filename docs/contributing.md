# Contributing

## Getting Started

Clone the repository and install dependencies with [uv](https://github.com/astral-sh/uv):

```bash
git clone git@github.com:DelgadoPanadero/GPokeT2.git
cd GPokeT2
uv sync --group dev
```

## Project Structure

The project follows a Domain-Driven Design (DDD) + medallion architecture:

```
src/
  domain/          # Entities and domain logic (no external dependencies)
  application/
    brz/           # Bronze layer: raw data ingestion
    slv/           # Silver layer: image → ASCII encoding
    gld/           # Gold layer: tokenization
    train/         # Training: model, collator, loss, callbacks
```

## Running Tests

```bash
uv run pytest tests/ -v
```

!!! info "Optional dependencies"
    Tests that depend on `torch` or `transformers` are automatically skipped if those packages
    are not installed. Only `test_pokemon_encoder.py` runs in the minimal environment
    (numpy + opencv).

## Code Style

The project uses [ruff](https://docs.astral.sh/ruff/) for linting and formatting:

```bash
uv run ruff check src/ tests/
uv run ruff format src/ tests/
```

## Data Pipeline

The full pipeline runs in stages:

1. **Bronze** (`gotta_catch_em_all_*.py`): download raw Pokémon sprites from the source.
2. **Silver** (`PokemonEncoder`): convert each PNG to an ASCII sprite (BGR quantization, blank detection).
3. **Gold** (`Pokenizer`): train a WordLevel tokenizer and produce HuggingFace datasets.
4. **Train** (`PokemonTrainer`): fine-tune `ConditionedGPT2` with a per-Pokémon conditioning embedding.

## Submitting Changes

1. Fork the repository.
2. Create a branch: `git checkout -b feat/your-feature`.
3. Keep commits focused; one logical change per commit.
4. Open a pull request against `main` with a clear description of what changed and why.

## Reporting Issues

Open an issue at [github.com/DelgadoPanadero/GPokeT2/issues](https://github.com/DelgadoPanadero/GPokeT2/issues)
describing the problem, the steps to reproduce it, and the environment (OS, Python version, GPU if relevant).
