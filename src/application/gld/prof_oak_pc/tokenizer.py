import json
from typing import List, Tuple

import numpy as np
import pyarrow as pa
from datasets import Dataset, DatasetDict
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.normalizers import NFKC
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import WhitespaceSplit

from src.domain.slv.pokedex import PokedexEntity


class Pokenizer:
    """
    Clase que encapsula la lógica para entrenar y usar un tokenizador
    personalizado para datos de Pokémon.
    """

    BOS_TOKEN = "[BOS]"
    EOS_TOKEN = "[EOS]"
    UNK_TOKEN = "[UNK]"
    PAD_TOKEN = "[PAD]"
    EOL_TOKEN = "00"
    BCK_TOKEN = "~"

    def __init__(
        self,
        context_length: int = 64,
    ):
        """
        Inicializa el tokenizador con las configuraciones base.
        """
        self.context_length = context_length
        self._tokenizer = Tokenizer(BPE())
        self._tokenizer.pre_tokenizer = WhitespaceSplit()  # type: ignore
        self._tokenizer.normalizer = NFKC()  # type: ignore

    def to_dict(self) -> dict:
        """
        Exporta la configuración completa del tokenizador a un diccionario.
        """
        return json.loads(self._tokenizer.to_str())

    def _clean_text(
        self,
        text: str,
    ) -> str:
        """
        Limpia el texto de los datos de Pokémon antes de la tokenización.
        """

        # Eliminamos filas y columnas vacías
        text_list = [row.split(" ") for row in text.split("\n")]
        text_array = np.array(text_list, dtype="<U6")
        rows_to_keep = ~(text_array == "~").all(axis=1)
        cols_to_keep = ~(text_array == "~").all(axis=0)
        text_array = text_array[rows_to_keep, :][:, cols_to_keep]

        # Añadimos caracteres especiales
        text_list = text_array.tolist()
        text_list = [["00"] + row for row in text_list]
        # text_list[0][0] = self.BOS_TOKEN
        text_list[-1][-1] = self.EOS_TOKEN

        # Convertimos a string
        text = " ".join([" ".join(row) for row in text_list])
        return text

    def train(
        self,
        pokedex_list: list[PokedexEntity],
    ):
        """
        Entrena el tokenizador BPE.
        """

        pokemon_data_list = [
            self._clean_text(pokedex_entity.data)
            for pokedex_entity in pokedex_list
            if pokedex_entity.data
        ]

        trainer = BpeTrainer(
            max_token_length=self.context_length,  # type: ignore
            special_tokens=[  # type: ignore
                self.BOS_TOKEN,
                self.EOS_TOKEN,
                self.UNK_TOKEN,
                self.PAD_TOKEN,
                self.EOL_TOKEN,
                self.BCK_TOKEN,
            ],
        )

        self._tokenizer.train_from_iterator(
            iterator=pokemon_data_list,
            trainer=trainer,
        )

        return self

    def _make_chunk_pairs(self, text: str) -> list[tuple[str, str]]:
        """
        Genera pares (input_chunk, label_chunk) a partir de un texto.
        El primer input es [BOS], el label es el primer chunk real.
        """
        text_list = text.split(" ")
        chunks = [
            " ".join(text_list[i : i + self.context_length])
            for i in range(0, len(text_list), self.context_length)
        ]

        pairs = []
        prev_chunk = self.BOS_TOKEN
        for current_chunk in chunks:
            pairs.append((prev_chunk, current_chunk))
            prev_chunk = current_chunk

        return pairs

    def _tokenize_function(self, batch) -> dict:

        inputs_encoding = self._tokenizer.encode_batch(
            batch["input_text"],
            add_special_tokens=False,
        )

        labels_encoding = self._tokenizer.encode_batch(
            batch["label_text"],
            add_special_tokens=False,
        )

        eol_token_id = self._tokenizer.token_to_id(self.EOL_TOKEN)
        bck_token_id = self._tokenizer.token_to_id(self.BCK_TOKEN)
        pad_token_id = self._tokenizer.token_to_id(self.PAD_TOKEN)

        attention_masks = [
            [
                (
                    1.0 if token_id == eol_token_id else
                    0.1 if token_id != bck_token_id else
                    0.3
                )
                for token_id in encoding.ids
            ]
            for encoding in inputs_encoding
        ]

        def pad(ids):
            return (
                ids + [pad_token_id] * (self.context_length - len(ids))
            )

        return {
            "name": batch["name"],
            "attention_mask": attention_masks,
            "input_ids": [pad(encoding.ids) for encoding in inputs_encoding],
            "labels": [pad(encoding.ids) for encoding in labels_encoding],
        }

    def tokenize(
        self,
        pokedex_list: list[PokedexEntity],
    ) -> DatasetDict:
        """
        Tokeniza el dataset de Pokémon, generando pares input-label.
        """

        names = []
        input_texts = []
        label_texts = []
        for pokedex_entity in pokedex_list:
            if not pokedex_entity.data:
                continue

            clean_text = self._clean_text(pokedex_entity.data)
            pairs = self._make_chunk_pairs(clean_text)

            for prev, curr in pairs:
                names.append(pokedex_entity.name)
                input_texts.append(prev)
                label_texts.append(curr)

        raw_dataset = Dataset.from_dict(
            {
                "name": names,
                "input_text": input_texts,
                "label_text": label_texts,
            }
        )

        tokenized_dataset = raw_dataset.map(
            self._tokenize_function, batched=True
        )

        return DatasetDict({"train": tokenized_dataset})
