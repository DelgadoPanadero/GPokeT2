import json
from typing import List, Tuple

import numpy as np
import pyarrow as pa
from datasets import Dataset
from datasets import DatasetDict
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
        token_size: int = 1,
        context_length: int = 1024,
    ):
        """
        Inicializa el tokenizador con las configuraciones base.
        """

        self._token_size = token_size
        self.context_length = context_length

        self._tokenizer = Tokenizer(BPE())
        self._tokenizer.pre_tokenizer = WhitespaceSplit()  # type: ignore
        self._tokenizer.normalizer = NFKC()  # type: ignore

        # self.tokenizer = PreTrainedTokenizerFast(
        #    tokenizer_object=self._tokenizer,
        #    bos_token=self.BOS_TOKEN,
        #    eos_token=self.EOS_TOKEN,
        #    unk_token=self.UNK_TOKEN,
        #    pad_token=self.PAD_TOKEN,
        #    model_max_length=self.context_length,
        #    max_token_length=1,
        # )

    def to_dict(self) -> dict:
        """
        Exporta la configuración completa del tokenizador a un diccionario.
        """
        return json.loads(self._tokenizer.to_str())

    def _clean_text(self, text: str) -> str:
        """
        Limpia el texto de los datos de Pokémon antes de la tokenización.
        """

        # Eliminamos filas y columnas vacias
        text_split = np.array([row.split(" ") for row in text.split("\n")])
        text_array = np.array(text_split, dtype="<U6")
        rows_to_keep = ~(text_array == "~").all(axis=1)
        cols_to_keep = ~(text_array == "~").all(axis=0)
        text_array = text_array[rows_to_keep, :][:, cols_to_keep]

        # Agrupamos múltiples caracteres en _token_size
        text_split = [
            [
                "".join(
                    (row + ["~"] * self._token_size)[i : i + self._token_size]
                )
                for i in range(0, len(row), self._token_size)
            ]
            for row in text_array.tolist()
        ]

        # Añadimos caracteres especiales. Lo hacemos aqui porque seguimos un
        # criterio que se sale del standard de los tokenizadores de HF
        text_split = [["00"] + row for row in text_split]
        text_split[0][0] = self.BOS_TOKEN
        text_split[-1][-1] = self.EOS_TOKEN

        # Convertirmos el texto a string
        text = "\n".join([" ".join(row) for row in text_split])
        text = text.replace("\n", " ")

        return text

    def train(
        self,
        pokedex_list: list[PokedexEntity],
    ):
        """
        Entrena el tokenizador BPE, configurando el post-procesador y
        limitando la longitud de los tokens.
        """

        pokemon_data_list = [
            self._clean_text(pokedex_entity.data)
            for pokedex_entity in pokedex_list
            if pokedex_entity.data
        ]

        trainer = BpeTrainer(
            max_token_length=self._token_size,  # type: ignore
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

    def _tokenize_function(
        self,
        batch,
    ) -> dict:

        all_names = []
        all_texts = []
        all_input_ids = []
        all_attention_masks = []
        for text, name in zip(batch["text"], batch["name"]):

            text_splitted = text.split(" ")
            for i in range(0, len(text_splitted), self.context_length):

                text_chunked = " ".join(
                    text_splitted[i : i + self.context_length]
                )

                encoding = self._tokenizer.encode(text_chunked)

                all_names.append(name)
                all_texts.append(text_chunked)
                all_input_ids.append(encoding.ids)
                all_attention_masks.append(
                    [
                        (
                            1.0
                            if i == self._tokenizer.token_to_id(self.EOL_TOKEN)
                            else (
                                0.1
                                if i
                                != self._tokenizer.token_to_id(self.BCK_TOKEN)
                                else 0.3
                            )
                        )
                        for i in encoding.ids
                    ]
                )

        return {
            "name": all_names,
            "text": all_texts,
            "input_ids": all_input_ids,
            "attention_mask": all_attention_masks,
        }

    def tokenize(
        self,
        pokedex_list: list[PokedexEntity],
    ) -> DatasetDict:
        """
        Tokeniza el dataset de Pokémon, aplicando la máscara de atención
        personalizada.
        """

        raw_dataset = Dataset.from_dict(
            {
                "name": [
                    pokedex_entity.name
                    for pokedex_entity in pokedex_list
                    if pokedex_entity.data
                ],
                "text": [
                    self._clean_text(pokedex_entity.data)
                    for pokedex_entity in pokedex_list
                    if pokedex_entity.data
                ],
            }
        )

        tokenized_dataset = raw_dataset.map(
            self._tokenize_function,
            batched=True,
            remove_columns=["text"],
        )

        return DatasetDict({"train": tokenized_dataset})
