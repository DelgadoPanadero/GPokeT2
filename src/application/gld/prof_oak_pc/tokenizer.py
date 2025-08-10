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
from tokenizers.pre_tokenizers import Split
from tokenizers.processors import PostProcessor
from tokenizers.pre_tokenizers import PreTokenizer
from tokenizers.pre_tokenizers import WhitespaceSplit
from transformers import PreTrainedTokenizerFast

from src.domain.slv.pokedex import PokedexEntity


class CustomAttentionProcessor(PostProcessor):
    def __init__(
        self,
        bos_token_id,
        eos_token_id,
        token_00_id,
    ):
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.token_00_id = token_00_id

    def __call__(
        self,
        encoding,
        other=None,
        add_special_tokens=True,
    ):
        # 1. Añadir tokens especiales (BOS y EOS)
        if add_special_tokens:
            encoding.ids = (
                [self.bos_token_id] + encoding.ids + [self.eos_token_id]
            )
            encoding.type_ids = [0] + encoding.type_ids + [0]
            encoding.tokens = ["[BOS]"] + encoding.tokens + ["[EOS]"]
            encoding.offsets = [(0, 0)] + encoding.offsets + [(0, 0)]

        # 2. Construir la máscara de atención personalizada
        attention_mask = [
            1.0 if token_id == self.token_00_id else 0.3
            for token_id in encoding.ids
        ]
        encoding.attention_mask = attention_mask

        return encoding


class GroupedWhitespaceSplit(PreTokenizer):
    """
    Divide el texto en grupos de 4 palabras en lugar de palabras individuales.
    """

    def pre_tokenize(
        self,
        pretok,  #: PreTokenizer.PreTokenizedString,
    ):
        # Primero divide el texto en palabras individuales por espacios.
        pretok.split(Split(pattern=" ", behavior="isolated"))

        # Almacena las palabras y sus offsets.
        temp_words: List[Tuple[Tuple[int, int], str]] = []
        for offset, word, _ in pretok.get_splits():
            temp_words.append((offset, word))

        # Limpia la lista para rellenarla con los nuevos grupos.
        pretok.clear()

        # Une las palabras en grupos de 4 y las añade como nuevos tokens.
        for i in range(0, len(temp_words), 4):
            chunk = temp_words[i : i + 4]
            if not chunk:
                continue

            start_offset = chunk[0][0][0]
            end_offset = chunk[-1][0][1]

            text_chunk = " ".join([word for _, word in chunk])
            pretok.add_pre_token(text_chunk, (start_offset, end_offset))


class Pokenizer:
    """
    Clase que encapsula la lógica para entrenar y usar un tokenizador
    personalizado para datos de Pokémon.
    """

    BOS_TOKEN = "[BOS]"
    EOS_TOKEN = "[EOS]"
    UNK_TOKEN = "[UNK]"
    PAD_TOKEN = "[PAD]"

    def __init__(self, context_length: int = 1024):
        """
        Inicializa el tokenizador con las configuraciones base.
        """
        self.context_length = context_length
        self._tokenizer = Tokenizer(BPE())
        self._tokenizer.pre_tokenizer = WhitespaceSplit()  # type: ignore
        self._tokenizer.normalizer = NFKC()  # type: ignore
        self._tokenizer.post_processor = TemplateProcessing(  # type: ignore
            single=f"{self.BOS_TOKEN} $A {self.EOS_TOKEN}",
            pair=f"{self.BOS_TOKEN} $A {self.EOS_TOKEN} $B:1 {self.EOS_TOKEN}:1",
            special_tokens=[
                (self.BOS_TOKEN, 0),
                (self.EOS_TOKEN, 1),
                (self.PAD_TOKEN, 2),
                (self.UNK_TOKEN, 3),
            ],
        )

        self.tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=self._tokenizer,
            bos_token=self.BOS_TOKEN,
            eos_token=self.EOS_TOKEN,
            unk_token=self.UNK_TOKEN,
            pad_token=self.PAD_TOKEN,
            model_max_length=self.context_length,
        )

    def to_dict(self) -> dict:
        """
        Exporta la configuración completa del tokenizador a un diccionario.
        """
        return json.loads(self._tokenizer.to_str())

    def _clean_text(self, text: str) -> str:
        """
        Limpia el texto de los datos de Pokémon antes de la tokenización.
        """
        array = np.array([row.split(" ") for row in text.split("\n")])
        rows_to_keep = ~(array == "~").all(axis=1)
        cols_to_keep = ~(array == "~").all(axis=0)
        array = array[rows_to_keep][:, cols_to_keep].tolist()
        array = [["00"] + row[1:] for row in array]
        text = "\n".join([" ".join(row) for row in array])
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
            self._clean_text(p.data) for p in pokedex_list if p.data
        ]

        trainer = BpeTrainer(
            max_token_length=1,  # type: ignore
            special_tokens=[  # type: ignore
                self.BOS_TOKEN,
                self.EOS_TOKEN,
                self.UNK_TOKEN,
                self.PAD_TOKEN,
            ],
        )
        self._tokenizer.train_from_iterator(
            iterator=pokemon_data_list, trainer=trainer
        )

        self.tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=self._tokenizer,
            bos_token=self.BOS_TOKEN,
            eos_token=self.EOS_TOKEN,
            unk_token=self.UNK_TOKEN,
            pad_token=self.PAD_TOKEN,
            model_max_length=self.context_length,
        )
        return self

    def tokenize(
        self,
        pokedex_list: list[PokedexEntity],
    ) -> DatasetDict:
        """
        Tokeniza el dataset de Pokémon, aplicando la máscara de atención personalizada.
        """
        data_dict = {
            "name": [p.name for p in pokedex_list if p.data],
            "text": [self._clean_text(p.data) for p in pokedex_list if p.data],
        }
        raw_dataset = Dataset.from_dict(data_dict)

        # Obtiene el ID numérico del token "00" para usarlo en la máscara.
        token_00_id = self.tokenizer.convert_tokens_to_ids("00")

        def tokenize_function(examples):
            # Tokenización estándar que incluye padding y truncado.
            tokenized_output = self.tokenizer(
                examples["text"],
                truncation=True,
                padding="max_length",
                max_length=self.context_length,
            )

            # Crea la máscara de atención personalizada.
            all_attention_masks = []
            all_truncated_texts = []
            for input_ids in tokenized_output["input_ids"]:  # type: ignore

                all_attention_masks.append([
                    1.0 if token_id == token_00_id else 0.3
                    for token_id in input_ids
                ])

                all_truncated_texts.append(
                    self.tokenizer.decode(
                        input_ids,
                        skip_special_tokens=True,
                    )
                )

            tokenized_output["attention_mask"] = all_attention_masks
            tokenized_output["truncated_text"] = all_truncated_texts

            return tokenized_output

        tokenized_dataset = raw_dataset.map(
            tokenize_function,
            batched=True,
        )

        return DatasetDict({"train": tokenized_dataset})
