import json
from typing import List
from datasets import Dataset, DatasetDict
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.normalizers import NFKC
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import WhitespaceSplit

from src.domain.slv.pokedex import PokedexEntity


class Pokenizer:
    BOS_TOKEN = "[BOS]"
    EOS_TOKEN = "[EOS]"
    UNK_TOKEN = "[UNK]"
    PAD_TOKEN = "[PAD]"
    BOL_TOKEN = "00"
    BCK_TOKEN = "~"

    def __init__(
        self,
        token_size: int = 1,
        context_length: int = 32,
        overlap_rows: int = 3,
        row_length: int = 8,
    ):
        self._token_size = token_size
        self.context_length = context_length
        self.overlap_rows = overlap_rows
        self.row_length = row_length

        self._tokenizer = Tokenizer(BPE())
        self._tokenizer.pre_tokenizer = WhitespaceSplit()
        self._tokenizer.normalizer = NFKC()

    def to_dict(self) -> dict:
        return json.loads(self._tokenizer.to_str())

    def _clean_text(self, text: str) -> str:
        text_split = text.split("\n")
        text_split = [[self.BOL_TOKEN]+row.split() for row in text_split]
        text_split = [row[0:self.row_length] for row in text_split]
        text_split[0][0] = self.BOS_TOKEN
        text_split[-1][-1] = self.EOS_TOKEN
        return "\n".join([" ".join(row) for row in text_split])

    def train(self, pokedex_list: list[PokedexEntity]):
        pokemon_data_list = [
            self._clean_text(pokedex_entity.data)
            for pokedex_entity in pokedex_list
            if pokedex_entity.data
        ]

        trainer = BpeTrainer(
            max_token_length=self._token_size,
            special_tokens=[
                self.BOS_TOKEN,
                self.EOS_TOKEN,
                self.UNK_TOKEN,
                self.PAD_TOKEN,
                self.BOL_TOKEN,
                self.BCK_TOKEN,
            ],
        )

        self._tokenizer.train_from_iterator(
            iterator=pokemon_data_list,
            trainer=trainer,
        )

        return self

    def _chunk_with_overlap(
        self,
        encoding_ids: List[int],
    ) -> List[List[int]]:

        # Número de filas por ventana de contexto:
        rows_per_window = self.context_length // self.row_length
        step_rows = rows_per_window - self.overlap_rows
        step = step_rows * self.row_length

        chunks = []
        for i in range(0,len(encoding_ids),step):
            chunks.append(encoding_ids[i:i+self.context_length])

        return chunks

    def _tokenize_function(
        self,
        batch,
    )->dict[str,list]:

        bol_token_id = self._tokenizer.token_to_id(self.BOL_TOKEN)
        bck_token_id = self._tokenizer.token_to_id(self.BCK_TOKEN)

        all_names = []
        all_labels = []
        all_input_ids = []
        all_attention_masks = []

        for text, name in zip(batch["text"], batch["name"]):
            encoding = self._tokenizer.encode(text)
            encoding_ids_chunked = self._chunk_with_overlap(encoding.ids)

            # Creamos pares input -> label (ventana t -> ventana t+1)
            for i in range(len(encoding_ids_chunked) - 1):

                all_names.append(f"{name}_pair{i+1}")
                all_input_ids.append(encoding_ids_chunked[i])
                all_labels.append(encoding_ids_chunked[i + 1])

                attention_weights = [
                    1.0 if token_id == bol_token_id else
                    0.1 if token_id == bck_token_id else
                    0.3
                    for token_id in encoding_ids_chunked[i]
                ]
                all_attention_masks.append(attention_weights)

        return {
            "name": all_names,
            "input_ids": all_input_ids,
            "labels": all_labels,
            "attention_mask": all_attention_masks,
        }

    def tokenize(
        self,
        pokedex_list: list[PokedexEntity],
    ) -> DatasetDict:

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