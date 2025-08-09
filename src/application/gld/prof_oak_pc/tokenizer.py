import json

import pyarrow as pa
from datasets import Dataset
from datasets import DatasetDict
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.normalizers import NFKC
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import WhitespaceSplit

from src.domain.slv.pokedex import PokedexEntity


def get_small_pokemon(text):
    import numpy as np

    new_array = [row.split(" ") for row in text.split("\n")]
    new_array = np.array(new_array)[:, 1:]
    rows_to_keep = ~(new_array == "~").all(axis=1)
    cols_to_keep = ~(new_array == "~").all(axis=0)
    new_array = new_array[rows_to_keep][:, cols_to_keep]
    if new_array.shape[0] > 22 or new_array.shape[1] > 22:
        new_array = []
    else:
        padded = np.full((22, 22), "~", dtype=str)
        row_size, col_size = new_array.shape
        row_start = (22 - row_size) // 2
        col_start = (22 - col_size) // 2
        row_end = row_start + row_size
        col_end = col_start + col_size
        padded[row_start:row_end, col_start:col_end] = new_array
        row_numbers = np.array([f"{i:02}" for i in range(22)]).reshape(22, 1)
        padded_with_row_nums = np.hstack((row_numbers, padded))
        new_array = padded_with_row_nums.tolist()
    return "\n".join([" ".join(r) for i, r in enumerate(new_array)])


class Pokenizer(object):

    BOS_TOKEN = "[BOS]"
    EOS_TOKEN = "[EOS]"
    UNK_TOKEN = "[UNK]"
    PAD_TOKEN = "[PAD]"

    def __init__(
        self,
        context_length: int = 1024,
    ):
        """ """

        self.context_length = context_length
        self._tokenizer = Tokenizer(BPE())
        self._tokenizer.normalizer = NFKC()  # type: ignore
        self._tokenizer.pre_tokenizer = WhitespaceSplit()  # type: ignore

    def to_dict(self) -> dict:
        return json.loads(self._tokenizer.to_str())

    def _preprocess_text(
        self,
        text: str,
    ) -> str:

        # text = get_small_pokemon(text)

        text_array = [["00"] + row.split(" ")[:-1] for row in text.split("\n")]
        text = "\n".join([" ".join(r) for i, r in enumerate(text_array)])

        text = text.replace("\n", " ")
        text_list = text.split(" ")
        text_list[+0] = self.BOS_TOKEN
        text_list[-1] = self.EOS_TOKEN
        text = " ".join(text_list)

        return text

    def _batch_map_dataset(
        self,
        batch,
    ) -> dict:

        all_input_ids = []
        all_attention_masks = []
        all_names = []
        all_texts = []

        for text, name in zip(batch["text"], batch["name"]):
            text_batches = []
            text_splitted = text.split(" ")

            for i in range(0, len(text_splitted), self.context_length):
                sub_text = text_splitted[i : i + self.context_length]
                sub_text += [self.EOS_TOKEN] * (
                    self.context_length - len(sub_text)
                )
                text_batch = " ".join(sub_text)
                text_batches.append(text_batch)

            for text_batch in text_batches:
                encoding = self._tokenizer.encode(text_batch)
                all_attention_masks.append(
                    [
                        1 if (i % 64) == 0 else 0.3
                        for i in range(len(encoding.attention_mask))
                    ]
                )
                all_input_ids.append(encoding.ids)
                all_texts.append(text_batch)
                all_names.append(name)

        return {
            "name": all_names,
            "text": all_texts,
            "input_ids": all_input_ids,
            "attention_mask": all_attention_masks,
        }

    def train(
        self,
        pokedex_list: list[PokedexEntity],
    ):
        """ """

        pokemon_data_list = []
        for pokedex_entity in pokedex_list:
            if pokemon_data := self._preprocess_text(pokedex_entity.data):
                pokemon_data_list.append(pokemon_data)

        tokenizer_trainer = BpeTrainer(
            show_progress=True,  # type: ignore
            max_token_length=2,  # type: ignore
            special_tokens=[  # type: ignore
                self.BOS_TOKEN,
                self.EOS_TOKEN,
                self.UNK_TOKEN,
                self.PAD_TOKEN,
            ],
        )

        self._tokenizer.train_from_iterator(
            iterator=(pokemon_data for pokemon_data in pokemon_data_list),
            trainer=tokenizer_trainer,
            length=len(pokemon_data_list),
        )

        return self

    def tokenize(
        self,
        pokedex_list: list[PokedexEntity],
    ) -> DatasetDict:
        """ """

        # Build th dataset
        pokemon_data_list = []
        pokemon_name_list = []
        for pokedex_entity in pokedex_list:
            if pokemon_data := self._preprocess_text(pokedex_entity.data):
                pokemon_data_list.append(pokemon_data)
                pokemon_name_list.append(pokedex_entity.name)

        dataset_dict = DatasetDict(
            {
                "train": Dataset(
                    pa.Table.from_pydict(
                        {
                            "name": pokemon_name_list,
                            "text": pokemon_data_list,
                        }
                    )
                ),
            }
        )

        # Tokenize the dataset
        dataset_dict = dataset_dict.map(
            function=self._batch_map_dataset,
            batched=True,
            remove_columns=["text"],
        )

        return dataset_dict
        # return {split_name: dataset.data.table for split_name, dataset in dataset_dict.items()}
