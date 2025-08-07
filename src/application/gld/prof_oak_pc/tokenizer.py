import json

import pyarrow as pa
from datasets import Dataset
from datasets import DatasetDict
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.normalizers import NFKC
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import WhitespaceSplit

from src.domain.slv.pokedex_entity import PokedexEntity


class Pokenizer(object):

    BOS_TOKEN = "[BOS]"
    EOS_TOKEN = "[EOS]"
    UNK_TOKEN = "[UNK]"
    PAD_TOKEN = "[PAD]"

    def __init__(
        self,
        sample_size: int = 1024,
        ):
        """ """

        self.sample_size = sample_size
        self._tokenizer = Tokenizer(BPE())
        self._tokenizer.normalizer = NFKC()               # type: ignore
        self._tokenizer.pre_tokenizer = WhitespaceSplit() # type: ignore


    def to_dict(self)->dict:
        return json.loads(self._tokenizer.to_str())


    def _preprocess_text(
        self,
        text: str,
    )->str:
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

            for i in range(0, len(text_splitted), self.sample_size):
                sub_text = text_splitted[i:i + self.sample_size]
                sub_text += [self.EOS_TOKEN] * (self.sample_size - len(sub_text))
                text_batch = " ".join(sub_text)
                text_batches.append(text_batch)
 
            for text_batch in text_batches:
                encoding = self._tokenizer.encode(text_batch)
                all_input_ids.append(encoding.ids)
                all_attention_masks.append(encoding.attention_mask)
                all_names.append(name)
                all_texts.append(text_batch)

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
        """
        """

        pokedex_list = pokedex_list.copy()

        for pokedex_entity in pokedex_list:
            pokedex_entity.data = self._preprocess_text(pokedex_entity.data)

        tokenizer_trainer = BpeTrainer(
            show_progress=True,                   # type: ignore
            max_token_length=2,                   # type: ignore
            special_tokens = [                    # type: ignore
                self.BOS_TOKEN,
                self.EOS_TOKEN,
                self.UNK_TOKEN,
                self.PAD_TOKEN,
            ],
        )

        self._tokenizer.train_from_iterator(
            iterator=(pokedex.data for pokedex in pokedex_list),
            trainer=tokenizer_trainer,
            length=len(pokedex_list),   
        )

        return self


    def tokenize(
        self,
        pokedex_list: list[PokedexEntity],
    )->DatasetDict:

        """
        """

        # Build th dataset
        text_list = []
        name_list = []
        for pokedex_entity in pokedex_list:
            text_list.append(self._preprocess_text(pokedex_entity.data))
            name_list.append(pokedex_entity.name)

        
        dataset_dict = DatasetDict({
            "train": Dataset(
                pa.Table.from_pydict({
                    "name": name_list,
                    "text": text_list,
                })
            ),
        })

        # Tokenize the dataset
        dataset_dict = dataset_dict.map(
            function=self._batch_map_dataset,
            batched=True,
            remove_columns=["text"],
        )

        return dataset_dict
        #return {split_name: dataset.data.table for split_name, dataset in dataset_dict.items()}