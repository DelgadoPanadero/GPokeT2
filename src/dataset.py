import os

from datasets import load_dataset
from transformers import PreTrainedTokenizerFast


class PokeCenter():


    def __init__(self, model_dir, row_length=64, context_length=1024):

        self._row_length = row_length
        self._context_length = context_length
        tokenizer_file = os.path.join(model_dir, 'tokenizer.json')
        self._tokenizer = PreTrainedTokenizerFast(tokenizer_file=tokenizer_file)


    def _tokenize(self, sample):

        """
        """

        outputs = self._tokenizer(
                      sample['text'],
                      return_overflowing_tokens=True,
                      return_length=True,
                      )

        input_batch = []
        for size, ids in zip(outputs["length"], outputs["input_ids"]):

            if size!=self._row_length:
                print(f'Missing Pokemon!!! Bad encoding size: {size}')
                return {"input_ids": []}

            input_batch += ids

        return {"input_ids": [input_batch]}


    def create_dataset(self,source_dir):

        """
        """

        pokedex = load_dataset(source_dir)

        batch_size = self._context_length//self._row_length
        pokedex = pokedex.map(self._tokenize,
                              batched=True,
                              batch_size=batch_size,
                              remove_columns=pokedex["train"].column_names)

        return pokedex
