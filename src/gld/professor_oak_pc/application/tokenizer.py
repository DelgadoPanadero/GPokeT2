import os
from pathlib import Path

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.normalizers import NFKC
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import WhitespaceSplit

from ..domain.pokedex_entity import PokedexEntity
WhitespaceSplit().pre_tokenize_str

class Pokenizer(object):

    def __init__(self):
        """ """

        self._vocab = {"[BOS]", "[EOS]"}

        self._tokenizer = Tokenizer(BPE(
            #vocab (Dict[str, int], optional) — A dictionary of string keys and their ids {"am": 0,...}
            vocab=None,
            #merges (List[Tuple[str, str]], optional) — A list of pairs of tokens (Tuple[str, str]) [("a", "b"),...]
            merges=None,
            #cache_capacity (int, optional) — The number of words that the BPE cache can contain. The cache allows to speed-up the process by keeping the result of the merge operations for a number of words.
            cache_capacity = None,
            #dropout (float, optional) — A float between 0 and 1 that represents the BPE dropout to use.
            dropout=None,
            #unk_token (str, optional) — The unknown token to be used by the model.
            unk_token = None,
            #continuing_subword_prefix (str, optional) — The prefix to attach to subword units that don’t represent a beginning of word.
            continuing_subword_prefix=None,
            #end_of_word_suffix (str, optional) — The suffix to attach to subword units that represent an end of word.
            end_of_word_suffix=None,
            #fuse_unk (bool, optional) — Whether to fuse any subsequent unknown tokens into a single one
            fuse_unk=False,
            #byte_fallback (bool, optional) — Whether to use spm byte-fallback trick (defaults to False)
            byte_fallback=False,
            #ignore_merges (bool, optional) — Whether or not to match tokens with the vocab before using merges.
            ignore_merges=True,
        ))

        self._tokenizer.normalizer = NFKC()
        self._tokenizer.pre_tokenizer = WhitespaceSplit()
    
    def preprocess(
        self,
        text: str,
    )->str:
        
        text = text.replace("\n", " ")
        text = "[BOS] " + text + " [EOS]"
        return text


    def update_vocab(
        self,
        pokedex_list: list[PokedexEntity],
    )->None:
        
        for pokedex_item in pokedex_list:

            text = pokedex_item.data

            text = self._tokenizer.normalizer.normalize_str(text)

            words = {word for word, position in 
                self._tokenizer.pre_tokenizer.pre_tokenize_str(text)
            }

            self._vocab.update(words)


    def train(
        self,
        pokedex_list: list[PokedexEntity],
    ):
        """
        """

        self.update_vocab(pokedex_list)

        for pokedex_entity in pokedex_list:
            pokedex_entity.data = self.preprocess(pokedex_entity.data)

        tokenizer_trainer = BpeTrainer(
            vocab_size=len(self._vocab),
            show_progress=True,
            special_tokens = ["[BOS]", "[EOS]"],
            max_token_length=2,
        )


        self._tokenizer.train_from_iterator(
            iterator=iter(pokedex_list),
            trainer=tokenizer_trainer,
            length=len(pokedex_list),   
        )

        return self

    def save(
        self,
        model_dir,
        prefix=None,
    ):
        """ """

        os.makedirs(model_dir, exist_ok=True)
        file_name = os.path.join(model_dir, "tokenizer.json")
        self._tokenizer.save(file_name)
        self._tokenizer.model.save(model_dir, prefix)
