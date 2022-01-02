import os
from pathlib import Path

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.normalizers import NFKC
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import WhitespaceSplit


class Pokenizer(object):


    def __init__(self, row_length=64, context_length=1024):

        """
        """

        self._row_length = row_length
        self._context_length = context_length

        self._tokenizer = Tokenizer(BPE())
        self._tokenizer.normalizer = NFKC()
        self._tokenizer.pre_tokenizer = WhitespaceSplit()

    def get_vocab(self):

        """
        """

        vocab  = [str(i) for i in range(10)]
        vocab += ['%02d' % i for i in range(65)]
        vocab += [chr(i+59) for i in range(64)]
        return vocab


    def train(self, images_dir):

        """
        """

        paths = [str(x) for x in Path(images_dir).glob("**/*.txt")]

        vocab = self.get_vocab()
        trainer = BpeTrainer(vocab_size=len(vocab),
                             initial_alphabet=vocab,
                             show_progress=True)

        self._tokenizer.train(files=paths, trainer=trainer)
        self._tokenizer.pad_token = '[PAD]'
        self._tokenizer.eos_token = '[EOS]'
        self._tokenizer.bos_token = '[BOS]'

        return self


    def save(self, model_dir, prefix=None):

        """
        """

        os.makedirs(model_dir, exist_ok=True)
        file_name = os.path.join(model_dir,'tokenizer.json')
        self._tokenizer.save(file_name)
        self._tokenizer.model.save(model_dir, prefix)
