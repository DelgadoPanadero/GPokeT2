import os
import json
import torch
from tokenizers import Tokenizer
from transformers import Trainer
from transformers import AutoConfig
from transformers import GPT2LMHeadModel
from transformers import TrainingArguments
from transformers import PreTrainedTokenizerFast
from transformers import DataCollatorForLanguageModeling
from .inference_callback import InferenceCallback
from src.domain.gld.prof_oak_pc import BoxEntity


class PokemonTrainer:

    def __init__(
        self,
        box_entity: BoxEntity,
        context_length=1024,
        row_length=64,
    ):

        self._row_length = row_length
        self._context_length = context_length

        tokenizer_object = Tokenizer.from_str(
            json.dumps(box_entity.tokenizer, ensure_ascii=False),
        )

        self._tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=tokenizer_object,
            bos_token="[BOS]",
            eos_token="[EOS]",
            unk_token="[UNK]",
            pad_token="[PAD]",
        )

        self._dataset = box_entity.dataset

        self._data_collator = DataCollatorForLanguageModeling(
            tokenizer=self._tokenizer,
            mlm=False,
        )

        from transformers import GPT2Config, GPT2LMHeadModel

        self._model = GPT2LMHeadModel(
            GPT2Config(
                vocab_size=len(self._tokenizer.get_vocab()),
                n_ctx=self._context_length,
                n_positions=self._context_length,
                n_embd=768,                                                      # tamaño del embedding (por defecto GPT2 usa 768)
                n_layer=6,                                                      # número de capas Transformer (por defecto 6)
                n_head=12,                                                       # número de cabezas de atención (por defecto 12)
                bos_token_id=self._tokenizer.bos_token_id,
                eos_token_id=self._tokenizer.eos_token_id,
                pad_token_id=self._tokenizer.pad_token_id,
            )
        )



    def create_trainer(self, **kwargs):
        model_dir = "/home/data/model"
        os.makedirs(model_dir, exist_ok=True)

        default_args = {
            "output_dir": model_dir,
            "per_device_train_batch_size": 1,
            "logging_steps": 10,
            "gradient_accumulation_steps": 4,
            "num_train_epochs": 20,
            "warmup_steps": 100,
            "weight_decay": 0.1,
            "lr_scheduler_type": "cosine",
            "learning_rate": 1e-5,
            "save_steps": 100,
            "fp16": torch.cuda.is_available(),
            "dataloader_pin_memory": torch.cuda.is_available(),
        }

        default_args.update(kwargs)
        trainer_args = TrainingArguments(**default_args)

        trainer = Trainer(
            model=self._model,
            processing_class=self._tokenizer,
            args=trainer_args,
            data_collator=self._data_collator,
            train_dataset=self._dataset["train"],
            callbacks=[
                InferenceCallback(
                    self._tokenizer,
                    interval_steps=10,
                    row_length=self._row_length,
                    context_length=self._context_length,
                )
            ],
        )

        return trainer
