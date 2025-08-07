import os
import json
import torch
from tokenizers import Tokenizer
from transformers import (
    Trainer,
    TrainingArguments,
    AutoConfig,
    GPT2LMHeadModel,
    PreTrainedTokenizerFast,
    DataCollatorForLanguageModeling,
)
from .inference_callback import InferenceCallback
from src.domain.gld.box_entity import BoxEntity


class PokemonTrainer:

    __base_model__ = "distilgpt2"

    def __init__(self, box_entity: BoxEntity, context_length: int = 1024,):
        self._context_length = context_length
        self._dataset = box_entity.dataset

        # 1. Reconstruir tokenizer desde string JSON
        tokenizer_object = Tokenizer.from_str(json.dumps(box_entity.tokenizer, ensure_ascii=False),)

        self._tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=tokenizer_object,
            bos_token="[BOS]",
            eos_token="[EOS]",
            unk_token="[UNK]",
            pad_token="[PAD]",
        )

        # 2. Configurar modelo con arquitectura de distilgpt2 pero pesos aleatorios
        config = AutoConfig.from_pretrained(self.__base_model__)
        config.vocab_size = len(self._tokenizer)
        config.bos_token_id = self._tokenizer.bos_token_id
        config.eos_token_id = self._tokenizer.eos_token_id
        config.pad_token_id = self._tokenizer.pad_token_id
        config.n_ctx = self._context_length
        config.max_length = self._context_length//2
        config.min_length = self._context_length//2
        config.n_positions = self._context_length

        self._model = GPT2LMHeadModel(config)

        # 3. Data collator
        self._data_collator = DataCollatorForLanguageModeling(
            tokenizer=self._tokenizer,
            mlm=False,
        )

    def create_trainer(self, **kwargs) -> Trainer:
        model_dir = "/home/data/model"
        os.makedirs(model_dir, exist_ok=True)

        default_args = {
            "output_dir": model_dir,
            "per_device_train_batch_size": 1,
            "per_device_eval_batch_size": 1,
            "logging_steps": 5000,
            "gradient_accumulation_steps": 8,
            "num_train_epochs": 3,
            "weight_decay": 0.1,
            "warmup_steps": 1000,
            "lr_scheduler_type": "cosine",
            "learning_rate": 5e-4,
            "save_steps": 5000,
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
            callbacks=[InferenceCallback(self._tokenizer, interval_steps=10)],
        )

        return trainer
