import os
import json

import torch
from tokenizers import Tokenizer
from transformers import Trainer  # type: ignore
from transformers import AutoConfig
from transformers import GPT2LMHeadModel
from transformers import TrainingArguments  # type: ignore
from transformers import PreTrainedTokenizerFast  # type: ignore
from transformers import DataCollatorForLanguageModeling  # type: ignore

from .inference_callback import InferenceCallback
from src.domain.gld.prof_oak_pc import BoxEntity


class WeightedLossTrainer(Trainer):

    def __init__(
        self,
        *args,
        loss_weights=None,
        **kwargs,
    ):

        super().__init__(*args, **kwargs)
        self.loss_weights = loss_weights

    def compute_loss(  # type: ignore
        self,
        model,
        inputs,
        return_outputs=False,
        *,
        num_items_in_batch=None,
    ):

        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")

        loss_fct = torch.nn.CrossEntropyLoss(
            weight=self.loss_weights.to(logits.device),
        )  # type: ignore
        loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))

        return (loss, outputs) if return_outputs else loss


class PokemonTrainer:

    __base_model__ = "distilgpt2"

    def __init__(
        self,
        box_entity: BoxEntity,
        context_length=1024,
    ):

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

        self._model = GPT2LMHeadModel(
            AutoConfig.from_pretrained(
                pretrained_model_name_or_path=self.__base_model__,
                vocab_size=len(self._tokenizer.get_vocab()),
                n_ctx=self._context_length,
                bos_token_id=self._tokenizer.bos_token_id,
                eos_token_id=self._tokenizer.eos_token_id,
            )
        )



        # Crear vector de pesos para la pérdida
        weights = torch.ones(len(self._tokenizer))
        weights[self._tokenizer.convert_tokens_to_ids("~")] = 0.5
        weights[self._tokenizer.convert_tokens_to_ids("00")] = (
            10  # penalizar menos el token "~"
        )

        self._loss_weights = weights

    def create_trainer(self, **kwargs):
        model_dir = "/home/data/model"
        os.makedirs(model_dir, exist_ok=True)

        default_args = {
            "output_dir": model_dir,
            "per_device_train_batch_size": 4,
            "logging_steps": 10,
            "gradient_accumulation_steps": 4,
            "num_train_epochs": 5,
            "weight_decay": 0.1,
            "warmup_steps": 1_000,
            "lr_scheduler_type": "cosine",
            "learning_rate": 1e-5,
            "save_steps": 5_000,
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
                    context_length=self._context_length,
                )
            ],
            # loss_weights=self._loss_weights,
        )

        return trainer
