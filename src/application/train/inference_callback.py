import torch
import logging

from transformers import TrainerState
from transformers import TrainerControl
from transformers import TrainerCallback

logger = logging.getLogger(__name__)

class InferenceCallback(TrainerCallback):

    def __init__(
        self,
        tokenizer,
        interval_steps=5000,
        context_length=1024,
    ):
        super().__init__()
        self.interval_steps = interval_steps
        self.tokenizer = tokenizer
        self.context_length = context_length

        # Crear input_ids una vez
        input_text = self.tokenizer.bos_token or "[BOS]"
        encoding = self.tokenizer(input_text, return_tensors="pt")
        self.input_ids = encoding["input_ids"]
        self.attention_mask = encoding.get("attention_mask", None)

    def on_step_end(
        self,
        args,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ) -> TrainerControl:

        if state.global_step > 0 and state.global_step % self.interval_steps == 0:
            model = kwargs["model"]
            device = next(model.parameters()).device

            model.eval()

            try:
                inputs = {
                    "input_ids": self.input_ids.to(device),
                }
                if self.attention_mask is not None:
                    inputs["attention_mask"] = self.attention_mask.to(device)

                with torch.no_grad():
                    output = model.generate(
                        **inputs,
                        max_length=self.context_length,  # margen extra
                        min_length=self.context_length,
                        do_sample=True,
                        top_k=50,
                        top_p=0.95,
                        temperature=0.9,
                        pad_token_id=self.tokenizer.pad_token_id,
                        eos_token_id=self.tokenizer.eos_token_id,
                    )
                decoded = self.tokenizer.decode(output[0], skip_special_tokens=True)

                logger.info(f"\n\n=== Inference @ step {state.global_step} ===")
                logger.info(decoded)
                logger.info("====================================\n\n")

            except Exception as e:
                logger.warning(f"Inference generation failed at step {state.global_step}: {e}")

            model.train()

        return control
