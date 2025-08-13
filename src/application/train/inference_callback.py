import torch

from transformers import TrainerState
from transformers import TrainerControl
from transformers import TrainerCallback


class InferenceCallback(TrainerCallback):
    def __init__(
        self,
        tokenizer,
        interval_steps=5000,
        row_length=64,
        context_length=1024,
    ):
        self.interval_steps = interval_steps
        self.tokenizer = tokenizer
        self.row_length = row_length
        self.context_length = context_length

    def on_step_end(
        self,
        args,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        if (
            state.global_step % self.interval_steps == 0
            and state.global_step > 0
        ):
            model = kwargs["model"]
            device = next(model.parameters()).device

            inputs = self.tokenizer("[BOS]", return_tensors="pt").to(device)
            model.eval()

            def run(inputs=inputs):
                with torch.no_grad():
                    output = model.generate(
                        **inputs,
                        max_length=self.context_length,
                        min_length=self.context_length,
                        do_sample=True,
                        top_k=50,
                        top_p=0.95,
                        temperature=0.9,
                        pad_token_id=self.tokenizer.pad_token_id,
                        eos_token_id=self.tokenizer.eos_token_id,
                    )
                return output

            output = run(inputs=inputs)

            decoded = self.tokenizer.decode(output[0], skip_special_tokens=False)
            print(f"\n\n=== Inference @ step {state.global_step} ===")

            try:
                print("\n".join([
                    " ".join(decoded.split(" ")[i:i+self.row_length])
                    for i in range(0,self.context_length,self.row_length)
                ]))

            except:
                print(decoded)
            print("====================================\n\n")

        return control
