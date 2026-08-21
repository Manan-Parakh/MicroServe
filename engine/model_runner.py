import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class ModelRunner:
    """The one boundary every engine version calls through instead of `transformers` directly."""

    def __init__(self, model_id: str, device: str | None = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(model_id).to(self.device)
        self.model.eval()

    def encode(self, prompt: str) -> list[int]:
        return self.tokenizer(prompt)["input_ids"]

    def decode(self, token_ids: list[int]) -> str:
        return self.tokenizer.decode(token_ids, skip_special_tokens=True)

    def generate(self, prompt_token_ids: list[int], max_new_tokens: int) -> list[int]:
        """Greedy-decode to completion; returns prompt + generated ids, like HF `generate()`."""
        input_ids = torch.tensor([prompt_token_ids], device=self.device)
        with torch.no_grad():
            output_ids = self.model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        return output_ids[0].tolist()
