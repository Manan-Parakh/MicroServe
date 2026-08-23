import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import StaticCache


class ModelRunner:
    """The one boundary every engine version calls through instead of `transformers` directly."""

    def __init__(
        self,
        model_id: str,
        device: str | None = None,
        attn_implementation: str | None = None,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, attn_implementation=attn_implementation
        ).to(self.device)
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
                repetition_penalty=1.0,  # true greedy argmax; see ADR-0008
                pad_token_id=self.tokenizer.eos_token_id,
            )
        return output_ids[0].tolist()

    def _left_pad(self, token_id_lists: list[list[int]]) -> tuple[torch.Tensor, torch.Tensor]:
        """Left-pad a batch of variable-length token id lists to a common width.

        Left, not right: it puts every row's real content at the same rightmost
        column, so `logits[:, -1, :]` reads "next-token prediction" for every row
        in the batch with one slice, no per-row bookkeeping needed.
        """
        max_len = max(len(ids) for ids in token_id_lists)
        pad_id = self.tokenizer.eos_token_id
        input_ids = torch.full(
            (len(token_id_lists), max_len), pad_id, dtype=torch.long, device=self.device
        )
        attention_mask = torch.zeros(
            (len(token_id_lists), max_len), dtype=torch.long, device=self.device
        )
        for row, ids in enumerate(token_id_lists):
            input_ids[row, max_len - len(ids) :] = torch.tensor(
                ids, dtype=torch.long, device=self.device
            )
            attention_mask[row, max_len - len(ids) :] = 1
        return input_ids, attention_mask

    @staticmethod
    def _position_ids_from_mask(attention_mask: torch.Tensor) -> torch.Tensor:
        """Real tokens are numbered 0, 1, 2, ... regardless of how much left-padding
        precedes them; padded columns get a harmless placeholder position since
        they're masked out of attention anyway."""
        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 1)
        return position_ids

    def prefill_batch(
        self, token_id_lists: list[list[int]], max_cache_len: int
    ) -> tuple[list[int], StaticCache, torch.Tensor]:
        """One batched forward pass over each sequence's *full* token history so far
        (a fresh admission's prompt, or a resync's prompt + already-generated
        tokens) -- compute-bound, the whole history processed in parallel in one
        call. Returns each row's greedy next-token id, a fresh `StaticCache` sized
        for this batch, and the attention mask to extend on later `decode_step`
        calls (ADR-0007: a new `StaticCache` per batch membership, not one long-
        lived cache with per-row positions).
        """
        input_ids, attention_mask = self._left_pad(token_id_lists)
        position_ids = self._position_ids_from_mask(attention_mask)
        cache = StaticCache(self.model.config, max_cache_len=max_cache_len)
        with torch.no_grad():
            out = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=cache,
                use_cache=True,
            )
        next_token_ids = out.logits[:, -1, :].argmax(dim=-1).tolist()
        return next_token_ids, cache, attention_mask

    def decode_step(
        self, cache: StaticCache, last_token_ids: list[int], attention_mask: torch.Tensor
    ) -> tuple[list[int], torch.Tensor]:
        """One batched forward pass, exactly one new token per row -- memory-
        bandwidth-bound. This is the call continuous batching amortizes across
        every currently-running sequence at once instead of paying its cost once
        per sequence. `cache` is updated in place by `StaticCache` itself.
        """
        input_ids = torch.tensor(last_token_ids, dtype=torch.long, device=self.device).unsqueeze(-1)
        attention_mask = torch.cat(
            [
                attention_mask,
                torch.ones((attention_mask.shape[0], 1), dtype=torch.long, device=self.device),
            ],
            dim=-1,
        )
        position_ids = self._position_ids_from_mask(attention_mask)[:, -1:]
        with torch.no_grad():
            out = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=cache,
                use_cache=True,
            )
        next_token_ids = out.logits[:, -1, :].argmax(dim=-1).tolist()
        return next_token_ids, attention_mask
