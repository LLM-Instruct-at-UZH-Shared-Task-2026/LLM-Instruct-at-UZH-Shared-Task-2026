from __future__ import annotations
from typing import Dict, Any, List, Optional
import re
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


class LocalLLM:
    """Wrapper around a local causal LM with chat-template support.

    Supports Qwen3-8B thinking mode: when enable_thinking=True the model
    generates <think>...</think> reasoning before the final JSON answer.
    The thinking block is automatically stripped so callers receive
    clean JSON. Batch inference is fully supported.
    """

    DEFAULT_SYSTEM = (
        "You are an expert in United Nations and UNESCO resolution argument structure. "
        "You analyse formal legal-political documents written in French or English. "
        "You MUST respond with strict JSON only — no markdown fences, no extra text."
    )

    # ── Thinking-block stripping ───────────────────────────────────────────
    _THINK_RE = re.compile(
        r"<\|?think(?:_start)?\|?>.*?<\|?/?\s*think(?:_end)?\|?>",
        re.DOTALL | re.IGNORECASE,
    )

    @classmethod
    def _strip_think(cls, text: str) -> str:
        """Remove Qwen3 <think>...</think> (or <|think_start|>...<|think_end|>) blocks."""
        return cls._THINK_RE.sub("", text).strip()

    def __init__(
        self,
        model_name: str,
        device: str = "auto",
        load_in_4bit: bool = True,
        enable_thinking: bool = False,
        thinking_budget: Optional[int] = None,
        batch_chunk_size: int = 16,
    ):
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.enable_thinking = enable_thinking
        # thinking_budget caps the <think> block length (Qwen3 native feature).
        # None means unlimited; 512 is a good balance of quality vs speed.
        self.thinking_budget = thinking_budget
        # Chunk large batches to avoid OOM when max_new_tokens is large.
        self.batch_chunk_size = batch_chunk_size

        kwargs: Dict[str, Any] = {}
        if load_in_4bit and device == "cuda":
            try:
                from transformers import BitsAndBytesConfig
                kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                )
            except ImportError:
                print("[LLM] bitsandbytes not available; loading in fp16 instead.")
                kwargs["torch_dtype"] = torch.float16
        elif device == "cuda":
            kwargs["torch_dtype"] = torch.float16

        self.tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto" if device == "cuda" else None,
            **kwargs,
        )
        if device != "cuda":
            self.model.to(device)
        self.model.eval()

        # Pad token fallback
        if self.tok.pad_token_id is None:
            self.tok.pad_token_id = self.tok.eos_token_id

    @torch.inference_mode()
    def chat(
        self,
        user_msg: str,
        system_msg: Optional[str] = None,
        max_new_tokens: int = 512,
        temperature: float = 0.2,
    ) -> str:
        """Generate a reply using the model's chat template.

        This is the preferred interface — avoids raw prompt formatting issues.
        'think' field in output comes from the model's own reasoning chain
        when enable_thinking=True (Qwen3/QwQ models).
        """
        sys = system_msg or self.DEFAULT_SYSTEM
        messages = [
            {"role": "system", "content": sys},
            {"role": "user", "content": user_msg},
        ]

        # Qwen3/QwQ thinking mode: pass enable_thinking kwarg if supported
        def _apply_template(extra_kwargs=None):
            kw = dict(tokenize=True, add_generation_prompt=True, return_tensors="pt")
            if extra_kwargs:
                kw.update(extra_kwargs)
            enc = self.tok.apply_chat_template(messages, **kw)
            device = self.model.device
            # apply_chat_template may return a Tensor or a BatchEncoding
            if hasattr(enc, "input_ids"):
                attn = enc.get("attention_mask", None)
                return (
                    enc.input_ids.to(device),
                    attn.to(device) if attn is not None else None,
                )
            tensor = enc.to(device)
            return tensor, None

        tk_kwargs: dict = {"enable_thinking": self.enable_thinking}
        if self.thinking_budget is not None:
            tk_kwargs["thinking_budget"] = self.thinking_budget
        try:
            input_ids, attn_mask = _apply_template(tk_kwargs)
        except TypeError:
            try:
                input_ids, attn_mask = _apply_template({"enable_thinking": self.enable_thinking})
            except TypeError:
                input_ids, attn_mask = _apply_template()

        attention_mask = attn_mask if attn_mask is not None else torch.ones_like(input_ids)
        out = self.model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0,
            temperature=temperature if temperature > 0 else 1.0,
            pad_token_id=self.tok.pad_token_id,
        )
        # Decode only the newly generated tokens
        new_tokens = out[0][input_ids.shape[-1]:]
        raw = self.tok.decode(new_tokens, skip_special_tokens=True).strip()
        return self._strip_think(raw) if self.enable_thinking else raw

    @torch.inference_mode()
    def chat_batch(
        self,
        user_msgs: List[str],
        system_msg: Optional[str] = None,
        max_new_tokens: int = 512,
        temperature: float = 0.2,
    ) -> List[str]:
        """Batch version of chat() — all prompts processed in one GPU pass.

        Handles left-padding correctly and strips think blocks when thinking
        mode is enabled. Falls back to sequential on any error.
        """
        if len(user_msgs) == 1:
            return [self.chat(user_msgs[0], system_msg, max_new_tokens, temperature)]

        # Process in chunks to avoid OOM with large max_new_tokens
        if len(user_msgs) > self.batch_chunk_size:
            results = []
            for i in range(0, len(user_msgs), self.batch_chunk_size):
                chunk = user_msgs[i:i + self.batch_chunk_size]
                results.extend(self.chat_batch(chunk, system_msg, max_new_tokens, temperature))
            return results

        sys = system_msg or self.DEFAULT_SYSTEM
        batch_messages = [
            [{"role": "system", "content": sys}, {"role": "user", "content": m}]
            for m in user_msgs
        ]
        try:
            # Build text strings per prompt (tokenize=False gives the template text)
            def _fmt(msgs):
                tk_kw = {"enable_thinking": self.enable_thinking}
                if self.thinking_budget is not None:
                    tk_kw["thinking_budget"] = self.thinking_budget
                try:
                    return self.tok.apply_chat_template(
                        msgs, tokenize=False, add_generation_prompt=True, **tk_kw
                    )
                except TypeError:
                    try:
                        return self.tok.apply_chat_template(
                            msgs, tokenize=False, add_generation_prompt=True,
                            enable_thinking=self.enable_thinking,
                        )
                    except TypeError:
                        return self.tok.apply_chat_template(
                            msgs, tokenize=False, add_generation_prompt=True,
                        )

            texts = [_fmt(msgs) for msgs in batch_messages]

            # Qwen tokenisers use left-padding for generation
            self.tok.padding_side = "left"
            enc = self.tok(
                texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=4096,
            ).to(self.model.device)

            out = self.model.generate(
                **enc,
                max_new_tokens=max_new_tokens,
                do_sample=temperature > 0,
                temperature=temperature if temperature > 0 else 1.0,
                pad_token_id=self.tok.pad_token_id,
            )

            # With left-padding, prompt occupies enc["input_ids"].shape[-1] tokens
            prompt_len = enc["input_ids"].shape[1]
            results = []
            for seq in out:
                new_tok = seq[prompt_len:]
                raw = self.tok.decode(new_tok, skip_special_tokens=True).strip()
                results.append(self._strip_think(raw) if self.enable_thinking else raw)
            return results
        except Exception as exc:
            print(f"[LLM] chat_batch fallback to sequential: {exc}")
            return [self.chat(m, system_msg, max_new_tokens, temperature) for m in user_msgs]

    @torch.inference_mode()
    def generate(self, prompt: str, max_new_tokens: int = 256, temperature: float = 0.2) -> str:
        """Legacy raw-text interface kept for backward compatibility."""
        inputs = self.tok(prompt, return_tensors="pt").to(self.model.device)
        out = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0,
            temperature=temperature if temperature > 0 else 1.0,
            pad_token_id=self.tok.pad_token_id,
        )
        text = self.tok.decode(out[0], skip_special_tokens=True)
        return text[len(prompt):].strip() if text.startswith(prompt) else text.strip()
