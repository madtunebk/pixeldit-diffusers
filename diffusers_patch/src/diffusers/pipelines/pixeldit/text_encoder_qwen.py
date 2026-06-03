"""
Qwen3-2B text encoder for PixelDiT.
Requires a trained projection (train_qwen_proj.py) to map 2048→2304.

Usage:
    from pixeldit.text_encoder_qwen import QwenEncoder
    enc  = QwenEncoder(proj_path="pixeldit/qwen_proj.pt")
    cond = enc.encode(["a dragon at sunset"])  # [1, 300, 2304]
    null = enc.encode_null(1)                  # [1, 300, 2304]
"""

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel

_QWEN_ID   = "Qwen/Qwen3-2B"
_QWEN_DIM  = 2048
_GEMMA_DIM = 2304
_TXT_MAX   = 300

_CHI_PROMPT = "\n".join([
    'Given a user prompt, generate an "Enhanced prompt" that provides detailed visual descriptions suitable for image generation. Evaluate the level of detail in the user prompt:',
    '- If the prompt is simple, focus on adding specifics about colors, shapes, sizes, textures, and spatial relationships to create vivid and concrete scenes.',
    '- If the prompt is already detailed, refine and enhance the existing details slightly without overcomplicating.',
    'Here are examples of how to transform or refine prompts:',
    '- User Prompt: A cat sleeping -> Enhanced: A small, fluffy white cat curled up in a round shape, sleeping peacefully on a warm sunny windowsill, surrounded by pots of blooming red flowers.',
    '- User Prompt: A busy city street -> Enhanced: A bustling city street scene at dusk, featuring glowing street lamps, a diverse crowd of people in colorful clothing, and a double-decker bus passing by towering glass skyscrapers.',
    'Please generate only the enhanced description for the prompt below and avoid including any additional commentary or evaluations:',
    'User Prompt: ',
])
_SELECT_IDX = [0] + list(range(-(_TXT_MAX - 1), 0))


class QwenEncoder:
    def __init__(
        self,
        model_id=_QWEN_ID,
        proj_path=None,           # path to trained qwen_proj.pt
        output_device="cuda",
        output_dtype=torch.bfloat16,
    ):
        self.output_device = torch.device(output_device)
        self.output_dtype  = output_dtype

        print(f"[QwenEncoder] loading {model_id} (CPU)")
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.tokenizer.padding_side = "right"
        self._model = AutoModel.from_pretrained(model_id, torch_dtype=torch.float32).eval()

        self.proj = nn.Linear(_QWEN_DIM, _GEMMA_DIM, bias=False)
        if proj_path:
            sd = torch.load(proj_path, map_location="cpu", weights_only=True)
            self.proj.load_state_dict(sd)
            print(f"[QwenEncoder] loaded projection: {proj_path}")
        else:
            with torch.no_grad():
                w = torch.zeros(_GEMMA_DIM, _QWEN_DIM)
                w[:_QWEN_DIM] = torch.eye(_QWEN_DIM)
                self.proj.weight.copy_(w)
            print("[QwenEncoder] projection: identity init — run train_qwen_proj.py for real quality")
        self._num_chi_tokens = len(self.tokenizer.encode(_CHI_PROMPT))
        self.proj = self.proj.to(self.output_device).to(output_dtype)
        print("[QwenEncoder] ready")

    @torch.no_grad()
    def encode(self, texts: list[str]) -> torch.Tensor:
        """Returns [B, 300, 2304]."""
        texts_full = [_CHI_PROMPT + t for t in texts]
        max_len = self._num_chi_tokens + _TXT_MAX - 2
        tok = self.tokenizer(
            texts_full, max_length=max_len,
            padding="max_length", truncation=True, return_tensors="pt",
        )
        emb = self._model(**tok).last_hidden_state
        emb = emb[:, _SELECT_IDX, :]
        emb = emb.to(self.output_device).to(self.output_dtype)
        return self.proj(emb)

    @torch.no_grad()
    def encode_null(self, batch_size: int) -> torch.Tensor:
        """Returns [B, 300, 2304] for empty string (CFG unconditional)."""
        tok = self.tokenizer(
            [""] * batch_size, max_length=_TXT_MAX,
            padding="max_length", truncation=True, return_tensors="pt",
        )
        emb = self._model(**tok).last_hidden_state
        emb = emb.to(self.output_device).to(self.output_dtype)
        return self.proj(emb)
