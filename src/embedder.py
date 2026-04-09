from __future__ import annotations

import torch
import torch.nn.functional as F
from adapters import AutoAdapterModel
from transformers import AutoTokenizer

SPECTER2_BASE_MODEL = "allenai/specter2_base"
SPECTER2_ADAPTER = "allenai/specter2"


class Specter2Embedder:
    def __init__(self) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(
            SPECTER2_BASE_MODEL,
            local_files_only=True,
        )
        self.model = AutoAdapterModel.from_pretrained(
            SPECTER2_BASE_MODEL,
            local_files_only=True,
        )

        adapter_name = self.model.load_adapter(
            SPECTER2_ADAPTER,
            source="hf",
            set_active=True,
            local_files_only=True,
        )
        self.model.set_active_adapters(adapter_name)
        self.model.eval()

    def _mean_pool(self, last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        summed = torch.sum(last_hidden_state * mask, dim=1)
        counts = torch.clamp(mask.sum(dim=1), min=1e-9)
        return summed / counts

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        inputs = self.tokenizer(
            texts,
            max_length=512,
            truncation=True,
            padding=True,
            return_tensors="pt",
        )

        with torch.no_grad():
            outputs = self.model(**inputs)
            pooled = self._mean_pool(outputs.last_hidden_state, inputs["attention_mask"])
            normalized = F.normalize(pooled, p=2, dim=1)

        return normalized.cpu().tolist()

    def embed_single(self, text: str) -> list[float]:
        return self.embed([text])[0]


_embedder: Specter2Embedder | None = None


def get_embedder() -> Specter2Embedder:
    global _embedder

    if _embedder is None:
        _embedder = Specter2Embedder()

    return _embedder
