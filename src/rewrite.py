import os
from typing import List, Optional


class SafeRewriter:
    """
    LLaMA-backed claim rewriter.
    - Lazy loads the model
    - Never blocks verification
    - Safe to disable if model is missing
    """

    def __init__(
        self,
        model_path: str = "models/llama.gguf",
        n_ctx: int = 1024,
        temperature: float = 0.1,
        enabled: bool = True,
    ):
        self.model_path = model_path
        self.n_ctx = n_ctx
        self.temperature = temperature
        self.enabled = enabled

        self.llm = None
        self._checked = False

    def _load_llm(self):
        """Load LLaMA only once, only if enabled."""
        if self._checked:
            return

        self._checked = True

        if not self.enabled:
            return

        if not os.path.exists(self.model_path):
            print("⚠️ LLaMA model not found. Rewrite disabled.")
            return

        try:
            from llama_cpp import Llama

            self.llm = Llama(
                model_path=self.model_path,
                n_ctx=self.n_ctx,
                temperature=self.temperature,
                verbose=False,
            )
        except Exception as e:
            print(f"⚠️ Failed to load LLaMA: {e}")
            self.llm = None

    def rewrite(self, claim: str, evidence_snippets: List[str]) -> Optional[str]:
        """
        Rewrite an overstated claim into a safer academic version.
        Returns None if rewriting is unavailable.
        """
        self._load_llm()

        if self.llm is None:
            return None

        evidence_text = "\n".join(
            f"- {e}" for e in evidence_snippets[:2]
        )

        prompt = f"""
You are an academic fact-checking assistant.

Original claim:
"{claim}"

Relevant evidence:
{evidence_text}

Rewrite the claim to be scientifically cautious and accurate.

Rules:
- Remove absolute or exaggerated language
- Do NOT add new facts
- Do NOT contradict the evidence
- Use neutral academic phrasing
- Output EXACTLY one sentence

Rewritten claim:
"""
        try:
            out = self.llm(prompt, max_tokens=80)
            return out["choices"][0]["text"].strip()
        except Exception as e:
            print(f"⚠️ Rewrite failed: {e}")
            return None