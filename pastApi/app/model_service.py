from pathlib import Path

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


class ReformulationModelService:
    def __init__(self, model_path: Path, max_input_tokens: int = 256):
        self.model_path = model_path
        self.max_input_tokens = max_input_tokens
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.tokenizer = AutoTokenizer.from_pretrained(str(model_path))
        self.model = AutoModelForSeq2SeqLM.from_pretrained(str(model_path)).to(self.device)
        self.model.eval()

    @staticmethod
    def build_prompt(input_text: str, tone: str, style: str) -> str:
        return f"reformule en ton {tone} et style {style} : {input_text}"

    @torch.inference_mode()
    def reformulate(
        self,
        input_text: str,
        tone: str,
        style: str,
        max_new_tokens: int = 128,
        num_beams: int = 4,
    ) -> tuple[str, str]:
        prompt = self.build_prompt(input_text=input_text, tone=tone, style=style)

        enc = self.tokenizer(
            [prompt],
            max_length=self.max_input_tokens,
            truncation=True,
            padding=True,
            return_tensors="pt",
        ).to(self.device)

        output_ids = self.model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            early_stopping=True,
        )

        reformulated_text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()
        return reformulated_text, prompt

