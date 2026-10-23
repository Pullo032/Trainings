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

        config = self.model.config
        configured_name = getattr(config, "_name_or_path", None)
        self.model_name = configured_name if configured_name and configured_name != str(model_path) else model_path.name
        self.model_version = getattr(config, "transformers_version", None)

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
    ) -> dict:
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
        return {
            "prediction": reformulated_text,
            "generation_parameters": {
                "max_new_tokens": max_new_tokens,
                "num_beams": num_beams,
            },
            "metadata": {
                "model_name": self.model_name,
                "model_version": self.model_version,
            },
        }
