from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


MODEL_NAME = "camembert-sentiment"


class SentimentModelService:
    def __init__(self, model_path: Path, max_input_tokens: int = 256):
        self.model_path = model_path
        self.max_input_tokens = max_input_tokens
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.tokenizer = AutoTokenizer.from_pretrained(str(model_path))
        self.model = AutoModelForSequenceClassification.from_pretrained(str(model_path)).to(self.device)
        self.model.eval()

        self.id2label = {int(index): label for index, label in self.model.config.id2label.items()}
        self.model_version = getattr(self.model.config, "transformers_version", None)

    @torch.inference_mode()
    def predict(self, input_text: str) -> dict:
        enc = self.tokenizer(
            input_text,
            max_length=self.max_input_tokens,
            truncation=True,
            return_tensors="pt",
        ).to(self.device)

        logits = self.model(**enc).logits

        probabilities = torch.softmax(logits, dim=-1)[0].tolist()
        best_index = max(range(len(probabilities)), key=probabilities.__getitem__)

        return {
            "label": self.id2label[best_index],
            "score": round(probabilities[best_index], 4),
            "probabilities": {
                self.id2label[index]: round(probability, 4)
                for index, probability in enumerate(probabilities)
            },
            "metadata": {
                "model_name": MODEL_NAME,
                "model_version": self.model_version,
            },
        }
