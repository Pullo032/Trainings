import string
from pathlib import Path

import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer


MODEL_NAME = "camembert-ner"
OUTSIDE_LABEL = "O"
# Ponctuation, espaces et guillemets retirés des bords d'une entité (ex. "Unesco." -> "Unesco").
EDGE_CHARACTERS = string.punctuation + string.whitespace + "«»“”‘’…"


class NERModelService:
    def __init__(self, model_path: Path, max_input_tokens: int = 256):
        self.model_path = model_path
        self.max_input_tokens = max_input_tokens
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.tokenizer = AutoTokenizer.from_pretrained(str(model_path))
        if not self.tokenizer.is_fast:
            raise RuntimeError("Le tokenizer NER doit être rapide (fast) pour fournir la position des entités.")

        self.model = AutoModelForTokenClassification.from_pretrained(str(model_path)).to(self.device)
        self.model.eval()

        self.id2label = {int(index): label for index, label in self.model.config.id2label.items()}
        self.model_version = getattr(self.model.config, "transformers_version", None)

    @torch.inference_mode()
    def predict(self, input_text: str) -> dict:
        enc = self.tokenizer(
            input_text,
            max_length=self.max_input_tokens,
            truncation=True,
            return_offsets_mapping=True,
            return_tensors="pt",
        )
        offsets = enc.pop("offset_mapping")[0].tolist()
        word_ids = enc.word_ids(batch_index=0)

        logits = self.model(**enc.to(self.device)).logits[0]
        scores, label_ids = torch.softmax(logits, dim=-1).max(dim=-1)

        return {
            "entities": group_word_entities(
                text=input_text,
                offsets=offsets,
                word_ids=word_ids,
                label_ids=label_ids.tolist(),
                scores=scores.tolist(),
                id2label=self.id2label,
            ),
            "metadata": {
                "model_name": MODEL_NAME,
                "model_version": self.model_version,
            },
        }


def group_word_entities(
    text: str,
    offsets: list[list[int]],
    word_ids: list[int | None],
    label_ids: list[int],
    scores: list[float],
    id2label: dict[int, str],
) -> list[dict]:
    # Comme à l'entraînement, seule la première sous-unité de chaque mot porte la prédiction,
    # mais le mot couvre toutes ses sous-unités.
    word_spans = {}
    for token_index, word_index in enumerate(word_ids):
        if word_index is None:
            continue
        token_start, token_end = offsets[token_index]
        if word_index not in word_spans:
            word_spans[word_index] = [token_start, token_end, id2label[label_ids[token_index]], scores[token_index]]
        else:
            word_spans[word_index][1] = token_end
    words = [tuple(span) for span in word_spans.values()]

    # Les mots consécutifs de même catégorie forment une seule entité.
    groups = []
    current = None
    for start, end, label, score in words:
        if label == OUTSIDE_LABEL:
            current = None
            continue
        if current is not None and current["label"] == label:
            current["end"] = end
            current["scores"].append(score)
            continue
        current = {"start": start, "end": end, "label": label, "scores": [score]}
        groups.append(current)

    entities = []
    for group in groups:
        raw_span = text[group["start"]:group["end"]]
        leading = len(raw_span) - len(raw_span.lstrip(EDGE_CHARACTERS))
        trailing = len(raw_span) - len(raw_span.rstrip(EDGE_CHARACTERS))
        start = group["start"] + leading
        end = group["end"] - trailing
        if start >= end:
            continue
        entities.append({
            "text": text[start:end],
            "label": group["label"],
            "score": round(sum(group["scores"]) / len(group["scores"]), 4),
            "start": start,
            "end": end,
        })
    return entities
