# FastAPI - Service de reformulation ADR

Ce projet expose en API le modèle entraîné dans le notebook `ARD.ipynb`.

## 1) Installation

```bash
cd API/pastApi
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## 2) Configuration

```bash
copy .env.example .env
```

Variable principale:

- `MODEL_DIR`: chemin vers le modèle exporté (`modele_final`).

## 3) Lancer l'API

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

## 4) Endpoints

- `GET /health` : statut du service et infos modèle.
- `POST /reformulate` : reformulation guidée par `tone` et `style`.

Exemple body:

```json
{
  "input_text": "Appartement 3 pièces de 78m2 proche centre ville, loyer 1200€.",
  "tone": "professionnel",
  "style": "persuasif",
  "max_new_tokens": 128,
  "num_beams": 4
}
```

## 5) Docs Swagger

- `http://localhost:8000/docs`

