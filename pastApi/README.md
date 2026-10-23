# ADR Reformulation API

Service FastAPI exposant, via une API REST, deux modèles de traitement automatique du français :

- la **reformulation** de texte (notebook `Models-copy/Trainings/ARD.ipynb`), qui reformule un texte source selon un **ton** et un **style** donnés ;
- l'**analyse de sentiment** d'avis clients (notebook `Models-copy/Trainings/sentiment.ipynb`), qui classe un texte comme `POSITIVE` ou `NEGATIVE`.

## Structure du projet

```
pastApi/
├── app/
│   ├── __init__.py
│   ├── config.py          # Paramètres de l'application (pydantic-settings, lecture de .env)
│   ├── main.py            # Application FastAPI et endpoints
│   ├── model_service.py   # Chargement du modèle de reformulation et génération
│   ├── schemas.py         # Schémas de requête et de réponse (Pydantic)
│   └── sentiment_service.py  # Chargement du modèle de sentiment et prédiction
├── modele_final/          # Modèle T5 de reformulation (format Hugging Face Transformers)
├── .env.example           # Modèle de configuration
└── requirements.txt       # Dépendances Python
```

## Prérequis

- Python 3.12 (version utilisée pour le développement)
- Un GPU CUDA est optionnel : le service utilise automatiquement le CPU si CUDA n'est pas disponible.

## 1) Installation

Depuis le dossier `API/` du dépôt :

```bash
cd pastApi
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## 2) Configuration

Créer le fichier `.env` à partir du modèle fourni :

```bash
copy .env.example .env
```

| Variable | Valeur par défaut | Description |
| --- | --- | --- |
| `APP_NAME` | `ADR Reformulation API` | Nom de l'application affiché dans la documentation OpenAPI. |
| `APP_VERSION` | `1.0.0` | Version de l'application affichée dans la documentation OpenAPI. |
| `MAX_INPUT_TOKENS` | `256` | Nombre maximal de tokens du prompt d'entrée ; au-delà, le texte est tronqué. |
| `MODEL_DIR` | `./modele_final` | Chemin vers le modèle de reformulation. Un chemin relatif est résolu depuis le répertoire de lancement : lancer le service depuis `pastApi/`. |
| `SENTIMENT_MODEL_DIR` | `../../Models-copy/camembert-sentiment/modele_final` | Chemin vers le modèle d'analyse de sentiment. Résolu depuis le répertoire de lancement. |
| `SENTIMENT_MAX_INPUT_TOKENS` | `256` | Nombre maximal de tokens analysés pour le sentiment ; au-delà, le texte est tronqué. |

Si aucun fichier `.env` n'est présent, les valeurs par défaut sont utilisées : `MODEL_DIR` et `SENTIMENT_MODEL_DIR` pointent alors vers les dossiers `modele_final` de `pastApi/` et de `Models-copy/camembert-sentiment/`.

## 3) Lancer l'API

Depuis le répertoire `pastApi/` :

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Pour le développement, ajouter `--reload` afin de relancer automatiquement le serveur à chaque modification du code.

Les deux modèles sont chargés une seule fois au démarrage. Si l'un des dossiers `MODEL_DIR` ou `SENTIMENT_MODEL_DIR` n'existe pas, le démarrage échoue avec un message explicite (`Le dossier modèle n'existe pas` ou `Le dossier modèle de sentiment n'existe pas`).

## Modèles

### Reformulation

| Caractéristique | Valeur |
| --- | --- |
| Architecture | `T5ForConditionalGeneration` (seq2seq) |
| Couches encodeur / décodeur | 6 / 6 |
| Dimension du modèle (`d_model`) | 512 |
| Taille du vocabulaire | 32 128 |
| Version Transformers à l'export | 4.57.1 |
| Périphérique d'inférence | CUDA si disponible, sinon CPU |

Le texte d'entrée est transformé en prompt avant inférence, selon le format suivant :

```
reformule en ton {tone} et style {style} : {input_text}
```

La génération utilise une recherche par faisceaux (*beam search*) avec `early_stopping=True`.

### Analyse de sentiment

Classification binaire d'avis clients en français, obtenue par fine-tuning de `almanach/camembert-base` (notebook `Models-copy/Trainings/sentiment.ipynb`).

| Caractéristique | Valeur |
| --- | --- |
| Architecture | `CamembertForSequenceClassification` |
| Classes | `NEGATIVE` (0), `POSITIVE` (1) |
| Données | 10 000 avis issus de `CATIE-AQ/amazon_reviews_multi_fr_prompt_sentiment_analysis`, mélangés (seed 42), découpés en 80 % entraînement / 20 % test |
| Hyperparamètres | Learning rate 2e-5, batch 8, 5 époques, weight decay 0,01 |
| Accuracy (jeu de test) | 0,8575 (meilleur checkpoint, étape 4000) |
| F1 (jeu de test) | 0,879 |
| Longueur maximale | 256 tokens (les textes plus longs sont tronqués) |

Le modèle a été entraîné sur des avis de produits Amazon. Ses performances sur d'autres domaines (restaurants, services, réseaux sociaux) n'ont pas été évaluées.

## 4) Endpoints

### `POST /reformulate`

Reformule un texte selon le ton et le style demandés.

**Corps de la requête** (`application/json`)

| Champ | Type | Obligatoire | Contraintes | Valeur par défaut | Description |
| --- | --- | --- | --- | --- | --- |
| `input_text` | string | Oui | Au moins 1 caractère | — | Texte source à reformuler. |
| `tone` | string | Oui | Au moins 1 caractère | — | Ton souhaité (ex. `professionnel`). |
| `style` | string | Oui | Au moins 1 caractère | — | Style souhaité (ex. `persuasif`). |
| `max_new_tokens` | integer | Non | 16 à 512 | `128` | Longueur maximale de la sortie générée. |
| `num_beams` | integer | Non | 1 à 8 | `4` | Nombre de faisceaux de la recherche par faisceaux. |

Exemple de requête :

```json
{
  "input_text": "Appartement 3 pièces de 78m2 proche centre ville, loyer 1200€.",
  "tone": "professionnel",
  "style": "persuasif",
  "max_new_tokens": 128,
  "num_beams": 4
}
```

**Réponse `200 OK`**

| Champ | Type | Description |
| --- | --- | --- |
| `input_text` | string | Texte source, tel que reçu. |
| `tone` | string | Ton demandé. |
| `style` | string | Style demandé. |
| `prediction` | string | Texte reformulé généré par le modèle. |
| `generation_parameters.max_new_tokens` | integer | Longueur maximale appliquée. |
| `generation_parameters.num_beams` | integer | Nombre de faisceaux appliqué. |
| `metadata.model_name` | string | Nom du modèle chargé. |
| `metadata.model_version` | string \| null | Version de Transformers utilisée à l'export. |

Exemple de réponse :

```json
{
  "input_text": "Appartement 3 pièces de 78m2 proche centre ville, loyer 1200€.",
  "tone": "professionnel",
  "style": "persuasif",
  "prediction": "Appartement de 3 pièces d'une surface de 78m2 proche de centre ville. Loyer de 1200€.",
  "generation_parameters": {
    "max_new_tokens": 128,
    "num_beams": 4
  },
  "metadata": {
    "model_name": "modele_final",
    "model_version": "4.57.1"
  }
}
```

**Erreurs**

| Code | Cause |
| --- | --- |
| `422 Unprocessable Entity` | Un champ est absent, vide ou hors des bornes autorisées. |
| `500 Internal Server Error` | Erreur survenue pendant la génération ; le détail est renvoyé dans le champ `detail`. |

### `POST /sentiment`

Détermine le sentiment d'un texte (classes `POSITIVE` ou `NEGATIVE`) et retourne la probabilité de chaque classe.

**Corps de la requête** (`application/json`)

| Champ | Type | Obligatoire | Contraintes | Valeur par défaut | Description |
| --- | --- | --- | --- | --- | --- |
| `input_text` | string | Oui | Au moins 1 caractère | — | Texte à analyser. Tronqué au-delà de `SENTIMENT_MAX_INPUT_TOKENS` tokens. |

Exemple de requête :

```json
{
  "input_text": "Produit excellent, livraison rapide et très bonne qualité."
}
```

**Réponse `200 OK`**

| Champ | Type | Description |
| --- | --- | --- |
| `input_text` | string | Texte analysé, tel que reçu. |
| `label` | string | Classe prédite : `POSITIVE` ou `NEGATIVE`. |
| `score` | number | Probabilité de la classe prédite, arrondie à 4 décimales. |
| `probabilities` | object | Probabilité de chaque classe (`NEGATIVE`, `POSITIVE`), dont la somme vaut 1. |
| `metadata.model_name` | string | Identifiant du modèle (`camembert-sentiment`). |
| `metadata.model_version` | string \| null | Version de Transformers utilisée à l'export. |

Exemple de réponse (valeurs illustratives) :

```json
{
  "input_text": "Produit excellent, livraison rapide et très bonne qualité.",
  "label": "POSITIVE",
  "score": 0.9961,
  "probabilities": {
    "NEGATIVE": 0.0039,
    "POSITIVE": 0.9961
  },
  "metadata": {
    "model_name": "camembert-sentiment",
    "model_version": "4.57.1"
  }
}
```

**Erreurs**

| Code | Cause |
| --- | --- |
| `422 Unprocessable Entity` | Le champ `input_text` est absent ou vide. |
| `500 Internal Server Error` | Erreur survenue pendant l'analyse ; le détail est renvoyé dans le champ `detail`. |

## 5) Documentation interactive

La documentation Swagger, générée automatiquement, est disponible à l'adresse :

- `http://localhost:8000/docs`

