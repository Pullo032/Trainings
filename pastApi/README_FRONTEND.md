# README Frontend — API de reformulation

Ce document est destiné aux développeurs frontend qui consomment l’API.

## Base URL

- Local: `http://localhost:8000`
- Swagger: `http://localhost:8000/docs`

## Authentification

- Aucune authentification requise.
- Aucun token nécessaire (usage éducatif).

## Endpoints utiles

### 1) Health check

- **GET** `/health`

Réponse exemple:

```json
{
  "status": "ok",
  "model_path": "C:/.../modele_final",
  "device": "cpu"
}
```

Usage: vérifier que l’API est en ligne avant d’activer la soumission côté UI.

---

### 2) Prédiction de reformulation

- **POST** `/reformulate`
- **Content-Type**: `application/json`

Body:

```json
{
  "input_text": "je vend chaussure nike noir taille 42 neuf prix 15000 fcfa",
  "tone": "professionnel",
  "style": "annonce commerciale",
  "target_text": "Chaussures Nike noires, taille 42, neuves, proposées au prix de 15 000 FCFA.",
  "max_new_tokens": 128,
  "num_beams": 4
}
```

Champs:

- `input_text` (string, requis)
- `tone` (string, requis)
- `style` (string, requis)
- `target_text` (string, requis)
- `max_new_tokens` (int, optionnel, défaut `128`, min `16`, max `512`)
- `num_beams` (int, optionnel, défaut `4`, min `1`, max `8`)

Réponse succès (`200`) — format:

```json
{
  "input_text": "je vend chaussure nike noir taille 42 neuf prix 15000 fcfa",
  "tone": "professionnel",
  "style": "annonce commerciale",
  "target_text": "Chaussures Nike noires, taille 42, neuves, proposées au prix de 15 000 FCFA.",
  "prediction": "Chaussures Nike noires, taille 42, neuves, proposées au prix de 15 000 FCFA."
}
```

Réponse réelle observée (exemple de test):

```json
{
  "input_text": "je vend chaussure nike noir taille 42 neuf prix 15000 fcfa",
  "tone": "professionnel",
  "style": "annonce commerciale",
  "target_text": "Chaussures Nike noires, taille 42, neuves, proposées au prix de 15 000 FCFA.",
  "prediction": "Je vends une chaussure nike noire taille 42 neuf prix 15000 fcfa. Elle est proposée à un professionnel ou à un professionnel."
}
```

Important pour le frontend:

- `prediction` est généré par le modèle et peut être imparfait.
- `target_text` est la référence envoyée par le client (vérité attendue).
- L’UI doit afficher `target_text` et `prediction` côte à côte pour comparaison.

## Gestion des erreurs côté frontend

### Validation (`422`)

Quand un champ est manquant ou invalide.

Exemples:

- `input_text` vide
- `target_text` vide
- `max_new_tokens` hors limites

### Erreur serveur (`500`)

Quand le modèle ne peut pas générer une réponse.

Format:

```json
{
  "detail": "Erreur de génération: ..."
}
```

## Exemple `fetch` (React / JS)

```js
const API_BASE_URL = "http://localhost:8000";

export async function reformulate(payload) {
  const response = await fetch(`${API_BASE_URL}/reformulate`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });

  const data = await response.json();

  if (!response.ok) {
    throw new Error(data?.detail || "Erreur API");
  }

  return data;
}
```

Payload minimal conseillé:

```js
{
  input_text: "Votre texte source",
  tone: "professionnel",
  style: "naturel",
  target_text: "Votre version corrigée de référence"
}
```

## Notes d’intégration

- Pas de streaming actuellement (réponse JSON unique).
- Le temps de réponse dépend de la taille du texte et des paramètres (`num_beams`, `max_new_tokens`).
- `target_text` est renvoyé tel quel pour faciliter la comparaison côté UI avec `prediction`.
