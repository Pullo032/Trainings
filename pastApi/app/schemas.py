from pydantic import BaseModel, Field


class ReformulationRequest(BaseModel):
    input_text: str = Field(..., min_length=1, description="Texte source à reformuler")
    tone: str = Field(..., min_length=1, description="Ton souhaité")
    style: str = Field(..., min_length=1, description="Style souhaité")
    max_new_tokens: int = Field(128, ge=16, le=512, description="Longueur max de sortie")
    num_beams: int = Field(4, ge=1, le=8, description="Beam search")


class GenerationParameters(BaseModel):
    max_new_tokens: int
    num_beams: int


class ModelMetadata(BaseModel):
    model_name: str
    model_version: str | None = None


class ReformulationResponse(BaseModel):
    input_text: str
    tone: str
    style: str
    prediction: str
    generation_parameters: GenerationParameters
    metadata: ModelMetadata


class SentimentRequest(BaseModel):
    input_text: str = Field(..., min_length=1, description="Texte à analyser")


class SentimentResponse(BaseModel):
    input_text: str
    label: str
    score: float
    probabilities: dict[str, float]
    metadata: ModelMetadata


class NERRequest(BaseModel):
    input_text: str = Field(..., min_length=1, description="Texte dans lequel détecter les entités")


class NEREntity(BaseModel):
    text: str = Field(..., description="Entité telle qu'elle apparaît dans le texte d'entrée")
    label: str = Field(..., description="Catégorie : PER, LOC, ORG ou MISC")
    score: float = Field(..., description="Confiance moyenne du modèle pour les mots de l'entité")
    start: int = Field(..., description="Index du premier caractère dans input_text (inclus)")
    end: int = Field(..., description="Index après le dernier caractère dans input_text (exclus)")


class NERResponse(BaseModel):
    input_text: str
    entities: list[NEREntity]
    metadata: ModelMetadata


class HealthResponse(BaseModel):
    status: str
    model_path: str
    device: str
    model_name: str
    model_version: str | None = None
