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


class HealthResponse(BaseModel):
    status: str
    model_path: str
    device: str
    model_name: str
    model_version: str | None = None
