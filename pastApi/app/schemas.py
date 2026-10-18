from pydantic import BaseModel, Field


class ReformulationRequest(BaseModel):
    input_text: str = Field(..., min_length=1, description="Texte source à reformuler")
    tone: str = Field(..., min_length=1, description="Ton souhaité")
    style: str = Field(..., min_length=1, description="Style souhaité")
    max_new_tokens: int = Field(128, ge=16, le=512, description="Longueur max de sortie")
    num_beams: int = Field(4, ge=1, le=8, description="Beam search")
    target_text: str = Field(..., min_length=1, description="Texte cible fourni par le visiteur")


class ReformulationResponse(BaseModel):
    input_text: str
    tone: str
    style: str
    target_text: str
    prediction: str


class HealthResponse(BaseModel):
    status: str
    model_path: str
    device: str
