from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException

from .config import get_settings
from .model_service import ReformulationModelService
from .schemas import (
    HealthResponse,
    ReformulationRequest,
    ReformulationResponse,
    SentimentRequest,
    SentimentResponse,
)
from .sentiment_service import SentimentModelService

settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    model_path = settings.model_dir.resolve()
    if not model_path.exists():
        raise RuntimeError(f"Le dossier modèle n'existe pas: {model_path}")

    sentiment_model_path = settings.sentiment_model_dir.resolve()
    if not sentiment_model_path.exists():
        raise RuntimeError(f"Le dossier modèle de sentiment n'existe pas: {sentiment_model_path}")

    app.state.model_service = ReformulationModelService(
        model_path=model_path,
        max_input_tokens=settings.max_input_tokens,
    )
    app.state.sentiment_service = SentimentModelService(
        model_path=sentiment_model_path,
        max_input_tokens=settings.sentiment_max_input_tokens,
    )
    yield


app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    service: ReformulationModelService = app.state.model_service
    return HealthResponse(
        status="ok",
        model_path=str(service.model_path),
        device=service.device,
    )


@app.post("/reformulate", response_model=ReformulationResponse)
def reformulate(payload: ReformulationRequest) -> ReformulationResponse:
    service: ReformulationModelService = app.state.model_service

    try:
        result = service.reformulate(
            input_text=payload.input_text,
            tone=payload.tone,
            style=payload.style,
            max_new_tokens=payload.max_new_tokens,
            num_beams=payload.num_beams,
        )
    except Exception as error:
        raise HTTPException(status_code=500, detail=f"Erreur de génération: {error}") from error

    return ReformulationResponse(
        input_text=payload.input_text,
        tone=payload.tone,
        style=payload.style,
        **result,
    )


@app.post("/sentiment", response_model=SentimentResponse)
def sentiment(payload: SentimentRequest) -> SentimentResponse:
    service: SentimentModelService = app.state.sentiment_service

    try:
        result = service.predict(input_text=payload.input_text)
    except Exception as error:
        raise HTTPException(status_code=500, detail=f"Erreur d'analyse de sentiment: {error}") from error

    return SentimentResponse(input_text=payload.input_text, **result)
