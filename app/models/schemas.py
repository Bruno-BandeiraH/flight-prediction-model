"""
Schemas Pydantic para validação de dados.
"""
from pydantic import BaseModel
from datetime import datetime


class FlightRequest(BaseModel):
    """Schema para requisição de predição de voo."""
    icao_empresa_aerea: str
    icao_aerodromo_origem: str
    icao_aerodromo_destino: str
    hora_prevista: datetime
    voos_no_slot: int
    tempo_voo_estimado: float


class PredictionResponse(BaseModel):
    """Schema para resposta de predição."""
    previsao: str
    probabilidade: float
    threshold_usado: float


class HealthResponse(BaseModel):
    """Schema para resposta de health check."""
    status: str
    model_loaded: bool

