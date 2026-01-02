from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from datetime import datetime
import pandas as pd
import joblib
import os

# =========================
# Configurações
# =========================

MODEL_PATH = os.path.join(
    "model",
    "modelo_atraso_voos_xgb_te_v1.pkl"
)

# =========================
# Inicialização da API
# =========================

app = FastAPI(title="Flight Delay Prediction API")

# =========================
# Carregamento do modelo
# =========================

artifacts = joblib.load(MODEL_PATH)

# Validação defensiva do artefato
if not isinstance(artifacts, dict):
    raise RuntimeError("Artefato do modelo inválido: esperado um dict.")

required_keys = {"pipeline", "threshold_recomendado", "features_esperadas"}
missing = required_keys - artifacts.keys()
if missing:
    raise RuntimeError(f"Artefato do modelo incompleto. Faltando: {missing}")

pipeline = artifacts["pipeline"]
THRESHOLD = artifacts.get("threshold_recomendado", 0.5)
FEATURES_ESPERADAS = artifacts.get("features_esperadas")

# Log útil em startup
print(type(artifacts))
print(artifacts.keys())

# =========================
# Schemas (Pydantic)
# =========================

class FlightRequest(BaseModel):
    icao_empresa_aerea: str
    icao_aerodromo_origem: str
    icao_aerodromo_destino: str
    hora_prevista: datetime
    voos_no_slot: int
    tempo_voo_estimado: float

# =========================
# Feature Engineering
# =========================

def gerar_features(data: dict) -> pd.DataFrame:
    df = pd.DataFrame([data])

    # Hora fracionada (ex: 14.5 = 14:30)
    df["hora_prevista_frac"] = (
        df["hora_prevista"].dt.hour +
        df["hora_prevista"].dt.minute / 60
    )

    # Faixa horária (exemplo simples)
    def faixa_horaria(h):
        if 6 <= h < 12:
            return "manha"
        elif 12 <= h < 18:
            return "tarde"
        elif 18 <= h < 24:
            return "noite"
        else:
            return "madrugada"

    df["faixa_horaria"] = df["hora_prevista"].dt.hour.apply(faixa_horaria)

    # Fim de semana
    df["eh_fim_de_semana"] = df["hora_prevista"].dt.weekday >= 5

    # Mês do voo
    df["mes"] = df["hora_prevista"].dt.month

    # Remover coluna original
    df = df.drop(columns=["hora_prevista"])

    # Garantir ordem e presença das features
    df = df[FEATURES_ESPERADAS]

    return df

# =========================
# Endpoints
# =========================

@app.get("/")
def health_check():
    return {
        "status": "ok",
        "model_loaded": True
    }

@app.post("/predict")
def predict_flight_delay(request: FlightRequest):

    try:
        # Geração das features
        df = gerar_features(request.dict())

        # Garantir ordem/contrato das features
        if FEATURES_ESPERADAS:
            df = df[FEATURES_ESPERADAS]

        # Predição
        prob = pipeline.predict_proba(df)[0][1]
        previsao = "Atrasado" if prob >= THRESHOLD else "Pontual"

        return {
            "previsao": previsao,
            "probabilidade": round(float(prob), 4),
            "threshold_usado": THRESHOLD
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )
