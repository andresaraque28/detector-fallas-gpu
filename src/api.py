"""API para clasificar una ventana de telemetría de GPU.

Desde la raíz del proyecto: uvicorn src.api:app --reload
Documentación interactiva: http://127.0.0.1:8000/docs
"""

from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, ConfigDict, Field

from .features import COLUMNAS_FEATURES, calcular_features


RUTA_MODELO = Path(__file__).resolve().parents[1] / "models" / "modelo.joblib"


class Lectura(BaseModel):
    """Una medición de un segundo; coincide con las señales de features.py."""

    model_config = ConfigDict(extra="forbid")

    temp_c: float = Field(ge=-40, le=125, strict=True, allow_inf_nan=False)
    power_w: float = Field(gt=0, strict=True, allow_inf_nan=False)
    util_pct: float = Field(ge=0, le=100, strict=True, allow_inf_nan=False)
    clock_mhz: float = Field(gt=0, strict=True, allow_inf_nan=False)
    ecc_errors: int = Field(ge=0, strict=True)


class Ventana(BaseModel):
    """Petición: diez o más lecturas de una ventana de una sola GPU."""

    model_config = ConfigDict(extra="forbid")

    lecturas: list[Lectura] = Field(min_length=10)


class Prediccion(BaseModel):
    estado_predicho: str
    confianza: float


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Carga el modelo una vez al iniciar el servidor."""
    if not RUTA_MODELO.is_file():
        raise RuntimeError(f"Falta {RUTA_MODELO}. Ejecuta primero python src/train.py")
    app.state.modelo = joblib.load(RUTA_MODELO)
    yield


app = FastAPI(title="Detector de fallas de GPU", lifespan=lifespan)


@app.get("/", include_in_schema=False)
def inicio() -> RedirectResponse:
    """Envía a la documentación al abrir la dirección principal."""
    return RedirectResponse(url="/docs")


@app.post("/predecir", response_model=Prediccion)
def predecir(ventana: Ventana) -> Prediccion:
    """Convierte la ventana en features y devuelve el estado estimado."""
    lecturas = pd.DataFrame([lectura.model_dump() for lectura in ventana.lecturas])
    x = calcular_features(lecturas)[COLUMNAS_FEATURES]
    modelo = app.state.modelo
    estado = str(modelo.predict(x)[0])
    indice = list(modelo.classes_).index(estado)
    confianza = float(modelo.predict_proba(x)[0][indice])
    return Prediccion(estado_predicho=estado, confianza=confianza)
