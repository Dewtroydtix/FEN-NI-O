"""API JSON preparada para recibir un visualizador geográfico 3D.

Ejecutar: uvicorn ninoprob_ia.api:app --reload
No simula datos ni genera una capa térmica píxel a píxel.
"""
from __future__ import annotations

import json
import os
import re
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from ninoprob_ia.core import BBOX_NINO12, FEATURES, NOAA_URL, THRESHOLD

HOME = Path(os.environ.get("NINOPROB_HOME", str(Path(__file__).resolve().parent.parent)))
RESULTS = HOME / "resultados"
app = FastAPI(
    title="NIÑOPROB-IA API",
    description="Estimación educativa Niño 1+2; no es una alerta ni un pronóstico oficial.",
    version="1.0.0",
)
origins = os.environ.get("NINOPROB_CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173")
app.add_middleware(
    CORSMiddleware, allow_origins=[s.strip() for s in origins.split(",") if s.strip()],
    allow_credentials=False, allow_methods=["GET"], allow_headers=["*"],
)


def month_valid(value: str) -> pd.Period:
    if not re.fullmatch(r"\d{4}-(0[1-9]|1[0-2])", value):
        raise HTTPException(status_code=422, detail="Usa YYYY-MM (ejemplo: 2024-03).")
    return pd.Period(value, freq="M")


@lru_cache(maxsize=1)
def load():
    model_path = RESULTS / "modelo_ninoprob_ia.joblib"
    data_path = RESULTS / "serie_mensual_noaa.csv"
    metric_path = RESULTS / "metricas.json"
    if not all(p.is_file() for p in (model_path, data_path, metric_path)):
        raise HTTPException(status_code=503, detail="Debes entrenar la IA con NOAA antes de usar la API.")
    # Solo cargar un modelo producido por este proyecto, no archivos de usuarios.
    artifact = joblib.load(model_path)
    if artifact["features"] != list(FEATURES):
        raise HTTPException(status_code=500, detail="Modelo incompatible con las variables previstas.")
    series = pd.read_csv(data_path)
    series["mes"] = series["mes"].astype(str)
    if series["mes"].duplicated().any():
        raise HTTPException(status_code=500, detail="Meses duplicados en la serie local.")
    metrics = json.loads(metric_path.read_text(encoding="utf-8"))
    if metrics["sha256_fuente"] != artifact["source_sha256"]:
        raise HTTPException(status_code=500, detail="Modelo y fuente NOAA no coinciden.")
    return artifact, series, metrics


@app.get("/health")
def health():
    return {
        "estado": "listo" if (RESULTS / "modelo_ninoprob_ia.joblib").is_file() else "pendiente_entrenamiento",
        "nombre": "NIÑOPROB-IA",
    }


@app.get("/api/v1/metadata")
def metadata():
    _, series, metrics = load()
    return {
        "modelo": "regresion_logistica",
        "nombre": "NIÑOPROB-IA",
        "fuente": metrics["fuente"],
        "url_noaa": NOAA_URL,
        "climatologia": "1991-2020",
        "registro_desde": series["mes"].min(),
        "registro_hasta": series["mes"].max(),
        "umbral_c": THRESHOLD,
        "bbox_nino12": BBOX_NINO12,
        "unidad_anomalia": "°C",
        "nota": "Media de la región Niño 1+2; no equivale a temperatura por ubicación.",
        "restriccion": "Uso escolar, no pronóstico oficial ENFEN.",
        "mapa_termico_detallado_disponible": False,
    }


@app.get("/api/v1/metricas")
def metrics():
    return load()[2]


@app.get("/api/v1/serie")
def series(desde: str = Query("1950-01"), hasta: str = Query("2025-12")):
    first = month_valid(desde)
    last = month_valid(hasta)
    if first > last:
        raise HTTPException(status_code=422, detail="desde debe ser anterior o igual a hasta.")
    _, df, _ = load()
    sub = df[(df["mes"] >= str(first)) & (df["mes"] <= str(last))].copy()
    return {
        "tipo": "observado_noaa",
        "unidad": "°C",
        "registros": [
            {"mes": str(r.mes),
             "anomalia_c": None if pd.isna(r.anomalia_c) else float(r.anomalia_c),
             "temperatura_c": None if pd.isna(r.temperatura_c) else float(r.temperatura_c)}
            for r in sub.itertuples(index=False)
        ],
    }


@app.get("/api/v1/prediccion/{mes_origen}")
def prediction(mes_origen: str):
    origin = month_valid(mes_origen)
    if origin < pd.Period("1950-03", "M") or origin > pd.Period("2025-12", "M"):
        raise HTTPException(status_code=422, detail="Mes fuera del intervalo de datos 1950–2025.")
    artifact, df, _ = load()
    series = df.set_index("mes")
    target = origin + 1
    dates = [str(origin), str(origin - 1), str(origin - 2)]
    if any(d not in series.index or pd.isna(series.loc[d, "anomalia_c"]) for d in dates):
        raise HTTPException(status_code=404, detail="No existen tres anomalías mensuales consecutivas.")
    values = [float(series.loc[d, "anomalia_c"]) for d in dates]
    m = target.month
    X = np.array([[values[0], values[1], values[2],
                   np.sin(2 * np.pi * (m - 1) / 12),
                   np.cos(2 * np.pi * (m - 1) / 12)]], dtype=float)
    p_ia = float(artifact["model"].predict_proba(X)[0, 1])
    observed = None
    if str(target) in series.index and pd.notna(series.loc[str(target), "anomalia_c"]):
        observed = float(series.loc[str(target), "anomalia_c"])
    stage = ("entrenamiento" if target.year <= 2015
             else "validacion" if target.year <= 2020
             else "prueba_independiente" if target.year <= 2025
             else "fuera_del_periodo_de_prueba")
    return {
        "mes_origen": str(origin),
        "mes_objetivo": str(target),
        "anomalias_entrada_c": {
            "actual": values[0], "anterior": values[1], "hace_dos_meses": values[2]
        },
        "probabilidad_ia": round(p_ia, 6),
        "probabilidad_frecuencia_historica": round(float(artifact["baseline"]), 6),
        "umbral_anomalia_c": THRESHOLD,
        "anomalia_objetivo_observada_c": observed,
        "evento_objetivo_observado": None if observed is None else int(observed >= THRESHOLD),
        "tipo_evaluacion": stage,
        "es_evaluacion_independiente": stage == "prueba_independiente",
        "aviso": "Predicción educativa regional, no pronóstico oficial ni mapa de temperaturas por punto."
    }


@app.get("/api/v1/region.geojson")
def region(mes: str = Query("2025-12")):
    """Área Niño 1+2 con un único índice regional: NO es un mapa espacial de calor."""
    target = month_valid(mes)
    _, df, _ = load()
    row = df.loc[df["mes"] == str(target)]
    if row.empty:
        raise HTTPException(status_code=404, detail="Sin observación para ese mes.")
    v = row.iloc[0]["anomalia_c"]
    west, south, east, north = BBOX_NINO12
    return {
        "type": "FeatureCollection",
        "features": [{
            "type": "Feature",
            "geometry": {"type": "Polygon", "coordinates": [[
                [west, south], [east, south], [east, north],
                [west, north], [west, south],
            ]]},
            "properties": {
                "nombre": "Región Niño 1+2",
                "mes": str(target),
                "anomalia_media_c": None if pd.isna(v) else float(v),
                "tipo_dato": "promedio_regional",
                "NO_es_mapa_termico_detallado": True,
                "fuente": "NOAA/CPC ERSSTv5",
            },
        }],
    }
