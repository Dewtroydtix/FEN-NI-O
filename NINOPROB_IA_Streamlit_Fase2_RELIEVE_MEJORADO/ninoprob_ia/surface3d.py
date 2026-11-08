"""Superficie térmica 3D de visualización (NO es una nueva medición NOAA).

NOAA provee temperatura superficial por celdas; interpolar suaviza la vista entre
celdas del océano sin crear observaciones nuevas, y conserva máscara de ausencia
de datos para no dibujar sistemáticamente sobre tierra.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.ndimage import zoom

from .phase2 import DISPLAY_MODES


def preparar_superficie(
    frame: pd.DataFrame,
    modo: str,
    exageracion: float = 1.3,
    limite: int = 140,
) -> dict:
    """Prepara malla geográfica interpolada conservando vacíos oceánicos.

    El interpolado se calcula solo para la presentación, no para entrenar la IA.
    El método no puede detallar el litoral mejor que la resolución NOAA original.
    """
    if modo not in DISPLAY_MODES:
        raise ValueError("Capa desconocida.")
    columna, _, _ = DISPLAY_MODES[modo]
    if columna not in frame or not {"lat", "lon"} <= set(frame.columns):
        raise ValueError("Faltan columnas del mapa.")
    if not 0.0 <= exageracion <= 3.0:
        raise ValueError("Exageración fuera de rango.")

    base = (
        frame[["lat", "lon", columna]]
        .drop_duplicates(["lat", "lon"])
        .pivot(index="lat", columns="lon", values=columna)
        .sort_index(axis=0)
        .sort_index(axis=1)
    )
    lat = base.index.to_numpy(dtype=float)
    lon = base.columns.to_numpy(dtype=float)
    z = base.to_numpy(dtype=float)
    valido = np.isfinite(z)
    if len(lat) < 2 or len(lon) < 2 or np.count_nonzero(valido) < 4:
        raise ValueError("Se requieren varias celdas oceánicas para una superficie.")
    if not np.isfinite(z[valido]).all():
        raise ValueError("La cuadrícula contiene valores inválidos.")

    # Nunca se añaden nuevas observaciones. Interpolación de valores NOAA
    # entre posiciones próximas donde existe soporte del dato original.
    escala = min(8, max(1, int(limite / max(z.shape))))
    if escala > 1:
        valores = zoom(np.where(valido, z, 0), escala, order=1)
        pesos = zoom(valido.astype(float), escala, order=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            suavizado = np.where(pesos > 0.90, valores / pesos, np.nan)
        eje_lat = np.linspace(lat.min(), lat.max(), suavizado.shape[0])
        eje_lon = np.linspace(lon.min(), lon.max(), suavizado.shape[1])
    else:
        suavizado = np.where(valido, z, np.nan)
        eje_lat, eje_lon = lat, lon

    if columna == "sst_c":
        # "Altura" inventada solo como transformación gráfica lineal,
        # NO representa altura del agua ni batimetría.
        altura = np.maximum(suavizado - 14.0, 0) * exageracion
    else:
        altura = (suavizado + 3.0) * exageracion

    return dict(lat=eje_lat, lon=eje_lon, valores=suavizado, altura=altura,
                mediciones=len(frame), originales=z.shape)


def grafico_superficie(
    frame: pd.DataFrame,
    modo: str,
    exageracion: float,
    inclinacion: int,
) -> go.Figure:
    """Figura 3D giratoria a partir de la cuadrícula NOAA recibida."""
    info = preparar_superficie(frame, modo, exageracion)
    columna = DISPLAY_MODES[modo][0]
    es_sst = columna == "sst_c"
    colores = [
        [0.00, "#184395"], [0.22, "#1e8fcf"], [0.45, "#2bccba"],
        [0.65, "#f8d059"], [0.82, "#ef6d31"], [1.00, "#a92337"],
    ] if es_sst else [
        [0.00, "#22489f"], [0.25, "#379ed6"], [0.50, "#e9f1f1"],
        [0.75, "#f6ad43"], [1.00, "#b12c35"],
    ]
    referencia = (14, 32) if es_sst else (-3, 3)
    fig = go.Figure(data=[
        go.Surface(
            x=info["lon"], y=info["lat"], z=info["altura"],
            surfacecolor=info["valores"],
            colorscale=colores, cmin=referencia[0], cmax=referencia[1],
            connectgaps=False, showscale=True,
            colorbar={"title": {"text": "°C"}, "len": 0.68, "thickness": 14,
                      "tickfont": {"color": "#cde6ec"}},
            hovertemplate=(
                "Longitud: %{x:.2f}°<br>"
                "Latitud: %{y:.2f}°<br>"
                "Valor interpolado: %{surfacecolor:.2f} °C"
                "<extra>Visualización, no medición adicional</extra>"
            ),
            lighting={"ambient": 0.72, "diffuse": 0.70, "roughness": 0.8,
                      "specular": 0.10},
        )
    ])
    altura_camara = 0.45 + (70 - min(max(inclinacion, 0), 70)) * 0.044
    fig.update_layout(
        margin={"l": 0, "r": 0, "t": 12, "b": 0},
        paper_bgcolor="#061321",
        font={"color": "#d7f2f6"},
        scene={
            "bgcolor": "#061321",
            "xaxis": {"title": "Longitud (°)", "gridcolor": "#284b62",
                      "backgroundcolor": "#071829"},
            "yaxis": {"title": "Latitud (°)", "gridcolor": "#284b62",
                      "backgroundcolor": "#071829"},
            "zaxis": {"title": "Altura visual", "showticklabels": False,
                      "gridcolor": "#244356", "backgroundcolor": "#071829"},
            "aspectmode": "manual",
            "aspectratio": {"x": 1.15, "y": 1, "z": 0.38},
            "camera": {"eye": {"x": 1.55, "y": -1.7, "z": altura_camara}},
        },
        height=660,
        uirevision="oceano_superficie",
    )
    return fig
