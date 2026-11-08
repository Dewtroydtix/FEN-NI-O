"""Capa de SST espacial real de NOAA/PSL para el visualizador Streamlit.

La fuente es NOAA ERSSTv5 en cuadrícula de 2° x 2°, en °C absolutos.
NO confundir con el índice regional Niño 1+2 ni con anomalías locales.
"""
from __future__ import annotations

import io
from typing import Any

import numpy as np
import pandas as pd
import requests
import xarray as xr

NCSS_URL = (
    "https://psl.noaa.gov/thredds/ncss/grid/"
    "Datasets/noaa.ersst.v5/sst.mnmean.nc"
)
# Coordenadas en la convención NOAA 0-360° E: 270° equivale a -90° O.
WEST_360, EAST_360 = 270.0, 294.0
SOUTH, NORTH = -24.0, 4.0
USER_AGENT = "NINOPROB-IA-Streamlit/1.0 (investigacion-escolar)"


def ncss_params(mes: str) -> dict[str, str | float]:
    """Parámetros fijos para un subconjunto costeño NOAA de un solo mes."""
    period = pd.Period(mes, freq="M")
    if not (pd.Period("1950-01", freq="M") <= period <= pd.Period("2025-12", freq="M")):
        raise ValueError("El mapa solo admite fechas del estudio: 1950-01 a 2025-12.")
    return {
        "var": "sst",
        "north": NORTH,
        "south": SOUTH,
        "west": WEST_360,
        "east": EAST_360,
        "time": f"{period.year:04d}-{period.month:02d}-15T00:00:00Z",
        "accept": "netcdf",
        "horizStride": 1,
    }


def decode_netcdf(raw: bytes, mes: str) -> pd.DataFrame:
    """Lee únicamente celdas oceánicas del subconjunto NetCDF descargado.

    La cuadrícula contiene temperatura SST ABSOLUTA, no anomalía climática.
    """
    if not (raw.startswith(b"CDF") or raw.startswith(b"\x89HDF")):
        raise ValueError("NOAA no devolvió un archivo NetCDF válido.")
    with xr.open_dataset(io.BytesIO(raw)) as ds:
        if "sst" not in ds.variables or not {"lat", "lon"} <= set(ds.coords):
            raise ValueError("NetCDF sin variable SST o coordenadas lat/lon.")
        sst = ds["sst"]
        # La consulta NCSS debe devolver solo una marca temporal.
        if "time" in sst.dims:
            if int(sst.sizes["time"]) != 1:
                raise ValueError("El subconjunto NOAA contiene más de un mes.")
            sst = sst.isel(time=0)
        frame = sst.to_dataframe(name="sst_c").reset_index()
    frame = frame[["lat", "lon", "sst_c"]].copy()
    frame["lon"] = ((frame["lon"].astype(float) + 180) % 360) - 180
    frame = frame.dropna(subset=["sst_c"])
    frame = frame[
        np.isfinite(frame["sst_c"])
        & (frame["sst_c"] > -2.5)
        & (frame["sst_c"] < 45)
        & frame["lat"].between(SOUTH, NORTH)
        & frame["lon"].between(-90, -66)
    ]
    if frame.empty:
        raise ValueError("El subconjunto NOAA no tiene celdas válidas frente a Perú.")
    frame["mes"] = mes
    return frame.reset_index(drop=True)


def fetch_gridded_sst(mes: str) -> pd.DataFrame:
    """Descarga remota opcional; nunca inventa valores si NOAA no responde."""
    response = requests.get(
        NCSS_URL,
        params=ncss_params(mes),
        headers={"User-Agent": USER_AGENT},
        timeout=(15, 70),
    )
    response.raise_for_status()
    if len(response.content) > 12_000_000:
        raise ValueError("La respuesta NOAA es inesperadamente grande.")
    return decode_netcdf(response.content, mes)


def thermal_color(sst_c: float) -> list[int]:
    """Escala fija, °C absolutos, sin reescalado engañoso por cada mes."""
    # Azul frío a rojo cálido. Color e intensidad son representativos, no causales.
    anchors = [
        (14., [24, 91, 160, 230]),
        (18., [28, 159, 188, 230]),
        (22., [35, 192, 160, 235]),
        (26., [247, 193, 69, 235]),
        (29., [240, 113, 55, 240]),
        (32., [177, 42, 60, 245]),
    ]
    value = float(np.clip(sst_c, 14, 32))
    for (lo, color_lo), (hi, color_hi) in zip(anchors[:-1], anchors[1:]):
        if lo <= value <= hi:
            fraction = (value - lo) / (hi - lo)
            return [
                int(round(a + fraction * (b - a)))
                for a, b in zip(color_lo, color_hi)
            ]
    return anchors[-1][1]


def spatial_polygons(frame: pd.DataFrame) -> list[dict[str, Any]]:
    """Crea polígonos con altura ilustrativa (NO profundidad real)."""
    if not {"lat", "lon", "sst_c"} <= set(frame.columns):
        raise ValueError("Faltan columnas de la cuadrícula NOAA.")
    polygons: list[dict[str, Any]] = []
    for lat, lon, value in frame[["lat", "lon", "sst_c"]].itertuples(index=False, name=None):
        if not np.isfinite(value) or not (-90 <= lon <= -66 and -24 <= lat <= 4):
            continue
        polygons.append({
            "polygon": [
                [float(lon - 1), float(lat - 1)],
                [float(lon + 1), float(lat - 1)],
                [float(lon + 1), float(lat + 1)],
                [float(lon - 1), float(lat + 1)],
            ],
            "lat": round(float(lat), 2),
            "lon": round(float(lon), 2),
            "sst_c": round(float(value), 2),
            "color": thermal_color(float(value)),
            # Únicamente para visualizar valores: altura vertical sin significado oceánico.
            "altura": max(50., (float(value) - 12) * 900.),
        })
    return polygons
