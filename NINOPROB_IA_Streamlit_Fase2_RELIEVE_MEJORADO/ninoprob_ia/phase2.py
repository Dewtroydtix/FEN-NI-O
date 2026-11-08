"""Visualizador geográfico de fase 2 para NIÑOPROB-IA.

Cada punto se obtiene de NOAA (ERSSTv5 mensual 2° / OISSTv2.1 diario 0.25°).
Las alturas de extrusión son metafóricas: NO representan relieve marino.
La climatología mensual espacial es NOAA ERSSTv5 1991–2020.
"""
from __future__ import annotations

import io
from datetime import date
from typing import Literal

import numpy as np
import pandas as pd
import requests
import xarray as xr

from .spatial import NCSS_URL, fetch_gridded_sst

REGIONS = {
    "Todo el mar peruano": (-90.0, -22.0, -69.0, 3.0),
    "Región Niño 1+2": (-90.0, -10.0, -80.0, 0.0),
    "Costa norte": (-85.0, -8.0, -77.0, 1.0),
    "Costa central": (-83.0, -16.0, -72.0, -6.0),
    "Costa sur": (-79.0, -21.0, -70.0, -13.0),
}
CLIMATOLOGY_URL = (
    "https://psl.noaa.gov/thredds/fileServer/Datasets/noaa.ersst.v5/"
    "sst.mon.ltm.1991-2020.nc"
)
OISST_NCSS_ROOT = (
    "https://psl.noaa.gov/thredds/ncss/grid/Datasets/"
    "noaa.oisst.v2.highres"
)
AGENT = "NINOPROB-IA-Educativo/2.0"
MAX_BYTES = 22_000_000


def validate_netcdf(raw: bytes) -> None:
    if len(raw) > MAX_BYTES or not raw.startswith((b"CDF", b"\x89HDF\r\n\x1a\n")):
        raise ValueError("La descarga no es un NetCDF válido o supera 22 MB.")


def clean_grid(raw: bytes, expected_date: str | None = None, decode_times: bool = True) -> pd.DataFrame:
    """Decodifica celdas reales; rechaza entradas incompletas o fuera de rango.

    expected_date valida el mes (ERSST) o día exacto (OISST) solicitado.
    """
    validate_netcdf(raw)
    with xr.open_dataset(io.BytesIO(raw), decode_times=decode_times) as ds:
        if "sst" not in ds or not {"lat", "lon"} <= set(ds.coords):
            raise ValueError("No se encontraron SST y coordenadas espaciales.")
        sst = ds["sst"]
        if "time" not in sst.dims or sst.sizes["time"] != 1:
            raise ValueError("La respuesta debe contener un único instante temporal.")
        if expected_date is not None:
            time_value = ds["time"].values[0]
            if not np.issubdtype(np.asarray(time_value).dtype, np.datetime64):
                raise ValueError("La respuesta NOAA no tiene una fecha interpretable.")
            observed = pd.Timestamp(time_value)
            expected = pd.Timestamp(expected_date)
            if len(expected_date) == 7:
                if observed.to_period("M") != expected.to_period("M"):
                    raise ValueError("NOAA respondió con un mes diferente al solicitado.")
            elif observed.date() != expected.date():
                raise ValueError("NOAA respondió con un día diferente al solicitado.")
        frame = sst.isel(time=0).to_dataframe(name="sst_c").reset_index()
    frame = frame[["lat", "lon", "sst_c"]].copy()
    frame["lon"] = ((frame["lon"].astype(float) + 180.0) % 360.0) - 180.0
    frame = frame.dropna(subset=["sst_c"]).copy()
    frame = frame[
        frame["lat"].between(-24.0, 4.0)
        & frame["lon"].between(-92.0, -68.0)
        & np.isfinite(frame["sst_c"])
        & frame["sst_c"].between(-2.0, 45.0)
    ]
    if frame.empty:
        raise ValueError("NOAA no devolvió temperatura oceánica en el área de estudio.")
    return frame.reset_index(drop=True)


def decode_climatology(raw: bytes, month: int) -> pd.DataFrame:
    """Lee el archivo oficial 1991–2020, incluyendo coordenadas y 12 meses.

    Importante: la climatología está fechada en el año 0001; decodificar
    tiempos como pandas datetime no es válido. Seleccionamos por índice de mes.
    """
    if not 1 <= month <= 12:
        raise ValueError("El mes debe estar entre 1 y 12.")
    validate_netcdf(raw)
    with xr.open_dataset(io.BytesIO(raw), decode_times=False) as ds:
        if "sst" not in ds or "time" not in ds["sst"].dims:
            raise ValueError("La climatología no contiene SST mensual.")
        if ds["sst"].sizes["time"] != 12:
            raise ValueError("Se requieren exactamente 12 meses climatológicos.")
        frame = ds["sst"].isel(time=month - 1).to_dataframe(name="climatologia_c").reset_index()
    frame = frame[["lat", "lon", "climatologia_c"]].copy()
    frame["lon"] = ((frame["lon"].astype(float) + 180) % 360) - 180
    frame = frame.dropna(subset=["climatologia_c"])
    frame = frame[
        np.isfinite(frame["climatologia_c"])
        & frame["climatologia_c"].between(-2.0, 45.0)
    ]
    return frame.reset_index(drop=True)


def fetch_climatology(month: int) -> pd.DataFrame:
    response = requests.get(
        CLIMATOLOGY_URL, headers={"User-Agent": AGENT},
        timeout=(15, 85),
    )
    response.raise_for_status()
    return decode_climatology(response.content, month)


def params_oisst(day: date, region: str) -> tuple[str, dict]:
    """Limita la consulta a un solo día y a la región seleccionada."""
    if not date(1982, 1, 1) <= day <= date(2025, 12, 31):
        raise ValueError("La capa diaria abarca 1982–2025.")
    if region not in REGIONS:
        raise ValueError("Región desconocida.")
    west, south, east, north = REGIONS[region]
    return (
        f"{OISST_NCSS_ROOT}/sst.day.mean.{day.year}.nc",
        {
            "var": "sst",
            "west": west % 360,
            "east": east % 360,
            "south": south,
            "north": north,
            "time": f"{day:%Y-%m-%d}T12:00:00Z",
            "accept": "netcdf",
            "horizStride": 1,
        },
    )


def fetch_oisst(day: date, region: str) -> pd.DataFrame:
    """Una descarga espacial de OISST de NOAA/PSL, sin inventar temperaturas."""
    url, params = params_oisst(day, region)
    response = requests.get(
        url, params=params, headers={"User-Agent": AGENT},
        timeout=(15, 100),
    )
    response.raise_for_status()
    frame = clean_grid(response.content, expected_date=day.isoformat())
    return crop_to_region(frame, region)


def crop_to_region(df: pd.DataFrame, region: str) -> pd.DataFrame:
    if region not in REGIONS:
        raise ValueError("Región desconocida.")
    west, south, east, north = REGIONS[region]
    return df[df["lon"].between(west, east) & df["lat"].between(south, north)].copy()


def fetch_ersst(month: str, region: str) -> pd.DataFrame:
    """Recupera cuadrícula ERSST mensual usando el motor ya verificado de fase 1."""
    frame = fetch_gridded_sst(month)
    return crop_to_region(frame, region).drop(columns=["mes"], errors="ignore")


def add_anomalies(observed: pd.DataFrame, climate: pd.DataFrame) -> pd.DataFrame:
    """Alinea la climatología y SST en la misma cuadrícula (sin interpolación)."""
    for name, frame in (("observada", observed), ("climatología", climate)):
        if not {"lat", "lon"} <= set(frame.columns):
            raise ValueError(f"Faltan coordenadas en {name}.")
    left = observed.copy()
    right = climate.copy()
    for frame in (left, right):
        frame["lat_key"] = frame["lat"].round(4)
        frame["lon_key"] = frame["lon"].round(4)
    joined = left.merge(
        right[["lat_key", "lon_key", "climatologia_c"]],
        on=["lat_key", "lon_key"], how="inner", validate="one_to_one",
    )
    if joined.empty:
        raise ValueError("No hay posiciones comunes entre SST y climatología.")
    joined["anomalia_c"] = joined["sst_c"] - joined["climatologia_c"]
    return joined.drop(columns=["lat_key", "lon_key"])


def compare_months(current: pd.DataFrame, previous: pd.DataFrame) -> pd.DataFrame:
    """Cambio de SST en la misma coordenada entre meses (no es anomalía)."""
    l, r = current.copy(), previous.copy()
    for frame in (l, r):
        frame["lat_key"] = frame["lat"].round(4)
        frame["lon_key"] = frame["lon"].round(4)
    result = l.merge(
        r[["lat_key", "lon_key", "sst_c"]].rename(columns={"sst_c": "sst_comparacion_c"}),
        on=["lat_key", "lon_key"], validate="one_to_one",
    )
    if result.empty:
        raise ValueError("No hay celdas comparables.")
    result["cambio_c"] = result["sst_c"] - result["sst_comparacion_c"]
    return result.drop(columns=["lat_key", "lon_key"])


def interpolate_rgb(value: float, stops: list[tuple[float, list[int]]]) -> list[int]:
    if not np.isfinite(value):
        raise ValueError("El valor del mapa debe ser un número finito.")
    v = float(np.clip(value, stops[0][0], stops[-1][0]))
    for (a, ca), (b, cb) in zip(stops, stops[1:]):
        if v <= b:
            ratio = (v - a) / (b - a)
            return [int(round(x + ratio * (y - x))) for x, y in zip(ca, cb)]
    return stops[-1][1]


TEMP_STOPS = [
    (14, [24, 67, 149]), (18, [30, 143, 207]), (22, [43, 204, 186]),
    (26, [248, 208, 89]), (29, [239, 109, 49]), (32, [169, 35, 55]),
]
DIFF_STOPS = [
    (-3, [34, 72, 159]), (-1.5, [55, 158, 214]), (0, [233, 241, 241]),
    (1.5, [246, 173, 67]), (3, [177, 44, 53]),
]
DISPLAY_MODES = {
    "Temperatura superficial (°C)": ("sst_c", TEMP_STOPS, "Temperatura superficial"),
    "Anomalía frente a 1991–2020 (°C)": ("anomalia_c", DIFF_STOPS, "Anomalía"),
    "Cambio respecto a otro mes (°C)": ("cambio_c", DIFF_STOPS, "Cambio entre meses"),
}


def ocean_tiles(
    frame: pd.DataFrame,
    mode: str,
    resolution: float,
    extrude: bool,
    exaggeration: float,
) -> list[dict]:
    """Prepara las teselas para deck.gl; la altura nunca significa profundidad."""
    if mode not in DISPLAY_MODES:
        raise ValueError("Capa desconocida.")
    value_col, colors, _ = DISPLAY_MODES[mode]
    if value_col not in frame:
        raise ValueError(f"La columna {value_col} no está disponible.")
    if resolution not in (0.25, 2.0):
        raise ValueError("Resolución espacial no soportada.")
    if len(frame) > 15000:
        raise ValueError("Demasiadas celdas para una visualización segura.")
    half = resolution / 2
    out = []
    for row in frame.itertuples(index=False):
        lat, lon = float(row.lat), float(row.lon)
        value = float(getattr(row, value_col))
        if not np.isfinite(value):
            continue
        # Extrusión visual. La altura es arbitraria y nunca km de profundidad.
        relative = max(0, value - 13) if value_col == "sst_c" else max(0, value)
        h = float(250 * relative * exaggeration) if extrude else 0.0
        out.append({
            "polygon": [
                [lon - half, lat - half], [lon + half, lat - half],
                [lon + half, lat + half], [lon - half, lat + half],
            ],
            "lat": round(lat, 3),
            "lon": round(lon, 3),
            "sst_c": round(float(row.sst_c), 2),
            "valor": round(value, 2),
            "unidad": "°C",
            "modo": DISPLAY_MODES[mode][2],
            "color": interpolate_rgb(value, colors) + [230],
            "altura": h,
        })
    return out


def weighted_mean(frame: pd.DataFrame, col: str) -> float:
    """Media de puntos ponderada por coseno de latitud; no promedio costero."""
    vals = frame[[col, "lat"]].dropna()
    if vals.empty:
        raise ValueError("No hay valores numéricos.")
    return float(np.average(vals[col], weights=np.cos(np.deg2rad(vals["lat"]))))


def area_center(region: str) -> tuple[float, float, float]:
    west, south, east, north = REGIONS[region]
    width = max(east - west, north - south)
    zoom = 6.7 if width < 9 else 5.9 if width < 13 else 4.75
    return ((west + east) / 2, (south + north) / 2, zoom)
