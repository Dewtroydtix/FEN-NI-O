"""Descarga, limpieza, entrenamiento y evaluación del índice Niño 1+2.

NOAA CPC ERSSTv5, anomalías mensuales 1991-2020, 1950-2025.
No se generan datos artificiales para el entrenamiento científico.
"""
from __future__ import annotations

import hashlib
import io
import json
import math
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

NOAA_URL = "https://www.cpc.ncep.noaa.gov/data/indices/ersst5.nino.mth.91-20.ascii"
SOURCE_TITLE = "NOAA/CPC ERSSTv5 Niño 1+2, climatología 1991–2020"
START = pd.Period("1950-01", "M")
END = pd.Period("2025-12", "M")
THRESHOLD = 0.5
FEATURES = ("anomalia_t", "anomalia_t_1", "anomalia_t_2", "mes_seno", "mes_coseno")
C_VALUES = (0.01, 0.1, 1.0, 10.0, 100.0)
BBOX_NINO12 = [-90.0, -10.0, -80.0, 0.0]  # oeste, sur, este, norte


def download_noaa(destination: str | Path, url: str = NOAA_URL) -> Path:
    """Descarga exclusivamente del origen NOAA previsto por el proyecto."""
    if url != NOAA_URL:
        raise ValueError("Solo se admite la URL oficial NOAA/CPC prevista.")
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        response = requests.get(
            url, headers={"User-Agent": "NINOPROB-IA/1.0 (investigacion-escolar)"},
            timeout=(15, 60),
        )
        response.raise_for_status()
        raw = response.content
        if len(raw) < 10000 or b"<html" in raw.lower()[:2000]:
            raise ValueError("La respuesta de NOAA no parece un archivo de datos válido.")
        parse_noaa(raw.decode("utf-8-sig", errors="replace"))
    except (requests.RequestException, ValueError) as exc:
        raise RuntimeError(
            "No se pudo descargar/verificar la serie NOAA. Comprueba Internet y "
            f"descarga manualmente {NOAA_URL} para usar --archivo RUTA."
        ) from exc
    destination.write_bytes(raw)
    return destination


def parse_noaa(contents: str) -> pd.DataFrame:
    """Procesa columnas: YR MON NINO1+2 ANOM NINO3 ANOM NINO4 ANOM NINO3.4 ANOM.

    Solo se usa la columna 4 (índice 3): anomalía Niño 1+2 en °C.
    Se ignoran encabezados, se rechazan registros duplicados y sentinelas.
    """
    rows = []
    for line in io.StringIO(contents):
        fields = line.split()
        if len(fields) < 10:
            continue
        try:
            year = int(fields[0])
            month = int(fields[1])
        except ValueError:
            continue
        if not (1900 <= year <= 2100 and 1 <= month <= 12):
            continue
        try:
            temperature = float(fields[2])
            anomaly = float(fields[3])
        except ValueError as exc:
            raise ValueError(f"Fila NOAA inválida: {line.strip()}") from exc
        if not (np.isfinite(temperature) and np.isfinite(anomaly)):
            continue
        if abs(anomaly) > 20 or temperature <= -90 or temperature > 50:
            # NOAA utiliza sentinelas para valores faltantes; nunca interpolarlos.
            continue
        rows.append((pd.Period(year=year, month=month, freq="M"), anomaly, temperature))
    if not rows:
        raise ValueError("No se encontraron registros mensuales NOAA Niño 1+2.")
    df = pd.DataFrame(rows, columns=["periodo", "anomalia_c", "temperatura_c"])
    if df["periodo"].duplicated().any():
        raise ValueError("El archivo NOAA contiene meses duplicados.")
    df = df.sort_values("periodo").reset_index(drop=True)
    return df


def prepare_monthly(raw: pd.DataFrame) -> pd.DataFrame:
    """Cubre 1950–2025 sin inventar valores si faltan meses."""
    df = raw[(raw["periodo"] >= START) & (raw["periodo"] <= END)].copy()
    expected = pd.period_range(START, END, freq="M")
    if df.empty or START not in set(df["periodo"]) or END not in set(df["periodo"]):
        raise ValueError("La fuente no tiene cobertura completa hasta diciembre de 2025.")
    if df["periodo"].duplicated().any():
        raise ValueError("Existen meses duplicados.")
    df = df.set_index("periodo").reindex(expected)
    df.index.name = "periodo"
    if df["anomalia_c"].isna().sum() > 24:
        raise ValueError("Demasiados meses faltantes (>24); revisar el archivo.")
    return df.reset_index()


def features_and_targets(monthly: pd.DataFrame) -> pd.DataFrame:
    """Objetivo y(t+1) calculado exclusivamente a partir del mes siguiente real.

    Los regresores son x(t), x(t-1), x(t-2), y seno/coseno del mes objetivo.
    Los periodos de entrenamiento/validación/prueba se asignan por mes objetivo.
    """
    df = monthly[["periodo", "anomalia_c"]].copy()
    df["anomalia_t"] = df["anomalia_c"]
    df["anomalia_t_1"] = df["anomalia_c"].shift(1)
    df["anomalia_t_2"] = df["anomalia_c"].shift(2)
    df["anomalia_mes_objetivo_c"] = df["anomalia_c"].shift(-1)
    df["mes_objetivo_periodo"] = df["periodo"] + 1
    target_month = df["mes_objetivo_periodo"].apply(lambda period: period.month)
    df["mes_seno"] = np.sin(2 * np.pi * (target_month - 1) / 12)
    df["mes_coseno"] = np.cos(2 * np.pi * (target_month - 1) / 12)
    mask = df[list(FEATURES) + ["anomalia_mes_objetivo_c"]].notna().all(axis=1)
    df = df.loc[mask].copy()
    df["evento"] = (df["anomalia_mes_objetivo_c"] >= THRESHOLD).astype(int)
    df["mes_origen"] = df["periodo"].astype(str)
    df["mes_objetivo"] = df["mes_objetivo_periodo"].astype(str)
    return df.reset_index(drop=True).drop(columns=["periodo", "mes_objetivo_periodo"])


def partition(dataset: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """La separación se define por mes de la ocurrencia real, no por azar."""
    year = dataset["mes_objetivo"].str[:4].astype(int)
    train = dataset.loc[year <= 2015].copy()
    validation = dataset.loc[(year >= 2016) & (year <= 2020)].copy()
    test = dataset.loc[(year >= 2021) & (year <= 2025)].copy()
    if min(len(train), len(validation), len(test)) < 24:
        raise ValueError("Insuficientes observaciones cronológicas para evaluar.")
    if len(np.unique(train["evento"])) < 2:
        raise ValueError("No hay ambas clases (evento y no evento) en entrenamiento.")
    return train, validation, test


def make_model(c: float) -> Pipeline:
    return Pipeline([
        ("scale", StandardScaler()),
        ("logistic", LogisticRegression(C=c, max_iter=4000, solver="lbfgs")),
    ])


def evaluation(y: np.ndarray, probs: np.ndarray) -> dict:
    y = np.asarray(y, dtype=int)
    probs = np.asarray(probs, dtype=float)
    if not np.isfinite(probs).all() or np.any((probs < 0) | (probs > 1)):
        raise ValueError("Probabilidades fuera de rango.")
    return {
        "brier": round(float(brier_score_loss(y, probs)), 6),
        "exactitud_umbral_50": round(float(accuracy_score(y, probs >= 0.5)), 6),
        "meses": int(y.size),
        "eventos": int(y.sum()),
    }


def make_calibration(y: np.ndarray, probs: np.ndarray, bins: int = 5) -> list[dict]:
    intervals = np.linspace(0, 1, bins + 1)
    result = []
    for i in range(bins):
        mask = (probs >= intervals[i]) & (
            (probs <= intervals[i + 1]) if i == bins - 1 else (probs < intervals[i + 1])
        )
        if np.any(mask):
            result.append({
                "desde": round(float(intervals[i]), 3),
                "hasta": round(float(intervals[i + 1]), 3),
                "numero_meses": int(mask.sum()),
                "probabilidad_media": round(float(probs[mask].mean()), 6),
                "frecuencia_observada": round(float(y[mask].mean()), 6),
            })
    return result


def save_graphs(test: pd.DataFrame, metrics: dict, results_dir: Path) -> None:
    # 1. Evolución mensual observada, con umbral didáctico.
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(pd.to_datetime(test["mes_objetivo"]), test["anomalia_mes_objetivo_c"],
            linewidth=1.5, label="Anomalía observada NOAA")
    ax.axhline(THRESHOLD, color="tomato", linestyle="--", label="Umbral educativo +0,5 °C")
    ax.set(ylabel="Anomalía (°C)", xlabel="Mes", title="Región Niño 1+2 | prueba 2021–2025")
    ax.legend()
    fig.tight_layout()
    fig.savefig(results_dir / "anomalías_prueba.png", dpi=160)
    plt.close(fig)

    # 2. Probabilidades IA y referencia frente al evento real.
    fig, ax = plt.subplots(figsize=(12, 4))
    x = pd.to_datetime(test["mes_objetivo"])
    ax.plot(x, test["probabilidad_ia"], label="IA (regresión logística)")
    ax.plot(x, test["probabilidad_historica"], linestyle=":", label="Frecuencia entrenamiento")
    ax.scatter(x, test["evento"], s=12, alpha=0.5, label="Evento observado (0/1)")
    ax.set(xlabel="Mes objetivo", ylabel="Probabilidad", ylim=(-0.08, 1.08),
           title="Probabilidad del evento cálido — prueba independiente")
    ax.legend()
    fig.tight_layout()
    fig.savefig(results_dir / "probabilidades_prueba.png", dpi=160)
    plt.close(fig)

    # 3. Fiabilidad por intervalos.
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], "--", color="gray", label="Calibración ideal")
    for key, title in [("ia", "IA"), ("historica", "Frecuencia histórica")]:
        bins = metrics["calibracion"][key]
        if bins:
            ax.plot([b["probabilidad_media"] for b in bins],
                    [b["frecuencia_observada"] for b in bins], "o-", label=title)
    ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="Probabilidad media estimada",
           ylabel="Frecuencia observada", title="Fiabilidad — prueba 2021–2025")
    ax.legend()
    fig.tight_layout()
    fig.savefig(results_dir / "fiabilidad_prueba.png", dpi=160)
    plt.close(fig)


def run_pipeline(
    base_dir: str | Path,
    input_file: str | Path | None = None,
    *,
    allow_download: bool = True,
) -> dict:
    """Ejecuta el proyecto completo y guarda resultados auditables.

    Si input_file es None, descarga de NOAA al directorio datos/.
    No crea resultados de entrenamiento cuando falla la descarga/verificación.
    """
    base_dir = Path(base_dir)
    data_dir = base_dir / "datos"
    results_dir = base_dir / "resultados"
    data_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    source_path = Path(input_file) if input_file is not None else data_dir / "ersst5.nino.mth.91-20.ascii"
    if not source_path.is_file():
        if not allow_download or input_file is not None:
            raise FileNotFoundError(f"No se encontró {source_path}. Descarga el archivo NOAA.")
        download_noaa(source_path)
    raw_bytes = source_path.read_bytes()
    raw = parse_noaa(raw_bytes.decode("utf-8-sig", errors="replace"))
    monthly = prepare_monthly(raw)
    data = features_and_targets(monthly)
    train, valid, test = partition(data)

    x_train = train[list(FEATURES)].to_numpy(dtype=float)
    y_train = train["evento"].to_numpy(dtype=int)
    x_valid = valid[list(FEATURES)].to_numpy(dtype=float)
    y_valid = valid["evento"].to_numpy(dtype=int)
    x_test = test[list(FEATURES)].to_numpy(dtype=float)
    y_test = test["evento"].to_numpy(dtype=int)
    base_rate = float(y_train.mean())

    # Parámetros escogidos SOLO con validación 2016–2020.
    candidates = []
    fitted_models = {}
    for c in C_VALUES:
        m = make_model(c).fit(x_train, y_train)
        brier_validation = float(brier_score_loss(y_valid, m.predict_proba(x_valid)[:, 1]))
        candidates.append({"C": c, "brier_validacion": round(brier_validation, 6)})
        fitted_models[c] = m
    best = min(candidates, key=lambda row: (row["brier_validacion"], row["C"]))
    selected_c = float(best["C"])
    # No re-entrenar con la prueba 2021-2025. Se conserva 1950-2015.
    model = fitted_models[selected_c]

    p_ia = model.predict_proba(x_test)[:, 1]
    p_hist = np.full(shape=len(test), fill_value=base_rate, dtype=float)
    test = test.copy()
    test["probabilidad_ia"] = p_ia
    test["probabilidad_historica"] = p_hist
    test["acierto_ia"] = ((p_ia >= 0.5) == y_test).astype(int)

    metrics = {
        "fuente": SOURCE_TITLE,
        "archivo_fuente": source_path.name,
        "sha256_fuente": hashlib.sha256(raw_bytes).hexdigest(),
        "periodo": {"inicio": str(START), "fin": str(END)},
        "definicion_evento": "anomalía Niño 1+2 del mes siguiente >= +0.5 °C",
        "climatologia": "1991-2020",
        "particion_por_mes_objetivo": {
            "entrenamiento": "1950-2015", "validacion": "2016-2020", "prueba": "2021-2025"
        },
        "cantidad_meses": {
            "origen": int(monthly["anomalia_c"].notna().sum()),
            "entrenamiento": len(train), "validacion": len(valid), "prueba": len(test)
        },
        "candidatos_C": candidates,
        "mejor_C": selected_c,
        "frecuencia_historica_entrenamiento": round(base_rate, 6),
        "prueba": {
            "ia": evaluation(y_test, p_ia),
            "frecuencia_historica": evaluation(y_test, p_hist),
        },
        "calibracion": {
            "ia": make_calibration(y_test, p_ia),
            "historica": make_calibration(y_test, p_hist),
        },
        "conclusion": (
            "En la prueba independiente, la IA obtuvo menor error Brier que la frecuencia histórica."
            if brier_score_loss(y_test, p_ia) < brier_score_loss(y_test, p_hist)
            else "La IA no mejoró la puntuación Brier de la frecuencia histórica en la prueba."
        ),
        "uso": "Investigación educativa; NO es pronóstico ni alerta oficial ENFEN.",
        "apto_mapa_espacial": False,
        "nota_mapa": "Índice agregado regional; no contiene píxeles ni temperatura de cada ubicación."
    }
    artifact = {
        "model": model,
        "features": list(FEATURES),
        "threshold": THRESHOLD,
        "baseline": base_rate,
        "training_end": "2015-12",
        "test_end": "2025-12",
        "source_sha256": metrics["sha256_fuente"],
    }
    joblib.dump(artifact, results_dir / "modelo_ninoprob_ia.joblib")
    monthly_export = monthly.assign(mes=monthly["periodo"].astype(str)).drop(columns="periodo")
    monthly_export[["mes", "anomalia_c", "temperatura_c"]].to_csv(
        results_dir / "serie_mensual_noaa.csv", index=False, float_format="%.6f")
    data.to_csv(results_dir / "variables_y_objetivos.csv", index=False, float_format="%.6f")
    test.to_csv(results_dir / "predicciones_prueba_2021_2025.csv", index=False, float_format="%.6f")
    (results_dir / "metricas.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    save_graphs(test, metrics, results_dir)
    return metrics
