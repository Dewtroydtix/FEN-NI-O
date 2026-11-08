"""Pruebas con datos SINTÉTICOS; nunca representan mediciones NOAA."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from ninoprob_ia.core import (
    FEATURES, NOAA_URL, features_and_targets, parse_noaa,
    partition, prepare_monthly, run_pipeline,
)


def fixture_text():
    """Solo para pruebas de software; datos inventados, NO NOAA."""
    rng = np.random.default_rng(42)
    lines = ["YR MON NINO1+2 ANOM NINO3 ANOM NINO4 ANOM NINO3.4 ANOM"]
    for i, month in enumerate(pd.period_range("1950-01", "2025-12", freq="M")):
        value = 0.65 * np.sin(i / 8.0) + 0.48 * rng.normal()
        lines.append(
            f"{month.year:04d} {month.month:02d} {25+value:.4f} {value:.4f} "
            f"25.0000 0.0000 25.0000 0.0000 25.0000 0.0000"
        )
    return "\n".join(lines) + "\n"


def test_parser_months_and_column_selection():
    raw = parse_noaa(fixture_text())
    assert len(raw) == 912
    assert str(raw.iloc[0]["periodo"]) == "1950-01"
    assert str(raw.iloc[-1]["periodo"]) == "2025-12"
    assert raw.iloc[0]["anomalia_c"] != 0.0  # lee anomalía Niño 1+2, no Niño3


def test_target_next_month_and_chronology():
    monthly = prepare_monthly(parse_noaa(fixture_text()))
    dataset = features_and_targets(monthly)
    train, valid, test = partition(dataset)
    assert dataset.iloc[0]["mes_origen"] == "1950-03"
    assert dataset.iloc[0]["mes_objetivo"] == "1950-04"
    assert dataset.iloc[-1]["mes_origen"] == "2025-11"
    assert dataset.iloc[-1]["mes_objetivo"] == "2025-12"
    assert train["mes_objetivo"].max() == "2015-12"
    assert valid["mes_objetivo"].min() == "2016-01"
    assert valid["mes_objetivo"].max() == "2020-12"
    assert test["mes_objetivo"].min() == "2021-01"
    assert test["mes_objetivo"].max() == "2025-12"
    assert set(FEATURES).issubset(dataset.columns)
    # La anomalía de 2025-12 NO se usa para predecirse a sí misma.
    dec25 = test.loc[test["mes_objetivo"] == "2025-12"].iloc[0]
    assert dec25["mes_origen"] == "2025-11"


def test_duplicates_rejected():
    row = "1950 01 24.00 0.15 24.00 0.15 24.00 0.15 24.00 0.15\n"
    with pytest.raises(ValueError, match="duplicados"):
        parse_noaa(row + row)


def test_full_pipeline_and_api_without_pretending_noaa(tmp_path, monkeypatch):
    input_file = tmp_path / "fixture_sintetico_SOLO_TEST.ascii"
    input_file.write_text(fixture_text(), encoding="utf-8")
    metrics = run_pipeline(tmp_path, input_file=input_file, allow_download=False)
    assert metrics["cantidad_meses"]["prueba"] == 60
    assert metrics["cantidad_meses"]["validacion"] == 60
    assert len(metrics["candidatos_C"]) == 5
    assert "brier" in metrics["prueba"]["ia"]
    assert len(list((tmp_path / "resultados").glob("*.png"))) == 3
    # Se prueban rutas y consistencia sin atribuir ningún resultado a NOAA.
    from ninoprob_ia import api
    monkeypatch.setattr(api, "RESULTS", tmp_path / "resultados")
    api.load.cache_clear()
    client = TestClient(api.app)
    assert client.get("/health").status_code == 200
    assert client.get("/api/v1/metadata").json()["mapa_termico_detallado_disponible"] is False
    result = client.get("/api/v1/prediccion/2025-11")
    assert result.status_code == 200
    assert result.json()["mes_objetivo"] == "2025-12"
    assert result.json()["es_evaluacion_independiente"] is True
    assert result.json()["evento_objetivo_observado"] in (0, 1)
    future = client.get("/api/v1/prediccion/2025-12")
    assert future.status_code == 200
    assert future.json()["mes_objetivo"] == "2026-01"
    assert future.json()["evento_objetivo_observado"] is None
    assert client.get("/api/v1/prediccion/2025-13").status_code == 422
    geo = client.get("/api/v1/region.geojson?mes=2025-03")
    assert geo.status_code == 200
    assert geo.json()["features"][0]["properties"]["tipo_dato"] == "promedio_regional"
    assert client.get("/api/v1/serie?desde=2025-01&hasta=2025-12").json()["registros"]
    api.load.cache_clear()
