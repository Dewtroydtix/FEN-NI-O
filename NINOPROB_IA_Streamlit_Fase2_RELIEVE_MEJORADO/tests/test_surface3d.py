"""Superficie científica: interpolación visual sin fabricar datos medidos."""
import numpy as np
import pandas as pd
import pytest

from ninoprob_ia.surface3d import grafico_superficie, preparar_superficie


def mapa_prueba():
    return pd.DataFrame([
        {"lat": lat, "lon": lon, "sst_c": 23 + (lat + 5) * .3 + (lon + 80) * .2}
        for lat in [-6., -5., -4., -3.]
        for lon in [-81., -80., -79., -78.]
        if not (lat == -3. and lon == -78.)
    ])


def test_superficie_es_mas_suave_pero_conserva_rango():
    frame = mapa_prueba()
    result = preparar_superficie(frame, "Temperatura superficial (°C)")
    assert len(result["lat"]) > 4
    assert len(result["lon"]) > 4
    assert np.nanmin(result["valores"]) >= frame["sst_c"].min() - 1.e-6
    assert np.nanmax(result["valores"]) <= frame["sst_c"].max() + 1.e-6
    assert np.isnan(result["valores"][-1,-1])


def test_sin_relieve_representa_altura_cero():
    result = preparar_superficie(mapa_prueba(), "Temperatura superficial (°C)", 0)
    assert np.nanmax(np.abs(result["altura"])) == 0


def test_figura_plotly_3d_y_errores():
    fig = grafico_superficie(mapa_prueba(), "Temperatura superficial (°C)", 1.5, 55)
    assert fig.data[0].type == "surface"
    assert fig.data[0].connectgaps is False
    with pytest.raises(ValueError, match="Capa desconocida"):
        preparar_superficie(mapa_prueba(), "Ficticio")
    with pytest.raises(ValueError, match="varias celdas"):
        preparar_superficie(mapa_prueba().head(1), "Temperatura superficial (°C)")
