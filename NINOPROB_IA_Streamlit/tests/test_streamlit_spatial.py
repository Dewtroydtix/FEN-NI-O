"""Pruebas deterministas del visualizador. No se inventan resultados científicos."""
import io

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from ninoprob_ia.spatial import (
    decode_netcdf, ncss_params, spatial_polygons, thermal_color,
)
from ninoprob_ia.core import FEATURES


def test_ncss_params_fixed_source_and_area():
    params = ncss_params("2025-11")
    assert params["var"] == "sst"
    assert params["west"] == 270.0
    assert params["east"] == 294.0
    assert params["time"] == "2025-11-15T00:00:00Z"
    with pytest.raises(ValueError):
        ncss_params("2026-01")


def test_color_gradient_is_bounded():
    assert len(thermal_color(27.2)) == 4
    assert thermal_color(27.0) == thermal_color(27.0)
    assert thermal_color(-10.0) == thermal_color(14.0)


def test_spatial_polygons_are_geographically_centered():
    grid = pd.DataFrame({
        "lat": [-12., -10.],
        "lon": [-80., -82.],
        "sst_c": [24., 28.],
    })
    polygons = spatial_polygons(grid)
    assert len(polygons) == 2
    assert polygons[0]["polygon"] == [
        [-81., -13.], [-79., -13.], [-79., -11.], [-81., -11.]
    ]
    assert polygons[0]["sst_c"] == 24.
    assert polygons[0]["altura"] > 0


def test_netcdf_real_shape_is_decoded_without_fake_values():
    ds = xr.Dataset(
        {
            "sst": (("time", "lat", "lon"),
                    np.array([[[21.5, 24.0], [22.0, np.nan]]], dtype=np.float32))
        },
        coords={
            "time": np.array(["2025-11-15"], dtype="datetime64[ns]"),
            "lat": np.array([-12.0, -10.0]),
            "lon": np.array([278.0, 280.0]),
        },
    )
    raw = bytes(ds.to_netcdf(engine="scipy"))
    assert raw.startswith(b"CDF")
    decoded = decode_netcdf(raw, "2025-11")
    assert len(decoded) == 3
    assert decoded["sst_c"].max() == 24.0
    assert decoded["lon"].min() == -82.0
    assert set(decoded["mes"]) == {"2025-11"}
    with pytest.raises(ValueError):
        decode_netcdf(b"<html>Error</html>", "2025-11")


def test_training_feature_contract_stays_compatible():
    assert list(FEATURES) == [
        "anomalia_t", "anomalia_t_1", "anomalia_t_2", "mes_seno", "mes_coseno"
    ]
