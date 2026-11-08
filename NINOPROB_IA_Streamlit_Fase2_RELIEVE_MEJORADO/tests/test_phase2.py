"""Pruebas locales de la fase 2. Los datos aquí son sintéticos SOLO para software."""
from datetime import date

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from ninoprob_ia.phase2 import (
    add_anomalies, area_center, clean_grid, compare_months, crop_to_region,
    decode_climatology, interpolate_rgb, ocean_tiles, params_oisst, weighted_mean,
    DISPLAY_MODES, DIFF_STOPS, TEMP_STOPS,
)


def _nc_month(when="2025-11-15"):
    ds = xr.Dataset(
        {"sst": (("time", "lat", "lon"), np.array([
            [[25., 26.], [27., np.nan]]
        ], dtype="float32"))},
        coords={
            "time": np.asarray([when], dtype="datetime64[ns]"),
            "lat": np.array([-8., -6.]),
            "lon": np.array([278., 280.]),
        },
    )
    return ds.to_netcdf(engine="scipy")


def test_real_month_or_day_must_be_correct():
    frame = clean_grid(bytes(_nc_month()), expected_date="2025-11")
    assert len(frame) == 3
    assert frame["lon"].min() == -82.0
    with pytest.raises(ValueError, match="mes diferente"):
        clean_grid(bytes(_nc_month()), expected_date="2025-12")
    with pytest.raises(ValueError, match="día diferente"):
        clean_grid(bytes(_nc_month()), expected_date="2025-11-16")
    with pytest.raises(ValueError, match="NetCDF"):
        clean_grid(b"<html>False</html>")


def test_climatology_aligns_correct_month():
    vals = np.zeros((12, 2, 2), dtype="float32")
    for i in range(12):
        vals[i, :, :] = 10 + i
    ds = xr.Dataset(
        {"sst": (("time", "lat", "lon"), vals)},
        coords={
            "time": np.arange(12, dtype="float64") * 30,
            "lat": [-8., -6.],
            "lon": [278., 280.],
        },
    )
    ref = decode_climatology(bytes(ds.to_netcdf(engine="scipy")), 11)
    assert set(ref["climatologia_c"]) == {20.0}
    frame = clean_grid(bytes(_nc_month()))
    matched = add_anomalies(frame, ref)
    assert set(np.round(matched["anomalia_c"], 1)) == {5., 6., 7.}
    with pytest.raises(ValueError):
        decode_climatology(bytes(ds.to_netcdf(engine="scipy")), 13)


def test_compare_only_same_position_not_unmatched():
    a = pd.DataFrame({"lat": [-10., -12.], "lon": [-80., -81.],
                      "sst_c": [25., 27.]})
    b = pd.DataFrame({"lat": [-10., -15.], "lon": [-80., -81.],
                      "sst_c": [24., 11.]})
    delta = compare_months(a, b)
    assert len(delta) == 1
    assert delta.iloc[0]["cambio_c"] == 1.


def test_high_resolution_params_use_noaa_specific_day_and_region():
    url, params = params_oisst(date(2025, 5, 2), "Costa norte")
    assert url.endswith("sst.day.mean.2025.nc")
    assert params["var"] == "sst"
    assert params["time"] == "2025-05-02T12:00:00Z"
    assert params["west"] == 275.0
    with pytest.raises(ValueError):
        params_oisst(date(2026, 1, 1), "Costa norte")
    with pytest.raises(ValueError):
        params_oisst(date(2025, 5, 2), "Mar ficticio")


def test_noaa_tiles_have_correct_cell_size_color_and_extrusion():
    frame = pd.DataFrame({
        "lat": [-10.125], "lon": [-80.125], "sst_c": [29.],
        "anomalia_c": [1.5], "cambio_c": [-1.]
    })
    cells = ocean_tiles(frame, "Temperatura superficial (°C)", .25, True, 2.)
    assert cells[0]["polygon"] == [
        [-80.25, -10.25], [-80., -10.25],
        [-80., -10.], [-80.25, -10.],
    ]
    assert cells[0]["altura"] > 0
    cold = ocean_tiles(frame, "Cambio respecto a otro mes (°C)", .25, False, 1.)
    assert cold[0]["altura"] == 0
    assert cold[0]["valor"] == -1.
    assert len(cold[0]["color"]) == 4
    with pytest.raises(ValueError):
        ocean_tiles(frame, "Inexistente", .25, True, 1.)


def test_filter_area_and_mean_uses_valid_ocean_cells():
    d = pd.DataFrame({"lat": [0., -20., -10.],
                      "lon": [-83., -73., -100.],
                      "sst_c": [27., 18., 3.]})
    filtered = crop_to_region(d, "Todo el mar peruano")
    assert len(filtered) == 2
    mean = weighted_mean(filtered, "sst_c")
    assert 18 <= mean <= 27
    lon, lat, zoom = area_center("Costa norte")
    assert -85 <= lon <= -77
    assert -8 <= lat <= 1
    assert zoom > 5


def test_palette_is_fixed_across_months():
    assert interpolate_rgb(19., TEMP_STOPS) == interpolate_rgb(19., TEMP_STOPS)
    assert interpolate_rgb(-3., DIFF_STOPS) == [34, 72, 159]
    assert interpolate_rgb(3., DIFF_STOPS) == [177, 44, 53]
