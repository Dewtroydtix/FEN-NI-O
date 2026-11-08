"""NIÑOPROB-IA | aplicación web Streamlit con visualizador oceánico 3D.

Ejecución: streamlit run app.py
Datos: NOAA/CPC (región Niño 1+2) y NOAA/PSL ERSSTv5 (cuadrícula SST).
Este producto es EDUCATIVO; no sustituye a ENFEN.
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pydeck as pdk
import requests
import streamlit as st

from ninoprob_ia.core import BBOX_NINO12, FEATURES, NOAA_URL, THRESHOLD, run_pipeline
from ninoprob_ia.spatial import (
    NCSS_URL, fetch_gridded_sst, spatial_polygons, thermal_color,
)

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "resultados"
DATA_DIR = ROOT / "datos"
MONTH_NAMES = (
    "Enero", "Febrero", "Marzo", "Abril", "Mayo", "Junio",
    "Julio", "Agosto", "Septiembre", "Octubre", "Noviembre", "Diciembre",
)

st.set_page_config(
    page_title="NIÑOPROB-IA | Océano peruano 3D",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
      .stApp {
        background: radial-gradient(circle at 78% 0%, #102d4b 0%, #071625 45%, #06131f 100%);
        color: #e6f3ff;
      }
      [data-testid="stSidebar"] {background:#071d30;}
      [data-testid="stMetric"] {
        background: #0b2940; padding: 1.1rem; border-radius: 14px;
        border: 1px solid #204960;
      }
      [data-testid="stMetricValue"] {color:#67e5e7;}
      h1, h2, h3 {letter-spacing:-.025em;}
      .kicker {color:#64dcdb; font-size:.82rem; font-weight:700; letter-spacing:.15em;}
      .soft {color:#abc8d9;font-size:.93rem}
      .hero-title {font-size:2.3rem;font-weight:800;line-height:1.12;margin:.3rem 0 .5rem}
      .pill {display:inline-block;padding:.34rem .8rem;margin:.25rem .25rem .25rem 0;
             background:#103950;border:1px solid #255e70;border-radius:999px;
             color:#bce8f0;font-size:.82rem}
      .legend {height:12px;border-radius:999px;
        background: linear-gradient(90deg,#185ba0 0%,#1c9fbc 24%,#23c0a0 45%,#f7c145 65%,#f07137 82%,#b12a3c 100%)}
      .small-muted {font-size:.8rem;color:#9ab4c7}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(ttl=86400, show_spinner=False)
def cached_spatial(month: str) -> pd.DataFrame:
    """El servicio NOAA se consulta solo a petición y se almacena por fecha."""
    return fetch_gridded_sst(month)


@st.cache_resource(show_spinner=False)
def load_model(model_path: str, modified_ns: int):
    """Solo abre modelos generados localmente por este proyecto."""
    del modified_ns  # Participa en la clave de caché.
    return joblib.load(model_path)


def local_results():
    """Regresa resultados solo si los archivos de entrenamiento están completos."""
    files = (
        RESULTS / "modelo_ninoprob_ia.joblib",
        RESULTS / "metricas.json",
        RESULTS / "serie_mensual_noaa.csv",
        RESULTS / "predicciones_prueba_2021_2025.csv",
    )
    if not all(file.is_file() for file in files):
        return None
    try:
        metrics = json.loads(files[1].read_text(encoding="utf-8"))
        model = load_model(str(files[0]), files[0].stat().st_mtime_ns)
        if model.get("features") != list(FEATURES):
            raise ValueError("Las variables del modelo no coinciden.")
        if model.get("source_sha256") != metrics.get("sha256_fuente"):
            raise ValueError("El archivo NOAA no corresponde al modelo entrenado.")
        monthly = pd.read_csv(files[2])
        predictions = pd.read_csv(files[3])
    except (OSError, ValueError, KeyError, TypeError) as exc:
        st.error(f"Los resultados locales no pudieron abrirse: {exc}")
        return None
    return model, metrics, monthly, predictions


def get_prediction(month: str, model: dict, monthly: pd.DataFrame) -> dict | None:
    """Mismos predictores y orden que el entrenamiento original."""
    origin = pd.Period(month, freq="M")
    indexed = monthly.set_index("mes")
    dates = [str(origin), str(origin - 1), str(origin - 2)]
    if any(x not in indexed.index or pd.isna(indexed.at[x, "anomalia_c"]) for x in dates):
        return None
    anomalies = [float(indexed.at[x, "anomalia_c"]) for x in dates]
    target = origin + 1
    angle = 2 * np.pi * (target.month - 1) / 12
    features = np.array([[
        anomalies[0], anomalies[1], anomalies[2], np.sin(angle), np.cos(angle),
    ]], dtype=float)
    probability = float(model["model"].predict_proba(features)[0, 1])
    observed = None
    if str(target) in indexed.index and pd.notna(indexed.at[str(target), "anomalia_c"]):
        observed = float(indexed.at[str(target), "anomalia_c"])
    stage = (
        "Entrenamiento" if target.year <= 2015 else
        "Validación" if target.year <= 2020 else
        "Prueba independiente" if target.year <= 2025 else
        "Fuera del periodo evaluado"
    )
    return {
        "probability": probability,
        "target": str(target),
        "observed": observed,
        "baseline": float(model["baseline"]),
        "stage": stage,
        "anomalies": anomalies,
    }


def regional_polygon(anomaly: float | None, spatial_loaded: bool) -> pdk.Layer:
    west, south, east, north = BBOX_NINO12
    polygon = [
        [west, south], [east, south], [east, north], [west, north],
    ]
    rgb = (
        thermal_color(22 + 4 * float(anomaly))[:3]
        if anomaly is not None else [42, 179, 211]
    )
    return pdk.Layer(
        "PolygonLayer",
        id="indice_nino12",
        data=[{
            "polygon": polygon, "name": "Región Niño 1+2",
            "valor": round(anomaly, 2) if anomaly is not None else None,
            "color": rgb + [14 if spatial_loaded else 115],
            "altura": 90 if spatial_loaded else (max(90, 1700 * (anomaly + 2))
                                               if anomaly is not None else 90),
        }],
        get_polygon="polygon",
        get_fill_color="color",
        get_line_color=[94, 237, 241, 250],
        get_line_width=4,
        line_width_min_pixels=2,
        stroked=True,
        filled=True,
        get_elevation="altura",
        extruded=not spatial_loaded,
        pickable=True,
        opacity=0.5 if spatial_loaded else 0.65,
    )


def map_3d(month: str, anomaly: float | None, spatial: pd.DataFrame | None, pitch: int):
    loaded = spatial is not None and not spatial.empty
    layers = []
    if loaded:
        polygons = spatial_polygons(spatial)
        layers.append(
            pdk.Layer(
                "PolygonLayer",
                id="temperatura_ersst",
                data=polygons,
                get_polygon="polygon",
                get_fill_color="color",
                get_line_color=[14, 39, 55, 45],
                line_width_min_pixels=0.2,
                stroked=True,
                extruded=True,
                get_elevation="altura",
                elevation_scale=1,
                pickable=True,
                auto_highlight=True,
                opacity=0.89,
            )
        )
    layers.append(regional_polygon(anomaly, loaded))
    deck = pdk.Deck(
        map_style="https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json",
        initial_view_state=pdk.ViewState(
            latitude=-10.0, longitude=-80.0, zoom=4.85, pitch=pitch, bearing=10,
        ),
        layers=layers,
        tooltip={
            "html": (
                "<div style='background:#0c2536;color:#ecffff;padding:10px;"
                "border-radius:10px'>"
                "<b>{name}</b><b>{sst_c} °C</b><br/>"
                "<span>Lat {lat} · Lon {lon}</span><br/>"
                "<span>Anomalía regional: {valor} °C</span></div>"
            ),
            "style": {"backgroundColor": "#0c2536", "color": "white"},
        },
    )
    st.pydeck_chart(
        deck, height=600, use_container_width=True,
        alt="Visualización inclinable del océano frente al Perú. "
            "Cada columna indica la temperatura superficial mensual NOAA "
            "cuando la cuadrícula está cargada.",
    )
    return loaded


def region_observation(month: str, monthly: pd.DataFrame | None) -> float | None:
    if monthly is None:
        return None
    row = monthly.loc[monthly["mes"] == month, "anomalia_c"]
    if row.empty or pd.isna(row.iloc[0]):
        return None
    return float(row.iloc[0])


with st.sidebar:
    st.markdown("### 🌊 NIÑOPROB-IA")
    st.caption("EXPOMATEMÁTICA · Inteligencia artificial y probabilidad")
    st.divider()
    st.markdown("**Fecha histórica**")
    year = st.selectbox("Año", list(range(2025, 1949, -1)), index=0)
    month_index = st.selectbox(
        "Mes", list(range(1, 13)), index=10, format_func=lambda i: MONTH_NAMES[i - 1]
    )
    selected_month = f"{year:04d}-{month_index:02d}"
    pitch = st.slider("Inclinación de cámara", min_value=0, max_value=70, value=56)
    st.divider()
    st.markdown("**Motor de IA**")
    files_present = (RESULTS / "modelo_ninoprob_ia.joblib").exists()
    st.caption("Modelo entrenado" if files_present else "Aún sin entrenar")
    with st.expander("Entrenamiento / archivo NOAA", expanded=not files_present):
        st.write("Entrena con NOAA/CPC 1950–2015, valida en 2016–2020 y prueba en 2021–2025.")
        upload = st.file_uploader(
            "Archivo NOAA opcional (si no puedes descargarlo)",
            type=["ascii", "txt"],
            help="Serie original NOAA/CPC ERSSTv5 Niño 1+2. No se aceptan datos inventados.",
        )
        if st.button("Entrenar IA con NOAA", type="primary", use_container_width=True):
            try:
                uploaded_path = None
                if upload is not None:
                    raw = upload.getvalue()
                    if len(raw) > 5_000_000:
                        raise ValueError("El archivo supera los 5 MB previstos.")
                    DATA_DIR.mkdir(parents=True, exist_ok=True)
                    uploaded_path = DATA_DIR / "noaa_subido_por_usuario.ascii"
                    uploaded_path.write_bytes(raw)
                with st.spinner("Validando NOAA y entrenando la regresión logística..."):
                    stats = run_pipeline(
                        ROOT, input_file=uploaded_path, allow_download=uploaded_path is None
                    )
                load_model.clear()
                st.success("¡Entrenamiento completado con datos NOAA!")
                st.caption(stats["conclusion"])
                st.rerun()
            except (OSError, ValueError, RuntimeError, requests.RequestException) as exc:
                st.error(f"No se completó el entrenamiento: {exc}")
    st.caption("Solo para fines educativos. No constituye una alerta ENFEN.")

trained = local_results()
artifact, metrics, monthly, predictions = (
    trained if trained else (None, None, None, None)
)
anomaly = region_observation(selected_month, monthly)

st.markdown('<div class="kicker">LABORATORIO OCEÁNICO · PERÚ</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="hero-title">Explorador 3D del mar peruano</div>'
    '<div class="soft">Temperaturas mensuales NOAA + probabilidad de '
    'calentamiento anómalo estimada por IA.</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div><span class="pill">NOAA / ERSSTv5</span>'
    '<span class="pill">Mapa 3D interactivo</span>'
    '<span class="pill">Regresión logística</span></div>',
    unsafe_allow_html=True,
)

tabs = st.tabs(["🌊 Océano 3D", "🧠 Inteligencia artificial", "📈 Evaluación", "📖 Metodología"])

with tabs[0]:
    col_head, col_action = st.columns([2.6, 1.1], vertical_alignment="bottom")
    with col_head:
        st.subheader(f"Océano frente al Perú · {MONTH_NAMES[month_index-1]} de {year}")
        st.caption(
            "Arrastra para explorar, utiliza la rueda para acercar y cambia "
            "la inclinación en el menú lateral."
        )
    with col_action:
        load_grid = st.button(
            "Cargar temperaturas reales NOAA",
            icon="🌡️", use_container_width=True,
            help="Descarga un subconjunto de la cuadrícula mensual ERSSTv5 del mes seleccionado.",
        )

    if load_grid:
        try:
            with st.spinner("Descargando la cuadrícula científica de NOAA/PSL..."):
                grid = cached_spatial(selected_month)
            st.session_state["spatial_month"] = selected_month
            st.session_state["spatial_grid"] = grid
        except (OSError, ValueError, requests.RequestException, ImportError) as exc:
            st.error(f"No se pudo obtener la cuadrícula NOAA: {exc}")
            st.info("El contorno Niño 1+2 sigue disponible si entrenaste el modelo.")
    spatial = (
        st.session_state.get("spatial_grid")
        if st.session_state.get("spatial_month") == selected_month
        else None
    )
    if not trained:
        st.info("Entrena primero la IA para visualizar el índice regional Niño 1+2. "
                "También puedes cargar directamente la cuadrícula geográfica NOAA.")
    loaded = map_3d(selected_month, anomaly, spatial, pitch)
    if loaded:
        st.markdown(
            '<div class="small-muted"><b>Escala de temperatura SST (°C absolutos)</b></div>'
            '<div class="legend"></div>'
            '<div class="small-muted">14 °C · 18 °C · 22 °C · 26 °C · 29 °C · 32 °C</div>',
            unsafe_allow_html=True,
        )
        st.caption(
            f"Capa NOAA/PSL ERSSTv5, {len(spatial)} puntos de la cuadrícula de 2° × 2°. "
            "El color indica temperatura superficial ABSOLUTA; "
            "las columnas tienen una altura ilustrativa, NO profundidad oceánica."
        )
        t1, t2, t3 = st.columns(3)
        t1.metric("SST mínima de la cuadrícula", f"{spatial['sst_c'].min():.1f} °C")
        t2.metric("SST máxima de la cuadrícula", f"{spatial['sst_c'].max():.1f} °C")
        t3.metric("Puntos oceánicos", f"{len(spatial)}")
    else:
        st.warning(
            "Solo está visible la delimitación Niño 1+2 (una anomalía media regional). "
            "Pulsa «Cargar temperaturas reales NOAA» para colorear celdas del océano. "
            "No inferimos temperaturas locales a partir de un promedio."
        )
    if anomaly is not None:
        st.metric("Anomalía media Niño 1+2 (NOAA/CPC)", f"{anomaly:+.2f} °C")
    st.caption(
        "NOAA ERSSTv5 no describe olas, corrientes, lluvia ni profundidad 3D. "
        "La extrusión ayuda a comparar temperaturas, pero no representa el relieve del mar."
    )

with tabs[1]:
    st.subheader("Probabilidad estimada por la IA")
    if trained:
        result = get_prediction(selected_month, artifact, monthly)
        if result is None:
            st.info("Se necesitan tres meses consecutivos de anomalías NOAA previas "
                    "para producir la estimación.")
        else:
            p = result["probability"]
            st.caption(f"Mes de origen: {selected_month} · Mes objetivo: {result['target']}")
            a, b, c = st.columns(3)
            a.metric("Probabilidad IA", f"{100*p:.1f} %")
            b.metric("Frecuencia histórica", f"{100*result['baseline']:.1f} %")
            c.metric("Anomalía observada del mes objetivo", (
                f"{result['observed']:+.2f} °C"
                if result["observed"] is not None else "No evaluada"
            ))
            st.progress(min(100, max(0, int(round(p * 100)))))
            st.markdown(
                "**Interpretación:** el modelo estima la probabilidad de que la "
                f"anomalía media de Niño 1+2 sea de **+{THRESHOLD:.1f} °C o más** "
                f"en {result['target']}."
            )
            st.caption(
                f"Periodo: {result['stage']}. Este cálculo es retrospectivo/educativo "
                "y no es un pronóstico operativo para la fecha actual."
            )
            with st.expander("Ver las tres entradas del modelo"):
                st.table(pd.DataFrame({
                    "Periodo": ["Mes seleccionado", "Mes anterior", "Hace dos meses"],
                    "Anomalía (°C)": result["anomalies"],
                }))
    else:
        st.info("Haz clic en «Entrenar IA con NOAA» en el panel izquierdo. "
                "Cuando termine, aquí aparecerán las probabilidades del modelo real.")

with tabs[2]:
    st.subheader("¿Qué tan bien predice la IA?")
    if trained:
        ia = metrics["prueba"]["ia"]
        base_rate = metrics["prueba"]["frecuencia_historica"]
        better = ia["brier"] < base_rate["brier"]
        a, b, c = st.columns(3)
        a.metric("Brier IA", f"{ia['brier']:.4f}")
        b.metric("Brier referencia", f"{base_rate['brier']:.4f}")
        c.metric("Exactitud IA", f"{100*ia['exactitud_umbral_50']:.1f} %")
        st.success(metrics["conclusion"]) if better else st.warning(metrics["conclusion"])
        st.caption("Menor puntuación de Brier significa menor error probabilístico.")
        plotted = predictions.copy()
        plotted["mes_objetivo"] = pd.to_datetime(plotted["mes_objetivo"])
        plotted = plotted.set_index("mes_objetivo")
        st.line_chart(
            plotted[["probabilidad_ia", "probabilidad_historica"]],
            height=290,
            y_label="Probabilidad",
            x_label="Mes de prueba (2021–2025)",
        )
        if "anomalia_mes_objetivo_c" in plotted:
            st.line_chart(
                plotted[["anomalia_mes_objetivo_c"]],
                height=250,
                y_label="Anomalía Niño 1+2 (°C)",
            )
        with st.expander("Tabla de pruebas cronológicas reales"):
            st.dataframe(predictions, use_container_width=True, hide_index=True)
            st.download_button(
                "Descargar predicciones CSV",
                predictions.to_csv(index=False).encode("utf-8"),
                file_name="ninoprob_predicciones_2021_2025.csv",
                mime="text/csv",
            )
    else:
        st.info("Las métricas aparecerán cuando el modelo sea entrenado y evaluado.")

with tabs[3]:
    st.subheader("Datos científicos y límites")
    st.markdown(
        "- **IA:** NOAA/CPC Niño 1+2 ERSSTv5, anomalías de referencia 1991–2020.\n"
        "- **Mapa 3D:** NOAA/PSL ERSSTv5, temperaturas SST absolutas en cuadrícula 2° × 2°.\n"
        "- **Entrenamiento:** objetivos de 1950–2015; validación 2016–2020; prueba 2021–2025.\n"
        "- **Regresión logística:** usa anomalías de los últimos tres meses y estacionalidad.\n"
        "- **Evento educativo:** próximo mes con anomalía regional ≥ +0,5 °C.\n"
        "- **Alturas del mapa:** convenciones visuales, no relieve submarino.\n"
        "- **Limitación:** el sistema no predice lluvias, daños, inundaciones ni alertas de ENFEN."
    )
    st.markdown(
        f"Fuentes oficiales: [índice Niño 1+2 NOAA/CPC]({NOAA_URL}) · "
        f"[cuadrícula NOAA/PSL ERSSTv5]({NCSS_URL}/dataset.html)"
    )
    if trained:
        st.markdown(f"**SHA-256 de la fuente del modelo:** `{metrics['sha256_fuente']}`")
        st.download_button(
            "Descargar métricas JSON",
            json.dumps(metrics, indent=2, ensure_ascii=False).encode("utf-8"),
            file_name="ninoprob_metricas.json", mime="application/json",
        )
