"""NIÑOPROB-IA | Fase 2 — Explorador científico 3D en Streamlit.

Ejecutar: streamlit run app.py
Los colores se calculan únicamente a partir de cuadrículas reales NOAA.
Nunca representa 3D volumétrico, corrientes o efectos biológicos medidos.
"""
from __future__ import annotations

import json
import hashlib
from datetime import date
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pydeck as pdk
import requests
import streamlit as st

from ninoprob_ia.core import BBOX_NINO12, FEATURES, NOAA_URL, THRESHOLD, run_pipeline
from ninoprob_ia.surface3d import grafico_superficie
from ninoprob_ia.phase2 import (
    CLIMATOLOGY_URL, DISPLAY_MODES, REGIONS, OISST_NCSS_ROOT,
    add_anomalies, area_center, compare_months, crop_to_region,
    fetch_climatology, fetch_ersst, fetch_oisst, ocean_tiles, weighted_mean,
)

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "resultados"
DATA_DIR = ROOT / "datos"
MESES = ("Enero", "Febrero", "Marzo", "Abril", "Mayo", "Junio",
         "Julio", "Agosto", "Septiembre", "Octubre", "Noviembre", "Diciembre")

st.set_page_config(
    page_title="NIÑOPROB-IA | Océano 3D",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.markdown("""
<style>
.stApp {
  background: radial-gradient(circle at 90% -10%, #14445b, #06192d 42%, #061321 90%);
  color: #e7f5fb;
}
[data-testid="stSidebar"] {background:#091e2d;}
[data-testid="stMetric"] {
  border: 1px solid #285269; border-radius: 16px;
  background: rgba(9,38,55,.88); padding: 14px;
}
[data-testid="stMetricValue"] {color:#8eeade}
h1,h2,h3 {letter-spacing:-.025em}
.eyebrow {font-size:.82rem; color:#73d7d8; font-weight:800; letter-spacing:.17em}
.big-header {font-weight:850; font-size:clamp(2rem,3vw,3.3rem);line-height:1.13;
 margin:.28rem 0 .6rem}
.subtitle {color:#b7d2df;font-size:1.05rem}
.badge-row span {display:inline-block; border-radius:99px; padding:.25rem .7rem;
 margin:.55rem .3rem .45rem 0; border:1px solid #2b647a; background:#103349;
 color:#bee9ed;font-size:.8rem}
.thermal-legend {height:14px; border-radius:100px;
 background:linear-gradient(90deg,#184395,#1e8fcf,#2bccba,#f8d059,#ef6d31,#a92337)}
.diff-legend {height:14px; border-radius:100px;
 background:linear-gradient(90deg,#22489f,#379ed6,#e9f1f1,#f6ad43,#b12c35)}
.legend-labels {display:flex;justify-content:space-between;font-size:.8rem;color:#b8d0dc}
.info-note {font-size:.83rem;color:#a9c5d4}
</style>
""", unsafe_allow_html=True)


@st.cache_data(ttl=60 * 60 * 24, show_spinner=False)
def grid_mensual(period: str, area: str) -> pd.DataFrame:
    return fetch_ersst(period, area)


@st.cache_data(ttl=60 * 60 * 24 * 7, show_spinner=False)
def clima_mensual(mes: int) -> pd.DataFrame:
    return fetch_climatology(mes)


@st.cache_data(ttl=60 * 60 * 24, show_spinner=False)
def grid_diaria(dia: date, area: str) -> pd.DataFrame:
    return fetch_oisst(dia, area)


@st.cache_resource(show_spinner=False)
def cargar_modelo(filename: str, mtime: int):
    del mtime
    # Solo archivos locales generados por el pipeline. Nunca subir modelos externos.
    return joblib.load(filename)


def resultados_del_modelo():
    paths = [
        RESULTS / "modelo_ninoprob_ia.joblib",
        RESULTS / "metricas.json",
        RESULTS / "serie_mensual_noaa.csv",
        RESULTS / "predicciones_prueba_2021_2025.csv",
    ]
    if not all(path.is_file() for path in paths):
        return None
    try:
        m = json.loads(paths[1].read_text(encoding="utf-8"))
        modelo = cargar_modelo(str(paths[0]), paths[0].stat().st_mtime_ns)
        if modelo.get("features") != list(FEATURES):
            raise ValueError("El modelo usa variables diferentes.")
        if modelo.get("source_sha256") != m.get("sha256_fuente"):
            raise ValueError("El archivo NOAA no corresponde al modelo guardado.")
        registro = pd.read_csv(paths[2])
        predicciones = pd.read_csv(paths[3])
        return modelo, m, registro, predicciones
    except (OSError, ValueError, KeyError, TypeError) as exc:
        st.error(f"No se pudieron abrir los resultados entrenados: {exc}")
        return None


def predecir_mes(period: str, modelo: dict, registro: pd.DataFrame) -> dict | None:
    p = pd.Period(period, freq="M")
    datos = registro.set_index("mes")
    origenes = [str(p - k) for k in (0, 1, 2)]
    if any(s not in datos.index or pd.isna(datos.at[s, "anomalia_c"]) for s in origenes):
        return None
    t = [float(datos.at[s, "anomalia_c"]) for s in origenes]
    siguiente = p + 1
    ang = 2 * np.pi * (siguiente.month - 1) / 12
    x = np.asarray([[t[0], t[1], t[2], np.sin(ang), np.cos(ang)]])
    prob = float(modelo["model"].predict_proba(x)[0, 1])
    real = float(datos.at[str(siguiente), "anomalia_c"]) if (
        str(siguiente) in datos.index and
        pd.notna(datos.at[str(siguiente), "anomalia_c"])
    ) else None
    return {
        "p": prob, "fecha": str(siguiente), "observacion": real,
        "historia": float(modelo["baseline"]), "entradas": t,
    }


def rango_nino12() -> pdk.Layer:
    west, south, east, north = BBOX_NINO12
    return pdk.Layer(
        "PolygonLayer",
        id="region_nino12",
        data=[{"polygon": [[west,south],[east,south],[east,north],[west,north]]}],
        get_polygon="polygon", filled=False, stroked=True,
        get_line_color=[91, 230, 244, 230], get_line_width=3,
        line_width_min_pixels=3, pickable=False,
    )


def render_mapa(celdas: list[dict], area: str, extruido: bool, camara: int, mostrar_region: bool, scene_id: str):
    lon, lat, zoom = area_center(area)
    layers = []
    if celdas:
        layers.append(pdk.Layer(
            "PolygonLayer",
            id="temperatura_oceano",
            data=celdas,
            get_polygon="polygon", get_fill_color="color",
            filled=True, stroked=False, pickable=True,
            extruded=extruido, get_elevation="altura", elevation_scale=1,
            auto_highlight=True, opacity=.93,
        ))
    if mostrar_region:
        layers.append(rango_nino12())
    deck = pdk.Deck(
        initial_view_state=pdk.ViewState(
            latitude=lat, longitude=lon, zoom=zoom,
            pitch=camara, bearing=8,
        ),
        map_style="https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json",
        layers=layers,
        tooltip={
            "html": (
                "<div style='padding:10px;background:#072437;color:#e9f9fc'>"
                "<b>{modo}: {valor} {unidad}</b><br/>"
                "SST: {sst_c} °C<br/>"
                "Latitud: {lat} · Longitud: {lon}<br/>"
                "<small>Dato superficial NOAA</small></div>"
            ),
            "style": {"backgroundColor": "#072437", "color": "#e9f9fc"},
        },
    )
    selected = st.pydeck_chart(
        deck, height=580, use_container_width=True,
        on_select="rerun", selection_mode="single-object", key="oceano3d_" + scene_id,
    )
    try:
        objects = selected.selection.get("objects", {})
        return (objects.get("temperatura_oceano") or [None])[0]
    except (AttributeError, KeyError, IndexError, TypeError):
        return None


def grafico_leyenda(capa: str):
    temp = capa.startswith("Temperatura")
    gradient_class = "thermal-legend" if temp else "diff-legend"
    labels = ["14°", "18°", "22°", "26°", "29°", "32°"] if temp else [
        "−3°", "−1,5°", "0°", "+1,5°", "+3°"
    ]
    st.markdown(
        f"<div class='{gradient_class}'></div><div class='legend-labels'>"
        + "".join(f"<span>{l}</span>" for l in labels) +
        "</div>", unsafe_allow_html=True,
    )


# Controles sin datos ficticios
with st.sidebar:
    st.markdown("### 🌊 NIÑOPROB-IA")
    st.caption("Fase 2 · Laboratorio oceanográfico")
    st.divider()
    st.markdown("**1. Fuente y resolución**")
    origen = st.radio(
        "Producto NOAA",
        ["Histórico mensual · ERSSTv5 (2°)", "Detalle diario · OISST (0,25°)"],
        help="El producto de 0,25° representa días observados; el modelo IA usa la serie mensual ERSSTv5.",
    )
    es_mensual = origen.startswith("Histórico")
    if es_mensual:
        periodo = st.select_slider(
            "Viajar por los meses", options=pd.period_range(
                "1950-01", "2025-12", freq="M"
            ).astype(str).tolist(),
            value="2025-11",
        )
        mes_fecha = int(periodo[-2:])
        fecha = None
    else:
        fecha = st.date_input(
            "Día de observación",
            min_value=date(1982, 1, 1), max_value=date(2025, 12, 31),
            value=date(2025, 11, 15),
        )
        periodo = fecha.strftime("%Y-%m")
        mes_fecha = fecha.month
    area = st.selectbox("2. Zona para explorar", list(REGIONS), index=0)
    capas = (
        list(DISPLAY_MODES) if es_mensual else ["Temperatura superficial (°C)"]
    )
    capa = st.selectbox("3. Capa oceánica", capas)
    comparar = None
    if es_mensual and capa.startswith("Cambio"):
        opciones = pd.period_range("1950-01", "2025-12", freq="M").astype(str).tolist()
        comparar = st.selectbox(
            "Mes de comparación", opciones, index=opciones.index("2024-11"),
        )
    st.divider()
    st.markdown("**Vista del océano**")
    tipo_vista = st.radio(
        "Estilo de visualización",
        ["Superficie 3D suave", "Mapa geográfico NOAA"],
        index=0,
        help="La superficie 3D interpola visualmente entre celdas observadas. "
             "El mapa geográfico conserva los polígonos NOAA originales.",
    )
    extruir = st.toggle("Relieve térmico 3D", value=True)
    angulo = st.slider("Inclinación", 0, 70, 55)
    exageracion = st.slider("Escala vertical ilustrativa", 0.5, 3.0, 1.3, 0.1)
    mostrar_celdas = st.checkbox("Ver contorno de región Niño 1+2", value=True,
                                 disabled=tipo_vista=="Superficie 3D suave",
                                 help="El contorno geográfico se muestra en el mapa NOAA.")
    st.divider()
    consulta = st.button(
        "🌡️ Cargar datos reales NOAA", type="primary",
        use_container_width=True,
    )
    st.caption("La descarga requiere Internet. Sin ella no se dibujan temperaturas ficticias.")

modelo_resultado = resultados_del_modelo()
modelo, metricas, registro, predicciones = (
    modelo_resultado if modelo_resultado else (None, None, None, None)
)
clave = (origen, periodo, area, capa, comparar)
if consulta:
    try:
        with st.spinner("Consultando los datos científicos NOAA y preparando el mapa..."):
            if es_mensual:
                base = grid_mensual(periodo, area)
                if capa.startswith("Anomalía"):
                    base = add_anomalies(base, crop_to_region(clima_mensual(mes_fecha), area))
                elif capa.startswith("Cambio"):
                    assert comparar is not None
                    anterior = grid_mensual(comparar, area)
                    base = compare_months(base, anterior)
            else:
                assert fecha is not None
                base = grid_diaria(fecha, area)
            if base.empty:
                raise ValueError("No hay celdas oceánicas para esta fecha y región.")
            st.session_state["mapa_clave"] = clave
            st.session_state["mapa_datos"] = base
        st.toast("Mapa NOAA cargado", icon="🌊")
    except (ValueError, OSError, requests.RequestException, ImportError) as exc:
        st.session_state.pop("mapa_clave", None)
        st.session_state.pop("mapa_datos", None)
        st.error(f"No fue posible cargar los datos: {exc}")
        st.info("Comprueba tu conexión o intenta otra fecha. No se generaron datos inventados.")

frame = st.session_state.get("mapa_datos") if st.session_state.get("mapa_clave") == clave else None

st.markdown('<div class="eyebrow">EXPOMATEMÁTICA JUANPABLINA 2026 · EDICIÓN STREAMLIT</div>',
            unsafe_allow_html=True)
st.markdown('<div class="big-header">El calor del mar peruano, en 3D</div>',
            unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Explora observaciones NOAA, viaja entre fechas '
    'y conecta los resultados a la inteligencia artificial NIÑOPROB-IA.</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="badge-row"><span>Océano interactivo</span>'
    '<span>Temperatura espacial NOAA</span><span>Predicción probabilística</span>'
    '<span>Proyecto escolar</span></div>',
    unsafe_allow_html=True,
)

tab_oceano, tab_ia, tab_evaluacion, tab_fuentes = st.tabs([
    "🌊 Visualizador 3D", "🧠 Inteligencia artificial",
    "📈 Evaluación científica", "📖 Fuentes y metodología",
])

with tab_oceano:
    titulo_fecha = (
        f"{MESES[mes_fecha-1]} de {periodo[:4]}" if es_mensual
        else fecha.strftime("%d/%m/%Y")
    )
    st.subheader(f"{area} · {titulo_fecha}")
    st.caption("En 'Superficie 3D suave' gira el gráfico con el mouse; "
               "en 'Mapa geográfico NOAA' selecciona celdas medidas "
               "y explora la costa peruana.")
    data_layer = []
    if frame is not None and tipo_vista == "Mapa geográfico NOAA":
        data_layer = ocean_tiles(
            frame, capa, (2.0 if es_mensual else 0.25),
            extruir, exageracion,
        )
    if frame is None:
        st.info(
            "Selecciona fuente, fecha y zona en el panel lateral, luego pulsa "
            "**Cargar datos reales NOAA**. El mapa mostrará la temperatura únicamente "
            "cuando la descarga científica sea satisfactoria."
        )
    # La superficie suavizada usa las MISMAS celdas NOAA, pero interpola
    # entre puntos cercanos SOLO como recurso gráfico (nunca como medición).
    # El mapa PyDeck original continúa disponible para inspeccionar posiciones.
    picked = None
    if frame is not None and tipo_vista == "Superficie 3D suave":
        if len(frame) >= 4:
            try:
                figura = grafico_superficie(frame, capa, exageracion if extruir else 0.0, angulo)
                st.plotly_chart(figura, use_container_width=True,
                                config={"displaylogo": False, "scrollZoom": True})
                st.info(
                    "Superficie 3D INTERPOLADA para la presentación: el suavizado "
                    "no agrega mediciones ni mejora la resolución de NOAA. "
                    "Los valores al pasar el cursor ENTRE celdas son interpolados. "
                    "Para ver cada celda real y el contorno Niño 1+2, elige "
                    "'Mapa geográfico NOAA'."
                )
            except ValueError as exc:
                st.warning(f"No es posible crear una superficie continua: {exc}")
        else:
            st.warning("Hay muy pocos puntos NOAA para interpolar una superficie.")
    elif tipo_vista == "Mapa geográfico NOAA" or frame is None:
        # El contorno geográfico se dibuja sobre un mapa base, sin valores ficticios.
        scene_id = hashlib.sha256(str(clave).encode("utf-8")).hexdigest()[:12]
        picked = render_mapa(data_layer, area, extruir, angulo, mostrar_celdas, scene_id)

    if frame is not None:
        grafico_leyenda(capa)
        st.caption(
            "Los colores representan la variable oceánica seleccionada en °C. "
            "La altura 3D es ILUSTRATIVA: no representa profundidad, "
            "altura del agua ni predicción de corrientes."
        )
        colname = DISPLAY_MODES[capa][0]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Temperatura mínima", f"{frame['sst_c'].min():.1f} °C")
        c2.metric("Temperatura máxima", f"{frame['sst_c'].max():.1f} °C")
        c3.metric(DISPLAY_MODES[capa][2] + " media*", f"{weighted_mean(frame, colname):+.2f} °C")
        c4.metric("Celdas oceánicas", f"{len(frame):,}")
        st.caption("*Media de celdas disponibles ponderada por latitud, no promedio "
                   "oficial del litoral ni índice Niño 1+2.")
        if picked and isinstance(picked, dict):
            st.success(
                f"Celda elegida: {picked.get('lat')}° lat., {picked.get('lon')}° lon. "
                f"· SST {picked.get('sst_c')} °C "
                f"· {DISPLAY_MODES[capa][2]} {picked.get('valor')} °C"
            )
        with st.expander("Explorar el patrón espacial de temperatura", expanded=False):
            lat_profile = frame.groupby("lat", as_index=False)["sst_c"].mean().sort_values("lat")
            st.line_chart(lat_profile.set_index("lat"), height=220,
                          y_label="SST media de celdas (°C)", x_label="Latitud (°)")
            st.caption("Promedio simple en cada franja de latitud, con las celdas oceánicas disponibles.")
        st.download_button(
            "⬇️ Descargar datos de este mapa (CSV)",
            data=frame.to_csv(index=False).encode("utf-8"),
            file_name=f"ninoprob_mapa_{periodo.replace('-','_')}_{'mensual' if es_mensual else 'diario'}.csv",
            mime="text/csv", use_container_width=False,
        )
    if not es_mensual:
        st.caption("OISST 0,25° aporta mayor detalle geográfico. "
                   "No alimenta directamente la IA mensual del proyecto.")
    st.caption(
        "Las temperaturas son de la superficie del océano. La aplicación "
        "no deduce impactos sobre peces, lluvias ni mareas a partir del mapa."
    )

with tab_ia:
    st.subheader("IA para probabilidad de anomalía cálida")
    if not es_mensual:
        st.info("La IA estudia anomalías *mensuales* Niño 1+2. "
                f"Para el día seleccionado, la consulta de IA usa el mes {periodo}, "
                "no una predicción diaria.")
    if modelo_resultado is None:
        st.warning("No hay modelo entrenado en esta instalación.")
        st.markdown("Entrénalo con la serie NOAA/CPC original para obtener resultados reales.")
        with st.form("entrenamiento"):
            subida = st.file_uploader(
                "Archivo NOAA/CPC opcional (.ascii o .txt)",
                type=["txt", "ascii"],
                help="Si no subes nada, el programa descargará el archivo oficial.",
            )
            ejecutar = st.form_submit_button(
                "🧠 Entrenar NIÑOPROB-IA con NOAA", type="primary",
                use_container_width=True,
            )
        if ejecutar:
            try:
                entrada = None
                if subida is not None:
                    raw = subida.getvalue()
                    if len(raw) > 5_000_000:
                        raise ValueError("El archivo NOAA supera el tamaño admitido (5 MB).")
                    DATA_DIR.mkdir(exist_ok=True, parents=True)
                    entrada = DATA_DIR / "noaa_original_usuario.ascii"
                    entrada.write_bytes(raw)
                with st.spinner("Entrenando y validando con separación cronológica..."):
                    calculo = run_pipeline(
                        ROOT, input_file=entrada,
                        allow_download=entrada is None,
                    )
                cargar_modelo.clear()
                st.success(f"Entrenamiento finalizado. {calculo['conclusion']}")
                st.rerun()
            except (OSError, ValueError, RuntimeError, requests.RequestException) as exc:
                st.error(f"El entrenamiento no se pudo completar: {exc}")
    else:
        futuro = predecir_mes(periodo, modelo, registro)
        if futuro is None:
            st.info("Este mes no tiene los tres valores previos que necesita el modelo.")
        else:
            c1,c2,c3 = st.columns(3)
            c1.metric(f"Probabilidad para {futuro['fecha']}", f"{futuro['p']*100:.1f} %")
            c2.metric("Frecuencia histórica", f"{futuro['historia']*100:.1f} %")
            c3.metric("Anomalía observada en mes objetivo", (
                "No disponible" if futuro["observacion"] is None
                else f"{futuro['observacion']:+.2f} °C"
            ))
            st.progress(float(futuro["p"]))
            st.markdown(
                f"La IA estima la probabilidad de que la anomalía *regional* Niño 1+2 "
                f"sea al menos **+{THRESHOLD:.1f} °C** durante **{futuro['fecha']}**."
            )
            st.caption(
                "Probabilidad estimada, no certeza. Es una evaluación histórica "
                "educativa, no una alerta oficial ni un pronóstico de daños."
            )
            with st.expander("Variables de entrada al modelo"):
                st.dataframe(pd.DataFrame({
                    "Referencia": ["Mes consultado", "Un mes atrás", "Dos meses atrás"],
                    "Anomalía NOAA/CPC (°C)": futuro["entradas"],
                }), hide_index=True, use_container_width=True)

with tab_evaluacion:
    st.subheader("¿La IA mejora la estimación histórica?")
    if modelo_resultado is None:
        st.info("La evaluación aparecerá después de entrenar el modelo.")
    else:
        ia = metricas["prueba"]["ia"]
        ref = metricas["prueba"]["frecuencia_historica"]
        c1,c2,c3 = st.columns(3)
        c1.metric("Brier IA", f"{ia['brier']:.4f}")
        c2.metric("Brier referencia", f"{ref['brier']:.4f}")
        c3.metric("Exactitud IA (umbral 50 %)", f"{ia['exactitud_umbral_50']*100:.1f} %")
        st.success(metricas["conclusion"]) if ia["brier"] < ref["brier"] else st.warning(metricas["conclusion"])
        st.caption("Cuanto menor sea la puntuación de Brier, mejor es el error "
                   "probabilístico en la prueba de 2021–2025.")
        data = predicciones.copy()
        data["mes_objetivo"] = pd.to_datetime(data["mes_objetivo"])
        st.line_chart(
            data.set_index("mes_objetivo")[["probabilidad_ia", "probabilidad_historica"]],
            x_label="Mes evaluado", y_label="Probabilidad", height=300,
        )
        with st.expander("Datos de prueba real"):
            st.dataframe(data, use_container_width=True, hide_index=True)
            st.download_button("⬇️ Descargar evaluación CSV",
                data=predicciones.to_csv(index=False).encode("utf-8"),
                file_name="ninoprob_evaluacion_2021_2025.csv", mime="text/csv")

with tab_fuentes:
    st.subheader("Cómo se interpreta la visualización")
    st.markdown("""
**Datos de la inteligencia artificial:** anomalías mensuales de Niño 1+2
en NOAA/CPC ERSSTv5, referencia 1991–2020. Entrenamiento 1950–2015,
validación 2016–2020 y prueba final 2021–2025.

**Mapa mensual:** NOAA/PSL ERSSTv5, temperatura superficial en celdas de
2° × 2°. La opción *Anomalía frente a 1991–2020* compara cada celda con
la climatología **espacial** del mismo mes (no con la anomalía regional del modelo).

**Mapa diario detallado:** NOAA/PSL OISST, 0,25° × 0,25°.
Muestra SST absoluta de un día; no es la misma serie ni la resolución
empleadas para entrenar la IA.

**Colores y alturas:** el color representa temperatura absoluta, anomalía
o diferencia según la capa elegida. La vista suave interpola únicamente
para mostrar cambios entre celdas originales; **no agrega observaciones NOAA**.
La elevación 3D exagera diferencias térmicas y **no representa profundidad, corrientes ni movimiento físico del agua**.

**Limitación:** el umbral +0,5 °C es un evento didáctico; no representa
por sí mismo los criterios oficiales de ENFEN. Ningún resultado es una
alerta sobre lluvias, inundaciones o daños.
""")
    st.markdown(
        f"- [Serie de anomalías NOAA/CPC]({NOAA_URL})\n"
        "- [Temperatura mensual ERSSTv5 NOAA/PSL]"
        "(https://psl.noaa.gov/thredds/ncss/grid/Datasets/noaa.ersst.v5/sst.mnmean.nc/dataset.html)\n"
        f"- [Climatología espacial NOAA/PSL 1991–2020]({CLIMATOLOGY_URL})\n"
        "- [Temperatura diaria 0,25° NOAA/PSL]"
        "(https://psl.noaa.gov/thredds/ncss/grid/Datasets/noaa.oisst.v2.highres/sst.day.mean.2025.nc/dataset.html)"
    )
    st.caption("Si el servidor oficial o los mosaicos del mapa no responden, "
               "es necesario conexión a Internet. No se rellenan datos faltantes con IA.")
    if modelo_resultado is not None:
        st.download_button(
            "Descargar métricas del modelo (JSON)",
            json.dumps(metricas, ensure_ascii=False, indent=2).encode("utf-8"),
            "ninoprob_metricas.json", "application/json",
        )
