# NIÑOPROB-IA — Fase 2 con Streamlit

**Visualizador oceánico 3D + motor probabilístico escolar ya creado en la Fase 1.**

El visualizador permite estudiar temperaturas superficiales REALES del océano frente a Perú.
La altura de los polígonos es una representación gráfica; **no** es profundidad,
nivel del mar, circulación oceánica, ni intensidad de un fenómeno meteorológico.

## Inicio rápido

1. Instala Python 3.10 o superior.
2. Abre una terminal dentro de esta carpeta.
3. Instala los paquetes:

   ```bash
   python -m pip install -r requirements.txt
   ```

4. Ejecuta Streamlit:

   ```bash
   python -m streamlit run app.py
   ```

5. Entra a http://localhost:8501 si no se abre automáticamente.

## Cómo usar el visualizador 3D (Fase 2)

1. Elige en el lateral **Histórico mensual · ERSSTv5 (2°)** o
   **Detalle diario · OISST (0,25°)**.
2. Elige fecha y zona: mar peruano completo, región Niño 1+2, costa norte,
   centro o sur.
3. En **Capas oceánicas** puedes elegir:
   - **Temperatura superficial (°C)**: temperatura absoluta del agua.
   - **Anomalía frente a 1991–2020 (°C)**: diferencia de la celda respecto
     a la climatología NOAA ERSSTv5 1991–2020 del mismo mes. Solo mensual.
   - **Cambio respecto a otro mes (°C)**: diferencia de SST en las mismas
     coordenadas entre dos meses. Solo mensual. Este cambio **no** es una anomalía.
4. Activa o desactiva el **Relieve térmico 3D** y ajusta la inclinación
   o la exageración vertical.
5. Pulsa **Cargar datos reales NOAA**. La consulta obtiene únicamente los datos
   solicitados. No hay imágenes prehechas con temperaturas imaginarias.
6. Arrastra, haz zoom, cambia la inclinación y haz clic en una celda. Se muestra
   su temperatura y su posición geográfica.
7. Revisa el perfil medio por latitud y descarga el archivo CSV de las celdas.

Las descargas se cachean temporalmente para que cambiar de cámara no obligue a
volver a consultar NOAA. Si cambias de fecha, región o capa, pulsa de nuevo
**Cargar datos reales NOAA**. Si el servidor NOAA no responde, se informa del
error y no se genera una cuadrícula falsa.

## Inteligencia artificial (Fase 1, conservada)

En la pestaña **Inteligencia artificial** haz clic en **Entrenar NIÑOPROB-IA
con NOAA**. El programa descarga automáticamente el índice regional Niño 1+2
o acepta el archivo ASCII NOAA original proporcionado por ti.

- Variables: anomalía regional actual, de un mes anterior y de dos meses anteriores,
  más seno/coseno del mes objetivo.
- Objetivo: anomalía regional Niño 1+2 del **mes siguiente ≥ +0,5 °C**.
- Entrenamiento: 1950–2015.
- Validación: 2016–2020.
- Evaluación completamente independiente: 2021–2025.
- Comparación contra la frecuencia histórica y puntuación de Brier.

**Separación crucial:** los mapas SST y la probabilidad de la IA proceden
de datos distintos y nunca se interpretan como si fueran una misma medición:

| Capa | Fuente | Significado | Resolución |
|---|---|---|---|
| IA Niño 1+2 | NOAA/CPC ERSSTv5 (1991–2020) | Anomalía media regional y probabilidad mensual | un índice por mes |
| SST mensual | NOAA/PSL ERSSTv5 | SST absoluta por celda | 2° × 2° |
| Anomalía mensual espacial | NOAA/PSL ERSSTv5 + clima 1991–2020 | Diferencia de cada celda contra su climatología mensual | 2° × 2° |
| SST diaria | NOAA/PSL OISST high-resolution | SST absoluta por celda, día elegido | 0,25° × 0,25° |

La opción detallada diaria no se usa directamente para entrenar la regresión
logística mensual del proyecto escolar. No se pretende estimar tendencias
biológicas o lluvias a partir de SST solamente.

## Fuentes oficiales

- NOAA/CPC Niño 1+2, ERSSTv5:
  https://www.cpc.ncep.noaa.gov/data/indices/ersst5.nino.mth.91-20.ascii
- NOAA/PSL ERSSTv5 mensual:
  https://psl.noaa.gov/thredds/ncss/grid/Datasets/noaa.ersst.v5/sst.mnmean.nc/dataset.html
- NOAA/PSL clima mensual ERSSTv5 1991–2020:
  https://psl.noaa.gov/thredds/fileServer/Datasets/noaa.ersst.v5/sst.mon.ltm.1991-2020.nc
- NOAA/PSL OISST v2 high-res diaria (ejemplo 2025):
  https://psl.noaa.gov/thredds/ncss/grid/Datasets/noaa.oisst.v2.highres/sst.day.mean.2025.nc/dataset.html

Se requiere Internet para consultar NOAA y cargar el mapa base CARTO.
Los datos son públicos; la aplicación no contiene claves privadas de servicio.

## Archivos importantes

```text
NINOPROB_IA_Streamlit_Fase2/
├── app.py                        # Lanzador PRINCIPAL de fase 2
├── app_fase1.py                  # Interfaz anterior conservada como respaldo
├── ninoprob_ia/
│   ├── core.py                   # Entrenamiento y validación original
│   ├── phase2.py                 # Datos SST espaciales y preparación de mapas
│   ├── spatial.py                # Cliente ERSST mensual fase 1
│   ├── api.py                    # API opcional para usos externos
│   └── cli.py
├── tests/
│   ├── test_pipeline.py
│   ├── test_streamlit_spatial.py
│   └── test_phase2.py
├── requirements.txt
├── pyproject.toml
├── .streamlit/config.toml
└── README.md
```

La carpeta `resultados/` se genera al entrenar. Streamlit Community Cloud puede
reiniciar su contenedor y perder los archivos creados en ejecución; en ese caso
se necesita repetir el entrenamiento o disponer de almacenamiento persistente
autorizado.

## Despliegue gratuito en Streamlit Community Cloud

1. Crea un repositorio GitHub privado o público para este proyecto.
2. Sube el contenido de la carpeta (no solo `app.py`).
3. Conecta el repositorio en https://share.streamlit.io.
4. Selecciona `app.py` como punto de entrada.
5. Espera la instalación automática de `requirements.txt`.
6. Abre la app y consulta los datos NOAA.

## Pruebas

```bash
python -m pytest -q
python -m py_compile app.py ninoprob_ia/phase2.py
```

Las pruebas usan NetCDF sintéticos exclusivamente para comprobar el programa.
**No son resultados de entrenamiento NOAA**. La descarga remota NOAA y el
renderizado dentro de Streamlit requieren ejecutar la app con Internet; no
han sido verificados en este entorno de creación.

## Limitaciones y advertencias

- NOAA ERSST 2° es una cuadrícula climática reconstruida, NO una foto
  satelital de alta resolución de cada porción del litoral.
- NOAA OISST 0,25° mejora la resolución espacial solo para fechas disponibles.
- Cambios entre dos meses y anomalías respecto a una climatología son
  magnitudes diferentes.
- La paleta de colores usa límites constantes para permitir comparaciones
  entre fechas; valores extremos se muestran en el color de los extremos.
- La extrusión vertical es simbólica y no significa profundidad del océano.
- La IA probabilística educativa **no sustituye** pronósticos ni alertas ENFEN.
