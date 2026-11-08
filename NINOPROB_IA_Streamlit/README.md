# NIÑOPROB-IA — Streamlit, IA real y explorador 3D de mar peruano

**Proyecto educativo EXPOMATEMÁTICA JUANPABLINA 2026.** La aplicación utiliza
anomalías históricas de la región Niño 1+2, entrenadas con regresión logística,
y un mapa inclinado interactivo con celdas de temperatura superficial oceánica
provenientes de NOAA. No constituye una predicción oficial de ENFEN.

## Inicio rápido en tu computadora

Necesitas **Python 3.10 o superior** y conexión a Internet.

1. Descomprime este ZIP y abre una terminal en la carpeta principal.
2. Instala los paquetes:

```bash
python -m pip install -r requirements.txt
```

3. Inicia la aplicación:

```bash
python -m streamlit run app.py
```

4. Abre la URL que aparezca en la terminal, normalmente
   **http://localhost:8501**.

5. En el panel lateral, pulsa **Entrenar IA con NOAA**. El programa obtiene
   la serie original NOAA/CPC, entrena y calcula el error con datos reales.
   Después puedes seleccionar un mes en el menú lateral.

6. En la pestaña **Océano 3D**, pulsa **Cargar temperaturas reales NOAA**
   para descargar la cuadrícula geográfica del mes elegido.
   Puedes usar el mouse para acercar, mover y cambiar de perspectiva.

## ¿Qué se ve?

- **Región Niño 1+2**: un rectángulo oceánico representa **una sola anomalía
  regional**, procedente de los datos NOAA/CPC. No son temperaturas locales.
- **Cuadrícula SST real**: al pulsar «Cargar temperaturas reales NOAA», se
  consultan temperaturas superficiales absolutas (°C) de NOAA/PSL ERSSTv5, a
  resolución **2° × 2°**. Cada celda se pinta con una escala fija 14–32 °C.
- **Columnas 3D**: la altura es una **metáfora visual para comparar SST**;
  no es profundidad marina ni altura de olas ni temperatura debajo del agua.
- **Pestaña Inteligencia artificial**: calcula la probabilidad de que la
  anomalía **regional** Niño 1+2 del mes siguiente sea al menos **+0,5 °C**.
- **Pestaña Evaluación**: expone Brier, exactitud, evolución de probabilidades
  y los registros del conjunto independiente 2021–2025.

**Importante:** el color de las celdas geográficas representa **temperatura
absoluta**, mientras que la anomalía regional de la IA es la **diferencia
respecto a la referencia climática 1991–2020**. No comparar directamente ambas
magnitudes ni deducir lluvias, fenómenos costeros o daños.

## Fuentes verificadas

- Datos de entrenamiento regional, NOAA/CPC ERSSTv5 Niño 1+2 1991–2020:
  https://www.cpc.ncep.noaa.gov/data/indices/ersst5.nino.mth.91-20.ascii
- Cuadrícula oceánica ERSSTv5, NOAA/PSL:
  https://psl.noaa.gov/thredds/ncss/grid/Datasets/noaa.ersst.v5/sst.mnmean.nc/dataset.html

El servidor NOAA/PSL puede sufrir demoras o caídas. La capa espacial requiere
Internet; en caso de error, la aplicación lo indica y no genera temperaturas
ficticias. El mapa base requiere conexión al proveedor de mosaicos CARTO.

## Entrenamiento científico

**Se mantiene el protocolo del PDF:** datos históricos 1950–2025; variables
de los últimos 3 meses y seno/coseno estacional; entrenamiento 1950–2015,
validación 2016–2020 y prueba independiente 2021–2025. El hiperparámetro de la
regresión logística se elige solo con la validación. El método se compara contra
una probabilidad constante basada en la frecuencia del entrenamiento.

Los archivos se guardan en `resultados/`. Si el servidor NOAA/CPC no responde,
puedes descargar manualmente la serie NOAA y subirla en el panel lateral.
Nunca se sustituyen registros NOAA por números simulados.

## Subir a Streamlit Community Cloud

1. Crea un repositorio GitHub con todos los archivos de esta carpeta.
2. Abre https://share.streamlit.io y conecta tu repositorio.
3. Selecciona `app.py` como archivo principal y publica.
4. La plataforma instalará los paquetes listados en `requirements.txt`.
5. Tras publicar, entrena la IA desde la interfaz.

**Nota sobre almacenamiento:** Streamlit Community Cloud puede reiniciar el
contenedor; los resultados generados en tiempo de ejecución no necesariamente
permanecen después de un reinicio. Si necesitas evitar volver a entrenar,
almacena los artefactos generados en un volumen persistente (con una política
adecuada de acceso) o distribúyelos como archivos confiables de tu proyecto.
Nunca subas secretos ni archivos personales.

## Comprobar el código

```bash
python -m pytest -q
```

Las pruebas utilizan un archivo NetCDF sintético muy pequeño para verificar
**la lectura del software**, no para declarar resultados científicos de la IA.

## Estructura

```text
NINOPROB_IA_Streamlit/
├── app.py
├── ninoprob_ia/
│   ├── core.py
│   ├── spatial.py
│   ├── api.py
│   └── cli.py
├── datos/
├── resultados/
├── tests/
├── notebooks/
├── integracion_3d/
├── requirements.txt
└── README.md
```

## Seguridad y límites

El motor carga modelos `joblib` solo desde rutas internas generadas por el proyecto.
No debes ejecutar archivos de modelos no confiables. El método es educativo y no
corresponde a un pronóstico oficial de El Niño costero.
