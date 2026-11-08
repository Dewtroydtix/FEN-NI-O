# Integración del futuro visualizador 3D

Este directorio tiene un cliente TypeScript que conecta la IA a un frontend **React + CesiumJS** o **React + MapLibre**. No dibuja aún el océano 3D: eso es la siguiente fase.

```ts
import { NinoprobAPI } from "./api-client";

const api = new NinoprobAPI("http://127.0.0.1:8000");
const region = await api.region("2025-03");  // polígono de índice regional
const serie = await api.serie("2021-01", "2025-12");
const prediccion = await api.prediccion("2025-11");
console.log(prediccion.probabilidad_ia);
```

**Regla de presentación científica:** el color uniforme del polígono puede representar la anomalía promedio de Niño 1+2, pero NO la temperatura específica de cada punto del océano.

Para un mapa térmico 3D detallado se necesitará un raster oceánico real de temperatura superficial del mar con latitud y longitud (p. ej. [NOAA OISST](https://www.ncei.noaa.gov/products/optimum-interpolation-sst)), procesado por un servicio de tiles/raster. La IA de fase 1 produce probabilidades **de la región completa**, no un campo de calor por coordenada.
