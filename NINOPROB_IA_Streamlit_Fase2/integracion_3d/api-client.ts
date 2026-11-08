/**
 * NIÑOPROB-IA: cliente tipado para un futuro visualizador CesiumJS/MapLibre 3D.
 *
 * Datos regionales NOAA != raster de temperaturas.
 * El mapa térmico espacial se integrará con datos cuadriculados en fase 2.
 */
export type Evaluacion =
  | "entrenamiento"
  | "validacion"
  | "prueba_independiente"
  | "fuera_del_periodo_de_prueba";

export interface Prediccion {
  mes_origen: string;
  mes_objetivo: string;
  anomalias_entrada_c: {
    actual: number;
    anterior: number;
    hace_dos_meses: number;
  };
  probabilidad_ia: number;
  probabilidad_frecuencia_historica: number;
  umbral_anomalia_c: number;
  anomalia_objetivo_observada_c: number | null;
  evento_objetivo_observado: 0 | 1 | null;
  tipo_evaluacion: Evaluacion;
  es_evaluacion_independiente: boolean;
  aviso: string;
}

export interface Observacion {
  mes: string;
  anomalia_c: number | null;
  temperatura_c: number | null;
}

export interface SerieTemporal {
  tipo: "observado_noaa";
  unidad: "°C";
  registros: Observacion[];
}

export interface Metadatos {
  modelo: string;
  nombre: string;
  fuente: string;
  url_noaa: string;
  climatologia: string;
  registro_desde: string;
  registro_hasta: string;
  umbral_c: number;
  bbox_nino12: [number, number, number, number];
  unidad_anomalia: string;
  nota: string;
  restriccion: string;
  mapa_termico_detallado_disponible: boolean;
}

export interface RegionNinoGeoJSON {
  type: "FeatureCollection";
  features: Array<{
    type: "Feature";
    geometry: {
      type: "Polygon";
      coordinates: number[][][];
    };
    properties: {
      nombre: string;
      mes: string;
      anomalia_media_c: number | null;
      tipo_dato: "promedio_regional";
      NO_es_mapa_termico_detallado: true;
      fuente: string;
    };
  }>;
}

/** Formato YYYY-MM, sin interpolar fechas ni fabricar meses faltantes. */
function validateMonth(month: string): string {
  if (!/^\d{4}-(0[1-9]|1[0-2])$/.test(month)) {
    throw new Error("Mes inválido: usa YYYY-MM");
  }
  return month;
}

export class NinoprobAPI {
  private readonly baseUrl: string;

  constructor(baseUrl = "http://127.0.0.1:8000") {
    this.baseUrl = baseUrl.replace(/\/+$/, "");
  }

  private async request<T>(path: string): Promise<T> {
    const response = await fetch(`${this.baseUrl}${path}`, {
      method: "GET",
      headers: { Accept: "application/json" },
    });
    if (!response.ok) {
      const error = await response.text();
      throw new Error(`Error NIÑOPROB-IA ${response.status}: ${error}`);
    }
    return (await response.json()) as T;
  }

  async metadatos(): Promise<Metadatos> {
    return this.request<Metadatos>("/api/v1/metadata");
  }

  async serie(desde = "1950-01", hasta = "2025-12"): Promise<SerieTemporal> {
    return this.request<SerieTemporal>(
      `/api/v1/serie?desde=${validateMonth(desde)}&hasta=${validateMonth(hasta)}`
    );
  }

  async prediccion(mesOrigen: string): Promise<Prediccion> {
    return this.request<Prediccion>(
      `/api/v1/prediccion/${validateMonth(mesOrigen)}`
    );
  }

  async region(mes: string): Promise<RegionNinoGeoJSON> {
    return this.request<RegionNinoGeoJSON>(
      `/api/v1/region.geojson?mes=${validateMonth(mes)}`
    );
  }

  async metricas(): Promise<unknown> {
    return this.request<unknown>("/api/v1/metricas");
  }
}
