"""Carga, valida, limpia y extrae features de telemetría (A.1 y A.2).

Ejecutar: python src/features.py
Ventanas de 30 segundos: python src/features.py --ventana-segundos 30
"""

import argparse
from pathlib import Path
import pandas as pd
from pandera.errors import SchemaErrors

if __package__:
    from .schema import validar_telemetria
else:
    from schema import validar_telemetria
RUTA_DATASET = Path(__file__).resolve().parents[1] / "data" / "telemetria_publica (1).csv"
SENALES = ("temp_c", "power_w", "util_pct", "clock_mhz", "ecc_errors")
ESTADISTICAS = ("mean", "std", "min", "max")
COLUMNAS_FEATURES = [
    f"{senal}_{estadistica}"
    for senal in SENALES
    for estadistica in ESTADISTICAS
] + ["ecc_errors_total", "power_w_range"]


def cargar_telemetria(ruta: str | Path = RUTA_DATASET) -> pd.DataFrame:
    """Lee únicamente el CSV indicado; por defecto, la telemetría pública."""
    datos = pd.read_csv(ruta)
    requeridas = {"episodio_id", "segundo", "estado", *SENALES}
    faltantes = requeridas.difference(datos.columns)
    if faltantes:
        raise ValueError(f"Faltan columnas: {', '.join(sorted(faltantes))}")
    return datos


def validar_y_limpiar_telemetria(
    datos: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Valida, descarta solo potencias no positivas y vuelve a validar.

    Cualquier otro error de Pandera detiene el proceso. Devuelve los datos
    válidos y las lecturas descartadas, sin modificar el DataFrame original.
    """
    try:
        return validar_telemetria(datos), datos.iloc[0:0].copy()
    except SchemaErrors as error:
        fallos = error.failure_cases
        solo_potencia = (
            not fallos.empty
            and fallos["column"].eq("power_w").all()
            and fallos["check"].eq("greater_than(0)").all()
        )
        if not solo_potencia:
            raise

    potencia_no_positiva = pd.to_numeric(datos["power_w"], errors="coerce") <= 0
    descartadas = datos.loc[potencia_no_positiva].copy()
    datos_limpios = datos.loc[~potencia_no_positiva].copy()
    return validar_telemetria(datos_limpios), descartadas


def calcular_features(lecturas: pd.DataFrame) -> pd.DataFrame:
    """Devuelve una fila de features de una ventana, sin exigir etiqueta ni IDs.

    Esta función se reutilizará en entrenamiento y API. La desviación es
    poblacional (ddof=0): una ventana de una lectura tiene desviación cero.
    """
    if lecturas.empty:
        raise ValueError("La ventana debe contener al menos una lectura.")
    senales = lecturas.loc[:, list(SENALES)]
    if senales.isna().any().any():
        raise ValueError("Las señales no pueden contener valores faltantes.")
    resumen = senales.agg(["mean", "min", "max"])
    resumen.loc["std"] = senales.std(ddof=0)
    features = {
        f"{senal}_{estadistica}": resumen.loc[estadistica, senal]
        for senal in SENALES
        for estadistica in ESTADISTICAS
    }
    features["ecc_errors_total"] = senales["ecc_errors"].sum()
    features["power_w_range"] = senales["power_w"].max() - senales["power_w"].min()
    return pd.DataFrame([features], columns=COLUMNAS_FEATURES)


def generar_features(
    telemetria: pd.DataFrame, ventana_segundos: int = 300
) -> pd.DataFrame:
    """Genera una fila por (episodio_id, ventana_id), con su etiqueta.

    Las ventanas son intervalos [k*N, (k+1)*N) según la columna segundo.
    Con N=300, el dataset público produce una fila por episodio. Se conservan
    las ventanas parciales. Los IDs son metadatos y estado es la etiqueta;
    para entrenar, seleccionar exclusivamente COLUMNAS_FEATURES como X.
    """
    if isinstance(ventana_segundos, bool) or not isinstance(ventana_segundos, int) or ventana_segundos <= 0:
        raise ValueError("ventana_segundos debe ser un entero positivo.")
    if telemetria[["episodio_id", "segundo", "estado"]].isna().any().any():
        raise ValueError("El episodio, el segundo y el estado son obligatorios.")
    segundos = telemetria["segundo"]
    if ((segundos < 0) | (segundos % 1 != 0)).any():
        raise ValueError("segundo debe contener enteros no negativos.")
    if telemetria.duplicated(["episodio_id", "segundo"]).any():
        raise ValueError("Hay segundos duplicados dentro de un episodio.")
    if (telemetria.groupby("episodio_id")["estado"].nunique() != 1).any():
        raise ValueError("Cada episodio debe tener un único estado.")

    datos = telemetria.assign(ventana_id=(segundos // ventana_segundos).astype(int))
    filas = []
    for (episodio_id, ventana_id), ventana in datos.groupby(
        ["episodio_id", "ventana_id"], sort=True
    ):
        features = calcular_features(ventana).iloc[0].to_dict()
        filas.append({
            "episodio_id": episodio_id,
            "ventana_id": ventana_id,
            **features,
            "estado": ventana["estado"].iloc[0],
        })
    return pd.DataFrame(
        filas, columns=["episodio_id", "ventana_id", *COLUMNAS_FEATURES, "estado"]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--entrada", type=Path, default=RUTA_DATASET)
    parser.add_argument("--ventana-segundos", type=int, default=30)
    parser.add_argument("--salida", type=Path, help="CSV opcional con las features.")
    args = parser.parse_args()
    telemetria = cargar_telemetria(args.entrada)
    datos_validos, descartadas = validar_y_limpiar_telemetria(telemetria)
    features = generar_features(datos_validos, args.ventana_segundos)
    print(f"Lecturas cargadas: {len(telemetria)}")
    print(f"Lecturas descartadas: {len(descartadas)}")
    if not descartadas.empty:
        print(descartadas[["episodio_id", "segundo", "power_w"]].to_string(index=False))
    print(f"Lecturas validadas: {len(datos_validos)}")
    print(f"Episodios: {datos_validos['episodio_id'].nunique()}")
    print(f"Registros generados: {len(features)}; features: {len(COLUMNAS_FEATURES)}")
    print(features.head().to_string(index=False))
    if args.salida:
        features.to_csv(args.salida, index=False)
        print(f"Features guardadas en: {args.salida.resolve()}")


if __name__ == "__main__":
    main()
