"""Contrato de datos de la telemetría cruda (punto A.2).

Uso antes de generar features:
    datos_validos = validar_telemetria(cargar_telemetria())

No corrige ni elimina datos inválidos. Los límites de temperatura son una
decisión del proyecto, no una especificación del fabricante.
"""

import numpy as np
import pandas as pd
import pandera.pandas as pa


ESTADOS_VALIDOS = (
    "normal",
    "sobrecalentamiento",
    "degradacion_memoria",
    "falla_alimentacion",
)
TEMP_MIN_C = -40.0
TEMP_MAX_C = 125.0


ESQUEMA_TELEMETRIA = pa.DataFrameSchema(
    {
        # Sin coerción de enteros: evita truncar valores fraccionarios corruptos.
        "episodio_id": pa.Column(int, pa.Check.ge(0)),
        "segundo": pa.Column(int, pa.Check.ge(0)),
        "temp_c": pa.Column(
            float, pa.Check.in_range(TEMP_MIN_C, TEMP_MAX_C), coerce=True
        ),
        "power_w": pa.Column(
            float,
            [pa.Check.gt(0), pa.Check(np.isfinite, name="potencia_finita")],
            coerce=True,
        ),
        "util_pct": pa.Column(float, pa.Check.in_range(0, 100), coerce=True),
        "clock_mhz": pa.Column(
            float,
            [pa.Check.gt(0), pa.Check(np.isfinite, name="frecuencia_finita")],
            coerce=True,
        ),
        "ecc_errors": pa.Column(int, pa.Check.ge(0)),
        "estado": pa.Column(str, pa.Check.isin(ESTADOS_VALIDOS)),
    },
    checks=[
        pa.Check(lambda df: not df.empty, name="telemetria_no_vacia"),
        pa.Check(
            lambda df: not df.duplicated(["episodio_id", "segundo"]).any(),
            name="segundo_unico_por_episodio",
        ),
        pa.Check(
            lambda df: df.groupby("episodio_id")["estado"].nunique().eq(1).all(),
            name="un_estado_por_episodio",
        ),
    ],
    strict=True,
    name="telemetria_gpu",
)


def validar_telemetria(datos: pd.DataFrame) -> pd.DataFrame:
    """Valida columnas, tipos, rangos, nulos y consistencia por episodio.

    Devuelve un DataFrame validado sin modificar el original. Si falla, lanza
    pandera.errors.SchemaErrors; su atributo failure_cases contiene el detalle
    de los errores acumulados gracias a lazy=True.
    """
    return ESQUEMA_TELEMETRIA.validate(datos, lazy=True, inplace=False)
