"""Entrena, evalúa por episodio y serializa el clasificador de fallas GPU.

Ejecutar desde la raíz del proyecto: python src/train.py
"""

import argparse
from pathlib import Path

import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

if __package__:
    from .features import (
        COLUMNAS_FEATURES,
        RUTA_DATASET,
        cargar_telemetria,
        generar_features,
        validar_y_limpiar_telemetria,
    )
else:
    from features import (
        COLUMNAS_FEATURES,
        RUTA_DATASET,
        cargar_telemetria,
        generar_features,
        validar_y_limpiar_telemetria,
    )


RUTA_MODELO = Path(__file__).resolve().parents[1] / "models" / "modelo.joblib"


def crear_pipeline() -> Pipeline:
    """Agrupa el preprocesamiento y el clasificador en un único objeto."""
    return Pipeline(
        steps=[
            ("imputar", SimpleImputer(strategy="median")),
            (
                "clasificador",
                RandomForestClassifier(
                    n_estimators=300,
                    class_weight="balanced",
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )


def entrenar(
    entrada: str | Path = RUTA_DATASET,
    salida: str | Path = RUTA_MODELO,
    ventana_segundos: int = 30,
) -> Pipeline:
    """Prepara ventanas, evalúa con episodios separados y guarda el modelo final."""
    datos = cargar_telemetria(entrada)
    datos_validos, descartadas = validar_y_limpiar_telemetria(datos)
    ventanas = generar_features(datos_validos, ventana_segundos=ventana_segundos)

    # Cada episodio tiene una etiqueta. La partición se hace con sus IDs para
    # que ninguna ventana de evaluación comparta episodio con entrenamiento.
    episodios = ventanas[["episodio_id", "estado"]].drop_duplicates()
    ids_entrenamiento, ids_evaluacion = train_test_split(
        episodios["episodio_id"],
        test_size=0.25,
        random_state=42,
        stratify=episodios["estado"],
    )
    es_entrenamiento = ventanas["episodio_id"].isin(ids_entrenamiento)
    es_evaluacion = ventanas["episodio_id"].isin(ids_evaluacion)

    # Solo las estadísticas entran a X: nunca estado ni identificadores.
    x_entrenamiento = ventanas.loc[es_entrenamiento, COLUMNAS_FEATURES]
    y_entrenamiento = ventanas.loc[es_entrenamiento, "estado"]
    x_evaluacion = ventanas.loc[es_evaluacion, COLUMNAS_FEATURES]
    y_evaluacion = ventanas.loc[es_evaluacion, "estado"]

    modelo_evaluacion = crear_pipeline()
    modelo_evaluacion.fit(x_entrenamiento, y_entrenamiento)
    predicciones = modelo_evaluacion.predict(x_evaluacion)
    print(f"Lecturas cargadas: {len(datos)}; descartadas: {len(descartadas)}")
    print(f"Ventanas: {len(ventanas)}; segundos por ventana: {ventana_segundos}")
    print(
        f"Episodios para evaluar: {len(ids_evaluacion)} de {len(episodios)}; "
        f"ventanas evaluadas: {len(x_evaluacion)}"
    )
    print(f"Exactitud en episodios separados: {accuracy_score(y_evaluacion, predicciones):.3f}")
    print(classification_report(y_evaluacion, predicciones, zero_division=0))

    # La evaluación queda terminada. El artefacto final aprende de todos los
    # episodios disponibles y conserva los mismos pasos de preprocesamiento.
    modelo_final = crear_pipeline()
    modelo_final.fit(ventanas[COLUMNAS_FEATURES], ventanas["estado"])
    ruta_salida = Path(salida)
    ruta_salida.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(modelo_final, ruta_salida)
    print(f"Pipeline guardado en: {ruta_salida.resolve()}")
    return modelo_final


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--entrada", type=Path, default=RUTA_DATASET)
    parser.add_argument("--salida", type=Path, default=RUTA_MODELO)
    parser.add_argument("--ventana-segundos", type=int, default=30)
    args = parser.parse_args()
    entrenar(args.entrada, args.salida, args.ventana_segundos)


if __name__ == "__main__":
    main()
