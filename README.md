# Detector de fallas de GPU
### Andres Felipe Araque Guerrero

API de FastAPI que clasifica ventanas de telemetría como `normal`,
`sobrecalentamiento`, `degradacion_memoria` o `falla_alimentacion`.

## Construir y ejecutar con Docker

El modelo entrenado `models/modelo.joblib` ya está incluido en el proyecto.
Desde la raíz del repositorio:

```bash
docker build -t detector-gpu .
docker run -p 8000:8000 detector-gpu
```

Abre <http://localhost:8000/docs> para probar `POST /predecir`. La petición
debe contener `lecturas`: una lista de al menos 10 mediciones. Cada medición
incluye `temp_c`, `power_w`, `util_pct`, `clock_mhz` y `ecc_errors`. La respuesta
contiene `estado_predicho` y `confianza`. Una petición inválida devuelve 422.

La imagen copia el código y el modelo ya entrenado. Los CSV de `data/` se usan
para entrenar localmente y quedan fuera de la imagen.

## Volver a entrenar

Si cambias los datos o las features, instala las dependencias y genera un
nuevo modelo antes de reconstruir la imagen:

```bash
python -m pip install -r requirements.txt
python src/train.py
docker build -t detector-gpu .
```

El entrenamiento lee `data/telemetria_publica (1).csv`, valida los registros,
descarta las lecturas de potencia no positiva y genera ventanas de 30 segundos.
El modelo final se guarda en `models/modelo.joblib`.

## Cómo funciona el proyecto

El objetivo es clasificar el estado de una GPU a partir de una **ventana de
telemetría**, no de una lectura aislada. El CSV público contiene 48 episodios
de 300 segundos; cada episodio tiene una etiqueta `estado`. El flujo de
entrenamiento es:

1. `src/features.py` carga el CSV desde `data/`. `src/schema.py` comprueba con
   Pandera los campos, tipos, rangos y etiquetas. El CSV contiene tres lecturas
   con potencia no positiva: se descartan solo esas lecturas y se vuelve a
   validar. Ningún otro error se omite.
2. `src/features.py` agrupa las lecturas por `episodio_id` y por intervalos de
   30 segundos. Nunca une episodios distintos ni desplaza lecturas entre
   ventanas. Se obtienen 480 ventanas; las tres afectadas por la limpieza
   conservan 29 lecturas.
3. Cada ventana se resume en 22 características: media, desviación estándar,
   mínimo y máximo de las cinco señales, más el total de errores ECC y el
   rango de potencia. Hereda el `estado` de su episodio. `episodio_id`,
   `ventana_id` y `estado` no se usan como entradas del modelo.
4. `src/train.py` separa episodios completos para evaluar, de modo que las
   ventanas de un mismo episodio no aparezcan a la vez en entrenamiento y
   evaluación. Usa un pipeline de scikit-learn con imputación por mediana y
   `RandomForestClassifier`. Después de evaluar, ajusta el pipeline final con
   todas las ventanas y lo guarda con Joblib en `models/modelo.joblib`.

En la partición local de evaluación se acertaron 120 de 120 ventanas de 12
episodios reservados. Este resultado describe esa partición; no garantiza el
mismo rendimiento en datos nuevos.

## Qué hace la API

`src/api.py` carga el `.joblib` una sola vez al iniciar. En `POST /predecir`,
Pydantic revisa que la petición tenga al menos 10 lecturas y que cada señal
tenga un valor válido. Si la entrada falla, FastAPI devuelve 422. Si pasa,
la API llama a la misma función `calcular_features()` usada en entrenamiento,
envía las 22 características al pipeline y devuelve el estado predicho y su
confianza estimada. El entrenamiento no se ejecuta al arrancar el contenedor
ni al recibir una petición.
