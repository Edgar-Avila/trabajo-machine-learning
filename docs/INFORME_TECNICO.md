---
title: "Informe técnico: Customer Churn Prediction"
author: ["Avila Edgar", "Condori Rodrigo", "Mamani Isaac"]
date: "09/12/2026"
subject: "Informe técnico"
keywords: [Informe, Customer Churn, ML]
subtitle: "MCD202 - Machine Learning"
lang: "en"
titlepage: true,
titlepage-text-color: "FFFFFF"
titlepage-rule-color: "360049"
titlepage-rule-height: 0
titlepage-background: "background.pdf"
---

# Entrenamiento del modelo

## 1. Fuente de datos y carga inicial

El dataset está en `data/customer_churn_dataset-testing-master.csv`. Tiene 64 374 filas y 12 columnas. Cada fila describe a un cliente con diez características y la variable objetivo `Churn`. El valor 1 indica abandono.

`ml/seed.py` carga las primeras 20 000 filas del CSV a la base SQLite `db/churn.db`. El script borra la tabla `customers` y la vuelve a llenar. Así los datos de entrenamiento siempre reflejan la última fuente antes de una ejecución.

## 2. Ingreso incremental de clientes

`ml/ingest.py` simula el crecimiento de la base. Cada 5 minutos inserta la primera fila del CSV que todavía no está en la base, comparando por `CustomerID`. Cuando el CSV se agota, el proceso sigue vivo pero no inserta nada.

## 3. Características y preprocesamiento

`ml/features.py` fija el contrato de columnas. Siete son numéricas y tres son categóricas:

| Tipo | Columnas |
| --- | --- |
| Numéricas | `Age`, `Tenure`, `Usage Frequency`, `Support Calls`, `Payment Delay`, `Total Spend`, `Last Interaction` |
| Categóricas | `Gender`, `Subscription Type`, `Contract Length` |

`train.py` arma el preprocesamiento con `ColumnTransformer` y lo deja dentro del pipeline:

- `StandardScaler` para las columnas numéricas.
- `OneHotEncoder(handle_unknown="ignore")` para las categóricas.

`handle_unknown="ignore"` es deliberado. Si en producción llega una categoría que el modelo nunca vio, la codificación no rompe la inferencia.

`train.py` ajusta el pipeline completo y lo guarda con joblib. La API no repite el encoding por su cuenta; lo hereda del artefacto. Por eso la transformación en producción coincide con la de entrenamiento.

## 4. Algoritmo y partición de datos

El clasificador es `RandomForestClassifier` con estos parámetros:

| Parámetro | Valor |
| --- | --- |
| `n_estimators` | 200 |
| `max_depth` | 12 |
| `min_samples_leaf` | 5 |
| `n_jobs` | -1 |
| `random_state` | 42 |

Elegimos RandomForest por una razón práctica: da resultados sólidos en tablas de este tamaño y se entrena sin GPU. Los fijamos probando sobre la partición de entrenamiento; no hay una búsqueda exhaustiva detrás.

`train.py` divide los datos con `train_test_split`: 80% para entrenar y 20% para probar, con `random_state=42` y `stratify=y`. El estratificado conserva la proporción de churn en los dos conjuntos.

## 5. Métricas y resultados

`train.py` calcula dos métricas sobre el conjunto de prueba:

- Accuracy, con umbral de 0.5 sobre la probabilidad predicha.
- ROC AUC, sobre la probabilidad directa.

En la prueba más reciente se entrenó con 16 004 filas. Dio accuracy de 0.9945 y ROC AUC de 0.99995.

Al principio se probó con menos filas, 1 600, y los resultados fueron más pobres: accuracy de 0.955. El salto llegó al pasar a 16 000 filas. Pero esas métricas miden el conjunto de prueba, que sale de la misma fuente que el entrenamiento. No hay validación cruzada ni análisis de errores por clase; queda como trabajo pendiente.

## 6. Versionado de artefactos

Cada ejecución que ve datos nuevos genera una versión nueva. La estructura por versión es:

- `model/<version>/model.pkl` — el pipeline serializado.
- `model/<version>/manifest.json` — hash de datos, filas, fecha y métricas.
- `model/current.txt` — puntero a la versión vigente.

Antes de entrenar, `train.py` calcula un hash md5 de las filas (características más objetivo). Si el hash coincide con el del último manifiesto, el entrenamiento se omite y se conserva la versión vigente. Eso ocurre cuando `ingest.py` ya no encuentra filas nuevas en el CSV.

La numeración de versiones es propia de cada entorno. El historial generado durante las pruebas no se arrastra a producción; ahí el conteo comienza de nuevo.

## 7. Reentrenamiento programado

`scheduler.py` corre en un ciclo de 12 horas (43 200 segundos). Cada ciclo ejecuta `train.py` y después `predict.py`. El primer ciclo corre al arrancar, de modo que la aplicación tiene modelo entrenado y predicciones desde el inicio.

El servicio `scheduler` del `docker-compose.yml` lo lanza así:

```powershell
python scheduler.py --db /db/churn.db --model-dir /opt/model --interval 43200
```

`predict.py` carga el modelo activo, calcula `predict_proba` para todos los clientes y guarda los resultados en la tabla `predictions`. Las predicciones se conservan por versión de modelo, así el historial de churn de cada cliente se puede rastrear entre versiones.

# Aplicación

## 1. Arquitectura de la aplicación

La capa de aplicación operacionaliza el modelo de predicción de abandono mediante una API REST construida con FastAPI y una interfaz web estática servida por la misma instancia. Esta capa no entrena el modelo: consume el pipeline serializado producido por el proceso de ML, valida las entradas de inferencia y presenta tanto estimaciones individuales como las predicciones masivas persistidas.

El flujo integra tres artefactos compartidos con los procesos de ML: la base SQLite `db/churn.db`, el directorio de modelos versionados `model/` y el archivo `model/current.txt`. El entrenamiento genera un pipeline que incluye preprocesamiento y clasificador; posteriormente, la predicción por lote calcula la probabilidad de churn de todos los clientes y la almacena en la tabla `predictions`, junto con la versión de modelo y la fecha de predicción. La aplicación consulta estos resultados, por lo que no recalcula ni modifica las predicciones persistidas.

![Diagrama de arquitectura](./assets/arquitectura.png)

La estructura bajo `/app` separa responsabilidades:

| Ruta | Responsabilidad |
| --- | --- |
| `main.py` | Punto de entrada ASGI. Construye la instancia FastAPI, registra rutas, manejadores globales de errores y publica el frontend. |
| `api/` | Contratos Pydantic en `schemas.py` y definición HTTP de los endpoints en `routes.py`. Los endpoints delegan la inferencia al servicio. |
| `services/` | Lógica de negocio independiente del transporte HTTP: carga e inferencia del modelo (`predictor.py`), clasificación de riesgo (`risk.py`) y consulta de clientes, agregados e historial de predicciones (`customers.py`). |
| `core/` | Configuración mediante variables de entorno (`config.py`) y excepciones/manejadores de errores comunes (`exceptions.py`). |
| `frontend/` | Interfaz estática: formulario de inferencia individual y página de clientes con tabla paginada, resumen agregado e historial por versión. |
| `tests/` | Pruebas unitarias y de endpoint. Usan un modelo mock, por lo que no requieren el artefacto real del equipo de ML. |
| `Dockerfile` | Definición de una imagen ejecutable de la aplicación. |

## 2. Datos, flujo de aplicación e integración con el modelo

La inferencia individual recibe las diez características utilizadas por el pipeline de entrenamiento, excluyendo `CustomerID`, que se conserva únicamente como identificador:

| Tipo | Características |
| --- | --- |
| Numéricas | `Age`, `Tenure`, `Usage Frequency`, `Support Calls`, `Payment Delay`, `Total Spend`, `Last Interaction` |
| Categóricas | `Gender`, `Subscription Type`, `Contract Length` |

`PredictorService` carga el artefacto con `joblib.load()`. Cuando se configura `MODEL_VERSIONS_DIR`, resuelve primero la versión indicada por `current.txt`; si este archivo no existe, selecciona el directorio numérico de versión más alto disponible. En ausencia de un directorio de versiones, utiliza `MODEL_PATH`. Para una solicitud válida, el servicio elimina `CustomerID`, construye un `pandas.DataFrame` de una fila con las características en el orden esperado y ejecuta `predict_proba`. La segunda posición de la salida binaria (`[0][1]`) se interpreta como la probabilidad de la clase positiva, `Churn`.

El artefacto generado por `ml/train.py` es un `Pipeline` de scikit-learn que encapsula el `ColumnTransformer` y `RandomForestClassifier`; por ello, el escalado de variables numéricas y la codificación one-hot de variables categóricas se ejecutan en la inferencia con la misma transformación usada en entrenamiento. La aplicación no reimplementa este procesamiento. De forma opcional, `FEATURE_LIST_PATH` permite proporcionar el contrato de columnas como una lista JSON o como un objeto `{"features": [...]}`; dicho contrato debe mantenerse consistente con los aliases y las validaciones de `api/schemas.py`.

Para la vista de clientes, la aplicación lee de SQLite la versión más reciente presente en `predictions`, une esa tabla con `customers` por `CustomerID` y ordena los resultados por probabilidad de churn descendente. Los agregados del dashboard —cantidad de clientes, probabilidad media, distribución por nivel de riesgo y medias por tipo de suscripción y duración de contrato— se calculan sobre esa misma versión. El historial de un cliente conserva las probabilidades generadas para cada versión y la diferencia respecto de la versión anterior. Por tanto, estas vistas representan resultados generados por `ml/predict.py`, no nuevas evaluaciones del modelo.

El flujo completo de aplicación es el siguiente:

1. El planificador entrena o conserva el pipeline vigente y ejecuta la predicción por lote sobre los registros de `customers`.
2. `ml/predict.py` persiste `CustomerID`, `ChurnProbability`, `ModelVersion` y `PredictedAt` en `predictions`.
3. Al iniciar, FastAPI carga el artefacto activo para atender inferencias individuales y crea el repositorio de consultas SQLite.
4. `POST /predict` valida el cliente, ejecuta el pipeline y devuelve la probabilidad y su nivel de riesgo.
5. `GET /customers`, `GET /customers/summary` y `GET /customers/{customer_id}/history` exponen, respectivamente, la última predicción masiva paginada, sus agregados y la evolución por versión; la interfaz web consume estos endpoints.

El nivel de riesgo es una regla de presentación posterior a la inferencia: `Low` para probabilidades menores de 0.35, `Medium` para valores desde 0.35 hasta menos de 0.65 y `High` para valores iguales o superiores a 0.65. Esta regla no altera la probabilidad producida por el modelo ni sustituye las métricas de evaluación descritas en la sección de entrenamiento.

Variables relevantes:

| Variable | Comportamiento |
| --- | --- |
| `MODEL_PATH` | Ruta de respaldo para un archivo joblib/pickle cuando no se usa un directorio de versiones. Por defecto apunta a `model/model.pkl` en la raíz del proyecto. |
| `MODEL_VERSIONS_DIR` | Directorio raíz de los modelos versionados; permite resolver el modelo activo mediante `current.txt`. |
| `DB_PATH` | Ubicación de la base SQLite que contiene `customers` y `predictions`; si no existe, las consultas de clientes devuelven colecciones vacías. |
| `FEATURE_LIST_PATH` | Ruta opcional al archivo JSON con el contrato y orden de columnas. |
| `ALLOW_FALLBACK_MODEL` | Si es `true` y no se carga el artefacto, activa `FallbackChurnModel`, un modelo heurístico determinista destinado exclusivamente a desarrollo. |

Con `ALLOW_FALLBACK_MODEL=false`, un artefacto ausente, ilegible o inválido impide inicializar el predictor y se informa un error de modelo no disponible. El fallback no es un modelo entrenado y no debe emplearse para decisiones de producción. En la configuración de Docker y Docker Compose se mantiene desactivado.

## 3. Endpoints de la API

### `GET /health`

Comprueba la disponibilidad de la aplicación y del servicio de inferencia. No recibe cuerpo de petición.

Respuesta exitosa (`200 OK`):

```json
{
  "status": "ok",
  "model_loaded": true,
  "model_source": "artifact",
  "model_version": "1"
}
```

`model_source` puede ser `artifact` si se cargó un artefacto, `fallback` si está activo el modelo local de demostración, o `injected` cuando se inyecta un modelo en pruebas. `model_version` se informa cuando el artefacto se resolvió desde el directorio versionado y, en los demás casos, puede ser `null`. El estado es `ok` mientras exista un modelo disponible; de lo contrario es `degraded`.

### `POST /predict`

Valida un cliente, llama al modelo y devuelve la probabilidad de churn junto con una banda de riesgo. Recibe JSON:

```json
{
  "CustomerID": "C-100",
  "Age": 35,
  "Gender": "Female",
  "Tenure": 12,
  "Usage Frequency": 15,
  "Support Calls": 2,
  "Payment Delay": 3,
  "Subscription Type": "Premium",
  "Contract Length": "Annual",
  "Total Spend": 650.5,
  "Last Interaction": 4
}
```

Respuesta exitosa (`200 OK`):

```json
{
  "customer_id": "C-100",
  "churn_probability": 0.78,
  "risk_level": "High"
}
```

La probabilidad se redondea a cuatro decimales en la respuesta. La clasificación implementada es: `Low` para valores menores de 0.35, `Medium` desde 0.35 hasta menos de 0.65 y `High` desde 0.65.

### `GET /customers`

Devuelve la última versión disponible de las predicciones persistidas, ordenada de mayor a menor probabilidad de churn. Recibe los parámetros opcionales `page` (mínimo 1; por defecto 1) y `page_size` (entre 1 y 100; por defecto 10). La respuesta incluye la página solicitada, el total de clientes y el número de páginas. Cada elemento conserva las características del cliente, la probabilidad, el nivel de riesgo derivado y la versión de modelo.

Cuando la base de datos no existe o aún no hay predicciones, devuelve una respuesta paginada vacía; no realiza inferencia en línea. Por ejemplo:

```json
{
  "items": [],
  "page": 1,
  "page_size": 10,
  "total": 0,
  "total_pages": 0
}
```

### `GET /customers/summary`

Calcula los indicadores del dashboard para la versión más reciente: total de clientes, probabilidad media de churn, conteos y porcentajes por nivel de riesgo, y medias de probabilidad agrupadas por `Subscription Type` y `Contract Length`. Cuando no existen predicciones devuelve valores y listas vacías.

### `GET /customers/{customer_id}/history`

Devuelve el historial de probabilidades almacenadas para un cliente a través de las versiones de modelo. Cada registro incluye versión, probabilidad, nivel de riesgo, fecha de predicción y cambio de probabilidad frente a la versión previa; también incluye la etiqueta `Churn` almacenada en `customers` cuando está disponible. Si el cliente no tiene predicciones persistidas, responde `404`.

## 4. Validación y manejo de errores

Pydantic valida el cuerpo de `POST /predict` antes de llamar al modelo. Los campos requeridos son todos los del ejemplo de cliente. Las principales reglas son:

| Campo | Regla implementada |
| --- | --- |
| `CustomerID` | Texto obligatorio, de 1 a 128 caracteres. |
| `Age` | Entero entre 18 y 120. |
| `Gender` | Exactamente `Male` o `Female`. |
| `Tenure` | Entero entre 0 y 1200. |
| `Usage Frequency`, `Support Calls` | Enteros entre 0 y 10000. |
| `Payment Delay`, `Last Interaction` | Enteros entre 0 y 3650. |
| `Subscription Type`, `Contract Length` | Texto obligatorio; tras quitar espacios no puede estar vacío y tiene máximo 100 caracteres. |
| `Total Spend` | Número entre 0 y 10 000 000. |

Los manejadores globales devuelven JSON uniforme:

| Código | Significado |
| --- | --- |
| `200` | Petición procesada correctamente. |
| `422` | Falló la validación Pydantic. La respuesta contiene `detail: "Input validation failed"` y una lista `errors` con los campos y reglas incumplidos. |
| `500` | No hay modelo disponible, el modelo no produjo una predicción válida o ocurrió un error interno inesperado. Se devuelve `detail` con una descripción segura del problema. |

La aplicación usa el módulo estándar `logging` para registrar errores de modelo, errores de inferencia y errores inesperados; no usa `print`.

## 5. Pruebas de funcionamiento

La suite se ejecuta desde `/app`:

```powershell
python -m pytest -q
```

Los casos actuales son:

| Archivo | Cobertura |
| --- | --- |
| `test_predict_endpoint.py` | `POST /predict` con cliente válido (200 y riesgo `High` con el mock) y con edad inválida (422 y mensaje de validación). |
| `test_health_endpoint.py` | `GET /health` informa estado `ok`, modelo cargado y fuente `injected`. |
| `test_predictor_service.py` | `PredictorService` crea el DataFrame y consume correctamente `predict_proba` del modelo mock. |
| `test_risk_classification.py` | Bandas Low/Medium/High, casos límite 0.35 y 0.65, y rechazo de probabilidad fuera de [0, 1]. |
| `conftest.py` | Cliente FastAPI, cliente de ejemplo y modelo mock reutilizables. |

Estas pruebas validan la API de inferencia, el contrato de entrada, la integración del servicio con un clasificador compatible y las reglas de riesgo, sin depender de `model.pkl`. La suite declarada contiene 11 pruebas. Los endpoints de consulta de clientes y dashboard dependen de la base SQLite poblada por el flujo de ML y no cuentan con pruebas automatizadas específicas en `app/tests`.

## 6. Despliegue con Docker

El `Dockerfile`:

1. Parte de `python:3.11-slim`.
2. Define variables de ejecución: salida Python sin búfer, `PORT=8000`, `MODEL_PATH=/opt/model/model.pkl` y fallback desactivado.
3. Establece `/opt/app` como directorio de trabajo.
4. Copia e instala `requirements.txt` sin caché de pip.
5. Copia el código de `/app`, expone el puerto 8000 y ejecuta Uvicorn enlazado a `0.0.0.0`.

Desde la raíz del repositorio:

```powershell
docker build -t churn-risk-app -f app/Dockerfile app
docker run --rm -p 8000:8000 -e MODEL_PATH=/opt/model/model.pkl -v "${PWD}/model:/opt/model:ro" churn-risk-app
```

El volumen monta el artefacto en modo solo lectura. Para producción se debe conservar `ALLOW_FALLBACK_MODEL=false`, especificar una ruta de modelo accesible y definir, si corresponde, `FEATURE_LIST_PATH`. `PORT` modifica el puerto de Uvicorn dentro del contenedor; si se modifica, también debe ajustarse el mapeo `-p`.

## 7. Guía de inicio para un integrante nuevo

1. Ubicarse en la carpeta de aplicación:

   ```powershell
   cd "D:\PROYECTO MAESTRIA\trabajo-machine-learning\app"
   ```

2. Crear y activar un entorno virtual:

   ```powershell
   python -m venv .venv
   .venv\Scripts\Activate.ps1
   ```

3. Instalar dependencias y crear configuración local:

   ```powershell
   pip install -r requirements.txt
   Copy-Item .env.example .env
   ```

4. Para desarrollo sin artefacto ML, mantener `ALLOW_FALLBACK_MODEL=true` en `.env`. Para usar el modelo real, establecer `MODEL_PATH` o `MODEL_VERSIONS_DIR` con una ruta accesible y definir `ALLOW_FALLBACK_MODEL=false`. Para consultar clientes y dashboard, configurar también `DB_PATH` hacia la base SQLite poblada.

5. Ejecutar pruebas:

   ```powershell
   python -m pytest -q
   ```

6. Iniciar el servidor:

   ```powershell
   uvicorn main:app --reload --port 8000
   ```

7. Abrir `http://localhost:8000` para la interfaz o `http://localhost:8000/docs` para probar la API mediante OpenAPI.

Antes de conectar el artefacto ml, el equipo debe confirmar el orden exacto de features, categorías válidas y que la clase positiva de `predict_proba` corresponde realmente a churn.

# Despliegue de la aplicación

## 1. Composición de servicios

Toda la aplicación corre con Docker Compose. El archivo `docker-compose.yml` define cinco servicios:

| Servicio | Rol |
| --- | --- |
| `db_seed` | Carga el CSV en SQLite una sola vez. Toma las primeras 20 000 filas. |
| `ingest` | Inserta un cliente nuevo cada 5 minutos, simulando el crecimiento de la base. |
| `scheduler` | Entrena cada 12 horas y predice en lote sobre todos los clientes. |
| `app` | API FastAPI y frontend. Escucha en el puerto 8000 dentro de la red interna. |
| `caddy` | Reverse proxy con HTTPS automático. |

Las dependencias están ordenadas. `db_seed` termina primero, de ahí dependen `ingest` y `scheduler`, y `app` arranca cuando el `scheduler` ya está corriendo. Caddy queda delante de todos.

`app`, `scheduler` e `ingest` montan los mismos volúmenes: `./model` y `./db`. Todos ven el mismo artefacto y los mismos datos.

## 2. Caddy como reverse proxy

El `Caddyfile` es corto:

```text
{$DOMAIN} {
    encode zstd gzip
    reverse_proxy app:8000
}
```

Caddy escucha en los puertos 80 y 443. Genera el certificado TLS por su cuenta con Let's Encrypt, comprime las respuestas con zstd o gzip y reenvía el tráfico al contenedor `app`. El dominio se pasa por la variable `CADDY_DOMAIN`.

El puerto 8000 de la aplicación no se publica hacia afuera. Solo Caddy queda expuesto; el resto de servicios vive en la red interna de Compose. `CADDY_LISTEN` controla la interfaz de escucha y por defecto apunta a `127.0.0.1` para pruebas locales.

## 3. Servidor y dominio

El despliegue corre en un VPS de Hetzner con Docker. El dominio se registró en no-ip. El hostname se pasa a Compose con `CADDY_DOMAIN` y Caddy lo usa para emitir el certificado y responder por HTTPS. La aplicación queda accesible en `https://customer-churn.ddns.net/`.

# Pruebas de funcionamiento

La suite automatizada con pytest se describe en la sección Aplicación, punto 5. Aparte de eso, se probó el sistema completo ya desplegado contra la API y la interfaz web:

- `GET /health` responde `200` con estado `ok` y el modelo cargado desde el artefacto.
- La interfaz web recibe el formulario, llama al endpoint y muestra la predicción en la tabla de la sesión.

![Prueba del endpoint de salud](./assets/prueba_health.png)

![Prueba de la interfaz web](./assets/prueba_frontend.png)

![Tabla de customers con churn](assets/20260912-tabla-customers-con-churn.png)

![Dashboard](assets/20260912-dashboard.png)
