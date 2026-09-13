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

La aplicación entrega predicciones individuales de abandono de clientes mediante una API REST construida con FastAPI y una interfaz web estática servida por la misma aplicación. La responsabilidad de esta capa es validar el contrato de entrada, convertirlo al formato de inferencia y presentar el resultado; no entrena ni reentrena el modelo.

![Diagrama de arquitectura](./assets/arquitectura.png)

La estructura bajo `/app` separa responsabilidades:

| Ruta | Responsabilidad |
| --- | --- |
| `main.py` | Punto de entrada ASGI. Construye la instancia FastAPI, registra rutas, manejadores globales de errores y publica el frontend. |
| `api/` | Contratos Pydantic en `schemas.py` y definición HTTP de los endpoints en `routes.py`. Los endpoints delegan la inferencia al servicio. |
| `services/` | Lógica de negocio independiente del transporte HTTP: carga e inferencia del modelo (`predictor.py`) y clasificación de riesgo (`risk.py`). |
| `core/` | Configuración mediante variables de entorno (`config.py`) y excepciones/manejadores de errores comunes (`exceptions.py`). |
| `frontend/` | Interfaz estática: formulario de cliente, llamada `fetch` al endpoint y tabla de predicciones realizadas durante la sesión del navegador. |
| `tests/` | Pruebas unitarias y de endpoint. Usan un modelo mock, por lo que no requieren el artefacto real del equipo de ML. |
| `Dockerfile` | Definición de una imagen ejecutable de la aplicación. |

## 2. Endpoints de la API

### `GET /health`

Comprueba la disponibilidad de la aplicación y del servicio de inferencia. No recibe cuerpo de petición.

Respuesta exitosa (`200 OK`):

```json
{
  "status": "ok",
  "model_loaded": true,
  "model_source": "artifact"
}
```

`model_source` puede ser `artifact` si se cargó el archivo configurado, `fallback` si está activo el modelo local de demostración, o `injected` cuando se inyecta un modelo en pruebas. El estado es `ok` mientras exista un modelo disponible; de lo contrario sería `degraded`.

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

Es un endpoint reservado para una futura lista de predicciones o clientes persistidos. Actualmente no recibe parámetros y devuelve siempre una lista vacía:

```json
[]
```

**Pendiente / trabajo futuro:** no existe todavía una base de datos, repositorio ni persistencia de clientes. La tabla del frontend conserva resultados solamente en el DOM de la sesión actual.

## 3. Integración con el modelo de ML

La clase `PredictorService` lee `MODEL_PATH` y carga el artefacto mediante `joblib.load()`. El objeto debe exponer `predict_proba(features)` y devolver una estructura binaria donde la columna/posición `[0][1]` es la probabilidad de churn. Se recomienda entregar un pipeline serializado que incluya el preprocesamiento, para que encoding y transformaciones se apliquen de la misma forma que durante el entrenamiento.

Antes de inferir, el servicio elimina `CustomerID` y crea un `pandas.DataFrame` de una fila. Por defecto, las columnas enviadas al modelo son:

```text
Age, Gender, Tenure, Usage Frequency, Support Calls, Payment Delay,
Subscription Type, Contract Length, Total Spend, Last Interaction
```

La ruta opcional `FEATURE_LIST_PATH` permite suministrar el orden/nombre de columnas como una lista JSON o como `{"features": [...]}`. Si el contrato del modelo cambia, se debe actualizar conjuntamente esa lista y los aliases/validaciones de `api/schemas.py`.

Variables relevantes:

| Variable | Comportamiento |
| --- | --- |
| `MODEL_PATH` | Ubicación del archivo joblib/pickle. Localmente el valor por defecto apunta a `../model/model.pkl` respecto a la carpeta `app`. |
| `ALLOW_FALLBACK_MODEL` | Si es `true` y no se puede cargar el artefacto, activa `FallbackChurnModel`, un modelo heurístico determinista para desarrollo. |
| `FEATURE_LIST_PATH` | Ruta opcional al archivo JSON con el contrato de columnas. |

Con `ALLOW_FALLBACK_MODEL=false`, un archivo ausente, ilegible o inválido impide inicializar el predictor y se genera un error de modelo no disponible. El fallback no es un modelo entrenado y no debe usarse para decisiones de producción. El Dockerfile lo desactiva de forma predeterminada.

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

El resultado esperado es una suite exitosa sin depender de `model.pkl`. En la versión implementada se obtienen 11 pruebas aprobadas.

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

4. Para desarrollo sin artefacto ML, mantener `ALLOW_FALLBACK_MODEL=true` en `.env`. Para usar el modelo real, establecer `MODEL_PATH` con su ruta y preferiblemente definir `ALLOW_FALLBACK_MODEL=false`.

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
