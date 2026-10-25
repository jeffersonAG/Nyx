# NYX 0.6.0 — Cognitive Brain v1 (Python, conocimiento con procedencia)

> **Estado:** implementación en la rama \`feature/nyx-0.2-crimson\`, aún pendiente de aceptación manual en Windows/Docker. No implica entrenamiento de pesos neuronales.

## Objetivo

Ampliar el núcleo cognitivo de NYX sin reemplazar el modelo Qwen2.5:3B ni modificar los otros módulos. NYX ahora puede conservar experiencias evaluadas, recuperar conocimiento de conversaciones anteriores y mantener relaciones justificadas con evidencia. La evaluación de los resultados afecta la **selección y recuperación de procedimientos**; no edita los pesos de Qwen.

## Arquitectura

\`\`\`text
                 Usuario autenticado (React)
                           |
                  FastAPI (permisos)
                           |
             +-------------+------------------------+
             |                                      |
    Cognition Python                         Cognitive Brain Python
    (orquestación/decisiones)                (brain.py + SQLite)
             |                                      |
             |                  +-------------------+----------------------+
             |                  |                   |                      |
        Ollama Qwen        Casos y resultados   Relaciones con        Feedback:
        2.5:3B             reportados           evidencia            útil/no útil
        (respuesta)             |                   |                      |
             |                  +-------------------+----------------------+
             |                                      |
             |                      BM25 (Python) + embeddings opcionales
             |                                      |
             |                     Ollama embeddinggemma:300m (local)
             +--------------------------|-----------+
                                        |
                          Contexto documentado, breve, etiquetado
                                        |
                        Qwen sintetiza y explica (no actúa solo)
\`\`\`

### Componentes reales, todos desarrollados en Python

- \`backend/app/brain.py\`: almacenamiento con validación, recuperador léxico BM25, vectores normalizados, similitud coseno, caché SQLite, valoración y gráfica de conocimiento.
- \`backend/app/cognition.py\`: integración del contexto recuperado con Qwen y las herramientas permitidas.
- \`backend/app/main.py\`: API autenticada \`/api/brain/*\` con CSRF y propietarios aislados.
- \`backend/app/brain_benchmark.py\`: benchmark repetible sobre una base **temporal y sintética**.
- \`frontend/src/BrainNotebook.jsx\`: registro y evaluación de experiencias, relaciones y recuperación; el chat incorpora 👍/corregir.

## Aprendizaje: qué significa realmente

**1. Memoria de experiencias.** Una experiencia tiene título, situación, procedimiento, resultado, evaluación y evidencia. \`FUNCIONÓ\`, \`FALLÓ\` o \`PARCIAL\` son **informes del usuario**, nunca pruebas científicas automáticas.

**2. Retroalimentación de conversación.** Los botones de respuesta útil/corregir crean evaluaciones asociadas al mensaje y la pregunta original. Una respuesta marcada incorrecta no se trata como ejemplo de solución: únicamente se recuperan el tema y el motivo de la corrección.

**3. Ejecución vs. éxito.** Cuando el backend confirma que creó una tarea o evento, registra \`executed\`. Eso **no** indica que se resolvió un problema del mundo real. No puede ascenderse a \`success\` sin un caso de resultado informado.

**4. Relaciones explícitas.** Dos experiencias pueden quedar vinculadas como \`related_to\`, \`supports\`, \`contradicts\` o \`prerequisite_for\`. El usuario debe escribir evidencia. Si incluye un enlace, el sistema **no lo verifica automáticamente**. Las relaciones sugeridas por coincidencias de palabras se marcan como **no verificadas** y no se insertan como hechos.

**5. Recuperación híbrida.** BM25 funciona sin modelos adicionales. Opcionalmente, \`embeddinggemma:300m\` genera vectores locales y NYX combina relevancia léxica y coseno. Los vectores sólo sirven para recuperar registros cercanos; no alteran Qwen. Indexación gradual de hasta ocho registros por petición para reducir la carga de CPU. Fuente del endpoint: https://github.com/ollama/ollama/blob/main/docs/api.md.

**6. Selección de procedimientos.** Para enfoques similares con evaluaciones, Python agrupa procedimientos iguales y ordena resultados mediante una estimación suavizada por evidencia (\`(éxitos + 0,5·parciales + 1) / (evaluados + 2)\`). No es una probabilidad calibrada ni medición de precisión científica; evita dar valor absoluto a una sola experiencia. Nunca ejecuta procedimientos automáticamente.

## Requisitos y recursos

- Conserva \`qwen2.5:3b\` dentro de Ollama; \`NYX_MODEL=nyx:0.4\` es una **etiqueta del modelo**, distinta de la versión \`NYX 0.6.0\` del programa.
- Los embeddings son optativos. \`embeddinggemma:300m\` requiere espacio adicional de descarga (aprox. 622 MB en el registro de Ollama) y consume RAM/CPU al indexar. Es un modelo orientado a varios idiomas, incluido español. No asumas el mismo rendimiento en CPU que una GPU.
- **Sin instalación:** todas las funciones de casos, comentarios, relaciones y recuperación léxica operan sin el modelo de embeddings.

## Instalación segura (PowerShell, raíz del proyecto)

1. Crear copia de seguridad antes de actualizar:

\`\`\`powershell
.\scripts\backup.ps1
git fetch origin
git switch feature/nyx-0.2-crimson
git pull --ff-only origin feature/nyx-0.2-crimson
docker compose up -d --build
\`\`\`

2. Abrir \`http://localhost:3000\`, usar Ctrl+F5, iniciar sesión y probar **MEMORIA → Cerebro**.

3. Activar el componente semántico **cuando lo decidas**:

\`\`\`powershell
.\scripts\enable_brain.ps1
\`\`\`

El script pide a Ollama descargar \`embeddinggemma:300m\`, guarda únicamente el nombre del modelo en el archivo local ignorado \`.env.brain\`, y recrea **solo** la API. No publica claves ni modifica \`.env.security\`. Si falla la descarga, no edita la configuración del cerebro. Se puede usar de nuevo para comprobar el modelo.

4. En **MEMORIA → Cerebro → Recuperar**, elige **INDIZAR 8 REGISTROS** y repite mientras existan pendientes. También hay indexación gradual durante las consultas. El estado informa cuándo el modelo local no está instalado.

5. Prueba un caso: registra «Corte del enlace inalámbrico», añade pasos de diagnóstico PoE, resultado, evidencias y evaluación; en otra conversación pregunta a NYX por experiencias sobre WiFi/PoE. Comprueba que indique el carácter **reportado**, no científicamente verificado. Califica la respuesta para producir feedback.

## Endpoints autenticados

| Método | Ruta | Acción |
|---|---|---|
| GET | \`/api/brain/status\` | Métricas, configuración y modelo instalado |
| GET/POST | \`/api/brain/cases\` | Ver o crear experiencias con evidencia |
| PATCH/DELETE | \`/api/brain/cases/{id}\` | Evaluar resultado o eliminar registro |
| GET/POST | \`/api/brain/relations\` | Consultar o vincular casos con evidencia |
| DELETE | \`/api/brain/relations/{id}\` | Borrar una relación |
| GET | \`/api/brain/suggestions/{id}\` | Asociaciones léxicas propuestas, no verificadas |
| GET/POST | \`/api/brain/feedback\` | Evaluaciones de respuestas de la cuenta |
| GET | \`/api/brain/recall?q=...\` | Búsqueda léxica/semántica con fuentes y estatus |
| GET | \`/api/brain/recommend?q=...\` | Ranking de procedimientos informados, no acciones |
| POST | \`/api/brain/index\` | Indexar un grupo pequeño de registros en Ollama local |

Todas las operaciones mutables requieren sesión autenticada y token CSRF. Las consultas y mutaciones a SQLite incluyen el \`owner_id\` de la cuenta. Los datos de investigación externos nunca adquieren permisos administrativos.

## Pruebas reproducibles

El repositorio ejecuta en GitHub Actions pruebas Python para: seguridad de sesiones, aislamiento de cuentas, CRUD, feedback real, grafo con evidencia, ejecución confirmada de herramientas, recuperación léxica y vectores simulados, y caída controlada del recuperador sin bloquear el chat.

Para una prueba sintética de recuperación **sin afectar tus datos**:

\`\`\`powershell
docker compose exec api python -m app.brain_benchmark
\`\`\`

Produce \`hit_at_3\` en cinco ejemplos inventados y latencia aproximada. **No** mide la precisión científica ni demuestra entrenamiento neuronal; sirve para detectar regresiones del algoritmo Python. Compara resultados reales con un conjunto de problemas de tu laboratorio antes de dar por mejorada la IA.

## Límites que persisten

1. Los pesos de Qwen2.5:3B no cambian por hablar, investigar, recibir feedback ni guardar embeddings. Reentrenamiento/LoRA es un proyecto separado, sujeto a hardware, dataset, métricas y autorización.
2. Una experiencia reportada puede ser incorrecta. NYX conserva la fuente y la incertidumbre; no determina automáticamente qué es una verdad científica.
3. La similitud vectorial es relevancia aproximada, **no** probabilidad de verdad ni prueba de causalidad.
4. La ventana de contexto de Qwen continúa limitada por su configuración; el cerebro selecciona un extracto breve para no saturarla.
5. No hay procesos autónomos que modifiquen redes, hagan experimentos ni ejecuten instrucciones recuperadas.
6. Los modelos adicionales de Ollama usan los recursos locales del equipo; no se descargan en segundo plano ni comparten datos por la nube.
7. El benchmark sintético y los modelos simulados en CI **no validan rendimiento semántico real** en Windows. Esa etapa necesita pruebas de aceptación en tu PC.
8. Los casos, relaciones y vectores tienen borrado explícito; las respuestas feedback se guardan por cuenta y no cambian automáticamente los mensajes originales.

## Archivos principales de esta iteración

\`backend/app/brain.py\`, \`backend/app/brain_benchmark.py\`,
\`backend/app/cognition.py\`, \`backend/app/main.py\`,
\`frontend/src/BrainNotebook.jsx\`, \`frontend/src/brain.css\`,
\`scripts/enable_brain.ps1\`, \`tests/test_brain*.py\`.

La rama \`main\` y la base \`feature/nyx-0.1-docker\` quedan intactas.
