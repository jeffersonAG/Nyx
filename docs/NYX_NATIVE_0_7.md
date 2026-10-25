# NYX 0.7.0 — Neural Core nativo, construido desde cero

**Estado:** alfa de investigación en la rama `feature/nyx-0.2-crimson`. No sustituye todavía a Qwen en el chat diario. Los usuarios deciden explícitamente cuándo entrenar y qué textos utilizar.

## 1. Diferencia fundamental con NYX 0.6

En NYX 0.6 existía una memoria episódica y una recuperación por embeddings, pero el generador de lenguaje seguía siendo un Qwen preentrenado. **NYX Native 0.7 implementa en Python/PyTorch una red neuronal original cuyos pesos nacen aleatoriamente y se ajustan con datos elegidos por el propietario.** No se descarga, copia ni afina Qwen para este laboratorio.

```text
                      APLICACIÓN NYX
                            |
                 Backend FastAPI + seguridad
                      /             \
            Chat normal           Banco de pruebas
           Qwen/Ollama            modelo PROPIO
                                   (perfil native)
                                         |
                      +------------------+-------------------+
                      |                  |                   |
               Tokenizador bytes    Transformer          Entrenamiento
               UTF-8 original        causal              PyTorch
                      |                  |                   |
                      +------------------+                   |
                                         |                    |
                                   Predicción de        Retropropagación
                                   siguiente token       + AdamW
                                         |                    |
                                         +--- pesos propios --+
                                              /models/*.pt
```

### Implementación real

- `native/nyx_native/tokenizer.py`: codificación UTF-8 byte a byte, 259 tokens con BOS/EOS/PAD; sin vocabulario descargado.
- `native/nyx_native/model.py`: Transformer causal autorregresivo propio con embeddings de entrada y posición, atención multi-cabeza, máscara causal, conexiones residuales, LayerNorm, capas feed-forward GELU y salida vinculada a embeddings. Usa `torch.nn.functional.scaled_dot_product_attention`, pero **las capas y los pesos se definen aquí**.
- `native/nyx_native/dataset.py`: valida carpetas y documentos UTF-8; separa entrenamiento y validación **por documentos** (no por fragmentos mezclados).
- `native/nyx_native/train.py`: pérdida de entropía cruzada, `loss.backward()`, `AdamW.step()`, clipping de gradientes, calendario de tasa de aprendizaje, pérdida en documento reservado y semillas reproducibles.
- `native/nyx_native/checkpoint.py`: checkpoints atómicos, pesos, optimizador, configuración, hash del corpus, paso de entrenamiento y evaluación; carga con `weights_only=True`.
- `native/nyx_native/inference.py` y `serve.py`: generación causal de tokens en servicio interno aislado; jamás ejecuta herramientas ni sistemas externos.
- `native/nyx_native/curate.py`: construye documentos supervisados **sólo** a partir de preguntas y respuestas explícitamente aprobadas en JSONL. Omite datos no aprobados y ejemplos duplicados.
- `native/tests/test_native.py`: pruebas reales de PyTorch con gradientes, cambio de pesos, atención causal, reanudación, adaptación, tokenizer y recuperación del checkpoint.

## 2. Lo que es y lo que NO es

| Propiedad | Estado |
|---|---|
| Red neuronal definida desde sus fundamentos en Python | Implementada |
| Pesos propios inicializados sin modelo de terceros | Implementada |
| Entrenamiento con retropropagación real | Implementada |
| Evaluación con documentos reservados | Implementada |
| Persistencia, recarga y continuación de entrenamiento | Implementada |
| Adaptación con otros datos elegidos expresamente | Implementada; requiere supervisión |
| Comprensión avanzada de español y conocimiento científico general | **No validado; no esperable con corpus demostrativo** |
| Aprendizaje autónomo de experiencias sin revisión | **No implementado intencionadamente** |
| Entrenamiento de miles de millones de parámetros en CPU de 16 GB | **Fuera del alcance actual** |
| Sustituir la conversación cotidiana protegida por Qwen | **No**, se conserva sin cambios |

Este modelo utiliza un **tokenizador simple por bytes**. Es excelente para comprender y depurar la arquitectura, pero es menos eficiente que tokenizadores subword/BPE para español. El modelo estándar empieza con ~0,6 millones de parámetros, **no** tres mil millones como Qwen2.5:3B, y un contexto inicial de sólo 128 bytes/tokens. Los ejemplos incluidos son sintéticos, no un conjunto de entrenamiento suficientemente amplio para un asistente competente.

Los números de parámetros reales y las pérdidas aparecen al entrenar. La reducción de pérdida demuestra ajuste estadístico al corpus, **no** que NYX haya adquirido razonamiento científico general.

## 3. Instalar en la MISMA rama sin borrar datos

Desde la carpeta raíz del proyecto, PowerShell:

```powershell
.\scripts\backup.ps1
git fetch origin
git switch feature/nyx-0.2-crimson
git pull --ff-only origin feature/nyx-0.2-crimson
docker compose up -d --build
```

Se actualizan frontend/API existentes; el contenedor con PyTorch **no se inicia automáticamente**. Para activar sólo el laboratorio opcional:

```powershell
docker compose --profile native up -d --build native
```

Esto instala PyTorch CPU en un **contenedor aparte**. Los pesos persisten en el volumen Docker `nyx_native_weights`; el corpus privado está en `nyx_native_corpus`. El servicio nativo no publica ningún puerto en el host. No se toca `.env.security`, EmailJS ni el modelo Qwen.

## 4. Primer experimento real: entrenar pesos NUEVOS

```powershell
docker compose --profile native exec native python -m nyx_native.train --data-dir /samples --checkpoint /models/native.pt --steps 200 --batch-size 8 --context 128 --d-model 128 --layers 3 --heads 4 --cpu-threads 4
```

Este comando **sí entrena una red neuronal original**. El corpus incluido tiene siete documentos sintéticos en español; el último se reserva para validación. El programa muestra por pantalla valores como:

```text
{"step": 25, "train_loss": ..., "validation_loss": ..., "parameters": ...}
```

No se muestran métricas inventadas: son valores que calcula PyTorch al ejecutar la prueba. Cuanto menor la pérdida, mejor se ajustó el modelo al conjunto observado, pero **una pérdida menor no garantiza generalización**.

Después abre `http://localhost:3000`, inicia sesión como administrador, entra a **MEMORIA → Red neuronal propia** y comprueba el checkpoint. Para generar texto debes **autorizar temporalmente cambios administrativos en SEGURIDAD**. Los resultados son experimentales y **no** pasan por las mismas capacidades conversacionales y controles semánticos de Qwen.

Si deseas continuar el **mismo corpus** tras interrumpir el proceso:

```powershell
docker compose --profile native exec native python -m nyx_native.train --data-dir /samples --checkpoint /models/native.pt --steps 350 --batch-size 8 --context 128 --d-model 128 --layers 3 --heads 4 --cpu-threads 4 --resume
```

`--steps` es el total final de pasos, no pasos adicionales. No modifiques el corpus ni la arquitectura para reanudar: el programa comprueba la huella digital y las dimensiones.

## 5. Incorporar experiencias verificadas por ti al aprendizaje neuronal

Crear en tu PC un archivo privado `datos_aprobados.jsonl`, con **una experiencia aprobada por línea**, por ejemplo:

```json
{"approved":true,"question":"¿Cómo comprobar una caída de enlace Wi-Fi?","answer":"Revisar alimentación PoE, estado del puerto y realizar mediciones de conectividad.","evidence":"Procedimiento contrastado en pruebas documentadas."}
{"approved":false,"question":"Prueba no revisada","answer":"No utilizar hasta su revisión."}
```

El ejemplo es ilustrativo. Se necesitan **al menos diez preguntas/respuestas únicas aprobadas** con cinco documentos de salida; no copies historiales privados, claves, datos personales, registros laborales confidenciales ni información sin autorización. `approved:true` es una **declaración de revisión humana**, no verificación científica automática.

Copia el archivo explícitamente al volumen privado:

```powershell
docker compose --profile native cp .\datos_aprobados.jsonl native:/corpus/aprobados.jsonl
docker compose --profile native exec native python -m nyx_native.curate --input /corpus/aprobados.jsonl --output /corpus/lote_01 --shards 5
```

Entrena **otra versión** de los pesos partiendo de un checkpoint propio anterior:

```powershell
docker compose --profile native exec native python -m nyx_native.train --data-dir /corpus/lote_01 --continue-from /models/native.pt --checkpoint /models/native_lote01.pt --steps 300 --batch-size 8 --context 128 --d-model 128 --layers 3 --heads 4 --cpu-threads 4
```

Aquí sucede lo que buscabas: **los pesos previamente aprendidos se vuelven a modificar mediante ejemplos nuevos**. El modelo base permanece inalterado. La salida informa pérdida de validación inicial y posterior en **los mismos fragmentos reservados del nuevo corpus**. Si la evaluación empeora, no promociones el candidato. Incluso si mejora, debes contrastarlo con pruebas anteriores para detectar **olvido catastrófico**.

### Probar el candidato sin sobrescribir el original

Crea `.env.native` en la carpeta raíz (está excluido de Git):

```ini
NYX_NATIVE_CHECKPOINT=/models/native_lote01.pt
```

Reinicia únicamente el servicio experimental:

```powershell
docker compose --profile native up -d --force-recreate native
```

Ahora el laboratorio utilizará esa versión. Para volver al checkpoint original, cambia la variable a `/models/native.pt` y recrea el servicio. **No se reescriben pesos anteriores**.

## 6. Evaluación reproducible y seguridad

Las pruebas de GitHub Actions incluyen:
- Retropropagación y comprobación numérica de modificación de pesos.
- Causalidad: modificar el último token no altera predicciones de posiciones anteriores.
- Tokenizador propio, división documental, checkpoints, reanudación y adaptación.
- Dataset curado: no incluye ejemplos sin `approved:true`, elimina preguntas duplicadas.
- El resto del sistema conserva sus pruebas de cuentas, permisos, voz, memoria, tareas y seguridad.

Las pruebas de CI utilizan ejemplos pequeños para demostrar **corrección de la implementación**, no aptitud para reemplazar un asistente preentrenado.

**Restricciones de seguridad:** el laboratorio se ejecuta dentro de la red Docker; no publica puertos al host. El gateway de la API requiere permisos administrativos y autorización temporal para generar. No tiene acceso a herramientas de sistema ni al historial, y no entrena datos de NYX automáticamente. Las rutas de pesos y corpus permanecen en volúmenes Docker y no en GitHub. Nunca ejecutes `docker compose down -v`.

Para apagar el laboratorio sin eliminar pesos:

```powershell
docker compose --profile native stop native
```

## 7. Hoja de ruta para convertir el prototipo en un asistente útil

**Fase A — Fundamentos (esta entrega):** red, retropropagación, tokenización, train/val, checkpoints y generación independientes de Qwen.

**Fase B — Aprendizaje medible:** conjunto de evaluación en español/tecnología con preguntas inéditas, métricas de respuestas correctas, tareas de razonamiento y pruebas de regresión entre versiones.

**Fase C — Escalar la red propia:** tokenizador subword/BPE, corpus limpio con licencias y permisos, entrenamiento de mayor duración, optimización CPU/GPU, contextos mayores, cuantización e inferencia optimizada.

**Fase D — Aprendizaje supervisado de experiencias:** elaborar ejemplos aprobados con evidencia, usar adaptación por etapas, medir olvido y activar nuevas versiones únicamente cuando superen la anterior.

**Fase E — Sustitución del modelo de conversación:** sólo tras demostrar calidad suficiente y preservar permisos, seguridad, herramientas y privacidad. El tamaño de la red y la calidad de los datos imponen límites reales al hardware disponible.

### Fuentes técnicas de PyTorch

- [Instalación oficial PyTorch](https://pytorch.org/get-started/locally/)
- [Atención causal](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html)
- [Autograd](https://docs.pytorch.org/docs/stable/autograd.html)
- [AdamW](https://docs.pytorch.org/docs/stable/generated/torch.optim.AdamW.html)
