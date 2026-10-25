# NYX 0.2 — Crimson Intelligence System

Asistente local para Windows 11 (Ryzen 5 4600G, 16 GB RAM), diseñado para crecer hacia otros equipos y servidores. La rama 0.2 conserva los archivos originales y la arquitectura Docker 0.1.

## Funciones implementadas
- **Interfaz Crimson 2.0:** paneles de terminal negra y carmesí, núcleo animado que refleja el estado real del asistente, contexto lateral, adaptación a móvil y modo reducido de animaciones.
- **Bitácora por sesiones:** nueva conversación, selección, búsqueda, cambio de título, eliminación y exportación TXT. Los mensajes previos de 0.1 se migran sin borrarlos.
- **Modelo local:** Qwen2.5 3B con identidad personalizada nyx:0.2. La información del creador no aparece en interfaz ni saludos; solo se responde cuando se pregunta explícitamente.
- **Memoria explícita y tareas:** SQLite persistente con volumen Docker independiente.
- **Lectura completa:** respuestas largas se sintetizan en segmentos (máximo 850 caracteres por solicitud), se reproducen en orden, incluyen indicador `LECTURA x/y` y admiten interrupción inmediata. Se precarga el siguiente segmento mientras se reproduce el actual.
- **Voz LOCAL optativa:** Piper, voz masculina española davefx-medium (modelo CC0), con modo natural y procesamiento de alertas mediante Web Audio. La onda se alimenta de la señal real reproducida o capturada.
- **Micrófono local optativo:** Vosk modelo pequeño en español; se activa manualmente, transcribe localmente y pone el texto reconocido en el editor para revisión (no lo envía automáticamente).
- **Respaldo:** volúmenes de modelos, memoria y voz; se preservan fuera de GitHub.

No incluye clonación de voz ni imitación exacta de ningún actor. No se ha entrenado desde cero el modelo Qwen: Modelfile define instrucciones, no pesos nuevos.

## Requisitos
- Windows 11, Docker Desktop + WSL 2. También Linux con Compose.
- RAM de Docker suficiente; se probará en CPU. Los modelos de voz se cargan bajo demanda.
- Acceso a Internet durante la descarga inicial de imágenes y modelos.

## Actualizar desde NYX 0.1 SIN PERDER EL HISTORIAL

**Haz una copia primero:**

    .\scripts\backup.ps1

El respaldo detiene momentáneamente los servicios. Conviene copiar la carpeta backups/ a almacenamiento externo protegido. NO publiques respaldos, voces ni bases de datos en GitHub.

Para cambiar de rama desde la copia local del repositorio:

    git fetch origin
    git switch feature/nyx-0.2-crimson

Comprueba los cambios y arranca:

    docker compose up -d --build
    docker compose exec ollama ollama create nyx:0.2 -f /config/Modelfile
    docker compose ps

Abre **http://localhost:3000**.

El modelo base qwen2.5:3b ya instalado en el volumen nyx_models se reutiliza; si no está presente:

    docker compose exec ollama ollama pull qwen2.5:3b
    docker compose exec ollama ollama create nyx:0.2 -f /config/Modelfile

**Nota:** la migración SQL se ejecuta al iniciar el API y agrupa los mensajes antiguos en "Conversaciones anteriores (0.1)" de forma que los registros se conserven.

## Activar voz natural, alertas robotizadas y micrófono

Instala las voces de forma expresa, una sola vez:

    docker compose exec api python -m app.voice_setup

O individualmente:

    docker compose exec api python -m app.voice_setup --tts
    docker compose exec api python -m app.voice_setup --stt

- TTS: Piper es_ES-davefx-medium (aprox. 63 MB, voz masculina de España).
- STT: Vosk vosk-model-small-es-0.42 (aprox. 39 MB comprimido).
- No se envía el audio a servicios externos; solo se usa Internet para descargar modelos.
- Los archivos descargados se guardan en nyx_voice.
- Tras instalar, pulsa **SINCRONIZAR** en la interfaz para actualizar el estado. En Configuración puedes probar la voz y una alerta.

**Flujo:** botón de micrófono → captura con permiso del navegador → audio WAV local → Vosk → texto editable → envío explícito → Ollama → texto → Piper → reproducción → onda real.

**Importante:** Piper y Vosk son software de terceros con licencias independientes. Piper-tts se distribuye bajo GPL-3.0 o posterior; el dataset de la voz davefx figura como CC0. Considera las obligaciones de licencia antes de redistribuir el programa.

## Privacidad y seguridad

- Solo expuesto en localhost:3000. NO publicar el servicio a Internet sin autenticación.
- El micrófono requiere autorización del navegador y está desactivado por defecto.
- No existe escucha permanente ni automatización de comandos del sistema.
- Los datos privados están en volúmenes Docker, ignorados por Git.
- El contenido de los recuerdos no debe incluir contraseñas ni credenciales.
- Para conservar los datos, NO ejecutes docker compose down -v.
- Los archivos históricos originales permanecen intactos.

## Verificación

En Windows 11:

    docker compose ps
    docker compose logs --tail=80 api

Pruebas Python (desde la raíz, con FastAPI y httpx disponibles):

    python -m unittest discover -s tests -v

La compilación completa y la calidad/performance de las voces deben validarse en el Ryzen 5 real antes de fusionar a main. Consulta docs/ARQUITECTURA.md.
