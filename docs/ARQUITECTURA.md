# Arquitectura de NYX 0.2

## Principios

- IA local portable.
- Negro, grafito y carmesí como identidad visual; interfaz útil y accesible.
- Privacidad del creador: no mostrar ni mencionar la autoría sin pregunta explícita.
- Persistencia de sesiones y memoria no equivale a entrenamiento de los pesos de un modelo.
- Interacciones con el micrófono siempre bajo control del usuario.
- Mantener la compatibilidad con hardware CPU Ryzen 5 4600G y los límites de Docker/WSL.

## Docker Compose

- frontend: React/Vite compilado en contenedor y servido por Nginx. Expone solo 127.0.0.1:3000.
- api: FastAPI; conversaciones, sesiones, tareas, memoria y endpoints locales de voz.
- ollama: Qwen2.5 3B configurado con el Modelfile como nyx:0.2.

La red nyx_internal no expone puertos del API o del LLM al host.

## Volúmenes persistentes

- nyx_models: modelos y configuraciones de Ollama (no están en GitHub).
- nyx_memory: base SQLite /data/nyx.db.
- nyx_voice: modelos descargados de Piper y Vosk, por defecto /voices.

scripts/backup.ps1 respalda los tres volúmenes con servicios detenidos y scripts/restore.ps1 permite restaurar. Este último requiere -Overwrite.

## Migración 0.1 → 0.2

Antes de cambiar de rama, respaldar nyx_memory.
La aplicación crea una tabla sessions, añade messages.session_id si no existe y crea una sesión "Conversaciones anteriores (0.1)" para todos los mensajes anteriores sin session_id. No se borra contenido histórico. init_db es idempotente.

API:
- GET/POST /api/sessions (GET con q para filtrar).
- PATCH/DELETE /api/sessions/{id}.
- GET /api/sessions/{id}/messages.
- POST /api/chat con session_id opcional.
- Endpoints existentes de tareas, memoria e historial continúan disponibles.

## Audio / micrófono

- GET /api/voice/status indica qué modelos están instalados.
- POST /api/voice/speak genera WAV local usando Piper. La voz base es es_ES-davefx-medium, masculino, dataset CC0.
- POST /api/voice/transcribe admite audio WAV PCM16 mono 16 kHz (máximo 2 MB) y usa Vosk local.
- frontend usa getUserMedia + MediaRecorder al pulsar botón de micrófono, convierte a WAV y envía al API Docker interno.
- Web Audio API analiza el micrófono y las respuestas sintetizadas para dibujar las ondas con datos reales.
- Alertas se procesan con filtros/delay suaves del navegador, manteniendo la misma voz.
- Pulsar "DETENER VOZ" interrumpe la reproducción. La voz está desactivada por defecto.

La voz debe probarse cualitativamente, ya que la naturalidad depende del modelo. La clonación de una voz cinematográfica exacta no forma parte del proyecto.

## Control de recursos

Piper y Vosk se cargan bajo demanda dentro del API. Sus archivos se descargan explícitamente, no durante docker build ni en el arranque. Evitar abrir múltiples conversaciones a la vez en CPU.

## Seguridad

La API carece de autenticación: NO exponerla a redes ni Internet. No ejecutar comandos en nombre del usuario. La memoria se gestiona mediante acciones explícitas y se pueden eliminar sesiones. No guardar secretos ni cargar datos privados en GitHub.

## Pruebas y límites

Tests Python de sesiones, migración, memoria y rutas HTTP. UI construible mediante npm run build. Integración Docker, Piper y Vosk deben validarse en el host real antes de liberar versión.
