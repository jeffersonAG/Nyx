# Arquitectura de NYX 0.1

Objetivo: IA local portable con personalidad Crimson, compatible con PC Ryzen 5 4600G y con migracion futura a servidor Linux.

Componentes Docker Compose:
- frontend: React/Vite compilado y servido por Nginx en 127.0.0.1:3000. Nginx redirige /api al backend.
- api: FastAPI y almacenamiento SQLite. Maneja chat, mensajes, tareas y notas manuales.
- ollama: motor Qwen2.5 3B basado en el Modelfile nyx:0.1. No expone 11434.

Red privada: nyx_internal.

Persistencia:
- nyx_models montado en /root/.ollama, contiene modelos y configuraciones.
- nyx_memory montado en /data, contiene nyx.db.
- backups/ esta ignorado en Git. Los scripts PowerShell permiten exportar/restaurar ambos volumenes.

Flujo: usuario en React -> /api/chat -> FastAPI -> historial/notas SQLite -> Ollama -> respuesta -> SQLite -> React.

Los recuerdos se guardan solo por acciones explicitas. No existe entrenamiento automatico. Guardar un Modelfile no equivale a entrenar el modelo.

Seguridad: API sin autenticacion; acceso localhost UNICAMENTE. No publicar en redes ni Internet. No escribir contrasenas ni secretos en las notas. No subir modelos, recuerdos, bases o copias a GitHub.

Voz: sintetizador opcional del navegador para vista previa. Whisper/Piper y voz hibrida (natural para chat, metalica para alertas) se implementaran en otra etapa. No se garantiza offline completo para la voz del navegador.

Evolucion: medir CPU/RAM, incorporar STT/TTS local, migrar modelos y memoria con backup, y despues construir la app de escritorio.

Esta arquitectura no elimina ni modifica los experimentos o la documentacion anterior.
