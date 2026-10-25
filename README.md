# NYX 0.1 — Crimson System

Primera version experimental portable del asistente NYX: Ollama + Qwen2.5 3B, FastAPI, SQLite y React. Diseno negro/grafito/carmesi inspirado en el portafolio V2.

## Requisitos
- Windows 11 con Docker Desktop y WSL 2, o Linux con Docker Compose.
- 16 GB RAM recomendados para pruebas CPU con Ryzen 5 4600G.
- Internet para descargar imagenes y modelo por primera vez; inferencia local despues.

## Instalacion
Desde PowerShell en la raiz del repositorio:

    docker --version
    docker compose version
    docker compose up -d --build
    docker compose exec ollama ollama pull qwen2.5:3b
    docker compose exec ollama ollama create nyx:0.1 -f /config/Modelfile

Abrir http://localhost:3000 en el mismo equipo. La primera respuesta en CPU puede tardar. Probar consola con:

    docker compose exec ollama ollama run nyx:0.1

Revisar estado:

    docker compose ps
    docker compose logs -f api

Tras modificar Modelfile, volver a ejecutar el comando ollama create.

## Alcance implementado
- Chat local con Qwen, historial de mensajes en SQLite.
- Tareas y notas guardadas expresamente, sin acciones autonomas.
- UI Crimson en React (negro #000000, gris #1E1E1E, carmesi #DC143C).
- Vista previa opcional de lectura de respuestas con la voz del navegador.
- Persistencia y respaldo de modelos y memoria mediante volumenes Docker.

## Pendiente
- Voz natural local con Piper TTS y modo metalico para alertas.
- Reconocimiento offline del microfono.
- Instalador de escritorio Tauri/Electron.
- Entrenamiento o fine-tuning y acciones sobre sistemas externos.

ATENCION: la vista previa de voz del navegador no garantiza operacion totalmente offline ni la voz final hibrida. Modelfile cambia configuracion de Qwen, NO entrena pesos nuevos.

## Datos y portabilidad
- nyx_models: archivos de Ollama (incluye modelos descargados).
- nyx_memory: SQLite con historial, tareas y recuerdos explicitamente guardados.
- Codigo: repositorio GitHub.

El Ollama instalado previamente en Windows y el de Docker tienen almacenamientos distintos. No desinstales aun Ollama para Windows; comprueba primero el contenedor.

Respaldar y restaurar desde PowerShell:

    .\scripts\backup.ps1
    .\scripts\restore.ps1 -BackupFolder ".\backups\nyx-AAAAMMDD-HHMMSS" -Overwrite
    docker compose up -d --build

La restauracion reemplaza la informacion existente; guarda respaldo primero. No publiques archivos de memoria ni respaldos en GitHub. Nunca utilices docker compose down -v si deseas conservar volumenes.

## Seguridad
La interfaz solo se publica en 127.0.0.1:3000. No hay autenticacion: no expongas NYX a Internet ni a la red local. El modelo no puede ejecutar comandos del equipo.

Se conservan el main.py y nyx_core originales, asi como Documentacion/. Consulta docs/ARQUITECTURA.md.
