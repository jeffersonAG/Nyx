# NYX 0.3 — Cognitive Core

NYX utiliza el diseño Crimson 0.2 y la misma rama Git de la versión 0.2.
Esta versión NO crea nuevos puertos, cuentas ni servicios externos.

## Flujo

1. Frontend POST /api/chat con texto y sesión.
2. Backend conserva el mensaje en SQLite y prepara contexto temporal verificable con zoneinfo (perfil local).
3. Para consultas simples de tiempo o ubicación, responde directamente sin modelo.
4. Las demás peticiones llaman a Ollama /api/chat con herramientas declaradas. Qwen2.5 3B puede producir tool_calls.
5. El orquestador valida nombre, argumentos y permiso textual de acciones, y ejecuta SOLO herramientas conocidas (no shell, no eval).
6. Se devuelven resultados al modelo para redactar respuesta. Las operaciones exitosas incluyen tool_events para actualizar tareas/notas en React.
7. Se guarda la respuesta final en la sesión.

## Herramientas

| Nombre | Tipo | Resultado |
|---|---|---|
| get_local_datetime | lectura | fecha, hora, zona IANA |
| get_user_profile | lectura | ciudad habitual, país, zona horaria, tono |
| list_tasks | lectura | hasta 40 tareas |
| create_task | escritura | tarea con fecha opcional, requiere petición expresa |
| complete_task | escritura | tarea por ID, requiere petición expresa |
| list_memories | lectura | hasta 25 recuerdos |
| remember_note | escritura | guarda nota solo a petición expresa |
| search_history | lectura | hasta 12 sesiones que coinciden |
| matrix_calculate | cálculo | SymPy, matrices numéricas 1..8 filas/columnas |

No hay modificación de ajustes desde el chat ni acciones destructivas (borrado) por tool calls. Los cambios de perfil son explícitos en el formulario de Configuración.

## Endpoints nuevos

- GET /api/context: contexto temporal real, sin datos de geolocalización.
- GET /api/profile: preferencias locales.
- PUT /api/profile: guardar zona horaria IANA válida, ciudad/país (opcionales) y tono.
- POST /api/chat: incluye tool_events de acciones exitosas; compatible con solicitudes anteriores.
- POST /api/tasks: acepta due_date opcional YYYY-MM-DD.

## Datos

Mismas rutas/volúmenes; cambios no destructivos e idempotentes:
- local_profile(id=1,timezone,city,country,tone) con ciudad/país vacíos por defecto.
- tasks.due_date TEXT NULL agregado si no existe.

Antes de actualizar ejecutar .\scripts\backup.ps1. No ejecutar docker compose down -v.

## Limitaciones y riesgos conocidos

- Herramientas disponibles no garantizan que el modelo 3B las use correctamente; revisar resultados.
- Sin navegación web, API de Google, geolocalización, ejecución de programas, recordatorios automáticos ni acceso directo al sistema operativo.
- Las memorias/historial se tratan como datos no fiables, nunca comandos para el sistema.
- El módulo matemático no acepta expresiones simbólicas abiertas por seguridad; admite únicamente valores numéricos validados.
- El rendimiento real y la calidad de llamadas a funciones requieren pruebas en el Ryzen 5.
- API sin autenticación: mantener siempre en 127.0.0.1.

## Próximas capacidades

Si se valida en hardware real: búsqueda de memorias con semántica, recordatorios reales con scheduler y notificaciones opcionales, pruebas A/B de modelos locales, mejora TTS para fórmulas y RAG documental.
