# NYX 0.5.0 — Núcleo funcional y agenda privada

Implementación incremental sobre la rama `feature/nyx-0.2-crimson`. Esta versión conserva
el historial, las cuentas, los permisos y el Security Sentinel de NYX 0.4.2.

## Cambios verificados por pruebas automatizadas

- **Tareas:** preguntas como «¿Cómo se llama cada tarea?» consultan directamente las
  filas de SQLite del usuario autenticado; se muestran nombres, fechas y estado.
- **Clima:** Open-Meteo consulta la localidad especificada; ya no obliga a que la
  ciudad se encuentre en el país habitual del usuario. La respuesta identifica
  la ciudad encontrada, hora y fuente.
- **Divisas:** fuente Frankfurter v2 mediante HTTPS y sin claves, admite USD/CRC
  y otros pares ISO. **Dato orientativo** de referencia, no compra/venta bancaria.
  Si falla el proveedor se informa del error, jamás se inventa la cotización.
- **Internet:** búsquedas explícitas con Bing RSS y alternativa Wikipedia.
  Se proporcionan extractos, URL y proveedor. Puede haber sitios bloqueados o
  resultados incompletos: no equivale a un navegador universal.
- **Investigación acumulativa:** extractos públicos solicitados se guardan por
  cuenta, con fecha y fuente, y se recuperan por coincidencia de términos.
  Se pueden revisar y eliminar en Memoria. **No reentrena Qwen/Ollama**.
- **Calendario local:** eventos personales por día, semana, mes y año; alta,
  modificación, borrado, detección de solapamientos y recordatorios.
  Las tareas ya existentes con fecha se visualizan sin migrarlas ni duplicarlas.
- **Avisos:** la interfaz consulta recordatorios mientras NYX esté abierto y
  ofrece notificación nativa del navegador tras consentimiento del usuario.
  No promete alertas si NYX, su API o el navegador están cerrados.
- **Interfaz:** estado ONLINE/LOCAL simplificado para usuarios estándar. Seguridad,
  herramientas de diagnóstico y controles administrativos siguen restringidos.
- **Tono:** instrucción al modelo para ofrecer respuestas más explicativas,
  sin afirmar que consultó fuentes cuando no lo hizo.

## Actualizar desde PowerShell (dentro de la carpeta NYX)

1. Detener actividades de edición de archivos locales del proyecto.
2. Ejecutar la copia de seguridad antes de reconstruir:

```powershell
.\scripts\backup.ps1
git fetch origin
git switch feature/nyx-0.2-crimson
git pull --ff-only origin feature/nyx-0.2-crimson
docker compose up -d --build
```

Abrir http://localhost:3000 y pulsar Ctrl+F5. **Nunca ejecutar
`docker compose down -v`**, porque borraría volúmenes con datos.

La rama principal `main` no se modifica mediante estos cambios; la
actualización debe hacerse sobre la rama indicada.

## Pruebas manuales recomendadas

Iniciar sesión con administrador y después con un usuario estándar distinto:

1. Guardar varias tareas y preguntar «¿Cómo se llama cada tarea?»; comprobar
   nombres reales y que otro usuario no reciba tareas ajenas.
2. Activar modo en línea desde Configuración con autorización administrativa.
3. Preguntar «¿Cuál es el clima actual en Madrid?»; comprobar localidad y fecha
   en la respuesta; repetir sin conexión y verificar que no invente mediciones.
4. Preguntar «¿Cuál es el tipo de cambio actual?» y «¿Cuánto son 200 dólares en colones?»;
   comprobar fuente, par y fecha. El valor varía por día de mercado.
5. Preguntar «Investiga en Internet sobre redes VLAN»; comprobar extractos y enlaces;
   revisar y borrar entradas en Memoria → Conocimiento consultado.
6. Crear una reunión el 2026-11-05 a las 13:00 por 60 minutos;
   comprobar el calendario y el rechazo de otra reserva solapada.
7. Crear un recordatorio próximo y mantener NYX abierto; permitir opcionalmente
   notificaciones en el navegador. Comprobar que al pulsar ENTENDIDO no se repita.
8. Revisar Security Sentinel en administración y ejecutar una prueba de correo.
9. Cerrar sesión y repetir la revisión del calendario desde un usuario estándar.

## Limitaciones y seguridad

- El acceso web **es por petición explícita**, a proveedores HTTPS predefinidos.
  No hay ejecución de código remoto, rastreo arbitrario ni acceso a redes locales.
- NYX no adquiere «aprendizaje neuronal continuo» sólo por almacenar extractos.
  La memoria asociativa y su trazabilidad son la base para una futura versión RAG,
  con evaluación, realimentación y controles de calidad.
- Los horarios del calendario son **locales al perfil**. Configurar la zona IANA
  correspondiente y evitar cambiarla después de crear eventos sin revisar horas.
- Los datos quedan en la base SQLite existente, aislados por propietario.
  Los contenidos web se tratan como **datos no confiables**, no instrucciones.
- Los recordatorios en pantalla no equivalen a tareas de fondo con correo/push
  cuando el navegador está cerrado; eso requiere otro servicio dedicado.
- GitHub Actions ejecuta unittest, build de React y `docker compose config`.
  Las APIs externas se simulan en CI: **no prueban conectividad real desde
  el equipo ni disponibilidad en tiempo real de los proveedores**.
