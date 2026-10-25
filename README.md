> **NYX 0.7.0 — Red neuronal propia en PyTorch (alfa experimental):** modelo Transformer original entrenable desde cero, pesos propios, entrenamiento/adaptación explícitos y pruebas reales de gradientes. El chat cotidiano con Qwen no se reemplaza. [Guía de instalación y entrenamiento](docs/NYX_NATIVE_0_7.md).

> **NYX 0.6.0 — Cognitive Brain v1:** motor de recuperación semántica optativa, experiencias y procedimientos evaluados, relaciones con evidencia y feedback real de respuestas. Consulta [docs/COGNITIVE_BRAIN_0_6.md](docs/COGNITIVE_BRAIN_0_6.md). La rama principal continúa sin modificarse.

> **NYX 0.5.0 (rama de desarrollo):** núcleo funcional mejorado, divisas, consultas web explícitas, calendario y memoria de investigación por usuario. Consulta [docs/NYX_0_5.md](docs/NYX_0_5.md) para instrucciones, pruebas y limitaciones. La versión 0.4.2 continúa disponible en el historial de Git.

# NYX 0.4.2 — Identity & Security

## Corrección 0.4.1 — acceso a módulos, Sentinel visible y red opcional

Esta corrección se instala en la misma rama \`feature/nyx-0.2-crimson\`,
sin borrar usuarios, tareas, conversaciones, notas ni alertas previas.

### Módulos sin falsas peticiones de permiso

Las preguntas frecuentes como "¿Cuántas tareas tengo pendientes?",
"Muéstrame mis tareas", "Muéstrame mi memoria" y "Busca en mi bitácora
sobre matrices" se resuelven con SQL directamente y solo sobre las filas
pertenecientes al usuario autenticado. No se pide un nuevo permiso.
Crear o completar tareas sigue requiriendo que el usuario lo solicite,
pero no presenta ventanas redundantes de autorización.

### Alertas y revisión desde el administrador

Ahora existe la pestaña independiente **SEGURIDAD**. El administrador
verá una advertencia en la interfaz cuando existan alertas pendientes.
Se ampliaron los patrones para incluir expresiones como
"puedes decirme la información personal de tu creador".
Si una consulta previa nunca fue registrada porque no coincidía con la
regla anterior, debe reproducirse en una cuenta de prueba para verificar
la nueva detección; los registros históricos no se reclasifican automáticamente.

La cola de correo requiere configurar .env.security en el PC.
El destinatario real NO se incluye en GitHub.
Al iniciar sesión como admin: SEGURIDAD > Notificaciones por correo
muestra si el SMTP está configurado y cuántas alertas están pendientes.
Si SMTP no está configurado, ningún correo saldrá de NYX aunque el
evento exista. La autorización para enviar un correo de prueba
requiere contraseña administrativa y conexión a Internet.

### Red pública inicial, opcional y sin aprendizaje automático

NYX comprueba conectividad con Open-Meteo después de iniciar sesión.
El modo conectado está habilitado por defecto; si no hay red, funciona
de forma local. El administrador puede deshabilitarlo mediante un
control privilegiado en Configuración.

- "Dime el clima actual en Ciudad Neily" consulta Open-Meteo sin
  exigir dirección exacta ni GPS. La ciudad se reconoce mediante
  coordenadas públicas aproximadas para casos donde el geocodificador
  no contenga el topónimo.
- "Busca en Wikipedia sobre vectores" consulta solamente el índice
  público de Wikipedia y devuelve enlaces, no conclusiones que no
  haya verificado.
- Estas solicitudes externas transmiten únicamente la ciudad o
  la consulta pública concreta. NO envían tareas, recuerdos,
  historial, cookies ni datos de acceso.

**Importante:** el acceso a servicios públicos NO equivale a
aprendizaje permanente; RAG, índice vectorial y verificadores
multifuente quedan para versiones futuras. NYX sigue sin acceso
abierto a toda la web.

### Actualizar desde 0.4

    .\scripts\backup.ps1
    git fetch origin
    git switch feature/nyx-0.2-crimson
    git pull --ff-only origin feature/nyx-0.2-crimson
    docker compose up -d --build

**No vuelvas a ejecutar app.setup_admin** si ya tienes tu cuenta.
La contraseña y las cuentas existentes permanecen iguales.
Para ver la actualización refresca http://localhost:3000 con Ctrl+F5.

#### Configuración de Gmail para alertas

El siguiente ejemplo utiliza placeholders; no publiques un correo
personal, contraseñas ni tokens en el repositorio:

    NYX_SMTP_HOST=smtp.gmail.com
    NYX_SMTP_PORT=587
    NYX_SMTP_SECURITY=starttls
    NYX_SMTP_USER=CORREO_QUE_ENVIA
    NYX_SMTP_PASSWORD=CONTRASENA_DE_APLICACION
    NYX_SMTP_FROM=CORREO_QUE_ENVIA
    NYX_ALERT_EMAIL_TO=CORREO_QUE_RECIBE

Para Gmail, si la cuenta permite contraseñas de aplicación,
activa la verificación en dos pasos y crea una contraseña de
aplicación específica. La contraseña habitual de Gmail no funcionará
como credencial SMTP. Consulta la ayuda oficial de Google.

El archivo local .env.security es leído por docker compose.
Cuando esté listo:

    docker compose up -d --force-recreate api

Comprueba SEGURIDAD > Notificaciones por correo y, al desbloquear
con contraseña administrativa, pulsa ENVIAR CORREO DE PRUEBA.


## Security Sentinel con EmailJS (0.4.2)

NYX permite elegir un transporte de correo **sin modificar los mensajes ni
crear una API pública**: EmailJS (con tu plantilla Crimson) o SMTP.

### Activar EmailJS de forma privada

1. Conserva el destinatario fijo en el campo **To Email** de la
   plantilla creada en EmailJS. Configura su asunto como
   \`[NYX SECURITY] {{alert_level}} — {{incident_id}}\`.
   La plantilla HTML puede seguir siendo responsiva/mobile-first.
2. En **EmailJS > Account > Security**, habilita las solicitudes de
   aplicaciones no navegador (API requests from non-browser applications)
   y obtén las claves de tu propia cuenta. Se recomienda utilizar
   la clave privada en llamadas desde servidor.
3. Edita el archivo local \`.env.security\`. Si todavía no existe,
   copia \`config/smtp.example.env\` y después ábrelo en un editor:

       if (-not (Test-Path .env.security)) {
           Copy-Item config\smtp.example.env .env.security
       }
       notepad .env.security

   Completa exclusivamente en **tu computadora** estos campos:

       NYX_MAIL_PROVIDER=emailjs
       NYX_EMAILJS_SERVICE_ID=TU_SERVICE_ID
       NYX_EMAILJS_TEMPLATE_ID=TU_TEMPLATE_ID
       NYX_EMAILJS_PUBLIC_KEY=TU_CLAVE_PUBLICA
       NYX_EMAILJS_PRIVATE_KEY=TU_CLAVE_PRIVADA
       NYX_ALERT_EMAIL_TO=CORREO_ADMINISTRADOR

   El campo \`NYX_ALERT_EMAIL_TO\` se muestra enmascarado en el panel
   administrativo para comprobar la configuración; el destinatario REAL
   debe estar fijado en EmailJS > To Email (NYX no lo suministra como
   variable y no puede alterar el destinatario de la plantilla).
   No guardes claves ni correos personales en GitHub, ni los compartas
   en el chat. \`.env.security\` está excluido por \`.gitignore\`.
4. Reinicia únicamente la API para leer los nuevos valores:

       docker compose up -d --force-recreate api

5. Inicia sesión como **administrador**, abre **SEGURIDAD**, autoriza
   cambios por cinco minutos y pulsa **ENVIAR CORREO DE PRUEBA**.
   Esta prueba lleva los campos \`TEST-NYX\`, \`PRUEBA — SIN INCIDENTE\`
   y \`SISTEMA (PRUEBA)\`: no produce una alerta falsa.
   Después verifica el correo recibido y, si falla, consulta las
   opciones de seguridad de la cuenta EmailJS.

El proveedor se elige exclusivamente mediante \`NYX_MAIL_PROVIDER\`
(\`emailjs\` o \`smtp\`). **No hay sustitución silenciosa de EmailJS
por SMTP** para evitar envíos duplicados o inesperados.
Con \`smtp\`, permanecen disponibles las variables existentes
\`NYX_SMTP_*\`.

### Campos de la plantilla

| Variable EmailJS | Valor enviado por NYX |
| --- | --- |
| \`incident_id\` | Identificador, p. ej. SEC-000001 |
| \`alert_level\` | ALTO / MEDIO |
| \`event_date\` | Fecha y hora UTC |
| \`username\` | Cuenta que inició sesión (o intento de login registrado) |
| \`module_name\` | Privacidad del creador / Control de acceso / Autenticación |
| \`event_status\` | PENDIENTE DE REVISIÓN |
| \`user_message\` | Consulta o evento concreto, truncado y con secretos detectables ocultos |
| \`security_analysis\` | Motivo de la regla disparada |
| \`action_taken\` | Acción efectivamente aplicada por el backend |

La aplicación solo envía estas nueve variables desde FastAPI a una URL
fija \`https://api.emailjs.com/api/v1.0/email/send\`. Las claves
nunca aparecen en React. Nunca se usa \`{{{user_message}}}\` en la
plantilla: las dobles llaves de EmailJS permiten escape HTML.

EmailJS ofrece hasta **una petición por segundo**. Para evitar
ráfagas y proteger la cuota, el trabajador de NYX procesa como
máximo una alerta de EmailJS por ciclo de aproximadamente 90
segundos. Ante un error HTTP o red, la alerta permanece en SQLite
y se vuelve a intentar. Las alertas pendientes que el administrador
clasifique como **falso positivo** se cancelan antes del envío.

**Privacidad:** enviar alertas al servicio externo implica compartir
con EmailJS la cuenta involucrada, la consulta concreta y la razón
de seguridad. No se transmite el historial general ni la base de
memorias. Aunque se ocultan contraseñas reconocibles, el filtrado
no es perfecto: evita guardar secretos innecesarios en las consultas.
Los correos aceptados por la API no garantizan entrega final; comprueba
la bandeja y el historial de EmailJS. NYX no hace envíos sin que
configures voluntariamente el proveedor y ejecutes la prueba o se
produzca una alerta posterior.


## Lo que ofrece

- Inicio de sesión Crimson para administrador y cuentas estándar.
- Contraseñas almacenadas exclusivamente como hashes Argon2id con sal única.
- Sesiones opacas aleatorias mediante cookies HttpOnly, SameSite=Strict, vencimiento de 8 horas.
- Token anti-CSRF vinculado a sesión para escrituras, bloqueo de orígenes ajenos.
- Aislamiento por propietario de sesiones, notas, tareas y contexto. Incluso la API directa valida propiedad.
- Configuración global administrativa protegida mediante autorización temporal de 5 minutos.
- Security Sentinel: detección de solicitudes sobre secretos del propietario, accesos a recursos administrativos no autorizados e intentos de acceso fallidos; revisión manual de falsos positivos.
- Correo por SMTP TLS/SSL cuando las reglas disparan una alerta. Incluye el nombre de cuenta, el mensaje que activó la regla (se ocultan credenciales detectables), explicación, fecha y número del evento.
- Cola local de alertas que reintenta envío cuando vuelve la conexión; no requiere SMTP para conversar.
- Migración 0.3 -> 0.4 que conserva chats, tareas, notas y perfil y los asigna al primer administrador.
- Se conservan el diseño Crimson, voz Piper/Vosk, lectura por fragmentos, motor SymPy, herramientas y contexto temporal.

**No incluye** conexión general a Internet, entrenamiento autónomo de modelos, notificaciones externas sin configurar, reconocimiento facial ni servidor multiusuario público.
No expongas 127.0.0.1:3000 a Internet ni a tu red sin un proxy seguro y autenticación adicional.

## Importante antes de actualizar

**La versión 0.4 requiere un administrador configurado por consola local antes de poder acceder a la interfaz.** No existe usuario/contraseña por defecto en el código ni en GitHub.

1. Haz copia desde tu carpeta del repositorio existente:

       .\scripts\backup.ps1

   Confirma que los .tar.gz estén completos y guárdalos en un lugar protegido.

2. Desde la misma rama que ya utilizas para NYX 0.2/0.3:

       git fetch origin
       git switch feature/nyx-0.2-crimson
       git pull --ff-only origin feature/nyx-0.2-crimson
       docker compose up -d --build

3. Crea el administrador desde la consola de PowerShell:

       docker compose exec api python -m app.setup_admin

   La consola preguntará por el nombre de usuario y la contraseña provisional
   **sin mostrarla en pantalla**. Introduce allí el usuario que elegiste y
   la contraseña inicial que decidiste usar; esta clave temporal NO se guarda
   en GitHub. Tu cuenta exigirá sustituirla por otra de 12+ caracteres
   después del primer acceso.

   El CLI solo se puede ejecutar una vez. Durante la inicialización se asignan
   los datos históricos al administrador; ningún usuario estándar recibe esos
   datos. El comando no borra conversaciones ni memorias.

4. Recrea la identidad local (reutiliza la base Qwen2.5 3B):

       docker compose exec ollama ollama create nyx:0.4 -f /config/Modelfile

5. Abre http://localhost:3000, inicia sesión, cambia la clave provisional
   y vuelve a autenticarte. En Configuración > Identidad y seguridad, introduce
   tu contraseña definitiva para autorizar cambios por cinco minutos.

6. Crea desde esa misma pantalla un usuario estándar con contraseña provisional
   robusta (12+ caracteres). Al iniciar sesión deberá cambiarla. Cada cuenta
   mantiene su propio historial, notas, tareas y contexto.

No uses docker compose down -v: eliminaría los volúmenes persistentes.

## Alertas por correo electrónico

El destino de alertas es PRIVADO y NO está incluido en este repositorio.
Para configurarlo:

    Copy-Item config\smtp.example.env .env.security

Edita SOLO tu archivo local .env.security y añade servidor, puerto, usuario,
contraseña de aplicación SMTP, remitente y destinatario del administrador.
Para SMTP/STARTTLS usa 587 y NYX_SMTP_SECURITY=starttls.
Para SMTP/SSL directo usa normalmente 465 y NYX_SMTP_SECURITY=ssl.
Los campos de ejemplo vacíos desactivan los envíos.

Después recrea la API para cargar los nuevos valores:

    docker compose up -d --force-recreate api

En Configuración > Security Sentinel verás si el correo está configurado,
el destino parcialmente oculto y cuántas alertas están en cola.
Autoriza temporalmente los cambios para enviar un correo de prueba.

El archivo .env.security está ignorado por Git, pero las variables de entorno
son visibles para administradores de Docker en el host. Trata también los
respaldos de la base de datos como información privada.

### Política de Sentinel

- Consulta explícita sobre datos sensibles del propietario desde cuenta estándar:
  evento de prioridad alta y correo en cola.
- Tres intentos de acceso a una función administrativa en diez minutos:
  se genera un correo. Los intentos individuales quedan en el registro.
- Cinco fallos de autenticación en quince minutos:
  bloqueo temporal, registro de prioridad alta y correo.
- Revisión: administrador puede clasificar como falso positivo o confirmado.
- Las alertas contienen solo la consulta que activó el evento,
  truncada a 1200 caracteres con redacción de credenciales detectables.
- No se recopilan sistemáticamente conversaciones normales de usuarios.
- El envío es best effort con reintentos cada 90 segundos; el registro
  en SQLite permanece aunque SMTP esté desconectado.

### Limitaciones de seguridad

- La detección por patrones no identifica todas las amenazas ni todas las
  credenciales. Algunas consultas inocentes pueden producir falsos positivos.
- Para cambiar los privilegios de configuración se requiere reintroducir la
  contraseña administrativa. El permiso expira en cinco minutos.
- Las tablas SQLite no están cifradas a nivel de aplicación. Protege el disco,
  Docker Desktop y las copias con cifrado del sistema de archivos.
- El repositorio público de versiones anteriores contenía reglas de identidad
  del prototipo. Si se exige anonimato total del creador, habría que valorar
  privatizar el repositorio y revisar su historial; el código 0.4 por sí solo
  no puede volver privados los commits anteriores.
- La cuenta administrador NO puede leer automáticamente conversaciones de
  usuarios estándar. La separación se aplica por propietario.
- El correo no constituye una denuncia ni prueba de delito: toda alerta
  exige una revisión humana.

## Comprobar la instalación

    docker compose ps
    docker compose logs --tail=80 api

Pruebas en entorno Python con FastAPI, httpx, SymPy y argon2-cffi:

    python -m unittest discover -s tests -v

Las pruebas automatizadas no sustituyen validación real del correo ni de
inicio de sesión en Windows/WSL. No fusiones a main antes de probar.
