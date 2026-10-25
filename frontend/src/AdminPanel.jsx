import React, { useEffect, useState } from 'react';

const categoryNames = {
  sensitive_request: 'Consulta de información sensible',
  restricted_access: 'Intento de acceso restringido',
  login_failures: 'Intentos fallidos de autenticación',
};

export default function AdminPanel({ auth, api, onElevated }) {
  const [events, setEvents] = useState([]);
  const [users, setUsers] = useState([]);
  const [mail, setMail] = useState({ configured:false, provider:'smtp', queued:0, recipient_masked:'' });
  const [password, setPassword] = useState('');
  const [newUser, setNewUser] = useState('');
  const [newPassword, setNewPassword] = useState('');
  const [error, setError] = useState('');
  const [notice, setNotice] = useState('');
  const [working, setWorking] = useState(false);

  async function load() {
    try {
      const [alerts, accounts, config] = await Promise.all([
        api('/api/security/events'), api('/api/auth/users'), api('/api/security/mail-status')
      ]);
      setEvents(alerts); setUsers(accounts); setMail(config);
    } catch (err) { setError(err.message); }
  }
  useEffect(() => { load(); }, [auth.id]);

  async function unlock(event) {
    event.preventDefault(); setWorking(true); setError(''); setNotice('');
    try {
      await api('/api/auth/elevate', {
        method: 'POST', body: JSON.stringify({ password })
      });
      setPassword(''); setNotice('Autorización activada durante 5 minutos.');
      await onElevated();
    } catch (err) { setError(err.message); }
    finally { setWorking(false); }
  }

  async function revoke() {
    try {
      await api('/api/auth/revoke-elevation', { method: 'POST' });
      await onElevated();
      setNotice('Acceso de configuración revocado.');
    } catch (err) { setError(err.message); }
  }

  async function createAccount(event) {
    event.preventDefault(); setWorking(true); setError(''); setNotice('');
    try {
      await api('/api/auth/users', {
        method: 'POST',
        body: JSON.stringify({ username: newUser, password: newPassword }),
      });
      setNotice('Cuenta estándar creada. La persona deberá cambiar su contraseña provisional.');
      setNewUser(''); setNewPassword(''); await load();
    } catch (err) { setError(err.message); }
    finally { setWorking(false); }
  }

  async function review(id, status) {
    if (!auth.elevated) return;
    try {
      await api('/api/security/events/' + id, {
        method: 'PATCH', body: JSON.stringify({ status }),
      });
      await load();
    } catch (err) { setError(err.message); }
  }
  async function testMail() {
    setWorking(true); setError(''); setNotice('');
    try {
      await api('/api/security/test-mail', { method:'POST' });
      setNotice('El proveedor confirmó la aceptación del mensaje de prueba. Comprueba la recepción en tu bandeja.');
    } catch (err) { setError(err.message); }
    finally { setWorking(false); }
  }
  return <section className="nyx-security-dashboard" aria-label="Centro de seguridad administrativa">
    <header className="nyx-security-heading">
      <div><div className="eyebrow">/ NYX SECURITY SENTINEL</div><h2>Identidad y seguridad</h2></div>
      <span className={auth.elevated ? 'security-tag elevated' : 'security-tag'}>
        {auth.elevated ? 'CAMBIOS AUTORIZADOS' : 'LECTURA ADMIN'}
      </span>
    </header>
    <p className="muted">Los módulos cotidianos se consultan sin autorizaciones adicionales. Solo los cambios administrativos requieren desbloqueo temporal.</p>
    {!auth.elevated ? <form className="nyx-admin-auth" onSubmit={unlock}>
      <label htmlFor="nyx-admin-password">Desbloquear configuración (5 minutos)</label>
      <div><input id="nyx-admin-password" type="password" value={password}
          autoComplete="current-password" placeholder="Contraseña administrativa"
          onChange={e => setPassword(e.target.value)} required />
        <button type="submit" disabled={working}>◆ AUTORIZAR CAMBIOS</button></div>
    </form> : <div className="nyx-security-unlocked">
      <span>◈ Permiso temporal activo. Los cambios quedan restringidos a esta sesión.</span>
      <button type="button" onClick={revoke}>REVOCAR AHORA</button>
    </div>}
    {error && <p role="alert" className="nyx-admin-error">⚠ {error}</p>}
    {notice && <p role="status" className="nyx-admin-notice">{notice}</p>}

    <div className="nyx-security-subheading"><h3>Usuarios</h3><span>{users.length} CUENTAS</span></div>
    <div className="nyx-users">
      {users.map(u => <div key={u.id} className="nyx-user-entry">
        <strong>{u.username}</strong>
        <span>{u.role === 'admin' ? 'ADMINISTRADOR' : 'ESTÁNDAR'}</span>
        <small>{u.must_change ? 'Contraseña provisional' : 'Activa'}</small>
      </div>)}
    </div>
    {auth.elevated && <form className="nyx-create-user" onSubmit={createAccount}>
      <h3>Registrar usuario estándar</h3>
      <label htmlFor="nyx-new-user">Nombre de usuario</label>
      <input id="nyx-new-user" value={newUser} maxLength={32}
        onChange={e => setNewUser(e.target.value)} required placeholder="Nuevo usuario"/>
      <label htmlFor="nyx-new-user-password">Contraseña provisional (mínimo 12 caracteres)</label>
      <input id="nyx-new-user-password" type="password" value={newPassword}
        onChange={e => setNewPassword(e.target.value)} autoComplete="new-password" required minLength={12}/>
      <button type="submit" disabled={working}>＋ CREAR CUENTA ESTÁNDAR</button>
    </form>}
    <div className="nyx-security-subheading">
      <h3>Incidentes</h3>
      <button type="button" onClick={load}>↻ ACTUALIZAR</button>
    </div>
    <div className="nyx-alert-list">
      {events.map(event => <article key={event.id} className="nyx-alert-entry">
        <div className="nyx-alert-head">
          <strong>SEC-{String(event.id).padStart(6, '0')}</strong>
          <span>{event.severity === 'high' ? 'ALTA' : 'MEDIA'} · {event.status === 'pending'
            ? 'PENDIENTE' : event.status === 'false_positive' ? 'FALSO POSITIVO' : 'CONFIRMADO'}</span>
        </div>
        <div className="nyx-alert-row">{categoryNames[event.category] || event.category} · {event.created_at} UTC</div>
        <div className="nyx-alert-row">USUARIO: {event.actor_username} · CORREO: {event.mail_state}</div>
        <p>{event.reason}</p>
        {event.message_excerpt && <blockquote>{event.message_excerpt}</blockquote>}
        {auth.elevated && <div className="nyx-alert-actions">
          <button onClick={() => review(event.id, 'false_positive')}>FALSO POSITIVO</button>
          <button onClick={() => review(event.id, 'confirmed')}>CONFIRMAR</button>
        </div>}
      </article>)}
      {!events.length && <p className="muted">No se han registrado incidentes.</p>}
    </div>
    <div className="nyx-security-subheading"><h3>Notificaciones por correo</h3></div>
    <div className="nyx-mail-info">
      <div>PROVEEDOR: {mail.provider === 'emailjs' ? 'EMAILJS + GMAIL' : mail.provider === 'smtp' ? 'SMTP' : 'DESCONOCIDO'}</div>
      <div>ESTADO: {mail.configured ? 'CONFIGURADO' : 'CONFIGURACIÓN INCOMPLETA'}</div>
      <div>DESTINO: {mail.recipient_masked || (mail.provider === 'emailjs' ? 'FIJADO EN LA PLANTILLA EMAILJS' : 'NO CONFIGURADO')}</div>
      <div>ALERTAS EN COLA: {mail.queued}</div>
      <p>Selecciona EmailJS o SMTP en el archivo local .env.security. Nunca introduzcas claves secretas en el navegador ni en GitHub. Si utilizas EmailJS, el destinatario se fija en la plantilla de EmailJS.</p>
      {auth.elevated && <button onClick={testMail} disabled={!mail.configured || working}>✉ ENVIAR CORREO DE PRUEBA</button>}
    </div>
  </section>;
}
