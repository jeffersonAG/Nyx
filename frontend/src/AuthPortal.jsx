import React, { useState } from 'react';
import './auth.css';

export default function AuthPortal({ mode = 'login', username = '', onLogin, onChangePassword }) {
  const [loginName, setLoginName] = useState('');
  const [password, setPassword] = useState('');
  const [oldPassword, setOldPassword] = useState('');
  const [newPassword, setNewPassword] = useState('');
  const [confirmPassword, setConfirmPassword] = useState('');
  const [showPassword, setShowPassword] = useState(false);
  const [error, setError] = useState('');
  const [working, setWorking] = useState(false);

  async function submit(event) {
    event.preventDefault();
    setWorking(true); setError('');
    try {
      if (mode === 'change') {
        if (newPassword !== confirmPassword) throw new Error('Las contraseñas nuevas no coinciden.');
        if (newPassword.length < 12) throw new Error('La contraseña definitiva debe tener al menos 12 caracteres.');
        await onChangePassword(oldPassword, newPassword);
      } else {
        await onLogin(loginName.trim(), password);
      }
      setPassword(''); setOldPassword(''); setNewPassword(''); setConfirmPassword('');
    } catch (err) {
      setError(err.message || 'No se pudo autenticar.');
    } finally {
      setWorking(false);
    }
  }

  return <div className="nyx-auth-page">
    <div className="nyx-auth-grid" aria-hidden="true" />
    <main className="nyx-auth-panel">
      <div className="nyx-auth-top"><span>NYX / AUTHENTICATION SYSTEM</span><span>◆ SECURITY v0.4</span></div>
      <div className="nyx-auth-core" aria-hidden="true"><div className="nyx-auth-diamond"><span>N</span></div></div>
      <div className="nyx-auth-title">
        <h1>NYX<span>_</span></h1>
        <p>CRIMSON INTELLIGENCE SYSTEM</p>
      </div>
      <div className="nyx-auth-heading">
        <h2>{mode === 'change' ? 'ACTUALIZAR CREDENCIALES' : 'CONTROL DE ACCESO'}</h2>
        <p>{mode === 'change'
          ? 'La cuenta requiere sustituir su clave provisional por una contraseña segura.'
          : 'Inicia sesión para acceder a tu espacio local protegido.'}</p>
      </div>
      <form onSubmit={submit} className="nyx-auth-form">
        {mode === 'login' ? <>
          <label htmlFor="nyx-login-user">IDENTIFICADOR DE USUARIO</label>
          <input id="nyx-login-user" autoComplete="username" autoFocus
            maxLength={32} value={loginName} onChange={e => setLoginName(e.target.value)}
            placeholder="Usuario" required />
          <label htmlFor="nyx-login-pass">CREDENCIAL DE ACCESO</label>
          <div className="nyx-secret-input">
            <input id="nyx-login-pass" type={showPassword ? 'text' : 'password'}
              autoComplete="current-password" value={password}
              onChange={e => setPassword(e.target.value)} placeholder="Contraseña" required />
            <button type="button" onClick={() => setShowPassword(v => !v)}
              aria-label={showPassword ? 'Ocultar contraseña' : 'Mostrar contraseña'}>
              {showPassword ? '◉' : '◎'}</button>
          </div>
        </> : <>
          <div className="nyx-auth-account">CUENTA AUTENTICADA · {username}</div>
          <label htmlFor="nyx-old-pass">CONTRASEÑA PROVISIONAL / ACTUAL</label>
          <input id="nyx-old-pass" type="password" autoComplete="current-password"
            value={oldPassword} onChange={e => setOldPassword(e.target.value)} required />
          <label htmlFor="nyx-new-pass">NUEVA CONTRASEÑA (MÍNIMO 12 CARACTERES)</label>
          <input id="nyx-new-pass" type="password" autoComplete="new-password"
            value={newPassword} onChange={e => setNewPassword(e.target.value)} required minLength={12}/>
          <label htmlFor="nyx-confirm-pass">CONFIRMAR CONTRASEÑA</label>
          <input id="nyx-confirm-pass" type="password" autoComplete="new-password"
            value={confirmPassword} onChange={e => setConfirmPassword(e.target.value)} required minLength={12}/>
        </>}
        {error && <div className="nyx-auth-error" role="alert">⚠ {error}</div>}
        <button className="nyx-auth-submit" type="submit" disabled={working}>
          {working ? 'PROCESANDO...' : mode === 'change' ? 'CAMBIAR CONTRASEÑA' : 'AUTENTICAR ACCESO ↗'}
        </button>
      </form>
      <div className="nyx-auth-foot">◈ LOCAL SESSION · ARGON2ID HASHED CREDENTIALS · PRIVATE DATA</div>
    </main>
    <p className="nyx-auth-bottom">NYX 0.4 · IDENTITY & SECURITY</p>
  </div>;
}
