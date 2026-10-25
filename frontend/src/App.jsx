import React, { useEffect, useRef, useState } from 'react';

const NAV = [
  ['chat', '⌘', 'CONVERSACIÓN'],
  ['tasks', '☷', 'TAREAS'],
  ['memory', '▤', 'MEMORIA'],
  ['history', '◷', 'HISTORIAL'],
];

async function request(path, options = {}) {
  let response;
  try {
    response = await fetch(path, {
      ...options,
      headers: { 'Content-Type': 'application/json', ...(options.headers || {}) },
    });
  } catch {
    throw new Error('No es posible conectar con el núcleo de NYX.');
  }
  if (!response.ok) {
    let detail = 'Error de comunicación (' + response.status + ')';
    try { detail = (await response.json()).detail || detail; } catch { /* Sin JSON */ }
    throw new Error(typeof detail === 'string' ? detail : 'Solicitud no válida');
  }
  return response.status === 204 ? null : response.json();
}

function speak(text, mode = 'chat') {
  // Voz preliminar del navegador; aun NO es Piper ni STT offline.
  if (!('speechSynthesis' in window)) return false;
  window.speechSynthesis.cancel();
  const utterance = new SpeechSynthesisUtterance(text);
  utterance.lang = 'es-ES';
  const voices = window.speechSynthesis.getVoices();
  const selected = voices.find(v => v.lang.toLowerCase().startsWith('es') && v.localService)
    || voices.find(v => v.lang.toLowerCase().startsWith('es'));
  if (selected) utterance.voice = selected;
  utterance.rate = mode === 'alert' ? 0.82 : 0.96;
  utterance.pitch = mode === 'alert' ? 0.68 : 0.9;
  window.speechSynthesis.speak(utterance);
  return true;
}

export default function App() {
  const [tab, setTab] = useState('chat');
  const [messages, setMessages] = useState([]);
  const [tasks, setTasks] = useState([]);
  const [memories, setMemories] = useState([]);
  const [online, setOnline] = useState(false);
  const [model, setModel] = useState('nyx:0.1');
  const [draft, setDraft] = useState('');
  const [itemDraft, setItemDraft] = useState('');
  const [busy, setBusy] = useState(false);
  const [voice, setVoice] = useState(false);
  const [error, setError] = useState('');
  const bottomRef = useRef(null);

  async function refresh() {
    const results = await Promise.allSettled([
      request('/api/history'), request('/api/tasks'),
      request('/api/memory'), request('/api/status')
    ]);
    if (results[0].status === 'fulfilled') setMessages(results[0].value);
    if (results[1].status === 'fulfilled') setTasks(results[1].value);
    if (results[2].status === 'fulfilled') setMemories(results[2].value);
    if (results[3].status === 'fulfilled') {
      setOnline(results[3].value.online);
      setModel(results[3].value.model);
    }
    if (results.some(r => r.status === 'rejected')) {
      setError('Algunos servicios de NYX no están disponibles.');
    }
  }

  useEffect(() => { refresh(); }, []);
  useEffect(() => {
    if (tab === 'chat') bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, tab]);

  async function send(event) {
    event.preventDefault();
    const text = draft.trim();
    if (!text || busy) return;
    setDraft(''); setBusy(true); setError('');
    setMessages(prev => [...prev, {
      id: 'pending-' + Date.now(), role: 'user', content: text
    }]);
    try {
      const answer = await request('/api/chat', {
        method: 'POST', body: JSON.stringify({ text })
      });
      setMessages(await request('/api/history'));
      if (voice) speak(answer.reply);
    } catch (err) {
      setError(err.message);
      if (voice) speak('Alerta. No fue posible completar la solicitud.', 'alert');
      try { setMessages(await request('/api/history')); } catch { /* offline */ }
    } finally { setBusy(false); }
  }

  async function createItem(event) {
    event.preventDefault();
    const text = itemDraft.trim();
    if (!text) return;
    setError('');
    try {
      await request('/api/' + (tab === 'tasks' ? 'tasks' : 'memory'), {
        method: 'POST', body: JSON.stringify({ text })
      });
      setItemDraft('');
      if (tab === 'tasks') setTasks(await request('/api/tasks'));
      else setMemories(await request('/api/memory'));
    } catch (err) { setError(err.message); }
  }

  async function toggleTask(task) {
    setError('');
    try {
      await request('/api/tasks/' + task.id, {
        method: 'PATCH', body: JSON.stringify({ completed: !task.completed })
      });
      setTasks(await request('/api/tasks'));
    } catch (err) { setError(err.message); }
  }

  async function removeItem(id) {
    setError('');
    try {
      await request('/api/' + (tab === 'tasks' ? 'tasks' : 'memory') + '/' + id, {
        method: 'DELETE'
      });
      if (tab === 'tasks') setTasks(await request('/api/tasks'));
      else setMemories(await request('/api/memory'));
    } catch (err) { setError(err.message); }
  }

  function changeTab(next) {
    setTab(next); setItemDraft(''); setError('');
  }

  return (
    <div className="shell">
      <header className="header">
        <div className="identity">
          <div className="brand-mark"><span>N</span></div>
          <div><div className="logo">NYX<span>_</span></div>
            <div className="kicker">PERSONAL INTELLIGENCE</div></div>
        </div>
        <div className="top-status">
          <div><span className={'status-dot ' + (online ? 'ok' : 'down')}></span>
            {online ? 'LOCAL AI ONLINE' : 'MODEL OFFLINE'}</div>
          <small>CRIMSON SYSTEM / 0.1</small>
        </div>
      </header>

      <div className="workspace">
        <aside className="side">
          <div className="section-label">/ NAVIGATION</div>
          {NAV.map(([id, glyph, name]) => (
            <button key={id} type="button"
              className={'nav-button ' + (tab === id ? 'selected' : '')}
              onClick={() => changeTab(id)}>
              <span className="nav-glyph">{glyph}</span><span>{name}</span>
            </button>
          ))}
          <div className="side-footer">
            <div className="section-label">/ ENGINE</div>
            <div className="engine">{model}</div>
            <div>CPU · LOCAL DOCKER</div>
            <button type="button" className="refresh-button" onClick={refresh}>
              ↻ ACTUALIZAR ESTADO
            </button>
          </div>
        </aside>

        <main className="main">
          <div className="main-top">
            <div><h1>/ {NAV.find(n => n[0] === tab)?.[2]}</h1>
              <p>NYX CORE / LOCAL INTERFACE</p></div>
            <span className="prototype">V0.1 / PROTOTYPE</span>
          </div>
          {error && <div className="alert" role="alert">
            <span>⚠ {error}</span>
            <button type="button" onClick={() => setError('')}>×</button>
          </div>}

          {tab === 'chat' && <>
            <div className="panel conversation" aria-live="polite">
              <div className="panel-meta">SESSION / LOCAL HISTORY</div>
              {messages.length === 0 && <div className="welcome">
                <div className="assistant-avatar">N</div>
                <div><strong>NYX / ASSISTANT</strong>
                  <p>Sistema preparado. Soy NYX, tu asistente local. ¿En qué puedo ayudarte?</p>
                </div>
              </div>}
              {messages.map(m => <div className={'message ' + m.role} key={m.id}>
                <div className="message-label">
                  {m.role === 'assistant' ? 'NYX / ASSISTANT' : 'OPERADOR / USER'}
                </div>
                <div className="message-text">{m.content}</div>
              </div>)}
              {busy && <div className="thinking">
                NYX está procesando la respuesta en tu CPU...
              </div>}
              <div ref={bottomRef}></div>
            </div>
            <div className="chat-controls">
              <label className="voice-control">
                <input type="checkbox" checked={voice}
                  onChange={e => setVoice(e.target.checked)}/>
                ◉ LEER RESPUESTAS (VOZ EXPERIMENTAL)
              </label>
              {voice && <button type="button"
                onClick={() => window.speechSynthesis?.cancel()}>
                ■ DETENER VOZ
              </button>}
            </div>
            <form className="composer" onSubmit={send}>
              <textarea rows="2" maxLength={4000} value={draft}
                placeholder="Escribe una instrucción para NYX..."
                onChange={e => setDraft(e.target.value)}
                onKeyDown={e => {
                  if (e.key === 'Enter' && !e.shiftKey) {
                    e.preventDefault();
                    e.currentTarget.form.requestSubmit();
                  }
                }}/>
              <button type="submit" disabled={!draft.trim() || busy}>
                {busy ? 'PROCESANDO' : 'ENVIAR ↗'}
              </button>
            </form>
          </>}

          {tab === 'history' && <div className="panel flex-panel">
            <p className="panel-meta">/ REGISTRO PERSISTENTE · SQLITE</p>
            <p className="explanation">El historial se conserva en el volumen Docker nyx_memory.</p>
            <div className="history-list">
              {messages.length ? messages.map(m =>
                <div className="history-item" key={m.id}>
                  <strong>{m.role === 'assistant' ? 'NYX' : 'TÚ'}</strong>
                  <span>{m.content}</span>
                </div>) : <p className="empty">Aún no hay conversaciones.</p>}
            </div>
          </div>}

          {(tab === 'tasks' || tab === 'memory') &&
            <div className="panel flex-panel">
              <p className="panel-meta">
                {tab === 'tasks' ? '/ GESTIÓN DE TAREAS' : '/ NOTAS DE MEMORIA EXPLÍCITAS'}
              </p>
              <p className="explanation">{tab === 'tasks'
                ? 'Crea y completa tareas. Todavía no se generan recordatorios automáticos.'
                : 'Solo se guardan las notas que agregues aquí. Se incluyen como contexto al conversar.'}
              </p>
              <form className="item-form" onSubmit={createItem}>
                <input maxLength={500} value={itemDraft}
                  placeholder={tab === 'tasks' ? 'Nueva tarea...' : 'Un dato que NYX debe recordar...'}
                  onChange={e => setItemDraft(e.target.value)}/>
                <button type="submit" disabled={!itemDraft.trim()}>+ GUARDAR</button>
              </form>
              <div className="item-list">
                {(tab === 'tasks' ? tasks : memories).length === 0 &&
                  <p className="empty">No hay elementos registrados.</p>}
                {(tab === 'tasks' ? tasks : memories).map(item =>
                  <div className="item" key={item.id}>
                    {tab === 'tasks' &&
                      <input aria-label="Marcar tarea" type="checkbox"
                        checked={item.completed} onChange={() => toggleTask(item)}/>}
                    <span className={item.completed ? 'completed' : ''}>
                      {tab === 'tasks' ? item.title : item.content}
                    </span>
                    <button type="button" className="delete"
                      onClick={() => removeItem(item.id)} aria-label="Eliminar">×</button>
                  </div>)}
              </div>
            </div>}

          <footer>NYX V0.1 // PERSONAL LOCAL ASSISTANT // CRIMSON SYSTEM</footer>
        </main>
      </div>
    </div>
  );
}
