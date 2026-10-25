import React, { useCallback, useEffect, useRef, useState } from 'react';
import './styles.css';
import { splitSpeechText } from './voiceChunks.js';
import AuthPortal from './AuthPortal.jsx';
import AdminPanel from './AdminPanel.jsx';
import CalendarPanel from './CalendarPanel.jsx';
import BrainNotebook from './BrainNotebook.jsx';

let csrfToken = '';


const MENU = [
  { id: 'chat', icon: '◇', label: 'CONVERSACIÓN' },
  { id: 'history', icon: '◷', label: 'BITÁCORA' },
  { id: 'tasks', icon: '☷', label: 'TAREAS' },
  { id: 'memory', icon: '▤', label: 'MEMORIA' },
  { id: 'settings', icon: '⚙', label: 'CONFIGURACIÓN' },
  { id: 'security', icon: '♢', label: 'SEGURIDAD' },
];

async function api(path, options = {}) {
  let res;
  try {
    res = await fetch(path, { credentials: 'same-origin', ...options,
      headers: { 'Content-Type': 'application/json',
        ...(csrfToken ? { 'X-NYX-CSRF': csrfToken } : {}), ...(options.headers || {}) },
    });
  } catch {
    throw new Error('No se pudo conectar con el núcleo de NYX.');
  }
  if (!res.ok) {
    if (res.status === 401 && path !== '/api/auth/login' && path !== '/api/auth/me') {
      window.dispatchEvent(new Event('nyx-session-expired'));
    }
    let detail = 'Error HTTP ' + res.status;
    try { detail = (await res.json()).detail || detail; } catch { /* respuesta no JSON */ }
    throw new Error(typeof detail === 'string' ? detail : 'Solicitud no válida');
  }
  return res.status === 204 ? null : res.json();
}

function toDate(date) {
  if (!date) return null;
  return new Date(date.replace(' ', 'T') + (date.includes('Z') ? '' : 'Z'));
}
function shortDate(date) {
  const d = toDate(date);
  return d ? d.toLocaleString('es-CR', { day: '2-digit', month: 'short', hour: '2-digit', minute: '2-digit' }) : '';
}
function time(date) {
  const d = toDate(date);
  return d ? d.toLocaleTimeString('es-CR', { hour: '2-digit', minute: '2-digit' }) : '';
}
function dateGroup(date) {
  const d = toDate(date);
  if (!d) return 'ANTERIORES';
  const now = new Date();
  const day = x => new Date(x.getFullYear(), x.getMonth(), x.getDate()).getTime();
  const diff = Math.floor((day(now) - day(d)) / 86400000);
  if (diff === 0) return 'HOY';
  if (diff === 1) return 'AYER';
  if (diff >= 0 && diff < 7) return 'ESTA SEMANA';
  return 'ANTERIORES';
}

function pcmWav(samples, sampleRate = 16000) {
  const dataSize = samples.length * 2;
  const buffer = new ArrayBuffer(44 + dataSize);
  const view = new DataView(buffer);
  const word = (offset, value) => view.setUint32(offset, value, true);
  const half = (offset, value) => view.setUint16(offset, value, true);
  const ascii = (offset, value) => { for (let i = 0; i < value.length; i++) view.setUint8(offset + i, value.charCodeAt(i)); };
  ascii(0, 'RIFF'); word(4, 36 + dataSize); ascii(8, 'WAVE');
  ascii(12, 'fmt '); word(16, 16); half(20, 1); half(22, 1);
  word(24, sampleRate); word(28, sampleRate * 2); half(32, 2); half(34, 16);
  ascii(36, 'data'); word(40, dataSize);
  for (let i = 0; i < samples.length; i++) {
    const x = Math.max(-1, Math.min(1, samples[i]));
    view.setInt16(44 + i * 2, x < 0 ? x * 0x8000 : x * 0x7fff, true);
  }
  return buffer;
}

function Core({ state }) {
  return <div className={'core-container core-' + state} aria-label={'Núcleo: ' + state}>
    <div className="core-halo" />
    <div className="core-orbit orbit-one" /><div className="core-orbit orbit-two" />
    <div className="core-diamond"><span>N</span></div>
    <div className="core-side left" /><div className="core-side right" />
  </div>;
}

export default function App() {
  const [tab, setTab] = useState('chat');
  const [auth, setAuth] = useState(null);
  const [authReady, setAuthReady] = useState(false);
  const [status, setStatus] = useState({ online: false, model: 'nyx:0.4' });
  const [networkStatus, setNetworkStatus] = useState({enabled: true, connected: false, mode: 'offline'});
  const [alertInfo, setAlertInfo] = useState({pending: 0, queued: 0});
  const [voiceReady, setVoiceReady] = useState({ tts_ready: false, stt_ready: false });
  const [sessions, setSessions] = useState([]);
  const [selectedId, setSelectedId] = useState(null);
  const [messages, setMessages] = useState([]);
  const [tasks, setTasks] = useState([]);
  const [dueAlerts, setDueAlerts] = useState([]);
  const [memories, setMemories] = useState([]);
  const [knowledge, setKnowledge] = useState([]);
  const [brainRatings, setBrainRatings] = useState({});
  const [ratingBusy, setRatingBusy] = useState(false);
  const [profile, setProfile] = useState({
    timezone: 'America/Costa_Rica', city: '', country: '', tone: 'natural',
  });
  const [clockInfo, setClockInfo] = useState(null);
  const [dueDate, setDueDate] = useState('');
  const [savedProfile, setSavedProfile] = useState(false);
  const [draft, setDraft] = useState('');
  const [itemDraft, setItemDraft] = useState('');
  const [search, setSearch] = useState('');
  const [error, setError] = useState('');
  const [busy, setBusy] = useState(false);
  const [voiceEnabled, setVoiceEnabled] = useState(false);
  const [voiceProfile, setVoiceProfile] = useState('hybrid');
  const [voiceState, setVoiceState] = useState('idle');
  const [sideOpen, setSideOpen] = useState(() => window.innerWidth > 680);
  const [rightOpen, setRightOpen] = useState(true);
  const bottomRef = useRef(null);
  const waveRef = useRef(null);
  const analyserRef = useRef(null);
  const audioCtxRef = useRef(null);
  const currentSourceRef = useRef(null);
  const voiceJobRef = useRef({ id: 0, abort: null });
  const [voiceProgress, setVoiceProgress] = useState({ current: 0, total: 0 });
  const recorderRef = useRef(null);
  const micStreamRef = useRef(null);
  const micCtxRef = useRef(null);
  const micTimerRef = useRef(null);
  const recordingRef = useRef(false);

  const activeSession = sessions.find(s => s.id === selectedId);
  const modelAvailable = Boolean(status.online);
  const systemOnline = modelAvailable && Boolean(networkStatus.enabled && networkStatus.connected);
  const systemLabel = !modelAvailable ? 'MODELO NO DISPONIBLE' : systemOnline ? 'ONLINE' : 'LOCAL';

  useEffect(() => {
    if (!auth || auth.must_change) return;
    let active = true;
    const seen = new Set();
    async function pollReminders() {
      try {
        const pending = await api('/api/calendar/reminders');
        if (!active) return;
        setDueAlerts(previous => {
          const known = new Set(previous.map(item => item.id));
          return [...previous, ...pending.filter(item => !known.has(item.id))].slice(-15);
        });
        for (const item of pending) {
          if (seen.has(item.id)) continue;
          seen.add(item.id);
          if (typeof Notification !== 'undefined' && Notification.permission === 'granted') {
            new Notification('NYX · Recordatorio', {
              body: item.title + ' · ' + item.start_at.replace('T', ' '),
            });
          }
        }
      } catch { /* Sin alarmas ficticias mientras el servidor está desconectado. */ }
    }
    pollReminders();
    const interval = setInterval(pollReminders, 60000);
    return () => { active = false; clearInterval(interval); };
  }, [auth?.username, auth?.must_change]);

  async function dismissReminder(item) {
    try {
      await api('/api/calendar/' + item.id + '/acknowledge', { method: 'POST' });
      setDueAlerts(previous => previous.filter(row => row.id !== item.id));
    } catch (err) { setError(err.message); }
  }

  const refreshSessions = useCallback(async (q = '') => {
    const rows = await api('/api/sessions' + (q ? '?q=' + encodeURIComponent(q) : ''));
    setSessions(rows);
    return rows;
  }, []);

  const refresh = useCallback(async () => {
    const results = await Promise.allSettled([
      api('/api/sessions'), api('/api/tasks'), api('/api/memory'),
      api('/api/status'), api('/api/voice/status'),
      api('/api/profile'), api('/api/context'), api('/api/network/status'),
      api('/api/knowledge'), api('/api/brain/feedback'),
    ]);
    if (results[0].status === 'fulfilled') {
      setSessions(results[0].value);
      if (results[0].value.length) {
        const latest = results[0].value[0].id;
        setSelectedId(latest);
        try { setMessages(await api('/api/sessions/' + latest + '/messages')); }
        catch { setMessages([]); }
      }
    }
    if (results[1].status === 'fulfilled') setTasks(results[1].value);
    if (results[2].status === 'fulfilled') setMemories(results[2].value);
    if (results[3].status === 'fulfilled') setStatus(results[3].value);
    if (results[4].status === 'fulfilled') setVoiceReady(results[4].value);
    if (results[5].status === 'fulfilled') setProfile(results[5].value);
    if (results[6].status === 'fulfilled') setClockInfo(results[6].value);
    if (results[7].status === 'fulfilled') setNetworkStatus(results[7].value);
    if (results[8].status === 'fulfilled') setKnowledge(results[8].value);
    if (results[9].status === 'fulfilled')
      setBrainRatings(Object.fromEntries(results[9].value.map(item => [item.assistant_message_id, item])));
    if (results.some(r => r.status === 'rejected')) setError('Algunos servicios no respondieron.');
  }, []);

  useEffect(() => {
    let active = true;
    async function resume() {
      try {
        const result = await api('/api/auth/me');
        if (active) { csrfToken = result.csrf; setAuth(result); }
      } catch { if (active) { csrfToken = ''; setAuth(null); } }
      finally { if (active) setAuthReady(true); }
    }
    resume();
    const expired = () => {
      csrfToken = '';
      setAuth(null);
      setSelectedId(null); setSessions([]); setMessages([]);
      setTasks([]); setMemories([]); setKnowledge([]); setDueAlerts([]); setBrainRatings({}); setTab('chat');
    };
    window.addEventListener('nyx-session-expired', expired);
    return () => { active = false; window.removeEventListener('nyx-session-expired', expired); };
  }, []);

  useEffect(() => {
    if (auth && !auth.must_change) refresh();
  }, [auth?.id, auth?.must_change, refresh]);

  useEffect(() => {
    if (auth?.role !== 'admin' || auth?.must_change) {
      setAlertInfo({pending: 0, queued: 0});
      return;
    }
    let active = true;
    api('/api/security/unreviewed').then(stats => {
      if (active) setAlertInfo(stats);
    }).catch(() => {});
    return () => { active = false; };
  }, [auth?.id, auth?.role, auth?.must_change, tab]);
  useEffect(() => {
    if (tab === 'chat') bottomRef.current?.scrollIntoView({ behavior: 'smooth', block: 'end' });
  }, [messages, tab, busy]);

  useEffect(() => {
    const canvas = waveRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    let raf;
    const draw = () => {
      const w = canvas.width, h = canvas.height;
      ctx.clearRect(0, 0, w, h);
      ctx.strokeStyle = 'rgba(220,20,60,.16)';
      ctx.lineWidth = 1;
      ctx.beginPath(); ctx.moveTo(0, h / 2); ctx.lineTo(w, h / 2); ctx.stroke();
      const analyser = analyserRef.current;
      if (analyser && (voiceState === 'speaking' || voiceState === 'listening')) {
        const values = new Uint8Array(analyser.fftSize);
        analyser.getByteTimeDomainData(values);
        ctx.strokeStyle = '#ff365f';
        ctx.lineWidth = 2;
        ctx.shadowColor = 'rgba(255,42,90,.75)'; ctx.shadowBlur = 9;
        ctx.beginPath();
        const step = w / (values.length - 1);
        values.forEach((v, i) => {
          const y = (v / 255) * h;
          if (i === 0) ctx.moveTo(0, y);
          else ctx.lineTo(i * step, y);
        });
        ctx.stroke(); ctx.shadowBlur = 0;
      }
      raf = requestAnimationFrame(draw);
    };
    draw();
    return () => cancelAnimationFrame(raf);
  }, [voiceState, tab, rightOpen, auth?.id]);

  useEffect(() => () => {
    voiceJobRef.current.abort?.abort();
    currentSourceRef.current?.stop?.();
    if (recorderRef.current?.state === 'recording') recorderRef.current.stop();
    micStreamRef.current?.getTracks().forEach(track => track.stop());
    micCtxRef.current?.close?.();
    audioCtxRef.current?.close?.();
    if (micTimerRef.current) clearTimeout(micTimerRef.current);
  }, []);

  function ensureAudio() {
    const AudioCtor = window.AudioContext || window.webkitAudioContext;
    if (!AudioCtor) throw new Error('Este navegador no admite Web Audio.');
    if (!audioCtxRef.current || audioCtxRef.current.state === 'closed') {
      audioCtxRef.current = new AudioCtor();
    }
    audioCtxRef.current.resume();
    return audioCtxRef.current;
  }

  function stopVoice() {
    voiceJobRef.current.id += 1;
    voiceJobRef.current.abort?.abort();
    voiceJobRef.current.abort = null;
    if (currentSourceRef.current) {
      try { currentSourceRef.current.stop(); } catch { /* ya terminó */ }
      currentSourceRef.current = null;
    }
    analyserRef.current = null;
    setVoiceProgress({ current: 0, total: 0 });
    setVoiceState('idle');
  }

  function playBuffer(ctx, audio, mode, jobId) {
    return new Promise((resolve) => {
      if (voiceJobRef.current.id !== jobId) {
        resolve();
        return;
      }
      const source = ctx.createBufferSource();
      source.buffer = audio;
      const analyser = ctx.createAnalyser();
      analyser.fftSize = 1024;
      const master = ctx.createGain();
      master.gain.value = 0.9;
      if (mode === 'system' && voiceProfile === 'hybrid') {
        const lowpass = ctx.createBiquadFilter();
        lowpass.type = 'lowpass';
        lowpass.frequency.value = 2600;
        const delay = ctx.createDelay(0.2);
        delay.delayTime.value = 0.038;
        const echo = ctx.createGain();
        echo.gain.value = 0.15;
        source.connect(master);
        source.connect(lowpass);
        lowpass.connect(delay);
        delay.connect(echo);
        echo.connect(master);
      } else {
        source.connect(master);
      }
      master.connect(analyser);
      analyser.connect(ctx.destination);
      currentSourceRef.current = source;
      analyserRef.current = analyser;
      source.onended = () => {
        if (currentSourceRef.current === source) {
          currentSourceRef.current = null;
          analyserRef.current = null;
        }
        resolve();
      };
      source.start();
      setVoiceState('speaking');
    });
  }

  async function playVoice(text, mode = 'conversation') {
    if (!voiceReady.tts_ready) {
      setError('Instala primero el modelo Piper para utilizar la voz local.');
      return;
    }
    stopVoice();
    const fragments = splitSpeechText(text);
    if (!fragments.length) return;
    const jobId = voiceJobRef.current.id;
    const controller = new AbortController();
    voiceJobRef.current.abort = controller;
    setVoiceProgress({ current: 0, total: fragments.length });
    setVoiceState('processing');

    try {
      const ctx = ensureAudio();
      // Los rechazos de la siguiente descarga quedan capturados antes de empezar
      // a reproducir el fragmento actual; así no hay promesas sin controlar.
      const synthesize = async (fragment) => {
        try {
          const response = await fetch('/api/voice/speak', {
            method: 'POST',
            signal: controller.signal,
            headers: { 'Content-Type': 'application/json', 'X-NYX-CSRF': csrfToken },
            body: JSON.stringify({ text: fragment, mode }),
          });
          if (!response.ok) {
            let detail = 'Fallo del sintetizador de voz (' + response.status + ')';
            try { detail = (await response.json()).detail || detail; } catch { /* sin JSON */ }
            throw new Error(detail);
          }
          const bytes = await response.arrayBuffer();
          return { audio: await ctx.decodeAudioData(bytes) };
        } catch (error) {
          return { error };
        }
      };

      let upcoming = synthesize(fragments[0]);
      for (let index = 0; index < fragments.length; index++) {
        const result = await upcoming;
        if (voiceJobRef.current.id !== jobId || controller.signal.aborted) break;
        if (result.error) throw result.error;
        upcoming = index + 1 < fragments.length
          ? synthesize(fragments[index + 1])
          : Promise.resolve(null);
        setVoiceProgress({ current: index + 1, total: fragments.length });
        await playBuffer(ctx, result.audio, mode, jobId);
        if (voiceJobRef.current.id !== jobId || controller.signal.aborted) break;
      }
    } catch (err) {
      if (!controller.signal.aborted && voiceJobRef.current.id === jobId) {
        setError('La lectura se interrumpió: ' + err.message);
      }
    } finally {
      if (voiceJobRef.current.id === jobId) {
        voiceJobRef.current.abort = null;
        analyserRef.current = null;
        currentSourceRef.current = null;
        setVoiceProgress({ current: 0, total: 0 });
        setVoiceState('idle');
      }
    }
  }

  async function selectSession(id, openChat = true) {
    try {
      setMessages(await api('/api/sessions/' + id + '/messages'));
      setSelectedId(id); if (openChat) setTab('chat'); setError('');
    } catch (err) { setError(err.message); }
  }
  function newSession() {
    stopVoice(); setSelectedId(null); setMessages([]); setDraft('');
    setTab('chat'); setError('');
  }

  async function loginUser(username, password) {
    csrfToken = '';
    const result = await api('/api/auth/login', {
      method: 'POST',
      body: JSON.stringify({ username, password }),
    });
    csrfToken = result.csrf;
    setAuth(result);
    setError('');
    setTab('chat');
    setSelectedId(null);
    setMessages([]); setSessions([]); setTasks([]); setMemories([]); setKnowledge([]); setDueAlerts([]); setBrainRatings({});
  }

  async function updateAuth() {
    const result = await api('/api/auth/me');
    csrfToken = result.csrf;
    setAuth(result);
  }

  async function changeOwnPassword(current_password, new_password) {
    await api('/api/auth/change-password', {
      method: 'POST',
      body: JSON.stringify({ current_password, new_password }),
    });
    csrfToken = '';
    setAuth(null);
    setSelectedId(null); setSessions([]); setMessages([]);
    setTasks([]); setMemories([]); setKnowledge([]); setDueAlerts([]); setBrainRatings({});
    setTab('chat');
  }

  async function logoutUser() {
    try {
      await api('/api/auth/logout', { method: 'POST' });
    } catch { /* invalidar igualmente el estado visible */ }
    stopVoice();
    csrfToken = '';
    setAuth(null);
    setSelectedId(null); setMessages([]); setSessions([]);
    setTasks([]); setMemories([]); setKnowledge([]); setDueAlerts([]); setBrainRatings({});
    setProfile({ timezone:'America/Costa_Rica', city:'', country:'', tone:'natural' });
    setTab('chat');
    setError('');
  }

  async function changeNetworkMode(enabled) {
    if (!auth?.elevated) {
      setError('Primero debes autorizar cambios administrativos en Seguridad.');
      return;
    }
    try {
      await api('/api/network/mode', {
        method: 'PUT', body: JSON.stringify({enabled}),
      });
      setNetworkStatus(await api('/api/network/status'));
      setError('');
    } catch (err) { setError(err.message); }
  }

  async function rateReply(message, rating) {
    if (ratingBusy || typeof message.id !== 'number') return;
    let note = '';
    if (rating === 'unhelpful') {
      const proposed = window.prompt(
        '¿Qué fue incorrecto o qué debería mejorar NYX? (opcional, máximo 400 caracteres)',
        brainRatings[message.id]?.note || ''
      );
      if (proposed === null) return;
      note = proposed.trim().slice(0, 400);
    }
    setRatingBusy(true); setError('');
    try {
      const saved = await api('/api/brain/feedback', {
        method: 'POST',
        body: JSON.stringify({ assistant_message_id: message.id, rating, note }),
      });
      setBrainRatings(prev => ({ ...prev, [message.id]: saved }));
    } catch (err) { setError(err.message); }
    finally { setRatingBusy(false); }
  }

  async function send(event) {
    event.preventDefault();
    const text = draft.trim();
    if (!text || busy) return;
    if (voiceEnabled && voiceReady.tts_ready) {
      try { ensureAudio(); } catch (err) { setError(err.message); }
    }
    stopVoice();
    setDraft(''); setBusy(true); setError('');
    const sessionAtSend = selectedId;
    setMessages(prev => [...prev, {
      id: 'pending-' + Date.now(), role: 'user', content: text,
    }]);
    setVoiceState('processing');
    try {
      const data = await api('/api/chat', {
        method: 'POST',
        body: JSON.stringify({ text, session_id: sessionAtSend }),
      });
      setSelectedId(data.session_id);
      setMessages(await api('/api/sessions/' + data.session_id + '/messages'));
      await refreshSessions();
      if (data.tool_events?.length) {
        const changed = await Promise.allSettled([api('/api/tasks'), api('/api/memory')]);
        if (changed[0].status === 'fulfilled') setTasks(changed[0].value);
        if (changed[1].status === 'fulfilled') setMemories(changed[1].value);
      }
      // La búsqueda pública puede actualizar la base de conocimiento incluso sin tool_calls.
      if (/^(?:nyx[,!:]?\s*)?(?:busca|buscar|investiga|investigar|consulta|averigua)/i.test(text))
        api('/api/knowledge').then(setKnowledge).catch(() => {});
      if (voiceEnabled && voiceReady.tts_ready) void playVoice(data.reply);
      else setVoiceState('idle');
    } catch (err) {
      setVoiceState('idle');
      setError(err.message);
      if (sessionAtSend) {
        try { setMessages(await api('/api/sessions/' + sessionAtSend + '/messages')); }
        catch { /* fuera de linea */ }
      }
      await refreshSessions().catch(() => {});
    } finally { setBusy(false); }
  }

  async function createItem(event) {
    event.preventDefault();
    const value = itemDraft.trim();
    if (!value) return;
    try {
      const resource = tab === 'tasks' ? 'tasks' : 'memory';
      await api('/api/' + resource, {
        method: 'POST', body: JSON.stringify(tab === 'tasks' ? { text: value, due_date: dueDate || null } : { text: value }),
      });
      setItemDraft('');
      if (tab === 'tasks') setDueDate('');
      if (tab === 'tasks') setTasks(await api('/api/tasks'));
      else setMemories(await api('/api/memory'));
    } catch (err) { setError(err.message); }
  }

  async function saveProfile(event) {
    event.preventDefault();
    if (!auth?.elevated) {
      setError('Autoriza primero los cambios desde el panel de seguridad.');
      return;
    }
    setSavedProfile(false);
    try {
      const result = await api('/api/profile', {
        method: 'PUT', body: JSON.stringify(profile),
      });
      setProfile(result);
      setClockInfo(await api('/api/context'));
      setSavedProfile(true);
      setError('');
    } catch (err) {
      setError(err.message);
    }
  }
  async function updateTask(task) {
    try {
      await api('/api/tasks/' + task.id, {
        method: 'PATCH', body: JSON.stringify({ completed: !task.completed }),
      });
      setTasks(await api('/api/tasks'));
    } catch (err) { setError(err.message); }
  }
  async function deleteItem(id) {
    if (!window.confirm('¿Eliminar este elemento?')) return;
    try {
      await api('/api/' + tab + '/' + id, { method: 'DELETE' });
      if (tab === 'tasks') setTasks(await api('/api/tasks'));
      else setMemories(await api('/api/memory'));
    } catch (err) { setError(err.message); }
  }
  async function renameSession(session) {
    const title = window.prompt('Nuevo título de la sesión:', session.title)?.trim();
    if (!title || title === session.title) return;
    try {
      await api('/api/sessions/' + session.id, {
        method: 'PATCH', body: JSON.stringify({ title: title.slice(0, 90) }),
      });
      await refreshSessions(search);
    } catch (err) { setError(err.message); }
  }
  async function deleteSession(session) {
    if (!window.confirm('¿Eliminar definitivamente la sesión y todos sus mensajes?')) return;
    try {
      await api('/api/sessions/' + session.id, { method: 'DELETE' });
      const rows = await refreshSessions(search);
      if (selectedId === session.id) {
        if (rows.length) await selectSession(rows[0].id);
        else newSession();
      }
    } catch (err) { setError(err.message); }
  }

  async function startMic() {
    if (!voiceReady.stt_ready) {
      setError('Instala el modelo local Vosk antes de activar el micrófono.');
      return;
    }
    try {
      stopVoice();
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      micStreamRef.current = stream;
      const AudioCtor = window.AudioContext || window.webkitAudioContext;
      const ctx = new AudioCtor();
      micCtxRef.current = ctx;
      const micNode = ctx.createMediaStreamSource(stream);
      const analyser = ctx.createAnalyser();
      analyser.fftSize = 1024;
      micNode.connect(analyser);
      analyserRef.current = analyser;
      const chunks = [];
      const recorder = new MediaRecorder(stream);
      recorderRef.current = recorder;
      recorder.ondataavailable = e => { if (e.data.size) chunks.push(e.data); };
      recorder.onstop = async () => {
        recordingRef.current = false;
        stream.getTracks().forEach(track => track.stop());
        micStreamRef.current = null;
        analyserRef.current = null;
        setVoiceState('processing');
        try {
          const recorded = new Blob(chunks, { type: recorder.mimeType });
          const audio = await ctx.decodeAudioData(await recorded.arrayBuffer());
          const length = Math.ceil(audio.duration * 16000);
          const offline = new OfflineAudioContext(1, length, 16000);
          const source = offline.createBufferSource();
          source.buffer = audio;
          source.connect(offline.destination);
          source.start();
          const resampled = await offline.startRendering();
          const res = await fetch('/api/voice/transcribe', {
            method: 'POST', headers: { 'Content-Type': 'audio/wav', 'X-NYX-CSRF': csrfToken },
            body: pcmWav(resampled.getChannelData(0)),
          });
          if (!res.ok) {
            const payload = await res.json();
            throw new Error(payload.detail || 'No se pudo reconocer la voz');
          }
          const result = await res.json();
          if (!result.text) throw new Error('No se reconocieron palabras. Intenta de nuevo.');
          setDraft(result.text);
          setTab('chat');
        } catch (err) { setError(err.message); }
        finally { setVoiceState('idle'); ctx.close(); }
      };
      recorder.start();
      recordingRef.current = true;
      setVoiceState('listening');
      setError('');
      micTimerRef.current = setTimeout(() => stopMic(), 18000);
    } catch (err) {
      micStreamRef.current?.getTracks().forEach(track => track.stop());
      setError('No se pudo utilizar el micrófono: ' + err.message);
      setVoiceState('idle');
    }
  }
  function stopMic() {
    if (micTimerRef.current) clearTimeout(micTimerRef.current);
    if (recorderRef.current?.state === 'recording') recorderRef.current.stop();
  }

  function exportSession() {
    if (!activeSession) return;
    const lines = messages.map(m => '[' + shortDate(m.created_at) + '] ' +
      (m.role === 'assistant' ? 'NYX' : 'USUARIO') + ':\n' + m.content);
    const blob = new Blob([activeSession.title + '\n\n' + lines.join('\n\n')], { type: 'text/plain;charset=utf-8' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url; a.download = 'nyx-sesion-' + activeSession.id + '.txt'; a.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  }

  const grouped = sessions.reduce((acc, session) => {
    const key = dateGroup(session.updated_at);
    (acc[key] ||= []).push(session);
    return acc;
  }, {});

  if (!authReady) {
    return <div className="nyx-auth-page"><p>INICIALIZANDO NYX SECURITY...</p></div>;
  }
  if (!auth) {
    return <AuthPortal onLogin={loginUser}/>;
  }
  if (auth.must_change) {
    return <AuthPortal mode="change" username={auth.username}
      onChangePassword={changeOwnPassword}/>;
  }

  return <div className="nyx-app">
    <header className="topbar">
      <div className="brand">
        <button type="button" className="nav-collapse" title="Mostrar navegación" onClick={() => setSideOpen(!sideOpen)}>☰</button>
        <div className="brand-sigil"><span>N</span></div>
        <div className="brand-text"><strong>NYX<span>_</span></strong></div>
      </div>
      <div className="top-center"><span className="bar-line" /></div>
      <div className="top-state" aria-live="polite">
        <span className={'led ' + (systemOnline ? 'on' : modelAvailable ? 'local' : 'off')} />
        <span>{systemOnline ? 'SISTEMA EN LÍNEA' : systemLabel}</span>
      </div>
    </header>
    <div className="shell">
      {sideOpen && <aside className="sidebar">
        <div className="section-caption">/ NAVIGATION MODULE</div>
        <button className="start-session" type="button" onClick={newSession}><span>＋</span> NUEVA CONVERSACIÓN</button>
        <nav aria-label="Navegación principal">
          {MENU.filter(item => !['settings','security'].includes(item.id) || auth.role === 'admin').map(item => <button type="button" key={item.id}
            className={'nav-item ' + (tab === item.id ? 'active' : '')}
            onClick={() => { setTab(item.id); setItemDraft(''); setError(''); }}>
            <span className="nav-icon">{item.icon}</span><span>{item.label}</span>
            {tab === item.id && <i />}
          </button>)}
        </nav>
        <div className="side-bottom">
          <div className="section-caption">/ NYX</div>
          <div className="meter"><span>VERSIÓN</span><strong>{status.version || '0.6.0'}</strong></div>
          <button type="button" className="plain-link" onClick={() => refresh()}>↻ SINCRONIZAR</button>
        </div>
      </aside>}
      <main className="content">
        <div className="content-head">
          <div><div className="eyebrow">NYX / {tab.toUpperCase()} / SYSTEM INTERFACE</div>
            <h1>{MENU.find(n => n.id === tab)?.label}</h1></div>
          <div className="head-actions">
            {auth.role === 'admin' && <span className="build-tag">◈ NYX {status.version || '0.6.0'}</span>}
            <span className="nyx-login-identity">{auth.username} · {auth.role === 'admin' ? 'ADMIN' : 'USUARIO'}</span>
            <button className="nyx-logout" title="Cerrar sesión" onClick={logoutUser}>⏻ SALIR</button>
            <button className="small-icon" title="Mostrar u ocultar panel de estado" onClick={() => setRightOpen(!rightOpen)}>▣</button>
          </div>
        </div>
        {auth.role === 'admin' && alertInfo.pending > 0 && tab !== 'security' &&
          <button className="nyx-alert-banner" onClick={() => setTab('security')}>
            ⚠ SECURITY SENTINEL · {alertInfo.pending} alerta{alertInfo.pending === 1 ? '' : 's'} pendiente{alertInfo.pending === 1 ? '' : 's'}
            {alertInfo.queued > 0 ? ' · ' + alertInfo.queued + ' correo(s) en cola' : ''} · REVISAR
          </button>}
        {error && <div className="error-bar" role="alert"><span>⚠ {error}</span>
          <button onClick={() => setError('')} aria-label="Cerrar alerta">×</button></div>}
        {dueAlerts.length > 0 && <div className="nyx-reminder-alerts" role="region" aria-label="Recordatorios pendientes">
          {dueAlerts.map(item => <div className="nyx-reminder-alert" key={item.id} role="alert">
            <div><b>◆ RECORDATORIO · {item.kind.toUpperCase()}</b>
              <strong>{item.title}</strong><small>{item.start_at.replace('T', ' · ')}</small></div>
            <button type="button" onClick={() => dismissReminder(item)}>ENTENDIDO</button>
          </div>)}
        </div>}

        {tab === 'chat' && <div className="chat-layout">
          <div className="chat-topline"><span>● SESSION / {activeSession ? String(activeSession.id).padStart(3,'0') : 'NEW'}</span>
            <span>{activeSession?.title || 'Conversación nueva'}</span></div>
          <div className="chat-scroll" aria-live="polite">
            {messages.length === 0 && <div className="intro">
              <div className="intro-badge">NYX · ASSISTANT</div>
              <h2>Sistema preparado<span>_</span></h2>
              <p>Tu asistente de inteligencia artificial local está listo. Escribe un mensaje o utiliza el micrófono.</p>
              <div className="intro-chips">
                <button onClick={() => setDraft('Ayúdame a organizar mis tareas de hoy')}>⌁ ORGANIZAR TAREAS</button>
                <button onClick={() => setDraft('Explícame un concepto de Python')}>⌁ ESTUDIAR PYTHON</button>
              </div>
            </div>}
            {messages.map(m => <div className={'bubble-line ' + m.role} key={m.id}>
              <div className="message-mark">{m.role === 'assistant' ? 'N' : '›'}</div>
              <div className="bubble">
                <div className="bubble-head"><b>{m.role === 'assistant' ? 'NYX / ASSISTANT' : 'USUARIO / OPERADOR'}</b>
                  <span>{time(m.created_at)}</span></div>
                <div className="bubble-body">{m.content}</div>
                {m.role === 'assistant' && voiceReady.tts_ready &&
                  <button className="speak-again" type="button" onClick={() => playVoice(m.content)}>▶ ESCUCHAR</button>}
                {m.role === 'assistant' && typeof m.id === 'number' && <div className="brain-feedback-actions" aria-label="Evaluar respuesta de NYX">
                  <button type="button" title="La respuesta fue útil"
                    aria-pressed={brainRatings[m.id]?.rating === 'helpful'}
                    className={brainRatings[m.id]?.rating === 'helpful' ? 'rated' : ''}
                    disabled={ratingBusy} onClick={() => rateReply(m, 'helpful')}>✓ ÚTIL</button>
                  <button type="button" title="Corregir o informar un problema"
                    aria-pressed={brainRatings[m.id]?.rating === 'unhelpful'}
                    className={brainRatings[m.id]?.rating === 'unhelpful' ? 'rated negative' : ''}
                    disabled={ratingBusy} onClick={() => rateReply(m, 'unhelpful')}>✕ CORREGIR</button>
                  {brainRatings[m.id] && <span>VALORACIÓN GUARDADA · AYUDA A MEJORAR FUTURAS RESPUESTAS</span>}
                </div>}
              </div>
            </div>)}
            {busy && <div className="processing"><div className="spinner" /> NYX está generando una respuesta en CPU...</div>}
            <div ref={bottomRef} />
          </div>
          <div className="voice-strip">
            <div className="voice-meta"><span>▥ VOICE CHANNEL / {voiceState.toUpperCase()}</span>
              <span>{voiceProgress.total > 0
                ? 'LECTURA ' + voiceProgress.current + '/' + voiceProgress.total
                : voiceReady.tts_ready ? 'PIPER LOCAL' : 'PIPER PENDIENTE'}</span></div>
            <canvas ref={waveRef} width="960" height="74" aria-label="Forma de onda del audio capturado o reproducido" />
          </div>
          <div className="compose-area">
            <form onSubmit={send} className="compose-form">
              <button type="button" className={'microphone ' + (voiceState === 'listening' ? 'live' : '')}
                title={voiceState === 'listening' ? 'Detener grabación' : 'Hablar a NYX'}
                disabled={!voiceReady.stt_ready || busy}
                onClick={() => voiceState === 'listening' ? stopMic() : startMic()}>
                {voiceState === 'listening' ? '■' : '♩'}
              </button>
              <textarea value={draft} onChange={e => setDraft(e.target.value)}
                placeholder="Escribe una instrucción para NYX..." rows="2" maxLength={4000}
                onKeyDown={e => { if (e.key === 'Enter' && !e.shiftKey) {
                  e.preventDefault(); e.currentTarget.form.requestSubmit();
                }}}/>
              <button className="send" type="submit" disabled={busy || !draft.trim()}>
                {busy ? '···' : 'ENVIAR ↗'}
              </button>
            </form>
            <div className="compose-foot">
              <label><input type="checkbox" checked={voiceEnabled && voiceReady.tts_ready}
                disabled={!voiceReady.tts_ready}
                onChange={e => { setVoiceEnabled(e.target.checked); if (!e.target.checked) stopVoice(); }} />
                RESPUESTAS HABLADAS</label>
              <div><button type="button" onClick={stopVoice} disabled={voiceProgress.total === 0}>■ DETENER VOZ</button>
                <span>ENTER: ENVIAR · SHIFT+ENTER: SALTO</span></div>
            </div>
          </div>
        </div>}

        {tab === 'history' && <div className="history-layout">
          <div className="sessions-panel">
            <div className="panel-top"><div><span className="eyebrow">/ ARCHIVO</span><h2>Sesiones</h2></div>
              <span className="count-chip">{sessions.length}</span></div>
            <input className="search-input" value={search}
              onChange={async e => { const q = e.target.value; setSearch(q); try { await refreshSessions(q); } catch (err) { setError(err.message); }}}
              placeholder="⌕ Buscar en conversaciones..." />
            <div className="session-list">
              {Object.entries(grouped).map(([label, rows]) =>
                <section key={label}>
                  <div className="group-label">{label}</div>
                  {rows.map(s => <div key={s.id}
                    className={'session-card ' + (selectedId === s.id ? 'current' : '')}>
                    <button type="button" className="session-main" onClick={() => selectSession(s.id, false)}>
                      <span className="session-id">SESSION_{String(s.id).padStart(3, '0')}</span>
                      <strong>{s.title}</strong>
                      <small>{shortDate(s.updated_at)} · {s.count} mensajes</small>
                      <span className="session-preview">{s.preview || 'Sesión vacía'}</span>
                    </button>
                    <div className="session-buttons">
                      <button title="Cambiar nombre" onClick={() => renameSession(s)}>✎</button>
                      <button title="Eliminar sesión" onClick={() => deleteSession(s)}>×</button>
                    </div>
                  </div>)}
                </section>)}
              {!sessions.length && <p className="muted">Aún no hay sesiones con ese filtro.</p>}
            </div>
          </div>
          <div className="timeline-panel">
            <div className="panel-top"><div><span className="eyebrow">/ SESSION TIMELINE</span>
              <h2>{activeSession?.title || 'Selecciona una sesión'}</h2></div>
              <button className="outline-button" onClick={exportSession} disabled={!activeSession}>↓ EXPORTAR TXT</button>
            </div>
            <div className="timeline">
              {messages.map(m => <div className="timeline-event" key={m.id}>
                <span className={'timeline-node ' + m.role} />
                <div className="timeline-card">
                  <div className="timeline-head"><strong>{m.role === 'assistant' ? 'NYX / ASSISTANT' : 'OPERADOR / USER'}</strong>
                    <span>{shortDate(m.created_at)}</span></div>
                  <p>{m.content}</p>
                </div>
              </div>)}
              {!messages.length && <p className="muted">Selecciona una conversación para examinar sus mensajes.</p>}
            </div>
          </div>
        </div>}

        {(tab === 'tasks' || tab === 'memory') && <div className="records-panel">
          <div className="panel-top"><div><span className="eyebrow">/ PERSONAL DATABASE</span>
            <h2>{tab === 'tasks' ? 'Gestión de tareas' : 'Memoria explícita'}</h2></div>
            <span className="count-chip">{(tab === 'tasks' ? tasks : memories).length} REGISTROS</span></div>
          <p className="muted">{tab === 'tasks'
            ? 'Organiza tareas locales. La versión actual todavía no ejecuta recordatorios.'
            : 'NYX solo guarda los datos que agregues aquí. Puedes eliminarlos cuando quieras.'}</p>
          <form className="record-form" onSubmit={createItem}>
            <input maxLength="500" value={itemDraft} onChange={e => setItemDraft(e.target.value)}
              placeholder={tab === 'tasks' ? 'Nueva tarea...' : 'Información que NYX debe recordar...'}/>
            {tab === 'tasks' && <label className="due-date-field">
              <span>FECHA OPCIONAL</span>
              <input type="date" value={dueDate} onChange={e => setDueDate(e.target.value)}
                aria-label="Fecha de vencimiento de la tarea" />
            </label>}
            <button type="submit" disabled={!itemDraft.trim()}>＋ GUARDAR</button>
          </form>
          <div className="record-list">{(tab === 'tasks' ? tasks : memories).map(item =>
            <div className="record" key={item.id}>
              <span className="record-glyph">{tab === 'tasks' ? '☷' : '◇'}</span>
              {tab === 'tasks' && <input type="checkbox" checked={item.completed}
                onChange={() => updateTask(item)} aria-label="Completar tarea"/>}
              <div><strong className={item.completed ? 'done' : ''}>
                {tab === 'tasks' ? item.title : item.content}</strong>
                <small>{shortDate(item.created_at)}{item.due_date
                  ? ' · FECHA: ' + item.due_date : ''}</small>
              </div>
              <button onClick={() => deleteItem(item.id)} title="Eliminar">×</button>
            </div>)}
            {(tab === 'tasks' ? tasks : memories).length === 0 && <p className="muted">Sin registros por ahora.</p>}
          </div>
          {tab === 'tasks' && <CalendarPanel api={api} tasks={tasks}/>}
          {tab === 'memory' && <BrainNotebook api={api}/>} 
        </div>}

        {tab === 'memory' && <section className="nyx-knowledge-panel">
          <div className="panel-top"><div><span className="eyebrow">/ INVESTIGACIONES PÚBLICAS</span>
            <h2>Conocimiento consultado</h2></div><span className="count-chip">{knowledge.length} FUENTES</span></div>
          <p className="muted">Extractos guardados únicamente después de búsquedas solicitadas. No entrenan el modelo, pueden quedar desactualizados y se pueden eliminar.</p>
          <div className="record-list">{knowledge.map(item=><div className="record" key={item.id}>
            <span className="record-glyph">⌕</span>
            <div><strong>{item.title}</strong><p>{item.excerpt}</p>
              <a href={item.source_url} target="_blank" rel="noopener noreferrer">{item.source_url}</a>
              <small>CONSULTADO: {item.retrieved_at} UTC · {item.provider}</small></div>
            <button title="Eliminar investigación" onClick={async()=>{
              if(!window.confirm('¿Eliminar este extracto guardado?'))return;
              try{await api('/api/knowledge/'+item.id,{method:'DELETE'});
                setKnowledge(await api('/api/knowledge'));}catch(err){setError(err.message);}
            }}>×</button>
          </div>)}
          {!knowledge.length&&<p className="muted">Todavía no hay investigaciones guardadas.</p>}
          </div>
        </section>}

        {tab === 'settings' && auth.role === 'admin' && <div className="settings-layout">
          <div className="settings-card"><div className="eyebrow">/ AUDIO ENGINE</div>
            <h2>Identidad vocal</h2>
            <p>Voz local masculina española con Piper. El modo híbrido aplica resonancia electrónica únicamente a alertas.</p>
            <div className="switch-row"><span>Leer respuestas automáticamente</span>
              <input type="checkbox" checked={voiceEnabled && voiceReady.tts_ready}
                disabled={!voiceReady.tts_ready}
                onChange={e => { setVoiceEnabled(e.target.checked); if (!e.target.checked) stopVoice(); }}/></div>
            <div className="switch-row"><span>Perfil de voz</span>
              <select value={voiceProfile} onChange={e => setVoiceProfile(e.target.value)}>
                <option value="hybrid">Híbrido / alertas metálicas</option>
                <option value="natural">Natural / sin efectos</option>
              </select></div>
            <button className="outline-button" disabled={!voiceReady.tts_ready}
              onClick={() => playVoice('Sistema NYX inicializado. Estoy preparado para ayudarte.')}>▶ PROBAR VOZ</button>
            <button className="outline-button" disabled={!voiceReady.tts_ready}
              onClick={() => playVoice('Atención. Se ha detectado una interrupción en el sistema.', 'system')}>
              ◆ PROBAR ALERTA</button>
            <div className="setup-note">
              <b>Instalación de modelos de voz (una sola vez):</b>
              <code>docker compose exec api python -m app.voice_setup</code>
              <p>Tras la descarga, pulsa Sincronizar. Piper y Vosk se ejecutan localmente, sin enviar audio a una API comercial.</p>
            </div>
          </div>
          <div className="settings-card"><div className="eyebrow">/ PERSONAL CONTEXT</div>
            <h2>Perfil local</h2>
            <p>Configura únicamente tu ciudad habitual y zona horaria. NYX no utiliza geolocalización actual y no necesita Google para conocer la fecha.</p>
            <form className="profile-form" onSubmit={saveProfile}>
              {!auth.elevated && <p className="muted">Para modificar el perfil, habilita temporalmente el permiso administrativo en Seguridad.</p>}
              <label>Zona horaria IANA
                <input value={profile.timezone} maxLength="70" required
                  onChange={e => { setSavedProfile(false); setProfile({...profile, timezone:e.target.value}); }}
                  placeholder="America/Costa_Rica" />
              </label>
              <label>Ciudad habitual (opcional)
                <input value={profile.city} maxLength="90"
                  onChange={e => { setSavedProfile(false); setProfile({...profile, city:e.target.value}); }}
                  placeholder="Sin configurar" />
              </label>
              <label>País habitual (opcional)
                <input value={profile.country} maxLength="90"
                  onChange={e => { setSavedProfile(false); setProfile({...profile, country:e.target.value}); }}
                  placeholder="Sin configurar" />
              </label>
              <label>Estilo conversacional
                <select value={profile.tone}
                  onChange={e => { setSavedProfile(false); setProfile({...profile, tone:e.target.value}); }}>
                  <option value="natural">Natural y cercano</option>
                  <option value="tecnico">Técnico y preciso</option>
                  <option value="academico">Académico y didáctico</option>
                </select>
              </label>
              <button type="submit" className="outline-button" disabled={!auth.elevated}>◆ GUARDAR PERFIL</button>
              {savedProfile && <p className="save-confirm" role="status">Perfil actualizado localmente.</p>}
            </form>
            <p className="muted">La ubicación física actual no se consulta ni se deduce de estos campos.</p>
          </div>
          <div className="settings-card"><div className="eyebrow">/ SYSTEM</div>
            <h2>Estado de los módulos</h2>
            <div className="detail-row"><span>Modelo</span><b>{status.model}</b></div>
            <div className="detail-row"><span>Motor</span><b>{status.online ? 'CONECTADO' : 'DESCONECTADO'}</b></div>
            <div className="detail-row"><span>Síntesis de voz</span><b>{voiceReady.tts_ready ? 'PIPER LISTO' : 'PENDIENTE'}</b></div>
            <div className="detail-row"><span>Reconocimiento</span><b>{voiceReady.stt_ready ? 'VOSK LISTO' : 'PENDIENTE'}</b></div>
            <div className="detail-row"><span>Persistencia</span><b>SQLITE / DOCKER</b></div>
            <div className="detail-row"><span>Herramientas</span><b>COGNITIVE CORE</b></div>
            <div className="detail-row"><span>Internet</span>
              <b>{networkStatus.mode === 'online' ? 'CONECTADO' : 'LOCAL / OFFLINE'}</b></div>
            <div className="switch-row"><span>Habilitar consultas públicas de red</span>
              <input type="checkbox" checked={Boolean(networkStatus.enabled)}
                disabled={!auth.elevated}
                onChange={e => changeNetworkMode(e.target.checked)}/>
            </div>
            <p className="muted">El modo conectado permite consultas meteorológicas y búsquedas públicas explícitas. No descarga ni aprende contenidos automáticamente.</p>
            <div className="detail-row"><span>Zona horaria</span><b>{clockInfo?.timezone || profile.timezone}</b></div>
            <p className="muted">Micrófono desactivado por defecto. La captura se inicia únicamente al pulsar el botón correspondiente.</p>
          </div>
        </div>}

        {tab === 'security' && auth.role === 'admin' &&
          <div className="nyx-security-view">
            <AdminPanel auth={auth} api={api} onElevated={updateAuth}/>
          </div>}

        <footer className="footer">NYX {status.version || '0.6.0'} <span>◆</span> SISTEMA DE ASISTENCIA LOCAL</footer>
      </main>
      {rightOpen && <aside className="inspector">
        <div className="section-caption">/ NYX CORE</div>
        <Core state={voiceState} />
        <div className="core-label">{voiceState === 'idle' ? 'EN ESPERA' : voiceState === 'listening' ? 'ESCUCHANDO' : voiceState === 'speaking' ? 'RESPONDIENDO' : 'PROCESANDO'}</div>
        {auth.role === 'admin' && <div className="inspector-section"><div className="section-caption">/ STATUS MATRIX</div>
          <div className="detail-row"><span>LLM</span><strong className={status.online ? 'positive' : 'negative'}>{status.online ? 'ONLINE' : 'OFFLINE'}</strong></div>
          <div className="detail-row"><span>INTERNET</span>
            <strong className={networkStatus.connected ? 'positive' : 'negative'}>
              {networkStatus.connected ? 'ONLINE' : 'LOCAL'}
            </strong></div>
          <div className="detail-row"><span>TTS LOCAL</span><strong>{voiceReady.tts_ready ? 'READY' : 'PENDING'}</strong></div>
          <div className="detail-row"><span>MIC / STT</span><strong>{voiceReady.stt_ready ? 'READY' : 'PENDING'}</strong></div>
          <div className="detail-row"><span>VOZ</span><strong>{voiceProfile.toUpperCase()}</strong></div>
          <div className="detail-row"><span>FECHA LOCAL</span><strong>{clockInfo?.date || '—'}</strong></div>
        </div>}
        {auth.role !== 'admin' && <div className="inspector-section">
          <div className="section-caption">/ ESTADO DEL SISTEMA</div>
          <div className="detail-row"><span>CONEXIÓN</span>
            <strong className={systemOnline ? 'positive' : modelAvailable ? 'nyx-local-state' : 'negative'}>{systemLabel}</strong></div>
        </div>}
        <div className="inspector-section"><div className="section-caption">/ QUICK STATUS</div>
          <div className="mini-stat"><span>CONVERSACIONES</span><b>{sessions.length}</b></div>
          <div className="mini-stat"><span>TAREAS PENDIENTES</span><b>{tasks.filter(t => !t.completed).length}</b></div>
          <div className="mini-stat"><span>NOTAS DE MEMORIA</span><b>{memories.length}</b></div>
        </div>
        <div className="inspector-bottom">
          <span className="signal-dot" /> LOCAL FIRST / PRIVATE DATA
        </div>
      </aside>}
    </div>
  </div>;
}
