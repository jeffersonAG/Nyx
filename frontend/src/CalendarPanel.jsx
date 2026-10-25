import React, { useEffect, useMemo, useState } from 'react';
import './calendar.css';

const KINDS = [
  ['task', 'Tarea'], ['meeting', 'Reunión'], ['reservation', 'Espacio reservado'],
  ['reminder', 'Recordatorio'],
];
const WEEK = ['LUN', 'MAR', 'MIÉ', 'JUE', 'VIE', 'SÁB', 'DOM'];
function iso(day) {
  return [day.getFullYear(), String(day.getMonth()+1).padStart(2,'0'),
    String(day.getDate()).padStart(2,'0')].join('-');
}
function dateFrom(value) {
  const [y,m,d] = value.split('-').map(Number);
  return new Date(y,m-1,d,12);
}
function plusDays(date, n) {
  const d = new Date(date);
  d.setDate(d.getDate()+n);
  return d;
}
function monday(date) { return plusDays(date, -((date.getDay()+6)%7)); }
function monthTitle(date) {
  return new Intl.DateTimeFormat('es-CR',{month:'long',year:'numeric'}).format(date);
}
function defaultForm(day) {
  return {title:'',description:'',kind:'task',start_at:day+'T09:00',
    end_at:day+'T10:00',reminder_minutes:15};
}

export default function CalendarPanel({api, tasks=[]}) {
  const [events,setEvents]=useState([]);
  const [view,setView]=useState('month');
  const [day,setDay]=useState(()=>iso(new Date()));
  const [editing,setEditing]=useState(null);
  const [form,setForm]=useState(()=>defaultForm(iso(new Date())));
  const [notice,setNotice]=useState('');
  const [error,setError]=useState('');
  const [saving,setSaving]=useState(false);
  const [reload,setReload]=useState(0);
  const base=dateFrom(day);

  useEffect(()=>{
    let alive=true;
    api('/api/calendar').then(list=>{if(alive)setEvents(list);})
      .catch(err=>{if(alive)setError(err.message);});
    return ()=>{alive=false;};
  },[api,reload]);

  const allEvents=useMemo(()=>[
    ...events,
    ...tasks.filter(t=>t.due_date).map(t=>({
      id:'legacy-'+t.id, title:t.title,kind:'task',
      start_at:t.due_date+'T09:00',end_at:t.due_date+'T10:00',
      legacy:true,completed:t.completed,
    })),
  ],[events,tasks]);
  const byDate=useMemo(()=>{
    const map={};
    for(const event of allEvents) {
      const key=event.start_at.slice(0,10);
      (map[key] ||= []).push(event);
    }
    for(const values of Object.values(map)) values.sort((a,b)=>a.start_at.localeCompare(b.start_at));
    return map;
  },[allEvents]);
  const days=useMemo(()=>{
    let begin,count;
    if(view==='day') {begin=base;count=1;}
    else if(view==='week') {begin=monday(base);count=7;}
    else {begin=monday(new Date(base.getFullYear(),base.getMonth(),1,12));count=42;}
    return Array.from({length:count},(_,i)=>plusDays(begin,i));
  },[day,view]);
  function navigate(steps) {
    const next=new Date(base);
    if(view==='day')next.setDate(next.getDate()+steps);
    else if(view==='week')next.setDate(next.getDate()+steps*7);
    else if(view==='year')next.setFullYear(next.getFullYear()+steps);
    else next.setMonth(next.getMonth()+steps,1);
    setDay(iso(next));
  }
  function openNew(selected=day) {
    setEditing('new');setForm(defaultForm(selected));setError('');setNotice('');
  }
  function openEdit(item) {
    if(item.legacy){setNotice('Esta tarea proviene de la lista de tareas; edítala allí.');return;}
    setEditing(item.id);
    setForm({title:item.title,description:item.description,kind:item.kind,
      start_at:item.start_at,end_at:item.end_at,reminder_minutes:item.reminder_minutes});
    setError('');setNotice('');
  }
  async function save(event) {
    event.preventDefault();
    if(saving)return;
    setSaving(true);setError('');
    try {
      const body={...form,reminder_minutes:Number(form.reminder_minutes)};
      if(editing==='new')await api('/api/calendar',{method:'POST',body:JSON.stringify(body)});
      else await api('/api/calendar/'+editing,{method:'PATCH',body:JSON.stringify(body)});
      setEditing(null);setNotice('Evento guardado correctamente.');
      setReload(n=>n+1);
    } catch(err){setError(err.message);}
    finally{setSaving(false);}
  }
  async function remove() {
    if(editing==='new'|| !window.confirm('¿Eliminar el evento?'))return;
    setSaving(true);setError('');
    try {
      await api('/api/calendar/'+editing,{method:'DELETE'});
      setEditing(null);setNotice('Evento eliminado.');
      setReload(n=>n+1);
    }catch(err){setError(err.message);}
    finally{setSaving(false);}
  }
  const title=view==='year' ? String(base.getFullYear()) :
    view==='day' ? new Intl.DateTimeFormat('es-CR',{day:'numeric',month:'long',year:'numeric'}).format(base) :
    view==='week' ? 'Semana del '+iso(monday(base)) : monthTitle(base);
  const inMonth=date=>date.getMonth()===base.getMonth();
  return <section className="nyx-calendar" aria-label="Calendario personal">
    <div className="calendar-heading">
      <div><span className="eyebrow">/ AGENDA PRIVADA</span><h2>Calendario y recordatorios</h2></div>
      <button className="calendar-primary" onClick={()=>openNew()}>＋ NUEVO EVENTO</button>
    </div>
    <p className="muted">Eventos almacenados únicamente en tu cuenta. Las alertas se muestran mientras NYX esté abierto; las notificaciones del sistema requieren permiso del navegador.</p>
    <div className="calendar-toolbar">
      <div className="calendar-arrows">
        <button onClick={()=>navigate(-1)} aria-label="Anterior">‹</button>
        <button onClick={()=>setDay(iso(new Date()))}>HOY</button>
        <button onClick={()=>navigate(1)} aria-label="Siguiente">›</button>
        <strong>{title.toUpperCase()}</strong>
      </div>
      <div className="calendar-views" aria-label="Vista de calendario">
        {[['day','DÍA'],['week','SEMANA'],['month','MES'],['year','AÑO']].map(([key,label])=>
          <button key={key} aria-pressed={view===key}
            className={view===key?'active':''} onClick={()=>setView(key)}>{label}</button>)}
      </div>
    </div>
    {view==='year' ? <div className="calendar-year">
      {Array.from({length:12},(_,i)=>{
        const first=new Date(base.getFullYear(),i,1,12);
        const yearEvents=allEvents.filter(e=>e.start_at.slice(0,7)===iso(first).slice(0,7));
        return <button key={i} className="calendar-year-month" onClick={()=>{setDay(iso(first));setView('month');}}>
          <strong>{new Intl.DateTimeFormat('es-CR',{month:'long'}).format(first)}</strong>
          <span>{yearEvents.length} EVENTOS/TAREAS</span></button>;
      })}
    </div> : <div className={'calendar-grid view-'+view}>
      {view!=='day'&&WEEK.map(v=><div key={v} className="calendar-day-label">{v}</div>)}
      {days.map(date=>{
        const key=iso(date);
        const listed=byDate[key]||[];
        return <div key={key} className={'calendar-day '+(!inMonth(date)&&view==='month'?'other-month ':'')+(key===iso(new Date())?'today':'')}>
          <div className="calendar-day-top">
            <button title={'Crear evento el '+key} onClick={()=>openNew(key)}>{date.getDate()}</button>
            {listed.length>0&&<small>{listed.length}</small>}
          </div>
          <div className="calendar-day-events">
            {listed.slice(0,view==='month'?3:15).map(item=>
              <button type="button" key={String(item.id)} title={item.title}
                className={'calendar-event kind-'+item.kind+(item.completed?' done':'')}
                onClick={()=>openEdit(item)}>
                <small>{item.legacy?'TAREA':item.start_at.slice(11,16)}</small>
                <span>{item.title}</span>
              </button>)}
            {listed.length>(view==='month'?3:15)&&<small className="calendar-more">+{listed.length-(view==='month'?3:15)} más</small>}
          </div>
        </div>;
      })}
    </div>}
    {notice&&<p role="status" className="calendar-notice">{notice}</p>}
    {error&&<p role="alert" className="calendar-error">{error}</p>}
    {editing!==null&&<form className="calendar-editor" onSubmit={save}>
      <h3>{editing==='new'?'Nuevo evento':'Editar evento'}</h3>
      <label>Título <input required maxLength="160" value={form.title}
        onChange={e=>setForm({...form,title:e.target.value})}/></label>
      <label>Tipo <select value={form.kind}
        onChange={e=>setForm({...form,kind:e.target.value})}>
        {KINDS.map(([key,label])=><option key={key} value={key}>{label}</option>)}
      </select></label>
      <label>Inicio <input required type="datetime-local" value={form.start_at}
        onChange={e=>setForm({...form,start_at:e.target.value})}/></label>
      <label>Fin <input required type="datetime-local" value={form.end_at}
        onChange={e=>setForm({...form,end_at:e.target.value})}/></label>
      <label>Recordar con anticipación <select value={form.reminder_minutes}
        onChange={e=>setForm({...form,reminder_minutes:Number(e.target.value)})}>
        <option value="0">En el momento</option>
        <option value="5">5 minutos</option>
        <option value="15">15 minutos</option>
        <option value="30">30 minutos</option>
        <option value="60">1 hora</option>
        <option value="1440">1 día</option>
        <option value="10080">1 semana</option>
      </select></label>
      <label className="calendar-description">Descripción <textarea maxLength="1200" rows="3"
        value={form.description} onChange={e=>setForm({...form,description:e.target.value})}/></label>
      <div className="calendar-editor-actions">
        <button className="calendar-primary" type="submit" disabled={saving}>GUARDAR</button>
        <button type="button" onClick={()=>setEditing(null)}>CANCELAR</button>
        {editing!=='new'&&<button type="button" className="danger" onClick={remove} disabled={saving}>ELIMINAR</button>}
      </div>
    </form>}
    <button className="calendar-notify-permission" type="button"
      onClick={async()=>{
        if(typeof Notification==='undefined') {setNotice('Tu navegador no admite notificaciones del sistema.');return;}
        const permission=await Notification.requestPermission();
        setNotice(permission==='granted'?'Notificaciones de NYX habilitadas.':
          'Puedes seguir viendo recordatorios dentro de NYX sin permiso del sistema.');
      }}>◈ HABILITAR NOTIFICACIONES DEL SISTEMA</button>
  </section>;
}
