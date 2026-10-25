import React, { useCallback, useEffect, useState } from 'react';
import './brain.css';

const TYPES=[['experience','EXPERIENCIA'],['procedure','PROCEDIMIENTO'],['hypothesis','HIPÓTESIS']];
const OUTCOMES=[['unknown','SIN EVALUAR'],['success','FUNCIONÓ'],['partial','PARCIAL'],['failure','FALLÓ']];
const RELATIONS=[['related_to','RELACIONADO'],['supports','APOYA'],['contradicts','CONTRADICE'],['prerequisite_for','ES REQUISITO DE']];
const EMPTY={kind:'experience',title:'',situation:'',action_taken:'',result:'',outcome:'unknown',evidence:''};
const LINK={from_case_id:'',to_case_id:'',relation:'related_to',evidence:'',source_url:''};
const RESULT={items:[],mode:'empty'};

export default function BrainNotebook({api}) {
  const [view,setView]=useState('cases');
  const [cases,setCases]=useState([]);
  const [relations,setRelations]=useState([]);
  const [status,setStatus]=useState(null);
  const [form,setForm]=useState(EMPTY);
  const [link,setLink]=useState(LINK);
  const [query,setQuery]=useState('');
  const [retrieved,setRetrieved]=useState(RESULT);
  const [recommendations,setRecommendations]=useState([]);
  const [warning,setWarning]=useState('');
  const [notice,setNotice]=useState('');
  const [error,setError]=useState('');
  const [working,setWorking]=useState(false);

  const refresh=useCallback(async()=>{
    const [records,edges,state]=await Promise.all([
      api('/api/brain/cases'), api('/api/brain/relations'),api('/api/brain/status')
    ]);
    setCases(records);setRelations(edges);setStatus(state);
  },[api]);
  useEffect(()=>{refresh().catch(err=>setError(err.message));},[refresh]);

  async function createCase(e) {
    e.preventDefault();if(working)return;
    setWorking(true);setError('');
    try {
      await api('/api/brain/cases',{method:'POST',body:JSON.stringify(form)});
      setForm(EMPTY);setNotice('Experiencia guardada como testimonio, no como verdad comprobada.');
      await refresh();
    }catch(err){setError(err.message);}
    finally{setWorking(false);}
  }

  async function reviewCase(item,outcome) {
    const newEvidence=window.prompt('¿Qué observaste? Anota evidencia o resultado de la revisión:',item.evidence||'');
    if(newEvidence===null)return;
    setWorking(true);setError('');
    try {
      await api('/api/brain/cases/'+item.id,{
        method:'PATCH',body:JSON.stringify({outcome,evidence:newEvidence})
      });
      setNotice('Evaluación actualizada. Este resultado continúa siendo un testimonio del usuario.');
      await refresh();
    }catch(err){setError(err.message);}
    finally{setWorking(false);}
  }

  async function removeCase(id) {
    if(!window.confirm('¿Eliminar esta experiencia y todas sus relaciones?'))return;
    setWorking(true);setError('');
    try{await api('/api/brain/cases/'+id,{method:'DELETE'});
      setNotice('Experiencia eliminada.');await refresh();}
    catch(err){setError(err.message);}
    finally{setWorking(false);}
  }

  async function createRelation(e) {
    e.preventDefault();if(working)return;
    setWorking(true);setError('');
    try {
      await api('/api/brain/relations',{method:'POST',
        body:JSON.stringify({...link,
          from_case_id:Number(link.from_case_id),to_case_id:Number(link.to_case_id)})});
      setLink(LINK);setNotice('Relación declarada con evidencia. NYX no la considerará automáticamente verificada.');
      await refresh();
    }catch(err){setError(err.message);}
    finally{setWorking(false);}
  }

  async function removeRelation(id) {
    if(!window.confirm('¿Eliminar esta relación?'))return;
    setWorking(true);setError('');
    try{await api('/api/brain/relations/'+id,{method:'DELETE'});await refresh();}
    catch(err){setError(err.message);}
    finally{setWorking(false);}
  }

  async function retrieve(e) {
    e.preventDefault();
    if(query.trim().length<3)return;
    setWorking(true);setError('');setNotice('');
    try {
      const q=encodeURIComponent(query.trim());
      const [search,advice]=await Promise.all([
        api('/api/brain/recall?q='+q), api('/api/brain/recommend?q='+q)
      ]);
      setRetrieved(search);setRecommendations(advice.recommendations||[]);
      setWarning(advice.warning||'');
    }catch(err){setError(err.message);}
    finally{setWorking(false);}
  }

  async function index() {
    setWorking(true);setError('');
    try{
      const result=await api('/api/brain/index',{method:'POST'});
      setNotice(result.message+' '+result.indexed+' indexados; '
        +(result.remaining ?? 'sin información')+' pendientes.');
      await refresh();
    }catch(err){setError(err.message);}
    finally{setWorking(false);}
  }

  return <section className="brain-notebook" aria-label="Cerebro cognitivo de NYX">
    <header className="brain-top">
      <div><div className="eyebrow">/ NYX · NÚCLEO COGNITIVO</div>
        <h2>Experiencias y aprendizaje verificable</h2></div>
      <span className="brain-version">COGNITIVE BRAIN v1</span>
    </header>
    <p className="muted">Aquí NYX conserva lo que ocurrió, lo que se intentó y cómo resultó.
      La evidencia siempre queda asociada al registro. Una experiencia declarada no demuestra por sí sola su exactitud
      y no reentrena los pesos de la red neuronal.</p>
    <div className="brain-metrics">
      <div><small>EXPERIENCIAS</small><strong>{status?.cases??'—'}</strong></div>
      <div><small>RELACIONES</small><strong>{status?.relations??'—'}</strong></div>
      <div><small>RESPUESTAS EVALUADAS</small><strong>{status?.rated_answers??'—'}</strong></div>
      <div><small>EMBEDDINGS</small><strong>{status?.embedding_model
        ? status.embedding_model_installed ? 'DISPONIBLE' : 'NO INSTALADO'
        : 'OPCIONAL'}</strong></div>
    </div>
    <div className="brain-tabs">
      {[['cases','EXPERIENCIAS'],['relations','RELACIONES'],['recall','RECUPERAR']].map(([id,label])=>
        <button type="button" key={id} className={view===id?'selected':''}
          onClick={()=>{setView(id);setError('');}}>{label}</button>)}
    </div>
    {error&&<p role="alert" className="brain-error">⚠ {error}</p>}
    {notice&&<p role="status" className="brain-notice">{notice}</p>}

    {view==='cases'&&<div className="brain-content">
      <form className="brain-form" onSubmit={createCase}>
        <h3>Registrar una experiencia</h3>
        <div className="brain-form-grid">
          <label>Tipo<select value={form.kind} onChange={e=>setForm({...form,kind:e.target.value})}>
            {TYPES.map(([id,title])=><option key={id} value={id}>{title}</option>)}
          </select></label>
          <label>Resultado observado<select value={form.outcome}
            onChange={e=>setForm({...form,outcome:e.target.value})}>
            {OUTCOMES.map(([id,title])=><option key={id} value={id}>{title}</option>)}
          </select></label>
        </div>
        <label>Título<input required minLength="3" maxLength="150" value={form.title}
          onChange={e=>setForm({...form,title:e.target.value})}
          placeholder="Ej.: Falla intermitente en una antena Wi-Fi"/></label>
        <label>¿En qué situación ocurrió?<textarea required maxLength="900" rows="2"
          value={form.situation} onChange={e=>setForm({...form,situation:e.target.value})}
          placeholder="Describe el problema, contexto y condiciones iniciales"/></label>
        <label>¿Qué procedimiento o hipótesis utilizaste?<textarea required maxLength="1300" rows="3"
          value={form.action_taken} onChange={e=>setForm({...form,action_taken:e.target.value})}
          placeholder="Pasos realizados o propuesta puesta a prueba"/></label>
        <label>¿Qué resultado obtuviste?<textarea maxLength="900" rows="2" value={form.result}
          onChange={e=>setForm({...form,result:e.target.value})}
          placeholder="Qué cambió y qué no se pudo comprobar"/></label>
        <label>¿Qué evidencia tienes?<textarea maxLength="900" rows="2" value={form.evidence}
          onChange={e=>setForm({...form,evidence:e.target.value})}
          placeholder="Mediciones, logs, pruebas, fechas o referencias"/></label>
        <button type="submit" className="brain-primary" disabled={working}>＋ GUARDAR EXPERIENCIA</button>
      </form>
      <div className="brain-list">
        <h3>Registro de experiencias</h3>
        {cases.map(item=><article key={item.id} className="brain-case">
          <div className="brain-case-head"><strong>#{item.id} · {item.title}</strong>
            <span>{item.kind.toUpperCase()} · {item.outcome.toUpperCase()}</span></div>
          <div><b>SITUACIÓN:</b> {item.situation}</div>
          <div><b>PROCEDIMIENTO:</b> {item.action_taken}</div>
          {!!item.result&&<div><b>RESULTADO:</b> {item.result}</div>}
          {!!item.evidence&&<div><b>EVIDENCIA:</b> {item.evidence}</div>}
          <small>ORIGEN: {item.origin==='tool_execution'?'Ejecución confirmada de herramienta':'Declaración del usuario'}.
            No es certificación científica independiente.</small>
          {item.origin==='user_report'&&<div className="brain-case-actions">
            <button type="button" disabled={working} onClick={()=>reviewCase(item,'success')}>FUNCIONÓ</button>
            <button type="button" disabled={working} onClick={()=>reviewCase(item,'partial')}>PARCIAL</button>
            <button type="button" disabled={working} onClick={()=>reviewCase(item,'failure')}>FALLÓ</button>
          </div>}
          <button type="button" className="brain-delete" onClick={()=>removeCase(item.id)}>ELIMINAR REGISTRO</button>
        </article>)}
        {cases.length===0&&<p className="muted">Aún no hay experiencias. Registra el primer procedimiento que hayas probado.</p>}
      </div>
    </div>}

    {view==='relations'&&<div className="brain-content">
      <form className="brain-form" onSubmit={createRelation}>
        <h3>Vincular conocimientos con evidencia</h3>
        <p className="muted">Las coincidencias automáticas no constituyen pruebas de causalidad.
          Las relaciones registradas aquí expresan una asociación que tú declaras y puedes revisar.</p>
        <label>Registro inicial<select required value={link.from_case_id}
          onChange={e=>setLink({...link,from_case_id:e.target.value})}>
          <option value="">Seleccionar experiencia</option>
          {cases.map(c=><option key={c.id} value={c.id}>#{c.id} · {c.title}</option>)}
        </select></label>
        <label>Tipo de relación<select value={link.relation}
          onChange={e=>setLink({...link,relation:e.target.value})}>
          {RELATIONS.map(([id,title])=><option key={id} value={id}>{title}</option>)}
        </select></label>
        <label>Registro relacionado<select required value={link.to_case_id}
          onChange={e=>setLink({...link,to_case_id:e.target.value})}>
          <option value="">Seleccionar experiencia</option>
          {cases.filter(c=>String(c.id)!==String(link.from_case_id)).map(c=>
            <option key={c.id} value={c.id}>#{c.id} · {c.title}</option>)}
        </select></label>
        <label>Justificación y evidencia<textarea required minLength="12" maxLength="750" rows="3"
          value={link.evidence} onChange={e=>setLink({...link,evidence:e.target.value})}
          placeholder="¿Qué observación demuestra que ambos casos se relacionan?"/></label>
        <label>URL de la fuente (opcional)<input type="url" maxLength="500"
          value={link.source_url} onChange={e=>setLink({...link,source_url:e.target.value})}
          placeholder="https://..."/></label>
        <button type="submit" className="brain-primary" disabled={working||cases.length<2}>
          ＋ REGISTRAR RELACIÓN
        </button>
      </form>
      <div className="brain-list"><h3>Relaciones documentadas</h3>
        {relations.map(item=><article key={item.id} className="brain-case">
          <div className="brain-case-head"><strong>{item.from_title}</strong>
            <span>{item.relation.toUpperCase().replaceAll('_',' ')}</span></div>
          <strong>{item.to_title}</strong>
          <p>Evidencia declarada: {item.evidence}</p>
          {!!item.source_url&&<a target="_blank" rel="noopener noreferrer" href={item.source_url}>Fuente citada (sin verificación automática)</a>}
          <small>ORIGEN: {item.basis==='user_report'?'Informe del usuario':'Fuente indicada por el usuario'}.</small>
          <button type="button" className="brain-delete" onClick={()=>removeRelation(item.id)}>ELIMINAR</button>
        </article>)}
        {!relations.length&&<p className="muted">No hay relaciones registradas todavía.</p>}
      </div>
    </div>}

    {view==='recall'&&<div className="brain-content">
      <form className="brain-form" onSubmit={retrieve}>
        <h3>Recuperar conocimientos relacionados</h3>
        <label>Pregunta o problema<input minLength="3" required maxLength="300"
          value={query} onChange={e=>setQuery(e.target.value)}
          placeholder="¿Cómo resolví los problemas de enlaces Wi-Fi?"/></label>
        <button type="submit" className="brain-primary" disabled={working}>⌕ BUSCAR EXPERIENCIAS</button>
        <p className="muted">La búsqueda combina coincidencias léxicas en Python y vectores neuronales
          locales cuando se ha instalado un modelo de embeddings.</p>
        <div className="brain-index-status"><span>MODELO: {status?.embedding_model||'Sin configurar; búsqueda léxica'}</span>
          {status?.embedding_model&&<button disabled={working||!status.embedding_model_installed} type="button" onClick={index}>INDIZAR 8 REGISTROS</button>}
        </div>
      </form>
      <div className="brain-list">
        <h3>Coincidencias {retrieved.mode!=='empty'&&<small>· {retrieved.mode.toUpperCase()}</small>}</h3>
        {retrieved.items.map((item,i)=><article key={item.kind+'-'+item.id+'-'+i} className="brain-case">
          <div className="brain-case-head"><strong>{item.title}</strong><span>{item.kind.toUpperCase()}</span></div>
          <p>{item.content}</p>
          <small>CONFIANZA DOCUMENTAL: {item.trust.replaceAll('_',' ')} · ESTADO: {item.outcome}</small>
          <small>FUENTE: {item.source}</small>
          <small>COINCIDENCIA: {item.score} (indicador relativo, no probabilidad de verdad)</small>
        </article>)}
        {retrieved.mode!=='empty'&&!retrieved.items.length&&<p className="muted">
          No se encontraron registros suficientemente relacionados con tu búsqueda.</p>}
        {!!recommendations.length&&<div className="brain-recommendations"><h3>Procedimientos previos</h3>
          <p className="muted">{warning}</p>
          {recommendations.map((rec,i)=><article key={i} className="brain-case">
            <strong>{rec.procedure}</strong>
            <div>Evaluación declarada: {rec.success} funcionó · {rec.partial} parcial · {rec.failure} falló.</div>
            <small>Muestra de {rec.sample_size} experiencias. Estimación suavizada de resultados favorables: {(rec.reported_success_estimate*100).toFixed(1)}% (NO es una probabilidad validada).</small>
          </article>)}
        </div>}
      </div>
    </div>}
  </section>;
}
