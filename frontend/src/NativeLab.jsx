import React, { useEffect, useState } from 'react';

/**
 * Banco de pruebas independiente. No sustituye el chat habitual y no
 * permite ejecutar herramientas ni entrenar con datos del usuario.
 */
export default function NativeLab({ api, elevated }) {
  const [status, setStatus] = useState(null);
  const [prompt, setPrompt] = useState('Explica qué es una red neuronal.');
  const [result, setResult] = useState(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');

  async function refresh() {
    try { setStatus(await api('/api/native/status')); }
    catch (err) { setError(err.message); }
  }

  useEffect(() => { refresh(); }, []);

  async function generate(e) {
    e.preventDefault();
    if (!elevated || !status?.ready || busy) return;
    setError(''); setResult(null); setBusy(true);
    try {
      const response = await api('/api/native/generate', {
        method: 'POST',
        body: JSON.stringify({ text: prompt.trim(), max_new_tokens: 64,
                               temperature: 0.9, seed: 2026 }),
      });
      setResult(response);
    } catch (err) { setError(err.message); }
    finally { setBusy(false); }
  }

  return <section className="nyx-native-lab" aria-label="Laboratorio de red neuronal propia">
    <div className="brain-top">
      <div>
        <span className="eyebrow">/ EXPERIMENTO AISLADO · PYTORCH</span>
        <h3>Red neuronal desarrollada desde cero</h3>
      </div>
      <button type="button" onClick={refresh}>↻ COMPROBAR ESTADO</button>
    </div>
    <p className="muted">Este laboratorio utiliza un Transformer construido y entrenado
      específicamente para NYX. No reutiliza los pesos de Qwen. El modelo debe
      entrenarse desde una colección de textos autorizados antes de generar
      respuestas. Su calidad inicial será limitada.</p>
    <div className="native-lab-info">
      <span>CONTENEDOR: <strong>{status?.service_running ? 'EJECUTÁNDOSE' : 'DETENIDO'}</strong></span>
      <span>CHECKPOINT: <strong>{status?.ready ? 'ENTRENADO' : 'NO DISPONIBLE'}</strong></span>
      <span>PASOS: <strong>{status?.step ?? '—'}</strong></span>
      <span>MODELO PRINCIPAL: <strong>SIN CAMBIOS</strong></span>
    </div>
    {!status?.ready &&
      <p className="muted">Para usar el laboratorio inicia el servicio optativo
        y ejecuta el entrenamiento en PowerShell siguiendo la guía técnica.
        No se descarga ni entrena nada desde esta pantalla.</p>}
    <form onSubmit={generate} className="brain-form">
      <label>Prueba de generación (sólo administrador)
        <textarea rows="2" maxLength="1500" required value={prompt}
          onChange={e=>setPrompt(e.target.value)} /></label>
      <button type="submit" className="brain-primary"
        disabled={!elevated || !status?.ready || busy || !prompt.trim()}>
        {busy ? 'GENERANDO…' : 'PROBAR MODELO PROPIO'}
      </button>
      {!elevated && <p className="muted">Autoriza primero cambios administrativos
        desde Seguridad para habilitar esta prueba experimental.</p>}
    </form>
    {error && <p role="alert" className="brain-error">{error}</p>}
    {result && <div role="status" className="native-lab-result">
      <strong>RESPUESTA EXPERIMENTAL</strong>
      <pre>{result.text || '(No generó texto visible)'}</pre>
      <small>Paso {result.step} · {result.parameters} parámetros.
        Resultado no validado; no utilizar para decisiones técnicas o de seguridad.</small>
    </div>}
  </section>;
}
