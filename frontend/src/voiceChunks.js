/**
 * Divide respuestas para Piper sin perder contenido hablado.
 * Cada fragmento queda por debajo del límite de 1400 caracteres de la API.
 * Se priorizan finales de frase; en segundo lugar comas y espacios.
 */
export function splitSpeechText(value, maxCharacters = 850) {
  const text = String(value ?? '').trim();
  if (!text) return [];
  const limit = Math.max(80, Math.min(1300, Math.floor(maxCharacters)));
  const result = [];
  let remaining = text;
  while (remaining.length > limit) {
    const sample = remaining.slice(0, limit);
    const candidates = [
      /[.!?。！？](?=\s)|\n+/g,
      /[,;:](?=\s)/g,
      /\s+/g,
    ];
    let cut = null;
    for (const pattern of candidates) {
      const matches = [...sample.matchAll(pattern)];
      const boundaries = matches
        .map(match => match.index + match[0].length)
        .filter(index => index >= limit * 0.45);
      if (boundaries.length) {
        cut = boundaries[boundaries.length - 1];
        break;
      }
    }
    if (!cut) {
      cut = limit;
      // No dividir en medio de un par sustituto Unicode.
      if (/[\uD800-\uDBFF]/.test(sample.charAt(limit - 1))) cut -= 1;
    }
    const fragment = remaining.slice(0, cut).trim();
    if (fragment) result.push(fragment);
    remaining = remaining.slice(cut).trimStart();
  }
  if (remaining.trim()) result.push(remaining.trim());
  return result;
}
