import test from 'node:test';
import assert from 'node:assert/strict';
import { splitSpeechText } from './voiceChunks.js';

test('conserva íntegramente una explicación larga en múltiples segmentos', () => {
  const input = Array.from({ length: 120 }, (_, index) =>
    'La matriz ' + index + ' permite describir operaciones lineales.').join(' ');
  const parts = splitSpeechText(input);
  assert.ok(parts.length > 1);
  assert.ok(parts.every(part => part.length > 0 && part.length <= 850));
  assert.equal(parts.join(' ').replace(/\s+/g, ' '), input.replace(/\s+/g, ' '));
});
test('las respuestas cortas no se fragmentan', () => {
  assert.deepEqual(splitSpeechText('Hola, ¿cómo estás?'), ['Hola, ¿cómo estás?']);
});
test('no elimina fórmulas, caracteres especiales ni párrafos', () => {
  const text = ('π × 𝑨 + B = C; \\\\(\\\\vec{a}\\\\cdot\\\\vec{b}\\\\).\\n' +
    'Explicación matemática completa. ').repeat(50);
  const parts = splitSpeechText(text);
  assert.ok(parts.every(part => part.length <= 850));
  assert.equal(parts.join(' ').replace(/\s+/g, ' ').trim(), text.replace(/\s+/g, ' ').trim());
});
test('admite respuestas sin espacios muy largas y no rompe emojis', () => {
  const text = 'a'.repeat(3000) + '🚀' + 'b'.repeat(2000);
  const parts = splitSpeechText(text);
  assert.equal(parts.join(''), text);
  assert.ok(parts.every(part => part.length <= 850));
});
test('ignora mensajes vacíos', () => {
  assert.deepEqual(splitSpeechText(' \n\t '), []);
  assert.deepEqual(splitSpeechText(null), []);
});
