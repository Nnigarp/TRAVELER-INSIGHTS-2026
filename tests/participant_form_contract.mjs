// Contract test of our generated offline form; all inputs are synthetic fixtures.
// No browser account, network, real consent or participant data is used.
import fs from 'node:fs';
import vm from 'node:vm';
import assert from 'node:assert/strict';

const html = fs.readFileSync(process.argv[2], 'utf8');
const payload = html.match(/<script id="consent-template" type="application\/json">([\s\S]*?)<\/script>/)[1];
const script = html.match(/<script>([\s\S]*?)<\/script>/)[1];
const original = JSON.parse(payload);
const keys = ['original_authorship', 'real_experience', 'annotation', 'training', 'evaluation', 'aggregate_results'];
const nodes = Object.fromEntries(keys.map(key => [key, { checked: false }]));
Object.assign(nodes, {'consent-template': {textContent: payload}, code: {}, consent: {}, text: {}, review: {value: ''}, status: {textContent: ''}});
const downloads = [];
const blobs = new Map();
let serial = 0;
const context = {
  Blob: class { constructor(parts, options) { this.content = parts.join(''); this.type = options.type; } },
  URL: {createObjectURL(blob) { const key = `blob:fixture-${++serial}`; blobs.set(key, blob); return key; }, revokeObjectURL(key) { blobs.delete(key); }},
  setTimeout(callback) { callback(); },
  document: {
    getElementById(id) { assert.ok(nodes[id], `Unexpected DOM access: ${id}`); return nodes[id]; },
    createElement(tag) { assert.equal(tag, 'a'); return {click() { downloads.push({name: this.download, content: blobs.get(this.href).content}); }}; },
  },
};
vm.runInNewContext(script, context, {timeout: 1000});
assert.equal(nodes.code.textContent, original.author_id);
nodes.consent.onclick();
assert.equal(downloads.length, 0, 'Unchecked form must not create consent');
nodes.annotation.checked = true;
nodes.consent.onclick();
assert.equal(downloads.length, 0, 'Partial consent must not be converted to complete consent');
keys.forEach(key => { nodes[key].checked = true; });
nodes.consent.onclick();
assert.equal(downloads.length, 1);
const consent = JSON.parse(downloads[0].content);
assert.equal(consent.author_id, original.author_id);
assert.equal(consent.notice_sha256, original.notice_sha256);
assert.equal(consent.public_raw_text, false);
assert.ok(keys.every(key => consent[key] === true));
assert.ok(Number.isFinite(Date.parse(consent.accepted_at)));
nodes.text.onclick();
assert.equal(downloads.length, 1, 'Empty text must not become a review');
nodes.review.value = '  Prueba sintética: todavía no hay una reseña humana. Ñ 🙂  ';
nodes.text.onclick();
assert.equal(downloads.length, 2);
assert.equal(downloads[1].content, nodes.review.value.trim() + '\n');
assert.equal(downloads[1].name, `${original.author_id}-review.txt`);
nodes.training.checked = false;
nodes.consent.onclick();
assert.equal(downloads.length, 2);
assert.ok(html.includes("default-src 'none'") && html.includes("form-action 'none'"));
console.log('OFFLINE_FORM_CONSENT_GATES_VERIFIED_WITH_SYNTHETIC_FIXTURES_ONLY');
