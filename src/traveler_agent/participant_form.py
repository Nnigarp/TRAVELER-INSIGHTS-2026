"""Generate a self-contained offline form; no server, tracking or submission."""
from __future__ import annotations

import html
import json
import os

from .human_intake import NOTICE, new_consent_form, private_path


def create_participant_form(output):
    output=private_path(output)
    template=new_consent_form()
    payload=json.dumps(template,ensure_ascii=True).replace("</","<\\/")
    page='''<!doctype html><html lang="es"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<meta http-equiv="Content-Security-Policy" content="default-src 'none'; style-src 'unsafe-inline'; script-src 'unsafe-inline'; form-action 'none'; base-uri 'none'">
<title>Aportación voluntaria — Traveler Insights</title>
<style>body{font:18px/1.5 system-ui;max-width:800px;margin:32px auto;padding:0 20px;color:#172033}pre{white-space:pre-wrap;font:inherit}label{display:block;margin:14px 0}textarea{box-sizing:border-box;width:100%;font:inherit;padding:12px}button{font:inherit;padding:10px;margin:8px 12px 8px 0}small{display:block}#status{font-weight:600}</style>
<main><h1>Tu experiencia turística, con tus palabras</h1>
<p>Formulario local: no envía nada ni guarda borradores. Si deseas participar, descarga los dos archivos y entrégalos al coordinador por el canal privado acordado.</p>
<pre>__NOTICE__</pre><p>Tu código: <strong id="code"></strong>. Consérvalo para consultas o retirada.</p>
<label><input type="checkbox" id="original_authorship"> Soy autor/a del texto original, sin copiarlo de una plataforma ni generarlo con IA.</label>
<label><input type="checkbox" id="real_experience"> La reseña describe una experiencia turística que viví.</label>
<label><input type="checkbox" id="annotation"> Acepto almacenamiento privado y lectura por anotadores humanos.</label>
<label><input type="checkbox" id="training"> Acepto su uso para entrenamiento académico de clasificadores.</label>
<label><input type="checkbox" id="evaluation"> Acepto su uso para evaluación académica de clasificadores.</label>
<label><input type="checkbox" id="aggregate_results"> Acepto comunicar resultados agregados. No se autoriza publicar el texto original por defecto.</label>
<label for="review">Reseña original en español</label>
<small>Cuenta una experiencia real. No se pide una opinión positiva o negativa. Evita nombres de personas, teléfonos, correos u otros datos personales innecesarios. No incluyas claves ni contraseñas.</small>
<textarea id="review" rows="9" maxlength="5000"></textarea>
<button type="button" id="consent">Descargar consentimiento</button>
<button type="button" id="text">Descargar reseña</button><p id="status" role="status"></p>
<small>Puedes cerrar el formulario sin participar. Nada se transmite automáticamente.</small></main>
<script id="consent-template" type="application/json">__PAYLOAD__</script>
<script>
const template=JSON.parse(document.getElementById('consent-template').textContent);
document.getElementById('code').textContent=template.author_id;
const keys=['original_authorship','real_experience','annotation','training','evaluation','aggregate_results'];
function download(name,content,type){
 const blob=new Blob([content],{type}); const url=URL.createObjectURL(blob);
 const a=document.createElement('a');a.href=url;a.download=name;a.click();
 setTimeout(()=>URL.revokeObjectURL(url),1000);
}
document.getElementById('consent').onclick=()=>{
 if(!keys.every(key=>document.getElementById(key).checked)){
  document.getElementById('status').textContent='No se genera consentimiento: falta una aceptación. Puedes no participar.';return;
 }
 const value={...template,accepted_at:new Date().toISOString()};keys.forEach(key=>value[key]=true);
 download(template.author_id+'-consent.json',JSON.stringify(value,null,2)+'\\n','application/json;charset=utf-8');
 document.getElementById('status').textContent='Consentimiento descargado. Falta descargar tu reseña y enviar ambos archivos por el canal privado acordado.';
};
document.getElementById('text').onclick=()=>{
 const value=document.getElementById('review').value.trim();
 if(value.length<3||value.length>5000){document.getElementById('status').textContent='Escribe una reseña de 3 a 5000 caracteres.';return;}
 download(template.author_id+'-review.txt',value+'\\n','text/plain;charset=utf-8');
 document.getElementById('status').textContent='Reseña descargada. El coordinador necesita también tu consentimiento efectivo.';
};
</script></html>'''.replace('__NOTICE__',html.escape(NOTICE)).replace('__PAYLOAD__',payload)
    output.parent.mkdir(parents=True,exist_ok=True,mode=0o700)
    fd=os.open(output,os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o600)
    with os.fdopen(fd,"w",encoding="utf-8",newline="\n") as stream: stream.write(page)
    return {"status":"OFFLINE_FORM_READY_NOT_CONSENT_OR_REVIEW","author_id":template["author_id"]}
