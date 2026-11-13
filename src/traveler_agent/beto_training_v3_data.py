"""New synthetic-only V3 pilot: schema, provenance, quality gates and blind review."""
from __future__ import annotations
from collections import Counter,defaultdict
from itertools import combinations
import hashlib,json,re,subprocess
from pathlib import Path
import numpy as np
from .corpus import CorpusError,file_sha,text_sha
from .labels import LABELS
from .synthetic_diversity import canonical_sha,ignored_target,write_json
from .research_metrics import summarize_predictions

RESOURCE=Path(__file__).with_name('research_data')
PROTOCOL=RESOURCE/'beto_v3_pilot_protocol.json'
SCENARIOS=RESOURCE/'beto_v3_scenarios.json'
REVIEW=RESOURCE/'beto_v3_ai_review.json'
QUALITY=RESOURCE/'beto_v3_quality.json'
STYLE_PROTOCOL=RESOURCE/'beto_v3_style_revision_protocol.json'
TAGS={'negation','double_negation','negative_concordance','explicit_contrast','implicit_contrast','contrast',
      'mixed_aspects','irony','sarcasm','resolved_incident','regionalism','typo','objective_chronology','expectation','litotes'}
LABEL_ROLES=('train','tune','calibration','policy_dev','assessment')


def protocol():return json.loads(PROTOCOL.read_text(encoding='utf-8'))

def historical_text_hashes():
    """Historical content is used only for an exclusion hash set, never fitting."""
    result=set()
    def walk(value):
        if isinstance(value,dict):
            if isinstance(value.get('text'),str):result.add(text_sha(value['text']))
            for child in value.values():walk(child)
        elif isinstance(value,list):
            for child in value:walk(child)
    for name in ('synthetic_diversity_v2.json','synthetic_diversity_v3.json','contextual_supervised_v1.json',
                 'beto_v2_train.json','beto_v2_development.json','beto_v2_assessment.json'):
        walk(json.loads((RESOURCE/name).read_text(encoding='utf-8')))
    return result


def load_data(*,require_review=False):
    p=protocol();plan=json.loads(SCENARIOS.read_text(encoding='utf-8'))
    scenarios={r['family']:r for r in plan['scenarios']}
    if len(scenarios)!=len(plan['scenarios']):raise CorpusError('Familia de escenario repetida.')
    paths=sorted(RESOURCE.glob('beto_v3_batch_*.json'))
    if len(paths)!=6:raise CorpusError('Se requieren seis lotes originales, incluida evaluación independiente.')
    rows=[];seen=set();group_roles={};banned=historical_text_hashes()
    for path in paths:
        batch=json.loads(path.read_text(encoding='utf-8'));prov=batch.get('provenance',{})
        if (batch.get('schema')!='traveler-beto-v3-author-batch-v1' or prov.get('synthetic') is not True
                or prov.get('human_authors') is not False or not prov.get('actual_task')
                or prov.get('protocol_version')!=p['version']):raise CorpusError('Procedencia de lote no acreditada.')
        for original in batch['reviews']:
            row=dict(original);scenario=scenarios.get(row['family'])
            if scenario is None:raise CorpusError('Escenario no registrado.')
            for field in ('role','region','experience_type','semantic_component','length_band'):
                if row.get(field)!=scenario[field]:raise CorpusError('Metadatos difieren del plan de escenarios.')
            text=row.get('text');label=row.get('intended_label');role=row['role']
            if not isinstance(text,str) or not 8<=len(text)<=5000:raise CorpusError('Texto inválido.')
            if role in LABEL_ROLES and label not in LABELS:raise CorpusError('Intención global inválida.')
            if role not in LABEL_ROLES and (role not in ('ambiguity_policy','ambiguity_assessment') or label is not None):
                raise CorpusError('Ambiguo sin sentimiento; no convertirlo en neutral.')
            if not row.get('author_rationale') or not row.get('structure') or not set(row.get('phenomena',[]))<=TAGS:
                raise CorpusError('Trazabilidad lingüística incompleta.')
            if re.search(r'(sentimiento|etiqueta|label)\s*[:=]\s*(positivo|neutral|negativo)',text,re.I):
                raise CorpusError('Marcador artificial de etiqueta en el texto.')
            n=len(text.split());lo,hi=p['length_words'][row['length_band']]
            if not lo<=n<=hi:raise CorpusError(f"Longitud fuera del plan: {row['family']} ({n}).")
            digest=text_sha(text)
            if digest in seen or digest in banned:raise CorpusError('Duplicado exacto o reutilización histórica.')
            seen.add(digest)
            group=row['semantic_component']
            if group in group_roles and group_roles[group]!=role:raise CorpusError('Fuga de componente entre roles.')
            group_roles[group]=role
            row.update(review_id='b3_'+canonical_sha([row['family'],digest])[:20],text_sha256=digest,
                       source=path.name,source_arm='v3_original_pilot',label_origin=p['label_origin'],
                       synthetic=True,generator_task=row.get('revision',{}).get('actual_task',prov['actual_task']),protocol_version=p['version'])
            if set(row['phenomena'])&{'explicit_contrast','implicit_contrast'} and 'contrast' not in row['phenomena']:
                row['phenomena']=[*row['phenomena'],'contrast']
            rows.append(row)
    parts={role:[r for r in rows if r['role']==role] for role in p['pilot_role_rows']}
    if {k:len(v) for k,v in parts.items()}!=p['pilot_role_rows']:raise CorpusError('Conteos distintos del protocolo piloto.')
    for family in scenarios:
        selected=[r for r in rows if r['family']==family]
        if not selected:raise CorpusError('Falta un escenario del plan.')
        if selected[0]['role'] in LABEL_ROLES and sorted(r['intended_label'] for r in selected)!=sorted(LABELS):
            raise CorpusError('Familia requiere tres experiencias distintas y tres intenciones.')
    manifest={'schema':'beto-v3-pilot-manifest','records_sha256':canonical_sha(rows),'protocol_sha256':file_sha(PROTOCOL),
              'scenario_sha256':file_sha(SCENARIOS),'source_sha256':{f.name:file_sha(f) for f in paths},
              'role_sha256':{role:canonical_sha(part) for role,part in parts.items()},
              'counts':{role:{'rows':len(part),'families':len({r['family'] for r in part}),
                        'classes':dict(Counter(r['intended_label'] for r in part if r['intended_label'])),
                        'regions':dict(Counter(r['region'] for r in part)),
                        'length_by_class':{label:dict(Counter(r['length_band'] for r in part if r['intended_label']==label)) for label in LABELS}}
                        for role,part in parts.items()},'human_gold':False,'population_independence_proven':False}
    if require_review:
        if any(r.get('revision',{}).get('phenomena_annotation_status') not in (None,'REFRESHED_BY_EDITOR') for r in rows):
            raise CorpusError('Anotaciones lingüísticas pendientes después de editar TRAIN.')
        review=json.loads(REVIEW.read_text(encoding='utf-8'))
        if review.get('records_sha256')!=manifest['records_sha256'] or review.get('admitted') is not True:
            raise CorpusError('Revisión independiente no admite este corpus.')
        validate_admission(parts,manifest,p,review)
        manifest['ai_review_sha256']=file_sha(REVIEW)
        manifest['quality_sha256']=file_sha(QUALITY)
        excluded=set(review.get('quarantined_families',[]))
        parts={role:[r for r in part if r['family'] not in excluded] for role,part in parts.items()}
        manifest['quarantined_families']=sorted(excluded)
        manifest['admitted_role_rows']={role:len(part) for role,part in parts.items()}
    manifest['assessment_membership']=[{k:r[k] for k in ('review_id','family','intended_label')} for r in parts['assessment']]
    return parts,manifest,p


def duplicate_audit(rows,threshold=.7):
    chars=[];words=[]
    for row in rows:
        text=' '.join(row['text'].casefold().split());tokens=re.findall(r'\w+',text)
        chars.append({text[i:i+5] for i in range(max(0,len(text)-4))})
        words.append({tuple(tokens[i:i+3]) for i in range(max(0,len(tokens)-2))})
    candidates=[];screened=0
    for i,j in combinations(range(len(rows)),2):
        a,b=rows[i],rows[j]
        scores=[]
        for sets in (chars,words):
            x,y=sets[i],sets[j]
            if not x or not y or min(len(x),len(y))/max(len(x),len(y))<threshold:scores.append(0.);continue
            scores.append(len(x&y)/len(x|y))
        screened+=1
        if max(scores)>=threshold:
            candidates.append({'left':a['review_id'],'right':b['review_id'],'roles':[a['role'],b['role']],
                               'same_family':a['family']==b['family'],'char5_jaccard':scores[0],'word3_jaccard':scores[1]})
    return {'pairs_screened':screened,'threshold':threshold,'candidates':candidates,
            'blocking_candidates':len(candidates),
            'limits':'Lexical screening and declared semantic families do not prove semantic independence.'}


def training_shortcuts(rows):
    if not rows or any(r['role']!='train' for r in rows):raise CorpusError('Diagnóstico de atajos solo TRAIN.')
    from sklearn.feature_extraction import DictVectorizer
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.model_selection import StratifiedGroupKFold,cross_val_predict
    cv=list(StratifiedGroupKFold(5,shuffle=True,random_state=42).split(rows,[r['intended_label'] for r in rows],groups=[r['family'] for r in rows]))
    shape=[{'words':len(r['text'].split()),'characters':len(r['text']),'sentences':len(re.findall(r'[.!?]+',r['text'])),
            'exclamations':r['text'].count('!'),'questions':r['text'].count('?'),'semicolons':r['text'].count(';'),
            'commas':r['text'].count(',')} for r in rows]
    metadata=[{k:r.get(k,'not_recorded') for k in ('region','experience_type','generator_task','length_band','structure','style_profile')} for r in rows]
    y=[r['intended_label'] for r in rows];result={}
    for name,features in [('length_and_punctuation',shape),('metadata_only',metadata)]:
        model=make_pipeline(DictVectorizer(),StandardScaler(with_mean=False),LogisticRegression(C=1,max_iter=1500,random_state=42))
        result[name]=summarize_predictions(y,cross_val_predict(model,features,y,cv=cv))
    endings=defaultdict(set)
    for r in rows:endings[(r['intended_label'],' '.join(re.findall(r'\w+',r['text'].casefold())[-4:]))].add(r['family'])
    result['repeated_endings']=[{'label':label,'ending':end,'families':len(groups)} for (label,end),groups in endings.items() if len(groups)>1]
    result['max_ending_family_fraction']=max((len(groups)/len({r['family'] for r in rows if r['intended_label']==label}) for (label,_),groups in endings.items()),default=0.)
    result['scope']='TRAIN_ONLY_NUISANCE_CORRELATIONS_NOT_CAUSAL_FEATURE_IMPORTANCE'
    return result


def diversity(rows):
    result={}
    for role in sorted({r['role'] for r in rows}):
        part=[r for r in rows if r['role']==role];tokens=[re.findall(r'\w+',r['text'].casefold()) for r in part]
        all_words=[w for words in tokens for w in words];ngrams={n:{tuple(words[i:i+n]) for words in tokens for i in range(max(0,len(words)-n+1))} for n in (1,2,3)}
        result[role]={'rows':len(part),'word_total':len(all_words),'vocabulary':len(set(all_words)),
          'type_token_ratio_length_sensitive':len(set(all_words))/len(all_words),
          'distinct_ngrams':{str(n):len(values) for n,values in ngrams.items()},
          'experience_types':dict(Counter(r['experience_type'] for r in part)),
          'self_reported_structures':dict(Counter(r['structure'] for r in part)),
          'phenomena_by_class':{tag:dict(Counter(r['intended_label'] for r in part if tag in r['phenomena'])) for tag in sorted({t for r in part for t in r['phenomena']})},
          'syntax_proxy_clause_markers':dict(Counter(t for words in tokens for t in words if t in {'aunque','mientras','porque','si','cuando','pero','sino'}))}
    return {'roles':result,'limits':['Lexical/structural proxies; no POS/dependency parser or human naturalness validation.','Structure and phenomena are AI author annotations, not independent human labels.']}


def blind_sample(parts,p):
    """Seeded role-stratified sample; balance class/region/length greedily within role."""
    rng=np.random.default_rng(p['data_admission']['review_seed']);chosen=[]
    for role,n in [('train',50),('tune',10),('calibration',10),('policy_dev',10),('assessment',10)]:
        pool=list(parts[role]);rng.shuffle(pool);counts=Counter()
        for _ in range(n):
            row=min(pool,key=lambda r:sum(counts[(field,r[field])] for field in ('region','intended_label','length_band')))
            chosen.append(row);pool.remove(row)
            for field in ('region','intended_label','length_band'):counts[(field,row[field])]+=1
    return chosen


def verify_git_registration(commit,manifest=None):
    if not re.fullmatch('[0-9a-f]{40}',commit or ''):raise CorpusError('Commit completo de registro requerido.')
    paths=[PROTOCOL] if manifest is None else [PROTOCOL,SCENARIOS,REVIEW,QUALITY,*sorted(RESOURCE.glob('beto_v3_batch_*.json'))]
    if manifest is not None and STYLE_PROTOCOL.exists():paths.append(STYLE_PROTOCOL)
    for path in paths:
        name='src/traveler_agent/research_data/'+path.name
        try:content=subprocess.check_output(['git','show',f'{commit}:{name}'],stderr=subprocess.DEVNULL)
        except subprocess.CalledProcessError as exc:raise CorpusError('Archivo fuera del registro Git.') from exc
        if hashlib.sha256(content).hexdigest()!=file_sha(path):raise CorpusError('Datos/protocolo/revisión cambiaron tras registrar.')


def automatic_audit(cache,*,download=False):
    from .beto_encoder_probe import checkpoint,REVISION
    from transformers import AutoTokenizer
    parts,manifest,p=load_data();rows=[r for part in parts.values() for r in part]
    checkpoint(cache,download=download);tokenizer=AutoTokenizer.from_pretrained(str(cache),local_files_only=True,trust_remote_code=False)
    lengths={role:[len(tokenizer(r['text'],truncation=False)['input_ids']) for r in part] for role,part in parts.items()}
    token_report={role:{'rows':len(values),'maximum':max(values),'over_128':sum(n>128 for n in values),
                       'over_256':sum(n>256 for n in values),'over_128_fraction':sum(n>128 for n in values)/len(values)}
                  for role,values in lengths.items()}
    duplicates=duplicate_audit(rows,p['data_admission']['near_duplicate_char5_or_word3_jaccard'])
    shortcuts=training_shortcuts(parts['train']);config=p['data_admission']
    checks={'near_duplicates':len(duplicates['candidates'])==0,
            'shape_shortcuts':shortcuts['length_and_punctuation']['macro_f1']<=config['shape_only_group_oof_macro_f1_max'],
            'long_training_context':token_report['train']['over_128_fraction']>=config['minimum_train_rows_over_128_tokens_fraction'],
            'repeated_endings':shortcuts['max_ending_family_fraction']<=config['max_repeated_four_word_ending_family_fraction_per_class']}
    return {'schema':'beto-v3-automatic-quality','manifest':manifest,'tokenizer_revision':REVISION,
            'tokenization':token_report,'duplicates':duplicates,'shortcuts':shortcuts,'diversity':diversity(rows),
            'checks':checks,'automatic_gates_pass':all(checks.values()),'human_validated':False,
            'limits':['This checks lexical and declared-family independence, not unseen semantic overlap or population independence.',
                      'No final sentiment classifier predictions or metrics were used for these audits.']}


def prepare_blind_packet(out):
    parts,manifest,p=load_data();sample=blind_sample(parts,p)
    sample+=parts['ambiguity_policy']+parts['ambiguity_assessment']
    # Random-looking text-derived identifiers disclose no source, role or intent.
    rng=np.random.default_rng(p['data_admission']['review_seed']);rng.shuffle(sample)
    out=ignored_target(out,kinds=('artifacts',))
    if out.exists():raise CorpusError('Paquete ciego existente; conservarlo.')
    write_json(out,{'scope':'AI_BLIND_REVIEW_NOT_HUMAN_ANNOTATION',
                    'items':[{'id':r['review_id'],'text':r['text']} for r in sample]})
    return manifest,len(sample)


def review_statistics(parts,manifest,first,second,decisions):
    """Audit real independent review files; decisions never silently rewrite intent."""
    from sklearn.metrics import cohen_kappa_score
    if first.get('actual_task')==second.get('actual_task') or not first.get('actual_task') or not second.get('actual_task'):
        raise CorpusError('Dos revisores reales distintos requeridos.')
    p=protocol();expected={r['review_id']:r for r in blind_sample(parts,p)}
    expected.update({r['review_id']:r for role in ('ambiguity_policy','ambiguity_assessment') for r in parts[role]})
    def index(value):
        records=value['annotations'];result={r['id']:r for r in records}
        if len(records)!=len(result) or set(result)!=set(expected):raise CorpusError('Paquete de revisión incompleto o duplicado.')
        if any(r.get('label') not in (*LABELS,'ambiguo') or not r.get('rationale') for r in records):raise CorpusError('Juicio de revisión inválido.')
        return result
    left,right=index(first),index(second);disagreement=[];quarantine=set();unresolved=0
    labeled=[key for key,row in expected.items() if row['intended_label'] in LABELS]
    for key,row in expected.items():
        a,b=left[key]['label'],right[key]['label'];intended=row['intended_label'] or 'ambiguo'
        if a!=b or a!=intended:
            decision=decisions.get(key)
            if (not isinstance(decision,dict) or decision.get('label') not in (*LABELS,'ambiguo')
                    or not decision.get('rationale')):raise CorpusError('Desacuerdo sin adjudicación explícita.')
            final=decision['label']
            disagreement.append({'id':key,'family':row['family'],'author_intent':intended,'reviewer1':a,'reviewer2':b,
                                 'adjudication':decision})
            if final!=intended:quarantine.add(row['family'])
            if key in labeled and final=='ambiguo':unresolved+=1
    agreement=[sum(v[key]['label']==expected[key]['intended_label'] for key in labeled)/len(labeled) for v in (left,right)]
    kappa=float(cohen_kappa_score([left[k]['label'] for k in labeled],[right[k]['label'] for k in labeled]))
    labeled_families={r['family'] for role in LABEL_ROLES for r in parts[role]}
    fraction=len(quarantine&labeled_families)/len(labeled_families)
    cfg=p['data_admission'];checks={'reviewer_agreement':all(a>=cfg['minimum_author_intent_agreement_each_reviewer'] for a in agreement),
        'interreviewer_kappa':np.isfinite(kappa) and kappa>=cfg['minimum_pairwise_ai_review_kappa'],
        'unresolved':unresolved/len(labeled)<=cfg['maximum_unresolved_sample_fraction'],
        'quarantined_labeled_families':fraction<=cfg['maximum_quarantined_labeled_families_fraction'],
        'nonempty_ambiguity_cohorts':all(any(r['family'] not in quarantine for r in parts[role]) for role in ('ambiguity_policy','ambiguity_assessment'))}
    return {'records_sha256':manifest['records_sha256'],'reviewers':[first,second],
            'reviewer_author_intent_agreement':agreement,'pairwise_review_kappa':kappa,'labeled_review_n':len(labeled),
            'unresolved_labeled':unresolved,'quarantined_families':sorted(quarantine),
            'quarantined_labeled_family_fraction':fraction,'disagreements':disagreement,'checks':checks,
            'review_gates_pass':all(checks.values()),'human_gold':False,
            'limits':'Agreement between AI reviewers is neither human gold nor a guarantee of regional/cultural naturalness.'}


def validate_admission(parts,manifest,p,review):
    """Recompute gate decisions from hashed evidence; an admitted boolean is insufficient."""
    try:
        automatic=json.loads(QUALITY.read_text(encoding='utf-8'))
        if (review['automatic_audit_sha256']!=file_sha(QUALITY)
                or automatic['manifest']['records_sha256']!=manifest['records_sha256']
                or automatic['manifest']['protocol_sha256']!=manifest['protocol_sha256']):
            raise CorpusError('Auditoría automática de otro corpus/protocolo o modificada.')
        cfg=p['data_admission']
        checks={'near_duplicates':not automatic['duplicates']['candidates'],
            'shape_shortcuts':automatic['shortcuts']['length_and_punctuation']['macro_f1']<=cfg['shape_only_group_oof_macro_f1_max'],
            'long_training_context':automatic['tokenization']['train']['over_128_fraction']>=cfg['minimum_train_rows_over_128_tokens_fraction'],
            'repeated_endings':automatic['shortcuts']['max_ending_family_fraction']<=cfg['max_repeated_four_word_ending_family_fraction_per_class']}
        if automatic.get('checks')!=checks or automatic.get('automatic_gates_pass') is not True or not all(checks.values()):
            raise CorpusError('Auditoría automática no supera todos los criterios registrados.')
        reviewers=review['reviewers']
        if len(reviewers)!=2 or any(r.get('human') is not False or r.get('synthetic_ai_review') is not True for r in reviewers):
            raise CorpusError('Se requieren dos revisiones AI declaradas, no gold humano.')
        expected=review_statistics(parts,manifest,reviewers[0],reviewers[1],review['adjudication_decisions'])
        if (not expected['review_gates_pass'] or review.get('review_gates_pass') is not True
                or review.get('checks')!=expected['checks']
                or review.get('quarantined_families')!=expected['quarantined_families']):
            raise CorpusError('Revisión/adjudicación no supera criterios o exclusiones inconsistentes.')
    except (KeyError,TypeError,IndexError,FileNotFoundError) as exc:
        raise CorpusError('Evidencia de admisión incompleta; no entrenar.') from exc
