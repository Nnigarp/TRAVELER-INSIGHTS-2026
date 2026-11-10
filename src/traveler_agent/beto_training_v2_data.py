"""Admission and immutable registration of the BETO Training V2 synthetic study."""
from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import re

from .corpus import CorpusError, file_sha, text_sha
from .labels import LABELS
from .research_v3_data import duplicate_screen
from .synthetic_diversity import canonical_sha, ignored_target, write_json

RESOURCE=Path(__file__).with_name('research_data')
PROTOCOL=RESOURCE/'beto_training_v2_protocol.json'
CATALOGS=tuple(RESOURCE/name for name in ('beto_v2_train.json','beto_v2_development.json','beto_v2_assessment.json'))


def load_data():
    protocol=json.loads(PROTOCOL.read_text(encoding='utf-8'))
    rows=[]
    v3=json.loads((RESOURCE/'synthetic_diversity_v3.json').read_text(encoding='utf-8'))
    for family in v3['families']:
        if family['role']!='train':continue
        for case in family['cases']:
            if case['variant']>=2:continue
            rows.append({**case,**{k:family[k] for k in ('family','semantic_component','domain','region')},
                         'role':'train','source':'V3_TRAIN_ONLY_120','source_arm':'existing120'})
    new_families={r['family'] for r in rows}
    for path in CATALOGS:
        value=json.loads(path.read_text(encoding='utf-8'))
        if (value.get('version')!='beto-training-v2' or value.get('provenance',{}).get('human_authors') is not False
                or value['provenance'].get('label_origin')!='AUTHOR_INTENT_NOT_HUMAN_GOLD'):
            raise CorpusError('Procedencia sintética no acreditada.')
        for family in value['families']:
            if family['family'] in new_families:raise CorpusError('Familia repetida en catálogos.')
            new_families.add(family['family'])
            role=family['role'];labels=[c['intended_label'] for c in family['cases']]
            if role not in protocol['role_rows']:raise CorpusError('Rol no registrado.')
            if (role.startswith('ambiguity') and labels!=[None]) or (not role.startswith('ambiguity') and sorted(labels)!=sorted(LABELS)):
                raise CorpusError('Cada familia requiere tres intenciones o ambigüedad sin sentimiento.')
            for case in family['cases']:
                if not case.get('rationale') or not case.get('structure'):raise CorpusError('Falta trazabilidad lingüística.')
                rows.append({**case,**{k:family[k] for k in ('family','semantic_component','domain','region','role')},
                             'source':path.stem,'source_arm':'new120' if role=='train' else role})
    fingerprints=set();components={};family_roles={}
    for row in rows:
        if not isinstance(row['text'],str) or not 5<=len(row['text'])<=5000:raise CorpusError('Texto inválido.')
        digest=text_sha(row['text'])
        if digest in fingerprints:raise CorpusError('Duplicación exacta entre textos.')
        fingerprints.add(digest)
        for key,index in (('semantic_component',components),('family',family_roles)):
            if row[key] in index and index[row[key]]!=row['role']:raise CorpusError('Fuga de componente/familia entre roles.')
            index[row[key]]=row['role']
        row['review_id']='b2_'+canonical_sha([row['family'],digest])[:20]
        row['text_sha256']=digest;row['label_origin']='AUTHOR_INTENT_NOT_HUMAN_GOLD'
        n=len(row['text'].split());row['length_band']='short' if n<=12 else 'long' if n>=85 else 'medium'
    # Frozen evaluated catalogues only as exact hash denylist, never training labels.
    banned={text_sha(c['text']) for f in v3['families'] if f['role']!='train' for c in f['cases']}
    banned.update(text_sha(c['text']) for c in v3['ambiguity'])
    for name in ('synthetic_diversity_v2.json','contextual_supervised_v1.json'):
        old=json.loads((RESOURCE/name).read_text(encoding='utf-8'))
        banned.update(text_sha(c['text']) for f in old['families'] for c in f['cases'])
        for field in ('ambiguity_cohort','illustrative_user_examples'):
            banned.update(text_sha(c['text']) for c in old.get(field,[]))
    if fingerprints&banned:raise CorpusError('Reutilización de ejemplos históricos evaluados.')
    parts={role:[r for r in rows if r['role']==role] for role in protocol['role_rows']}
    if {k:len(v) for k,v in parts.items()}!=protocol['role_rows']:raise CorpusError('Presupuestos distintos del registro.')
    for arm in ('existing120','new120'):
        subset=[r for r in parts['train'] if r['source_arm']==arm]
        if len(subset)!=120 or Counter(r['intended_label'] for r in subset)!=Counter({label:40 for label in LABELS}):
            raise CorpusError('TRAIN no conserva presupuesto y equilibrio predefinidos.')
    audit=duplicate_screen(rows,protocol['near_duplicate_jaccard'])
    if audit['candidate_count']:raise CorpusError('Cercanía léxica entre roles pendiente de revisión antes de entrenar.')
    counts={role:{'rows':len(part),'components':len({r['semantic_component'] for r in part}),
                  'classes':dict(Counter(r['intended_label'] for r in part if r['intended_label'])),
                  'length_by_class':{label:dict(Counter(r['length_band'] for r in part if r['intended_label']==label)) for label in LABELS},
                  'phenomena_by_class':{tag:dict(Counter(r['intended_label'] for r in part if r['intended_label'] in LABELS and tag in r['phenomena']))
                                        for tag in sorted({p for r in part for p in r['phenomena']})}}
            for role,part in parts.items()}
    manifest={'schema':'beto-training-v2-data','protocol_sha256':file_sha(PROTOCOL),
        'catalog_sha256':{p.name:file_sha(p) for p in CATALOGS},'records_sha256':canonical_sha(rows),
        'role_hashes':{k:canonical_sha(v) for k,v in parts.items()},'counts':counts,'duplicates':audit,
        'human_gold':False,'historical_test_used_for_selection':False,'population_independence_proven':False}
    return parts,manifest,protocol


def register(output:Path):
    output=ignored_target(output,kinds=('artifacts',))
    if output.exists():raise CorpusError('Registro ya existente; conservarlo.')
    _,manifest,protocol=load_data();output.mkdir(parents=True)
    write_json(output/'data-manifest.json',manifest);write_json(output/'protocol.json',protocol)
    return manifest


def verify_registration(path:Path):
    parts,manifest,protocol=load_data()
    if json.loads((path/'data-manifest.json').read_text())!=manifest or json.loads((path/'protocol.json').read_text())!=protocol:
        raise CorpusError('Catálogo/protocolo cambiados tras registrar; no entrenar/evaluar.')
    return parts,manifest,protocol


def training_diagnostics(rows):
    """TRAIN-only nuisance probes; no causal claim or predictor input metadata."""
    if not rows or any(r['role']!='train' for r in rows):raise CorpusError('Diagnóstico solo TRAIN.')
    import numpy as np
    from sklearn.feature_extraction import DictVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedGroupKFold,cross_val_predict
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from .research_metrics import summarize_predictions
    y=np.asarray([r['intended_label'] for r in rows]);groups=[r['semantic_component'] for r in rows]
    cv=list(StratifiedGroupKFold(n_splits=5,shuffle=True,random_state=42).split(rows,y,groups))
    shape=[{'log_words':float(np.log1p(len(r['text'].split()))),'sentences':len(re.findall(r'[.!?]+',r['text'])),
            'exclamations':r['text'].count('!'),'questions':r['text'].count('?'),'semicolons':r['text'].count(';')} for r in rows]
    metadata=[{k:r[k] for k in ('domain','region','source_arm')} for r in rows]
    scores={}
    for name,features in (('shape_only',shape),('source_domain_region_only',metadata)):
        model=make_pipeline(DictVectorizer(),StandardScaler(with_mean=False),LogisticRegression(C=1,max_iter=1000,random_state=42))
        scores[name]=summarize_predictions(y,cross_val_predict(model,features,y,cv=cv))
    endings={}
    for row in rows:
        words=re.findall(r'\w+',row['text'].casefold());key=(row['intended_label'],' '.join(words[-4:]))
        endings.setdefault(key,set()).add(row['family'])
    repeated=[{'label':label,'ending':ending,'families':len(families)} for (label,ending),families in endings.items() if len(families)>1]
    return {'scope':'TRAIN_ONLY_NO_EVALUATION_OR_FEATURE_LEAKAGE','group_oof':scores,
            'repeated_four_word_endings':repeated,'shape_macro_f1_above_0_6_flag':scores['shape_only']['macro_f1']>.6,
            'interpretation':'Nuisance correlation diagnostic, not feature contribution or proof of shortcut-free data. Metadata never enters sentiment models.'}
