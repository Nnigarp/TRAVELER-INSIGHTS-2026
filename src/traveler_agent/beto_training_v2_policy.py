"""New-role calibration/abstention study, never installed in DEMO ControlAgent."""
from itertools import product

import numpy as np

from .corpus import CorpusError
from .labels import LABELS
from .linguistic import contextual_signals
from .research_metrics import summarize_predictions
from .research_v3_selective import valid_probabilities


def reviewed(rows,probabilities,policy):
    p=valid_probabilities(probabilities)
    if len(rows)!=len(p):raise CorpusError('Cohorte desalineada.')
    pred=p.argmax(axis=1);positive=pred==LABELS.index('positivo')
    threshold=np.where(positive,policy['positive'],policy['global'])
    ordered=np.sort(p,axis=1)
    mask=(p.max(axis=1)<threshold)|((ordered[:,-1]-ordered[:,-2])<policy['margin'])
    if policy['guard']:
        cues=[contextual_signals(r['text']) for r in rows]
        mask|=positive&np.asarray([bool(c['negaciones'] or c['contrastes']) for c in cues])
    return mask


def describe(rows,p,ambiguity,ambiguous_p,policy,criteria):
    mask=reviewed(rows,p,policy);guess=np.asarray(LABELS)[np.argmax(p,axis=1)]
    metric=summarize_predictions([r['intended_label'] for r in rows],guess,reviewed=mask)['abstention']
    metric.update(total_negatives=sum(r['intended_label']=='negativo' for r in rows),
        accepted_negatives=sum(r['intended_label']=='negativo' and not m for r,m in zip(rows,mask)),
        accepted_components=len({r['semantic_component'] for r,m in zip(rows,mask) if not m}),
        ambiguity_review_rate=float(reviewed(ambiguity,ambiguous_p,policy).mean()))
    checks={
        'accepted_support':metric['accepted']>=criteria['min_accepted'],
        'negative_support':metric['total_negatives']>=criteria['min_total_negatives'],
        'accepted_negative_support':metric['accepted_negatives']>=criteria['min_accepted_negatives'],
        'component_support':metric['accepted_components']>=criteria['min_accepted_components'],
        'coverage':metric['coverage']>=criteria['min_coverage'],
        'risk':metric['error_risk_among_accepted'] is not None and metric['error_risk_among_accepted']<=criteria['max_empirical_risk'],
        'dangerous_errors':metric['automatic_negative_to_positive_rate_over_all_negatives'] is not None and metric['automatic_negative_to_positive_rate_over_all_negatives']<=criteria['max_automatic_negative_to_positive_rate'],
        'ambiguity_review':metric['ambiguity_review_rate']>=criteria['min_ambiguity_review_rate']}
    return {'metrics':metric,'criteria':checks,'all_empirical_criteria_pass':all(checks.values()),
            'human_or_population_risk_certified':False}


def select(rows,p,ambiguity,ambiguous_p,protocol):
    if any(r['role']!='policy_dev' for r in rows) or any(r['role']!='ambiguity_policy' for r in ambiguity):
        raise CorpusError('La política solo se selecciona en desarrollo independiente.')
    config=protocol['policy'];candidates=[]
    for global_,positive,guard in product(config['global_thresholds'],config['positive_thresholds'],config['text_guards']):
        policy={'global':global_,'positive':positive,'guard':guard,'margin':config['margin']}
        result=describe(rows,p,ambiguity,ambiguous_p,policy,config)
        candidates.append({'policy':policy,**result})
    eligible=[r for r in candidates if r['all_empirical_criteria_pass']]
    chosen=max(eligible,key=lambda r:(r['metrics']['coverage'],-r['metrics']['error_risk_among_accepted'])) if eligible else None
    return {'selected':chosen['policy'] if chosen else None,
            'status':'EMPIRICAL_DEVELOPMENT_CANDIDATE_ONLY' if chosen else 'NO_POLICY_ADMITTED_NOT_REVIEW_ALL_SOLUTION',
            'candidates':candidates,'guarantee':'NONE; multiplicity and synthetic group dependence preclude a deployment guarantee'}
