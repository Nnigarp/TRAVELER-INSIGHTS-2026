"""Replay original TRAIN-only TF-IDF vocabularies and calibration fold state.

CountVectorizer's frequency cutoff uses an unstable argsort for tied terms.
Keeping its already fitted state avoids CPU-dependent feature reselection.
No vocabulary or fold is learned from tune, calibration, policy or assessment.
"""
from pathlib import Path
import json
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion,Pipeline
from sklearn.svm import LinearSVC
from .corpus import CorpusError
from .synthetic_diversity import canonical_sha

STATE_PATH=Path('src/traveler_agent/research_data/beto_v3_train_representation.json')


def training_fingerprint(rows):
    if not rows or any(r.get('role')!='train' for r in rows):raise CorpusError('Representation replay requires TRAIN only.')
    return canonical_sha([{k:r[k] for k in ('review_id','text','intended_label','family','semantic_component')} for r in rows])


class BoundTrainTfidf(TfidfVectorizer):
    """Bind a fitted vocabulary only to its exact original fit document set."""
    def __init__(self,snapshots,branch):
        self.snapshots=snapshots;self.branch=branch
        if branch=='words':super().__init__(ngram_range=(1,2),strip_accents='unicode',sublinear_tf=True)
        elif branch=='chars':super().__init__(analyzer='char_wb',ngram_range=(3,5),sublinear_tf=True)
        else:raise CorpusError('Unknown TF-IDF feature branch.')

    def _bind(self,documents):
        documents=list(documents);fingerprint=canonical_sha(documents)
        matches=[s for s in self.snapshots if s['fit_documents_sha256']==fingerprint]
        if len(matches)!=1:raise CorpusError('Unregistered fit documents: never reuse a global vocabulary in a held fold.')
        terms=matches[0]['features'][self.branch]
        if len(terms)!=len(set(terms)) or not all(isinstance(t,str) for t in terms):raise CorpusError('Invalid frozen TRAIN vocabulary.')
        observed=set();analyzer=self.build_analyzer()
        for document in documents:observed.update(analyzer(document))
        if not set(terms)<=observed:raise CorpusError('Vocabulary includes a feature absent from this TRAIN fold.')
        self.vocabulary={term:i for i,term in enumerate(terms)}
        return documents

    def fit_transform(self,raw_documents,y=None):return super().fit_transform(self._bind(raw_documents),y)

    def fit(self,raw_documents,y=None):
        super().fit_transform(self._bind(raw_documents),y)
        return self


def frozen_features(snapshots):
    return FeatureUnion([('words',BoundTrainTfidf(snapshots,'words')),('chars',BoundTrainTfidf(snapshots,'chars'))])


def validate_state(train,state,choices):
    if (state.get('schema')!='beto-v3-fitted-train-representation-v1'
            or state.get('training_input_sha256')!=training_fingerprint(train)
            or state.get('human_gold') is not False
            or state.get('source_roles')!=['train']
            or state.get('classical_C')!={k:choices[k]['selected']['C'] for k in ('lr','svm')}):
        raise CorpusError('Representation state differs from the registered TRAIN/recipe.')
    ids=[r['review_id'] for r in train];byid={r['review_id']:r for r in train}
    if state['full']['fit_review_ids']!=ids:raise CorpusError('Full representation membership changed.')
    held=[];splits=[]
    if len(state['svm_folds'])!=3:raise CorpusError('Three original TRAIN calibration folds required.')
    positions={rid:i for i,rid in enumerate(ids)}
    for fold in state['svm_folds']:
        fit=fold['fit_review_ids'];test=fold['held_review_ids']
        if (len(fit)!=len(set(fit)) or len(test)!=len(set(test)) or set(fit)&set(test)
                or set(fit)|set(test)!=set(ids)):
            raise CorpusError('Invalid original TRAIN fold membership.')
        if {byid[i]['semantic_component'] for i in fit}&{byid[i]['semantic_component'] for i in test}:
            raise CorpusError('Related TRAIN families overlap inside SVM calibration.')
        if fold['fit_documents_sha256']!=canonical_sha([byid[i]['text'] for i in fit]):raise CorpusError('Fold text fingerprint changed.')
        held+=test;splits.append((np.asarray([positions[i] for i in fit]),np.asarray([positions[i] for i in test])))
    if sorted(held)!=sorted(ids):raise CorpusError('Original held folds must partition TRAIN exactly once.')
    if state['full']['fit_documents_sha256']!=canonical_sha([r['text'] for r in train]):raise CorpusError('Full text fingerprint changed.')
    return splits


def fit_represented_baselines(train,choices,state=None):
    state=json.loads(STATE_PATH.read_text(encoding='utf-8')) if state is None else state
    splits=validate_state(train,state,choices);texts=[r['text'] for r in train];labels=[r['intended_label'] for r in train]
    models={}
    for kind in ('dummy','lr','svm'):
        if kind=='svm':
            inner=Pipeline([('features',frozen_features(state['svm_folds'])),
                ('classifier',LinearSVC(C=choices['svm']['selected']['C'],class_weight='balanced',random_state=42))])
            model=CalibratedClassifierCV(inner,cv=splits,method='sigmoid')
        else:
            classifier=(DummyClassifier(strategy='most_frequent') if kind=='dummy' else
                LogisticRegression(C=choices['lr']['selected']['C'],max_iter=1500,class_weight='balanced',random_state=42))
            model=Pipeline([('features',frozen_features([state['full']])),('classifier',classifier)])
        model.fit(texts,labels);models[kind]=model
    return models
