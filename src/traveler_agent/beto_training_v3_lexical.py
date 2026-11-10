"""Exploratory TRAIN-only word/phrase associations, never a selection gate.

These descriptive correlations do not distinguish legitimate sentiment evidence
from generator shortcuts. No assessment text or model prediction is inspected.
"""
from collections import Counter,defaultdict
import argparse,math,re
from pathlib import Path

from .corpus import CorpusError
from .labels import LABELS
from .synthetic_diversity import ignored_target,write_json


def analyze_train(rows,records_sha256,*,minimum_document_frequency=6):
    if (not rows or any(r.get('role')!='train' for r in rows)
            or set(r.get('intended_label') for r in rows)!=set(LABELS)):
        raise CorpusError('Lexical associations require three-class TRAIN only.')
    if type(minimum_document_frequency) is not int or minimum_document_frequency<2:
        raise CorpusError('Minimum document frequency must be at least two.')
    totals=Counter(r['intended_label'] for r in rows)
    terms=defaultdict(Counter);phrases=defaultdict(Counter);families=defaultdict(set)
    for row in rows:
        words=re.findall(r'\w+',row['text'].casefold());label=row['intended_label']
        # Document presence, not token repetition, prevents a verbose review
        # from counting as independent support for the same word or phrase.
        for term in set(words):terms[term][label]+=1
        unique={' '.join(words[i:i+n]) for n in (2,3,4) for i in range(len(words)-n+1)}
        for phrase in unique:
            phrases[phrase][label]+=1;families[phrase].add(row['family'])
    associated={}
    for label in LABELS:
        candidates=[]
        for term,counts in terms.items():
            frequency=sum(counts.values())
            if frequency<minimum_document_frequency:continue
            yes=counts[label];other=frequency-yes;other_n=len(rows)-totals[label]
            log_odds=math.log((yes+.5)/(totals[label]-yes+.5))-math.log((other+.5)/(other_n-other+.5))
            candidates.append({'term':term,'documents':frequency,'documents_by_class':{k:counts[k] for k in LABELS},
                'class_fraction_given_term':yes/frequency,'smoothed_one_vs_rest_log_odds':log_odds})
        associated[label]=sorted(candidates,key=lambda r:(-r['smoothed_one_vs_rest_log_odds'],-r['documents'],r['term']))[:12]
    repeated=[{'phrase':term,'documents':sum(counts.values()),'families':len(families[term]),
               'documents_by_class':{k:counts[k] for k in LABELS}}
              for term,counts in phrases.items() if len(families[term])>=3]
    repeated.sort(key=lambda r:(-r['families'],-r['documents'],r['phrase']))
    return {'schema':'beto-v3-train-lexical-descriptive','scope':'TRAIN_ONLY_NO_FITTING_SELECTION_OR_ASSESSMENT',
        'records_sha256':records_sha256,'rows':len(rows),'classes':dict(totals),
        'minimum_document_frequency':minimum_document_frequency,'top_terms_by_class':associated,
        'most_repeated_two_to_four_word_phrases':repeated[:30],
        'model_or_data_decisions_changed':False,
        'limits':['Descriptive associations with synthetic author intent, not causal importance or human validity.',
                  'A sentiment-bearing word can be legitimate evidence; a high association alone is not proof of leakage.',
                  'No significance test or independent generalization claim; short word ngrams do not prove syntactic diversity.']}


def main(argv=None):
    from .beto_training_v3_data import load_data
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args(argv);parts,manifest,_=load_data(require_review=True)
    write_json(ignored_target(args.out,kinds=('artifacts',)),analyze_train(parts['train'],manifest['records_sha256']))


if __name__=='__main__':main()
