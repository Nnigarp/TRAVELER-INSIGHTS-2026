from copy import deepcopy
import pytest
from traveler_agent.beto_training_v3_lexical import analyze_train
from traveler_agent.corpus import CorpusError


def rows():
    return [{'role':'train','intended_label':label,'family':str(i),
             'text':('sucio '*20 if label=='negativo' else 'limpio ' if label=='positivo' else 'horario ')+ 'llegamos al museo'}
            for label in ('negativo','neutral','positivo') for i in range(6)]


def test_word_associations_use_document_support_not_repetition():
    report=analyze_train(rows(),'hash');term=next(x for x in report['top_terms_by_class']['negativo'] if x['term']=='sucio')
    assert term['documents']==6 and term['documents_by_class']=={'negativo':6,'neutral':0,'positivo':0}
    assert term['class_fraction_given_term']==1
    assert report['model_or_data_decisions_changed'] is False


def test_repeated_phrases_count_semantic_families_separately_from_documents():
    report=analyze_train(rows(),'hash')
    phrase=next(x for x in report['most_repeated_two_to_four_word_phrases'] if x['phrase']=='al museo')
    assert phrase['documents']==18 and phrase['families']==6


def test_nontraining_text_is_rejected_before_any_association():
    data=rows();data[-1]['role']='assessment'
    with pytest.raises(CorpusError,match='TRAIN'):analyze_train(data,'hash')


def test_insufficient_class_support_is_rejected_and_source_rows_are_unchanged():
    data=rows();before=deepcopy(data);analyze_train(data,'hash');assert data==before
    with pytest.raises(CorpusError,match='three-class'):analyze_train(data[:6],'hash')
