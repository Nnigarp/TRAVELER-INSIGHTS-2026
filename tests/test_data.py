from pathlib import Path
import pandas as pd
import pytest
from traveler_agent.dataset import read_labeled_csv
from traveler_agent.experiments import holdout_evaluate

def test_conflicting_duplicate_labels(tmp_path):
    p=tmp_path/'a.csv'
    pd.DataFrame({'reseña':['Mismo texto','Mismo texto','Otra reseña'], 'sentimiento':['positivo','negativo','neutral']}).to_csv(p,index=False)
    with pytest.raises(ValueError,match='contradictorias'): read_labeled_csv(p)

def test_holdout_with_too_few_rows(tmp_path):
    p=tmp_path/'a.csv'
    pd.DataFrame({'reseña':['bien','mal','normal'], 'sentimiento':['positivo','negativo','neutral']}).to_csv(p,index=False)
    df=read_labeled_csv(p)
    with pytest.raises(ValueError,match='30'): holdout_evaluate(df)
