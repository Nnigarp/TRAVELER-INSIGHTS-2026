"""Evaluación de entrenamiento real; nunca reportar desempeño del dataset demo."""
import json
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from .labels import LABELS
from .model import train

def holdout_evaluate(df, text_col='reseña',label_col='sentimiento', test_size=0.2, *, model_kind='lr'):
    counts=df[label_col].value_counts()
    if len(df)<30 or (counts<5).any():
        raise ValueError('Para holdout se exigen al menos 30 registros únicos y 5 por clase; no inflar resultados con pocos ejemplos.')
    # Las reseñas idénticas fueron deduplicadas en dataset.read_labeled_csv.
    train_ids,test_ids=train_test_split(np.arange(len(df)),test_size=test_size,random_state=42,stratify=df[label_col])
    a,b=df.iloc[train_ids],df.iloc[test_ids]
    estimator=train(a[text_col].tolist(), a[label_col].tolist(), model_kind=model_kind)
    predicted=estimator.predict(b[text_col].tolist())
    metrics={"model_kind":model_kind,"n_total":len(df),"n_train":len(a),"n_test":len(b),
        "accuracy":round(float(accuracy_score(b[label_col],predicted)),4),
        "macro_f1":round(float(f1_score(b[label_col],predicted,labels=LABELS,average='macro',zero_division=0)),4),
        "classification_report":classification_report(b[label_col],predicted,labels=LABELS,output_dict=True,zero_division=0),
        "confusion_matrix_labels":list(LABELS),
        "confusion_matrix":confusion_matrix(b[label_col],predicted,labels=LABELS).tolist(),
        "limitations":["Holdout único; extender a validación cruzada y conjunto externo.","Sin calibración de probabilidades.","Revisar cambios de dominio para Cartagena."]}
    return estimator,metrics
