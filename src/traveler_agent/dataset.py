"""Ingreso de CSV; etiquetas textuales verificadas y controles básicos de calidad."""
import pandas as pd
from .labels import normalize_label,LABELS

def read_labeled_csv(path, text_col='reseña', label_col='sentimiento'):
    df = pd.read_csv(path, encoding='utf-8-sig')
    missing = set([text_col,label_col])-set(df.columns)
    if missing: raise ValueError(f"Faltan columnas: {sorted(missing)}. Disponibles: {list(df.columns)}")
    df = df[[text_col,label_col]].copy()
    df = df.dropna(subset=[text_col,label_col])
    df[text_col] = df[text_col].astype(str).str.strip()
    df = df[(df[text_col].str.len()>=3) & (df[text_col].str.len()<=5000)]
    df[label_col] = df[label_col].map(normalize_label)
    conflicts = df.groupby(df[text_col].str.lower().str.replace(r'\s+',' ',regex=True))[label_col].nunique()
    if (conflicts>1).any():
        raise ValueError('Hay reseñas idénticas con etiquetas contradictorias; auditar antes de entrenar.')
    df['_group'] = df[text_col].str.lower().str.replace(r'\s+',' ',regex=True)
    df = df.drop_duplicates(subset=['_group']).reset_index(drop=True)
    if set(df[label_col])!=set(LABELS):
        raise ValueError('Faltan clases; requieren positivo, neutral y negativo.')
    return df
