# Traveler Insights — Dataset / Training V4 Handoff

**Fecha:** 2026-11-25 (V4.3 desde `b06be80`)  
**Rama:** `feat/dataset-training-v4`  
**Restricciones:** sin UI/`main`; sin merge; sin pesos en Git; V3/V4.0–V4.2 conservados.

## Worktree

`C:\UTB-2026\INTELIGENCIA ARTIFICIAL\traveler_insights_github`

| Recurso | Ruta |
| --- | --- |
| V4.0–V4.2 refs | `data/processed/synthetic-tourism-v4{,.1,.2}/` · `artifacts/…` |
| V4.3 | `data/processed/synthetic-tourism-v4.3/` · `artifacts/synthetic-tourism-v4.3/` |
| Eval v1 (72, observada) | `src/traveler_agent/research_data/v4_independent_eval.json` |
| Eval v2 (300, observada) | `src/traveler_agent/research_data/v4_independent_eval_v2.json` |
| **Eval v3 (300, congelada)** | `src/traveler_agent/research_data/v4_independent_eval_v3.json` |

### Modelos locales (gitignored)

| Artefacto | Notas |
| --- | --- |
| `artifacts/synthetic-tourism-v4.2/beto-v4-partial256/` | **Conservado** — 7/7 en IE-v2 |
| `artifacts/synthetic-tourism-v4.3/lr.joblib` · `svm.joblib` | TF-IDF + RandomizedSearchCV + calib |
| `artifacts/synthetic-tourism-v4.3/beto_emb_{lr,svm}.joblib` | BETO congelado mean-pool + LR/SVM |
| `artifacts/synthetic-tourism-v4.3/ensemble_softvote.joblib` | Soft-vote TF-IDF LR ⊕ emb SVM (pesos en val) |

## V4.3 — cambios

- Léxico PRAISE/FAILURE ampliado (sinónimos / naturalidad); seed `202611253`.
- Corpus `rows_sha256=2f80aa25798d56823192b767d328d2ea181fa04cb7d4ab798fb7202cf9db2b70`.
- `classical_v43_training`: TF-IDF words+chars, n-gram grids, `StratifiedGroupKFold`, `RandomizedSearchCV`, `CalibratedClassifierCV`.
- Variantes embedding BETO congelado (checkpoint `beto-v3-pr28-local`) + LR/SVM calibrados — **separadas** de TF-IDF.
- Ensemble soft-vote con peso elegido **solo** en validation.
- BETO V4.2 partial256 **no reentrenado**.

### Tiempos de entrenamiento (medidos)

| Modelo | Wall (s) | Val Macro-F1 | Test Macro-F1 |
| --- | ---: | ---: | ---: |
| TF-IDF LR | 169.4 | 0.957 | 0.967 |
| TF-IDF SVM | 88.9 | 0.964 | 0.968 |
| BETO-emb LR | 32.1 | 0.874 | 0.883 |
| BETO-emb SVM | 39.7 | 0.918 | 0.931 |
| Ensemble soft-vote | 11.5 | 0.965 | 0.963 |

HP TF-IDF LR (CV train): `C=4`, words `(1,2)`/16k, chars `(4,6)`/20k.  
HP TF-IDF SVM: `C=2`, words `(1,3)`/8k, chars `(3,4)`/12k.  
Emb: `C=8` ambos. Ensemble: `w_tfidf=0.7`.

## Eval independiente v3 (congelada **antes** de puntuar)

- **n=300** (100/clase) · banco **tercero** (no train / no IE-v1/v2)
- `records_sha256=6422a188f79b76b93584bb62e481771f13c2f5e08e3215bb86f57993764b129a`
- IE-v1/v2 **no** usados para tuning.

## Resultados en IE-v3 (honestos; no se fuerzan metas)

Evidencia: `docs/research/v4/independent_v3_comparison_2026-11-25.json`

| Modelo | Acc | Macro-F1 | Macro-P | Macro-R | Rec.neg | Neg→Pos | Lat ms | Goals |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| demo_lr | 0.347 | 0.195 | 0.281 | 0.347 | 0.04 | 0.00 | ~0.3 | 1/7 |
| v4_2_lr / svm | 0.467 / 0.437 | 0.400 / 0.358 | — | — | 0.20 / 0.13 | 0.07 / 0 | ~0.3–1.0 | 0–1/7 |
| v4_3_tfidf_lr | 0.573 | 0.525 | 0.713 | 0.573 | 0.52 | **0.00** | ~0.9 | 1/7 |
| v4_3_tfidf_svm | 0.527 | 0.463 | 0.608 | 0.527 | 0.45 | 0.05 | ~0.9 | 1/7 |
| v4_3_beto_emb_lr | 0.663 | 0.645 | 0.751 | 0.663 | 0.81 | **0.00** | ~6.3 | 1/7 |
| **v4_3_beto_emb_svm** | **0.697** | **0.673** | **0.774** | **0.697** | **0.89** | **0.00** | ~5.8 | **1/7** |
| v4_3_ensemble | 0.583 | 0.540 | 0.721 | 0.583 | 0.53 | 0.00 | ~6.5 | 1/7 |
| beto_v4_2_partial256 | 0.843 | 0.842 | 0.865 | 0.843 | **1.00** | **0.00** | ~5.7 | **2/7** |

**Ningún clásico V4.3 alcanza Acc/Macro-F1 ≥90% en IE-v3.** No se fabrican métricas.

### Lectura causal

- TF-IDF sigue acoplado al banco léxico del train; IE-v3 usa fraseología distinta → transferencia limitada (aunque mejor que V4.2 refs en este set).
- Embeddings BETO congelados + SVM mejoran recall negativo (0.89) y Acc (~0.70) sin tocar partial256.
- Soft-vote con peso alto TF-IDF (óptimo en val in-generator) **empeora** IE-v3 vs emb-SVM solo → confirma desajuste dominio.
- BETO partial256 V4.2 sigue siendo el mejor candidato global en IE-v3 (2/7; 7/7 en IE-v2). Caída IE-v2→v3 es esperable (eval nueva, no tuned).

Matrices de confusión, errores por fenómeno y muestras: JSON de comparación.

## Pytest (`.venv-research`)

- Antes (`.venv` base): 608 passed, **10 skipped**, 1 xfailed.
- Ahora (`.venv-research`): **617+ passed**, **0 skipped** por torch/matplotlib (10 omitidas recuperadas), 1 xfailed.
- Test de cierres compartidos actualizado a `VERSION >= 4.2.0`.

## Pendiente Codex Astra

1. Clásicos TF-IDF **no** cumplen metas de transferencia; valorar encoder FT o datos humanos.
2. Conservar `beto-v4-partial256` V4.2 como candidato de integración (aún **no** cablear Gradio).
3. No reutilizar IE-v1/v2/v3 para tuning adicional.
4. No fusionar a `main` hasta auditoría Astra.
