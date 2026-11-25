# Traveler Insights — Dataset / Training V4 Handoff

**Fecha:** 2026-11-25 (continuación post-`aae9bcd`)  
**Rama:** `feat/dataset-training-v4`  
**Restricciones:** sin UI/`app.py`/`main`; sin merge; sin pesos/cachés en Git; V1–V3 y V4.0 locales conservados.

## Worktree y rutas absolutas

| Recurso | Ruta absoluta |
| --- | --- |
| Worktree | `C:\UTB-2026\INTELIGENCIA ARTIFICIAL\traveler_insights_github` |
| Corpus V4.0 referencia (no sobrescribir) | `...\data\processed\synthetic-tourism-v4\` |
| Artefactos V4.0 referencia | `...\artifacts\synthetic-tourism-v4\` |
| Corpus V4.1 (cierres compartidos) | `...\data\processed\synthetic-tourism-v4.1\` |
| Artefactos V4.1 | `...\artifacts\synthetic-tourism-v4.1\` |
| Eval independiente congelada | `...\src\traveler_agent\research_data\v4_independent_eval.json` |
| Checkpoint BETO base (V3, no tocar) | `...\artifacts\beto-v3-pr28-local\checkpoint` |

### Modelos locales V4.1 (gitignored)

- `...\artifacts\synthetic-tourism-v4.1\lr.joblib`
- `...\artifacts\synthetic-tourism-v4.1\svm.joblib`
- `...\artifacts\synthetic-tourism-v4.1\beto-v4-frozen128\`
- `...\artifacts\synthetic-tourism-v4.1\beto-v4-frozen256\`
- `...\artifacts\synthetic-tourism-v4.1\beto-v4-partial256\`

## 1) Atajos léxicos V4.0 → corrección V4.1

**Problema medido en V4.0:** cierres `NON_RETURN` 100 % exclusivos por etiqueta (`no volveré` solo negativo, `sí lo recomendaría` solo positivo, etc.).

**Corrección V4.1 (`VERSION=4.1.0`, seed `202611251`):**

- Cierres `SHARED_CLOSINGS` compartidos entre clases.
- Fillers de longitud sin polaridad.
- Gate `lexical_closing_audit` en `audit_records` (FAIL si exclusividad ≥ 0.92).

| Corpus | `rows_sha256` | In-gen Val Macro-F1 LR | Nota |
| --- | --- | ---: | --- |
| V4.0 ref | `fd3ea2c0…52b6a` | 1.000 | Conservado localmente |
| V4.1 | `1eff643d…aa9b7c` | 0.953 | Atajos de cierre eliminados |

Evidencia: `docs/research/v4/v4_0_reference_pointer.json`, `v4_1_training_metrics_summary.json`.

## 2) Evaluación independiente (congelada antes de predecir)

- Archivo: `src/traveler_agent/research_data/v4_independent_eval.json`
- **72** reseñas (24/clase), textos originales (sin plantillas V4)
- `records_sha256`: `d377dc71e5c7584c78eb991def3100a371c4f2dc9cabbdc604eecce318a280ac`
- `frozen: true` **antes** de cualquier predicción
- Caso diagnóstico: `ie_cartagena_fail` (hotel elogiado + transporte/excursión/reclamación)
- **No** se usó para fitting ni selección de hiperparámetros

## 3) CUDA / RTX 4050

| Ítem | Valor medido |
| --- | --- |
| GPU | NVIDIA GeForce RTX 4050 Laptop GPU |
| VRAM | 6.0 GB |
| Driver / CUDA | 560.94 / 12.6 |
| torch research | `2.14.1+cu126`, `cuda_available=True` |
| frozen128 wall | ~25 s (encode train ~13 s) |
| frozen256 wall | ~28 s |
| partial256 | 2 épocas, peak **654 MiB**, wall ~98 s — **viable** |

## 4) Comparación en eval independiente (no optimizada)

Evidencia completa: `docs/research/v4/independent_comparison_2026-11-25.json`

| Modelo | Accuracy | Macro-F1 | Neg→Pos | Latencia media (ms) |
| --- | ---: | ---: | ---: | ---: |
| demo_lr (Gradio default) | 0.500 | 0.501 | 0 | 0.38 |
| v1_lr / v1_svm | 0.458 | 0.443 / 0.447 | 11 | 0.22 / 0.68 |
| v4_0_lr / v4_0_svm | **0.639** / 0.611 | **0.631** / 0.594 | 0 | 0.23 / 0.74 |
| v4_1_lr / v4_1_svm | 0.583 / 0.528 | 0.560 / 0.492 | 0 | 0.22 / 0.70 |
| beto_v3_pr28 | 0.556 | 0.479 | 9 | 15.2 |
| beto_v4_0_frozen256 | **0.736** | **0.684** | 1 | 14.0 |
| beto_v4_1_frozen128 | 0.583 | 0.492 | 0 | 14.3 |
| beto_v4_1_frozen256 | 0.597 | 0.500 | 0 | 11.3 |
| beto_v4_1_partial256 | 0.694 | 0.620 | 1 | 11.4 |

Matrices, precision/recall por clase y reportes: ver JSON de comparación (no inventar cifras adicionales aquí).

**Lectura:** corregir atajos baja el Macro-F1 in-generator y también el transfer a esta eval; V4.0 frozen256 sigue siendo el mejor en este set congelado. No se retocó nada tras ver estas cifras.

## 5) Reseña Cartagena + artefactos Gradio

Evidencia: `docs/research/v4/cartagena_gradio_diagnosis_2026-11-25.json`

| Binding Gradio | Artefacto efectivo | Predicción `ie_cartagena_fail` |
| --- | --- | --- |
| LR | `artifacts/demo_model.joblib` (DEMO) | negativo |
| SVM | train in-memory sobre DEMO (`get_demo_model`) | **neutral** |
| BETO sin env | fallback LR (`FileNotFoundError`) | negativo (vía LR) |
| BETO + `TRAVELER_BETO_ARTIFACT=artifacts/beto-v3-pr28-local` | BETO V3 PR28 | **positivo** (fallo) |
| V4.* locales | **no cableados a la UI** | negativo (todos los V4 medidos) |

## 6) Pruebas

```
608 passed, 10 skipped, 1 xfailed
```

Log: `docs/research/v4/pytest_full_2026-11-25.txt`. Suite verde; no se eliminaron pruebas históricas.

## Archivos versionables de esta iteración

- Generador V4.1 + protocol + eval independiente
- `beto_training_v4.py` (frozen128/256 + partial256 CUDA)
- `scripts/compare_v4_independent.py`
- Evidencias bajo `docs/research/v4/`
- Tests V4 actualizados

**No versionar:** `artifacts/*`, `data/processed/*`, `*.joblib`, `*.npz`, pesos, plan BETO con textos (~MB).

## Pendiente para Codex Astra

1. Auditoría final del generador V4.1 y de la eval independiente (hashes, fugas, naturalidad).
2. Decidir si la integración UI debe apuntar a V4 (hoy sigue DEMO + BETO V3 opcional) — **sin fusionar aún**.
3. No optimizar sobre `v4_independent_eval.json`.
4. Conservar V4.0 como baseline de referencia local.
5. Commit/push solo en `feat/dataset-training-v4`; no tocar `main`.
