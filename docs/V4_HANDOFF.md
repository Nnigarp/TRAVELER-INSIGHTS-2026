# Traveler Insights — Dataset / Training V4 Handoff

**Fecha:** 2026-11-25 (V4.2 desde `2fa3a545`)  
**Rama:** `feat/dataset-training-v4`  
**Restricciones:** sin UI/`main`; sin merge; sin pesos en Git; V3/V4.0/V4.1 conservados.

## Worktree

`C:\UTB-2026\INTELIGENCIA ARTIFICIAL\traveler_insights_github`

| Recurso | Ruta |
| --- | --- |
| V4.0 ref | `data/processed/synthetic-tourism-v4/` · `artifacts/synthetic-tourism-v4/` |
| V4.1 ref | `data/processed/synthetic-tourism-v4.1/` · `artifacts/synthetic-tourism-v4.1/` |
| V4.2 | `data/processed/synthetic-tourism-v4.2/` · `artifacts/synthetic-tourism-v4.2/` |
| Eval v1 (72, observada) | `src/traveler_agent/research_data/v4_independent_eval.json` |
| Eval v2 (300, congelada) | `src/traveler_agent/research_data/v4_independent_eval_v2.json` |

### Modelos V4.2 locales (gitignored)

- `artifacts/synthetic-tourism-v4.2/lr.joblib` · `svm.joblib`
- `artifacts/synthetic-tourism-v4.2/beto-v4-frozen256/`
- `artifacts/synthetic-tourism-v4.2/beto-v4-partial256/` ← **candidato principal**

## V4.2 — cambios

- Neutrales factuales ampliados + plantillas `factual_neutral`, `resolved_incident`, `contradiction_pos/neg`, `sarcasm`.
- Cierres compartidos (sin atajos de etiqueta); gate `lexical_closing_audit`.
- Seed `202611252` · `rows_sha256=a497f9e7…b4f1`.
- HP clásicos (`C∈{0.25,1,4}`) **solo** sobre validation.

## Eval independiente v2 (congelada antes de puntuar)

- **n=300** (100/clase) · bancos **alternos** (no plantillas train)
- `records_sha256=7fc9cf2130c4a1ffd7dedf5f6a8296bcbc02f88f88e267037af488f5997a036b`
- Truncación: 103/300 >128 tokens; **0 >256**; max 182 WordPiece

## CUDA / entrenamiento V4.2

| Modo | Val Macro-F1 | Test Macro-F1 | Peak VRAM | Wall |
| --- | ---: | ---: | ---: | ---: |
| LR (C sel. val) | 0.972 | 0.961 | — | ~30 s |
| SVM (C sel. val) | 0.978 | 0.964 | — | ~30 s |
| BETO frozen256 | 0.810 | 0.790 | — | ~36 s |
| BETO **partial256** (3 ep, last-2 blocks) | **0.982** | **0.971** | **655 MiB** | ~144 s |

Full FT de todo el encoder: **no ejecutado** (partial ya viable en 6 GB y cumple metas en eval v2; full aumentaría riesgo VRAM sin evidencia previa de necesidad).

## Resultados en eval v2 (no optimizada sobre este set)

Evidencia: `docs/research/v4/independent_v2_comparison_2026-11-25.json`

| Modelo | Acc | Macro-F1 | Macro-P | Macro-R | Rec.neg | Neg→Pos | Goals met |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| demo_lr | 0.417 | 0.315 | — | — | 0.30 | 0 | 1/7 |
| v4_0_lr | 0.733 | 0.728 | — | — | 0.74 | 25 | 0/7 |
| v4_1_lr | 0.677 | 0.584 | — | — | **0.95** | 0 | 2/7 |
| v4_2_lr / svm | 0.53 / 0.52 | 0.46 / 0.43 | — | — | 0.48 / 0.52 | 0 | 1/7 |
| beto_v3_pr28 | 0.603 | 0.591 | — | — | 0.40 | **60** | 0/7 |
| beto_v4_2_frozen256 | 0.590 | 0.570 | — | — | **0.98** | 0 | 2/7 |
| **beto_v4_2_partial256** | **0.910** | **0.909** | **≥0.90** | **≥0.90** | **1.00** | **0** | **7/7** |
| beto_v4_1_partial256 | 0.880 | 0.883 | — | — | 1.00 | 0 | 4/7 |

**Causas honestas (clásicos V4.2 débiles en IE-v2):** TF-IDF depende de bancos léxicos del train; la eval v2 usa fraseología alterna → baja transferencia. BETO partial adapta representación y generaliza mejor a neutrales/contrastes de la eval.

Matrices, errores por fenómeno, latencias: JSON de comparación (no inventar cifras).

## 10 pruebas omitidas (investigadas)

Con `.venv` base (sin torch/matplotlib):

| Skip | Motivo |
| --- | --- |
| `test_beto_training_v2.py` ×2 | `importorskip(torch)` / entorno research |
| `test_beto_training_v3_model.py` ×6 | `importorskip(torch)` research CPU |
| `test_beto_training_v3_analysis.py` ×1 | `importorskip(matplotlib)` |
| `test_contextual_supervised.py` ×1 | `importorskip(torch)` research opcional |

**No son fallos funcionales del producto Gradio.** Suite: **608 passed, 10 skipped, 1 xfailed**.

## Pendiente Codex Astra

1. Revisión de `beto-v4-partial256` V4.2 como candidato de integración (aún **no** cablear Gradio).
2. Valorar si clásicos necesitan encoder features o más diversidad léxica cross-bank.
3. No reutilizar eval v1/v2 para tuning.
4. No fusionar a `main` hasta auditoría Astra.
