# Traveler Insights — Dataset / Training V4 Handoff

**Fecha:** 2026-11-25  
**Rama:** `feat/dataset-training-v4` (base `bf97231`, PR #35 ya en main)  
**Agente:** continuación tras límite de uso de Codex  
**Restricciones respetadas:** sin cambios de UI/`app.py`/`main`; sin merge de PR; artefactos V1–V3 conservados; sin pesos/cachés en Git.

## Worktree y rutas absolutas locales

| Recurso | Ruta absoluta |
| --- | --- |
| Worktree / raíz del repo | `C:\UTB-2026\INTELIGENCIA ARTIFICIAL\traveler_insights_github` |
| Rama experimental | `feat/dataset-training-v4` |
| Corpus V4 (gitignored) | `C:\UTB-2026\INTELIGENCIA ARTIFICIAL\traveler_insights_github\data\processed\synthetic-tourism-v4\` |
| Artefactos V4 (gitignored) | `C:\UTB-2026\INTELIGENCIA ARTIFICIAL\traveler_insights_github\artifacts\synthetic-tourism-v4\` |
| LR V4 | `...\artifacts\synthetic-tourism-v4\lr.joblib` (+ `lr_report.json`, `lr.meta.json`) |
| SVM V4 | `...\artifacts\synthetic-tourism-v4\svm.joblib` (+ `svm_report.json`, `svm.meta.json`) |
| BETO V4 frozen256 | `...\artifacts\synthetic-tourism-v4\beto-v4-frozen256\` |
| Informe BETO V4 | `...\artifacts\synthetic-tourism-v4\beto_v4_frozen256_report.json` |
| Caché features TRAIN (no subir) | `...\artifacts\synthetic-tourism-v4\beto_v4_train_features_256.npz` |
| Plan BETO completo con textos (no subir; ~4.8 MB) | `...\artifacts\synthetic-tourism-v4\beto_v4_training_plan.json` |
| Checkpoint base reutilizado (V3, no tocar) | `C:\UTB-2026\INTELIGENCIA ARTIFICIAL\traveler_insights_github\artifacts\beto-v3-pr28-local\checkpoint` |

**No versionar:** `artifacts/*`, `data/processed/*`, `*.joblib`, `*.npz`, pesos BETO, ni la carpeta anidada `traveler-insights-2026/`.

## Estado al retomar

| Ítem | Estado |
| --- | --- |
| Rama experimental | Reutilizada: `feat/dataset-training-v4` en `bf97231` |
| Cambios previos de Codex | Solo la rama vacía; **ningún archivo V4** dejado |
| Workspace Cursor | Trabajo en el worktree padre anterior; no usar el clone anidado `traveler-insights-2026/` |

## Auditoría histórica (medida / documentada, no inventada)

### Datasets previos

| Recurso | Filas | Notas |
| --- | ---: | --- |
| Synthetic tourism V1 | 336 | Template composicional; 56 familias; balance 112/clase; `data/processed/synthetic-tourism-v1/` |
| Synthetic diversity V2/V3 | catálogos en `research_data/` | Familias de 3 intenciones; labels = intención de autor |
| BETO V3 piloto | 564 generadas → 562 admitidas | TRAIN 300 + tune/cal/policy/assessment 60 c/u + ambiguos; ver `docs/BETO_V3_PILOTO_Y_ESCALADO_2026-11-10.md` |

### Entrenamientos previos (assessment V3, cohorte sintética)

| Modelo | Macro-F1 | Neg→Pos /20 | Fuente |
| --- | ---: | ---: | --- |
| TF-IDF + LR | 0.9158 | 2 | docs BETO V3 |
| TF-IDF + SVM | 0.9158 | 2 | docs BETO V3 |
| BETO V3 full · seed 42 | 1.0000 | 0 | docs BETO V3 |

V3 TRAIN: **99/300** textos >128 tokens WordPiece; ninguno >256. Artefacto recuperado PR28 usa `max_length=256`, schema `traveler-beto-supervised-v2`.

## Hallazgo de truncamiento (medido 2026-11-25)

Evidencia: `docs/research/v4/truncation_audit_2026-11-25.json`

Caso corto (~122 tokens BETO, elogio hotel + fallo transporte/excursión/reclamación):

- Cabe entero en 128.
- BETO V3 recuperado → **negativo**.
- LR/SVM V1 históricos → **neutral** (no positivo).

Caso largo (prefijo de elogios ×6 + mismo fallo, ~206 tokens):

| max_length | Predicción BETO V3 | ¿Conserva transporte/reclamación? |
| ---: | --- | --- |
| 128 | **positivo** | No (corte a mitad del elogio) |
| 256 | **negativo** | Sí |

**Conclusión:** el truncamiento a 128 **sí puede** volcar elogio-inicial→fallo-final a positivo cuando el prefijo positivo es largo. No es la única causa posible de errores en reseñas cortas mixtas.

## Dataset V4 construido

| Campo | Valor |
| --- | --- |
| Generador | `src/traveler_agent/synthetic_tourism_v4.py` |
| Protocolo | `src/traveler_agent/research_data/synthetic_v4_protocol.json` |
| Semilla | `20261125` |
| Filas | **5001** (= 1667 familias × 3; tamaño ternario más cercano a 5000) |
| Balance | 1667 negativo / 1667 neutral / 1667 positivo |
| Salida | `data/processed/synthetic-tourism-v4/` (gitignored) |
| `rows_sha256` | `fd3ea2c01e980b89929968c5773f772623928c583f351942f839b02aab552b6a` |

### Particiones (por familia, sin fuga)

| Rol | Familias | Filas |
| --- | ---: | ---: |
| train | 1120 | 3360 |
| validation | 240 | 720 |
| test | 240 | 720 |
| hard_holdout | 67 | 201 |

`hard_holdout`: plantilla `praise_then_failure` (elogio → fallo de transporte/excursión → reclamación sin respuesta → no volver). **No entra en fitting.**

Controles: hashes históricos V1/V2/V3/contextual excluidos; `text_sha256` disjuntos entre roles; pantalla near-duplicate Jaccard bloqueada (umbral 0.84).

### Regenerar

```bat
python -m traveler_agent.synthetic_tourism_v4 --out data/processed/synthetic-tourism-v4
```

## Entrenamiento V4

### LR / SVM (completado)

```bat
python -m traveler_agent.dataset_v4_training train-classical --model-kind lr
python -m traveler_agent.dataset_v4_training train-classical --model-kind svm
```

Artefactos (gitignored): `artifacts/synthetic-tourism-v4/{lr,svm}.joblib` + `*_report.json`  
Resumen versionado: `docs/research/v4/classical_v4_metrics_summary.json`

| Modelo | Val Macro-F1 | Test Macro-F1 | Hard Macro-F1 | Hard Neg→Pos |
| --- | ---: | ---: | ---: | ---: |
| LR | 1.0 | 1.0 | 1.0 | 0 |
| SVM | 1.0 | 1.0 | 1.0 | 0 |

**Limitación científica:** Macro-F1=1.0 **in-generator** no implica validez poblacional. Persisten cierres léxicos por clase y plantillas compartidas (atajo conocido del generador composicional). Sobre el caso de fallo usuario, LR/SVM V4 predicen **negativo** (short y long).

### BETO V4 (frozen256 completado)

```bat
python -m traveler_agent.dataset_v4_training prepare-beto-plan
.\.venv-research\Scripts\python.exe -m traveler_agent.beto_training_v4 --max-length 256 --checkpoint artifacts/beto-v3-pr28-local/checkpoint
```

- Plan: `artifacts/synthetic-tourism-v4/beto_v4_training_plan.json`
- Índice docs: `docs/research/v4/beto_v4_training_index.json`
- Informe: `artifacts/synthetic-tourism-v4/beto_v4_frozen256_report.json`
- Resumen docs: `docs/research/v4/beto_v4_frozen256_metrics_summary.json`
- Artefacto: `artifacts/synthetic-tourism-v4/beto-v4-frozen256/`
- Caché de features TRAIN: `artifacts/synthetic-tourism-v4/beto_v4_train_features_256.npz` (reutilizable)

| Split | Macro-F1 | Neg→Pos |
| --- | ---: | ---: |
| validation | 0.9945 | ver informe |
| test | 0.9972 | 1 |
| hard_holdout | 1.0000 | 0 |

Encoder congelado + cabeza logística sklearn; `max_length=256`. Ablación 128 y `partial256` siguen pendientes.

Comprobación post-entrenamiento del caso usuario (medida, no inventada):

| Texto | Inferencia max_length | Predicción BETO V4 frozen |
| --- | ---: | --- |
| failure short | 128 / 256 | negativo / negativo |
| failure long (prefijo×6) | 128 | **positivo** (sigue truncando) |
| failure long (prefijo×6) | 256 | **negativo** |

LR/SVM V4: negativo en short y long (sin truncamiento Transformer).

## Archivos tocados en esta sesión

**Nuevos**

- `src/traveler_agent/synthetic_tourism_v4.py`
- `src/traveler_agent/dataset_v4_training.py`
- `src/traveler_agent/beto_training_v4.py`
- `src/traveler_agent/research_data/synthetic_v4_protocol.json`
- `scripts/audit_v4_truncation.py`
- `tests/test_synthetic_tourism_v4.py`
- `docs/V4_HANDOFF.md`
- `docs/research/v4/truncation_audit_2026-11-25.json`
- `docs/research/v4/beto_v4_training_index.json`
- `docs/research/v4/classical_v4_metrics_summary.json`
- `docs/research/v4/beto_v4_frozen256_metrics_summary.json`

**Generados locales (gitignored)**

- `data/processed/synthetic-tourism-v4/`
- `artifacts/synthetic-tourism-v4/`

**No modificados:** `app.py`, UI, modelos/datasets/métricas V1–V3 históricos.

## Pruebas

```bat
python -m pytest tests/test_synthetic_tourism_v4.py -q
```

6 passed (sesión 2026-11-25). Suite completa del repo **no** re-ejecutada aquí tras el handoff.

## Decisiones

1. **5001 filas** en lugar de 5000 exactas para preservar familias de 3 etiquetas.
2. Generador **rule-based local** (sin LLM API), alineado con V1/V3 del repo.
3. Anclas escénicas por familia **sin codificar la etiqueta** (se corrigió una fuga inicial `NEG/POS/NEU`).
4. BETO V4 prioriza **256 tokens**; 128 se mantiene como candidato de ablación.
5. Hard set independiente; LR/SVM y BETO frozen256 ya entrenados localmente (artefactos fuera de Git).

## Pendiente para Codex / siguiente agente

1. Ablación BETO V4 `frozen128` (comparar truncamiento vs 256) reutilizando el mismo plan/corpus.
2. Opcional: adaptación `partial256` / últimos bloques bajo presupuesto VRAM 6 GB (torch CUDA si se habilita; el entorno actual usó torch CPU).
3. Auditoría de atajos léxicos OOF (estilo V3 `beto-v3-train-lexical.json`) sobre TRAIN V4 **sin** retocar assessment.
4. Diversificar cierres/neutralidad para reducir Macro-F1 trivial in-generator antes de congelar un assessment “oficial” V4.
5. No fusionar PR; no tocar UI; no sobrescribir `artifacts/beto-v3-*` ni V1; no subir pesos/cachés.

## Comandos rápidos de verificación

```bat
git status
git branch --show-current
python -m traveler_agent.synthetic_tourism_v4 --help
python -m traveler_agent.dataset_v4_training --help
python -m traveler_agent.beto_training_v4 --help
.\.venv-research\Scripts\python.exe scripts\audit_v4_truncation.py
```
