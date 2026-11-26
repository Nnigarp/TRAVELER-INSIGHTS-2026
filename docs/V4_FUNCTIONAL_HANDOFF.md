# Handoff — mejora funcional final (PR #36)

**Base:** `692af86` · **Commit funcional:** `9bf2ca7` · **Rama:** `feat/dataset-training-v4` (PR #36)  
**Alcance:** mejoras de semántica/control/acción/Qwen/UI/agentes. **Sin reentrenamiento.** Artefactos V4.2/V4.3 intactos.  
**Pytest `.venv-research`:** 627 passed, 1 xfailed.

## Qué cambió

| Área | Archivos | Efecto |
| --- | --- | --- |
| Semántica contextual | `semantic_context.py` (nuevo), `multiagent.py` | Polaridad exploratoria por aspecto, incidentes resueltos/no, consecuencias, evidencias rankeadas |
| Política | `policy.py` | Recomendaciones específicas con/sin Qwen; duda/ambigüedad → revisión; incidente abierto vs etiqueta |
| Control | `multiagent.py` | Razones nuevas (`AMBIGUEDAD_O_DUDA`, `INCIDENTE_ABIERTO_VS_ETIQUETA`, verify discrepancia/incierto/no_disponible). **No sobrescribe etiqueta** |
| Verify ≠ informe | `agent.py`, `multiagent.py` | `context_verifier` independiente de `reporter` / checkbox Informe Qwen. `TRAVELER_QWEN_VERIFY=0` desactiva verify |
| Qwen content | `report_content.py` | Más patrones generalizables; neutrales factuales → `informativo`; CITA evita volcar reseña completa cuando hay cláusulas; barreras anti-alucinación intactas |
| Vista agentes | `multiagent_view.py` | Clasificador efectivo/fallback/representación, evidencias, incidentes, verify pre-Control, sin etiqueta DEMO en copy |
| UI | `ui_presenter.py`, `ui_style.py`, `app.py` | Sin “Sentimiento DEMO”; panel idioma sin overflow; ASR obtenida/confirmada/corregida |

## Contratos conservados

- Plan de 4 agentes + traza de 10 eventos.
- Campos legacy del JSON.
- Código de razón histórico `SCORE_DEMO_BAJO` (texto UI sin “DEMO”).
- Enums Qwen / `validate_report` / ActionAgent revalidación.
- Artefactos y digests V4.3 emb + BETO V4.2 partial256.

## Limitaciones científicas (no ocultar)

- Aspectos/incidentes = reglas exploratorias, no ABSA ni entailment.
- Scores no calibrados; labels de train sintéticas.
- Verify Qwen puede fallar o discrepar; solo escala revisión.
- Truncación BETO 256 sigue derivando a revisión.

## Cómo auditar (Astra)

```powershell
.\.venv-research\Scripts\python.exe -m pytest -q --tb=line
# Smoke sin informe (Ollama irrelevante si ES claro)
$env:TRAVELER_QWEN_VERIFY='0'
.\.venv-research\Scripts\python.exe -c "from traveler_agent.demo_data import DEMO; from traveler_agent.model import train; from traveler_agent.agent import inspect_review; m=train([t for t,_ in DEMO],[y for _,y in DEMO]); print(inspect_review('El hotel era cómodo, pero el transporte nunca llegó y perdimos la excursión.', m)['recomendacion'][:120])"
# Verify desacoplado del informe: TRAVELER_QWEN_VERIFY=1, qwen_enabled=False, classifier con context_verification
# Ollama down + qwen_enabled=True → generacion_qwen.estado=ERROR, clasificación OK
# Casos de aceptación (chat previo, 30 reseñas): NO entrenar con ellos
```

## Pruebas añadidas

- `tests/test_semantic_context.py`
- Ajustes de copy en tests UI/Qwen que esperaban “Sentimiento DEMO”

## Checklist auditoría definitiva

1. [ ] Pytest verde en `.venv-research`
2. [ ] LR/SVM V4.3 + BETO V4.2 en Gradio (perfil V4)
3. [ ] Informe Qwen ON/OFF e Ollama parado
4. [ ] Verify activo con informe OFF (desacople)
5. [ ] Audio: estados obtenida → confirmada → corregida
6. [ ] Panel idioma sin barra horizontal en ambos temas
7. [ ] Texto largo → `TEXTO_TRUNCADO_POR_ENCODER`
8. [ ] Lote ES sin Qwen
9. [ ] No merge hasta OK de Astra
