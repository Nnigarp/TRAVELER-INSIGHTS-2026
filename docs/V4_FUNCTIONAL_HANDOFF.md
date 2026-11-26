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

## Aceptación técnica posterior a 4fdb8e4

Los resultados por entrada, etiqueta supervisada, versión, verificación e informe están en [el registro medido](research/v4/functional_acceptance.json). Se usaron las frases disponibles en las pruebas funcionales y en la batería Qwen. No constituyen una muestra de validación científica.

* En el incidente del transporte, las tres variantes devolvieron negativo. El informe identifica la ausencia del traslado y la pérdida de la excursión.
* En la llave averiada y resuelta, LR y SVM devolvieron positivo. BETO devolvió negativo. Su informe reconoció el fallo y su resolución. La etiqueta se mostró junto a la solicitud de revisión.
* En la descripción de duración, las tres variantes devolvieron positivo. Control derivó la contradicción factual a revisión incluso sin Ollama. Esta regla exige cláusulas descriptivas reconocidas y cobertura completa.
* Las recomendaciones con contexto se apoyan en los aspectos desfavorables, mixtos o vinculados al incidente abierto. Se corrigieron inferencias sobre devoluciones negadas, cobros, transporte y resolución. Ante sintaxis desconocida el informe puede limitarse a una cita.
* El verificador opcional actúa antes de Control. El informe generativo se solicita por separado después de Control y ActionAgent revalida su contenido. En lotes los dos servicios comparten presupuesto. Con varios incidentes reconocidos, el esquema de generación vincula la propuesta al primero en orden textual, preservando el resumen de todos.
* Simulando un endpoint local desconectado, las nueve combinaciones de clasificador y reseña devolvieron resultado y diez eventos principales. El informe solicitado registró error. El español reconocido o indicado manualmente, un lote y la traducción aportada por el usuario funcionaron. La traducción automática requiere Ollama. Para presentar en modo autónomo usar las variables siguientes y dejar desmarcado Informe Qwen.
* Chromium verificó la interfaz, los lotes y exportaciones. En móvil a 390 px, el panel abierto de idioma midió 362 px visibles y 362 px de contenido en cada tema. La pestaña Agentes mostró BETO V4.2 partial256, diez eventos y el motivo de revisión factual. Con un WAV sintético se verificaron las rutas confirmada y corregida. La prueba de navegador conservó el hash de audio tras retranscripción y corrección, tradujo inglés, francés y chino y exigió revisión en el caso italiano.
* Los cuatro cambios previos a pruebas existentes afectaban textos de interfaz. Se restauró la comprobación exacta de la etiqueta de sentimiento en audio. Las nuevas pruebas cubren negación de resolución, aspecto favorable frente a incidente, afirmaciones sin evidencia y descripción factual. Artefactos y métricas deduplicadas se conservaron.

### Ejecución en el Lenovo

```powershell
Set-Location 'C:\UTB-2026\INTELIGENCIA ARTIFICIAL\traveler_insights_github'
.\scripts\iniciar_demo.ps1
```

Para una sesión sin Ollama, antes del comando de inicio:

```powershell
$env:TRAVELER_QWEN_VERIFY = '0'
$env:TRAVELER_LANGUAGE_QWEN = '0'
```

Para la regresión y la batería real con Ollama activo, desde una nueva consola en el repositorio:

```powershell
$env:PYTHONUTF8 = '1'
$env:TRAVELER_BETO_ARTIFACT = 'artifacts/synthetic-tourism-v4.2/beto-v4-partial256'
.\.venv-research\Scripts\python.exe -m pytest tests -q --basetemp=.pytest_cache/aceptacion-final
.\.venv-research\Scripts\python.exe -m scripts.check_qwen_report_quality --actual --output artifacts/qwen-aceptacion-final.json
```

Regresión final en .venv-research: 631 aprobadas, 1 xfail y 27 advertencias en 130,37 s. Batería real BETO + Qwen: 12/12. Chromium: 80/80 comprobaciones, además de los recorridos de voz, traducción y Agentes indicados arriba.
