# Incremento 09 — Benchmark privado del corpus turístico

Estado: implementación ejecutable con tests sintéticos. NO hay datos turísticos reales ni resultados científicos verificados.

Dependencia: este PR deriva de la rama del corpus, PR #7. No fusionar antes de incorporar y validar PR #7.

## Componentes

- Auditoría de procedencia: valida fuentes, revisión de permisos declarados, adjudicaciones, esquemas y SHA-256 de archivos antes de entrenar.
- Fuga: prohíbe repetir review_id, group_id o texto normalizado idéntico en diferentes particiones.
- Tres modelos: DummyClassifier (mayoría), TF-IDF + regresión logística y TF-IDF + LinearSVC. Para SVM, la calibración sigmoid usa folds estratificados por componentes de grupos y duplicados, exclusivamente dentro de train, con vectorización TF-IDF dentro de cada fold.
- Búsqueda de hiperparámetros: C = 0.5 y 2.0 para LR y SVM; únicamente Macro-F1 de validation decide qué candidato conservar.
- Test: una única evaluación por modelo seleccionado, sin reentrenar ni seleccionar ajustes con test.
- Métricas: accuracy, Macro-F1, precision, recall y F1 por clase, matriz de confusión y Macro-F1 por idioma.
- Metadatos: semilla, versiones Python/scikit-learn/numpy, hashes de datasets, hash de modelos, parámetros, limitaciones y advertencias.
- Exportación: modelos joblib e informes JSON locales, nunca en Git.

## Uso reproducible en Windows PowerShell

Primero deben existir fuentes autorizadas, reseñas adjudicadas y particiones preparadas con PR #7. Los ejemplos sintéticos sólo comprueban software.

~~~powershell
git fetch origin
git switch --track origin/feat/corpus-benchmark
.\.venv\Scripts\python.exe -m pip install -e ".[test]"
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m traveler_agent.scientific_benchmark --corpus-dir data/private/tourism-corpus-v1 --split-dir data/processed/tourism-corpus-v1 --output-dir artifacts/corpus-benchmark-v1 --seed 42
~~~

Si son ejemplos sintéticos, añadir --demo-only. No reutilizar un directorio de salida con archivos: se rechaza la sobrescritura. Las salidas son dummy.joblib, lr.joblib, svm.joblib, experiment_report.json y model_manifest.json.

Para integrar de forma opcional un artefacto propio en el agente:

~~~python
from traveler_agent.trusted_models import load_own_benchmark_model
from traveler_agent.agent import inspect_review
model = load_own_benchmark_model("artifacts/corpus-benchmark-v1", "lr", trust_local_artifacts=True)
resultado = inspect_review("La atención fue buena, pero esperamos demasiado.", model)
~~~

La comprobación SHA detecta alteraciones pero no autenticidad ante adversarios capaces de cambiar modelo y manifiesto. Joblib puede ejecutar código al cargar; **no aceptar modelos de terceros**. Gradio sigue usando DEMO salvo integración posterior y explícita.

## Límites científicos

- Las etiquetas son las de adjudicación humana; no se crean etiquetas automáticamente.
- Las particiones están aisladas por grupos declarados y duplicados exactos normalizados, no por similitud semántica o autores ocultos.
- El manifiesto indica RESEARCH_CANDIDATE_NOT_INDEPENDENTLY_AUDITED aun cuando el usuario omite --demo-only; ese valor no certifica la fuente ni la calidad de las etiquetas.
- No hay intervalos de confianza, validación entre fuentes, análisis de error por negación/emoji, ABSA supervisado o fine-tuning XLM-R en este incremento.
- Calibración interna no significa fiabilidad probabilística demostrada.
- Las métricas calculadas sobre fixtures sintéticos no deben presentarse como resultados turísticos.
- Se requiere revisión de PII, permisos, casi duplicados y anotaciones antes de una evaluación científica externa.

## Siguientes incrementos

Implementar anotación visual de aspectos con spans y offsets, acuerdo entre anotadores, revisión de casi duplicados, benchmark externo y posterior ajuste fino de un modelo contextual multilingüe si hay datos suficientes.
