# Experimento sellado fuera del generador — 5 de noviembre de 2026

**Resultado real de GitHub Actions:** PR de investigación con base en main `dfac831aa6050a7242234e95122d97e43fa7a877`.
**Naturaleza:** evaluación **SINTÉTICA INTERNA**, NO datos humanos, NO precisión científica sobre turistas reales, NO conjunto de anotaciones humanas gold.
**Evidencia:** https://github.com/Nnigarp/TRAVELER-INSIGHTS-2026/actions/runs/42009698229
**Artefacto de desafío:** https://github.com/Nnigarp/TRAVELER-INSIGHTS-2026/actions/runs/42009698229/artifacts/13405289140
**Baseline congelado:** workflow 42006977003, semilla 42, 336 reseñas, SHA-256 CSV original `315012f9d553c51906e49bac617ef7ed9107ee2fc3cfd4dfd5a1368c7a4be059`.

## Pregunta

¿El clasificador de tres clases conserva su rendimiento cuando cambia el repertorio de expresiones del generador sintético original?

### Diseño de desafío separado

- Inventario nuevo y explícito de **60 situaciones turísticas** redactadas independientemente de los textos fuente del generador inicial: 10 por dominio, seis dominios (hotel, restaurante, playa, museo, transporte y experiencia).
- Cada situación posee tres descripciones de intención negativa, neutral y positiva, **sin adjudicación humana independiente**: 180 núcleos semánticos.
- Siete encuadres narrativos por núcleo, con introducción coloquial, frase interrogativa, variación narrativa, alguna negación/contraste y faltas ortográficas sencillas: 1.260 registros, **420 por clase**. Estas siete reformulaciones comparten una frase central por núcleo; no se deben contar como 1.260 unidades independientes de evidencia.
- Registro explícito de semilla 913, fecha, versión, generador NO LLM, dominio, identificador del núcleo, variante, etiqueta de intención, hash de texto, región, carácter sintético.
- El desafío tiene un único test, reservado tras la primera ejecución. **No se ajustaron plantillas ni hiperparámetros en función de sus errores.**
- Control cruzado: huellas SHA256 exactas y cribado de similitud de caracteres TF-IDF 3–5 grams contra los 336 ejemplos originales, umbral de similitud 0.96. **0 coincidencias exactas, 0 candidatos >=0.96**, máximo 0.2409 bajo esa métrica. No excluye semejanza semántica de otro tipo.
- El diseño **no incluye todavía un conjunto riguroso de sarcasmo/ironía verdaderamente contextual, opiniones mixtas equilibradas y ambigüedad adjudicada**. Algunas variaciones incorporan formas de negación o contraste superficial; no afirmar cobertura exhaustiva. El español regional se representa mediante marcadores coloquiales limitados, no por textos humanos colombianos/peruanos.

## Modelo congelado y procedimiento exacto

Se descargan desde el workflow original **únicamente** el CSV original, su manifiesto y su reporte de métricas, fijados por run-id. Se verifica el SHA original; no se modifica ningún registro. Se reconstruye la partición original por familias con seed=42. Se leen los **parámetros ganadores originales** del reporte, sin búsqueda de hiperparámetros nueva. El ajuste LR/SVM y Dummy se reproduce una vez sobre el **TRAIN ORIGINAL de 204 ejemplos**; SVM usa calibración interna de grupos, sin desafío. Se verifica que el Macro-F1 de la validación original sea exactamente igual al histórico.

**Limitación operacional crítica:** el workflow original **no archivó los modelos joblib**. Por ello NO se cargan pesos serializados del experimento original: se reconstruyen con datos/parámetros originales congelados. El desafío no participa en fitting ni selección. No se reevalúa el test original; se utiliza únicamente la métrica guardada en el informe histórico.

Se evalúa cada modelo UNA VEZ sobre el desafío. Reportes completos contienen Macro-F1, accuracy, precision, recall y F1 por clase, matriz de confusión orden [negativo, neutral, positivo], errores identificados sin texto, recuentos por fenómeno y dominio, hashes y versiones reales de dependencias.

## Resultados — 1.260 textos de desafío sintético

| Modelo | Macro-F1 test sintético inicial | Macro-F1 desafío | Accuracy desafío | Errores |
| --- | ---: | ---: | ---: | ---: |
| Dummy | 0.1667 | 0.1667 | 0.3333 | 840 |
| LR | 1.0000 | 0.392560 | 0.405556 | 749 |
| SVM | 1.0000 | 0.388164 | 0.404762 | 750 |

F1 por clase en **desafío**:

| Modelo | F1 negativo | F1 neutral | F1 positivo |
| --- | ---: | ---: | ---: |
| Dummy | 0.5000 | 0.0000 | 0.0000 |
| LR | 0.307927 | 0.406250 | 0.463504 |
| SVM | 0.289389 | 0.407595 | 0.467509 |

Matrices de confusión del desafío, filas etiqueta prevista / columnas predicción, orden [negativo, neutral, positivo]:

- Dummy: `[[420,0,0],[420,0,0],[420,0,0]]`.
- LR: `[[101,87,232],[74,156,190],[61,105,254]]`.
- SVM: `[[90,94,236],[66,161,193],[46,115,259]]`.

**Interpretación:** existe una caída de Macro-F1 de ~1.0 a ~0.39 para los dos modelos, con sesgo hacia predecir positivo. El primer 100 % probablemente reflejaba reutilización de pistas léxicas en el corpus inicial. No se puede inferir qué proporción de la caída proviene de distribución, lenguaje, cambios de longitud o ambigüedad de etiquetas sintéticas, porque no hay adjudicación humana independiente.

**No comparar estas cifras como un mismo estimador de precisión turística:** test inicial y desafío son distribuciones diferentes, ambos sintéticos. Los errores se usaron **solo** para el reporte; no para ajustar el desafío después de abrirlo.

## Ejecutar en Windows (Python 3.12)

```powershell
git fetch origin
git switch --track origin/feat/out-of-generator-challenge
.\.venv\Scripts\python.exe -m pip install -e ".[test]"
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m traveler_agent.out_of_generator_challenge --seed 913
```

Para reproducir además la evaluación, descargar previamente desde el enlace oficial del workflow original los tres archivos a `data/processed/frozen-original/`: `synthetic_reviews.csv`, `synthetic_manifest.json` y `synthetic_experiment_report.json`. **No copiar los nuevos ejemplos a entrenamiento original.** Luego:

```powershell
.\.venv\Scripts\python.exe -m traveler_agent.evaluate_out_of_generator --baseline data/processed/frozen-original --challenge data/processed/out-of-generator-challenge-v1 --out artifacts/out-of-generator-v1
```

No sobrescribir resultados ya generados. La ejecución automática en GitHub Actions descarga el artefacto congelado y verifica hash antes de entrenar/reconstruir cualquier modelo.

## Datos humanos auténticos y licencias: vía pública no individual

- **TASS español**: licencia explícita de investigación académica/interna no comercial; el sitio oficial exige completar y aceptar el acuerdo para descargar. No equivale a permiso ya concedido; datos generales de tweets, NO experiencias turísticas. https://sepln.org/workshops/tass/tass_data/download.php
- **SemEval-2016 Task 5, restaurantes españoles**: existen particiones oficiales train/test y anotaciones de **aspectos** (ABSA), no necesariamente una etiqueta global por reseña ni un permiso libre general. Página: https://alt.qcri.org/semeval2016/task5/index.php?id=data-and-tools ; licencia exacta de textos originales por confirmar.
- **COAR SINAI/UJA**: la ficha Hugging Face y convenio BOE declaran variantes CC BY-NC-SA vs CC BY. Sin elegir arbitrariamente licencia, no descargar textos.
- Los tres grupos son de procedencia, etiqueta y licencia diferentes; la futura comparación sintético/autorizado/híbrido debe conservar sus categorías y evaluar en un test auténtico humano disjunto, nunca en el desafío sintético.

## Resumen para revisión interna — viernes 6 de noviembre

1. Prototipo y Gradio Midnight Indigo: funcionalidad básica validada y regresión en main.
2. Corpus propio: 336 ejemplos sintéticos, Dummy/LR/SVM con semillas y manifiestos; el 100 % inicial no es precisión real.
3. Desafío fuera del generador: 1.260 textos inventados, 180 núcleos, sin duplicados exactos con baseline; F1 cae a ~0.39 para LR/SVM, evidencia de limitada transferencia sintética.
4. Control metodológico: parámetros anteriores congelados, fitting solo TRAIN original, una evaluación del desafío, provenance y hashes; no se usaron textos humanos como si fueran gold.
5. Próximo paso: obtener corpus auténtico con permiso académico verificable, privacidad y anotación humana independiente en español. Después comparar entrenamiento sintético/real/híbrido frente al mismo test humano cerrado.

**Conclusión:** primer test sintético de generalización adverso y útil; aún no existe evidencia de generalización a opiniones turísticas reales.
