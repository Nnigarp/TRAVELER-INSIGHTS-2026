# Corpus turístico sintético y primer benchmark interno — 2026-11-05

**Estado:** primer experimento reproducible ejecutado sobre DATOS TOTALMENTE INVENTADOS.
**Prohibición metodológica:** no presenta datos humanos, sentimiento gold, turismo real ni generalización a Cartagena.
**Rama:** feat/synthetic-tourism-research (no modifica Gradio, audio, modelos DEMO ni corpus adjudicado).
**Python:** 3.12 · sin API paga, scraping, ChatGPT ni Gemini usados en esta ejecución.

## 1. Diseño del generador

Módulo: src/traveler_agent/synthetic_tourism.py. Generador de escenarios **curados y composicionales**, no un LLM externo. La diversidad es limitada: los mismos fragmentos descriptivos pueden aparecer en distintos contextos; no confundir 336 filas con 336 testimonios independientes.

- Siete categorías: hotel, restaurante, museo, playa, transporte, recorrido, guía turístico.
- Ocho aspectos: servicio, higiene, precio, espera, seguridad, acceso, alimentación, instalaciones (acceso representa accesibilidad en esta versión).
- Tres clases objetivo en cada familia: negativo, neutral y positivo; dos variaciones por clase.
- 7 × 8 = 56 familias, cada una con seis registros: total **336 registros**.
- Casos controlados: negación, contraste, opinión mixta *simulada*, intensificadores, regionalismos colombianos/peruanos, faltas ortográficas e ironía escrita.
- Procedencia: review_id seudónimo, seed, fecha explícita, método, versión, modelo=NO_LLM_LOCAL_RULE_BASED, familia, aspecto, región, estilo, etiqueta prevista, huella SHA256 y synthetic=YES.
- No inventar participantes, empresas reales, licencias ni anotaciones humanas.
- Etiquetas son objetivos de composición: GENERATOR_INTENT_NOT_HUMAN_GOLD.
- Controles: IDs únicos, SHA de texto normalizado, longitud, PII heurística, diversidad léxica, recuentos de grupos y clases, candidatos a casi duplicados y cola de negaciones/ironías/contrastes que requieren revisión de coherencia. La detección de contradicciones no equivale a comprobación semántica exhaustiva.

Se almacena en data/processed/synthetic-tourism-v1/ (ignorado por Git). La CI publica artefactos identificados SYNTHETIC-ONLY-NOT-HUMAN-GOLD-seed42 con el CSV, manifiesto y reporte; no exporta datos humanos.

## 2. Benchmark ejecutado

Reutiliza el MODEL_GRID, _new_estimator, _score y _training_calibration_folds del benchmark científico. No altera las implementaciones originales. Se entrenan DummyClassifier (majority), TF-IDF+LR (C=0.5/2) y TF-IDF+LinearSVC calibrado, con calibración interna por grupos. Se optimiza Macro-F1 usando solamente VALIDATION. Test se evalúa una única vez por estimador seleccionado. Se guardan accuracy, Macro-F1, precision/recall/F1 por clase, matrices de confusión, errores con ID sin texto y hashes de modelos.

Particiones por familia completa, hash reproducible con seed 42:

| Split | Familias disjuntas | Total | Por clase |
| --- | ---: | ---: | ---: |
| train | 34 | 204 | 68 |
| validation | 11 | 66 | 22 |
| test | 11 | 66 | 22 |

**Primera ejecución, versión sintética original (run 42006272108, seed 42):**

| Modelo | Macro-F1 validation | Macro-F1 test | Accuracy test | Errores de test |
| --- | ---: | ---: | ---: | ---: |
| Dummy (clase mayoritaria) | 0.1667 | 0.1667 | 0.3333 | 44/66 |
| TF-IDF + LR | 1.0000 | 1.0000 | 1.0000 | 0/66 |
| TF-IDF + SVM calibrado | 1.0000 | 1.0000 | 1.0000 | 0/66 |

Matriz de confusión test, orden de etiquetas = negativo, neutral, positivo:
- Dummy: [[22,0,0],[22,0,0],[22,0,0]].
- LR y SVM: [[22,0,0],[0,22,0],[0,0,22]].

El dataset original SHA256 es 315012f9d553c51906e49bac617ef7ed9107ee2fc3cfd4dfd5a1368c7a4be059.
Artefactos de la primera ejecución: https://github.com/Nnigarp/TRAVELER-INSIGHTS-2026/actions/runs/42006272108/artifacts/13405480095

### Advertencia científica esencial

**Macro-F1 test 1.0 NO acredita excelencia de los modelos.** El generador repite expresiones descriptivas de cada polaridad entre familias diferentes. La separación por familias impide que variaciones de una *misma* familia crucen particiones, pero NO garantiza independencia semántica: LR/SVM pueden explotar pistas léxicas generadas de manera repetitiva. Se incorporó una auditoría explícita de atajos léxicos; el resultado inicial debe interpretarse como un **control de software / evaluación sintética interna con fuerte riesgo de shortcut**, no como evaluación científica del mundo turístico.

No modificar las plantillas ni seleccionar prompts por haber visto test 1.0. La próxima evaluación representativa debe usar un nuevo test auténtico anotado independientemente en español, sin haber sido utilizado para definir vocabulario, plantillas, hiperparámetros ni criterios de selección. Un nuevo generador evaluado sobre este mismo test sería una comparación sesgada.

Datos sintéticos no entrenan la versión desplegada de Gradio automáticamente. Los modelos científicos adjudicados tampoco se mezclan con este experimento.

## 3. Reproducir en Windows PowerShell (Python 3.12)

Tras revisar estado Git y proteger cambios locales:

~~~powershell
git fetch origin
git switch --track origin/feat/synthetic-tourism-research
.\.venv\Scripts\python.exe -m pip install -e ".[test]"
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m traveler_agent.synthetic_tourism --seed 42 --variants 2 --date 2026-11-05 --out data/processed/synthetic-tourism-v1
.\.venv\Scripts\python.exe -m traveler_agent.synthetic_evaluation --seed 42 --input data/processed/synthetic-tourism-v1 --out artifacts/synthetic-tourism-v1
~~~

No sobrescribir reportes existentes: cambiar rutas si la ejecución ya ocurrió. Archivos:
- synthetic_reviews.csv y synthetic_manifest.json, con tipos claramente sintéticos.
- dummy.joblib, lr.joblib, svm.joblib, synthetic_experiment_report.json, bajo artifacts/.
- El informe incluye ambiente de Python, versiones reales scikit-learn/numpy/scipy/joblib, semillas y SHA de entrada y modelos.

## 4. Extensión a LLM (NO ejecutada)

Si posteriormente se utiliza ChatGPT, Gemini u otro proveedor, mantener cada lote aislado y registrar proveedor/modelo e identificador de versión disponible, fecha, temperature/top_p/seed si aplican, prompt íntegro o hash+archivo versionado, política de retención, identificación sintética, coste y derechos de uso. Un seed de generación remoto no necesariamente garantiza reproducibilidad literal. Nunca copiar entradas personales a servicios externos sin consentimiento ni publicar secretos/API keys. La generación con LLM no convierte los objetivos de sentimiento en etiquetas humanas.

## 5. Datos humanos originales opcionales

Protocolo de recolección voluntaria separado del sintético:
1. Invitación neutra explicando finalidad de investigación y clasificación; sin coacción académica.
2. Consentimiento informado específico antes de aportar un relato, autorización para procesar el texto, periodo de conservación y eventual publicación de resultados agregados. No prometer anonimato absoluto.
3. Retirar nombres, contactos, detalles de localización personal y demás identificadores; revisión humana de PII y acceso restringido a data/private (ignorado por Git). Permitir retirada conforme a política documentada.
4. Registrar source_id voluntario, id de consentimiento seudónimo y base de uso sin subir consentimientos al repositorio público.
5. Dos anotadores independientes y adjudicación en una muestra de referencia; acuerdo Cohen kappa, spans verificables y reserva del test auténtico.
6. Mantener test auténtico exclusivamente para una evaluación final, no para diseñar prompts, escoger plantillas o modificar modelos.

Nunca usar predicciones DEMO ni etiquetas previstas del generador como truth humana.

## 6. Investigación paralela sin dependencia de solicitudes personales

- COAR (SINAI/UJA) español: su ficha de Hugging Face declara CC BY-NC-SA 4.0 y requiere aceptar condiciones; el BOE menciona CC BY 4.0. Continúa sin resolver compatibilidad jurídica y derechos de textos TripAdvisor. No descargar ni marcar permisos falsos.
- HotelReview UNED: anotaciones manuales globales y acceso de investigación, comprobar idioma, alcance del permiso y textos Booking.com.
- SemEval-2016 Task 5: reseñas de restaurantes españoles con anotaciones de aspecto, comprobar licencia del paquete original, no inferir etiqueta global.
- Audiala Places: datos de lugares turísticos bajo CC BY 4.0, pero **no es corpus de reseñas etiquetadas**; útil solo para futuros metadatos, no entrenamiento de sentimiento.
- Dataset of Digital Reviews in Tourism (Telkom Dataverse, CC0 declarado): 56.448 reseñas Bali; comprobar derechos sobre UGC de origen, idiomas y privacidad antes de importar, pues CC0 del depositante no resuelve automáticamente derechos ajenos.

**Regla de salida:** no hay datos turísticos auténticos autorizados ni métricas externas reportadas en este incremento.
