# Traveler Insights 2026 — agente turístico (v0.1.0)

**Proyecto académico de Inteligencia Artificial (ISCO-A06A, UTB).** Proponente: Nilver García Pimentel. Caso de aplicación futura: Cartagena de Indias, Colombia.

> **Estado:** prototipo funcional de demostración, NO modelo científico validado. La opción 3 del reto pide clasificar reseñas **positivas, neutrales o negativas**. Esta implementación agrega aspectos por reglas, fragmentos literales y recomendaciones auditables. No afirma contar con datos locales de Cartagena.

## Qué funciona

1. **Percepción:** recibe reseñas e identifica validaciones de entrada (vacío, longitud).
2. **Clasificación supervisada:** `TfidfVectorizer` de palabras y caracteres + regresión logística (tres etiquetas semánticas).
3. **Aspectos exploratorios:** diccionario visible y extensible (servicio, precio, higiene, acceso, espera y seguridad). **No equivale a ABSA entrenado.**
4. **Evidencia descriptiva:** muestra fragmentos **literales** que contienen términos del diccionario; no ofrece una explicación causal del modelo.
5. **Decisión:** reglas deterministas y abstención/revisión humana cuando la puntuación es baja o hay indicios de opinión mixta.
6. **Trazabilidad:** salida JSON, manifiesto del modelo y hash del dataset (cuando exista CSV real).
7. **Evaluación separada:** holdout estratificado, Macro-F1, accuracy, métricas por clase y matriz de confusión; **únicamente para CSV real**.

## Instalación (Windows PowerShell)

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e ".[ui,test]"
pytest -q
traveler demo-train
traveler predict --text "El precio fue excesivo y esperamos horas."
python app.py
```

Abrir la URL local que muestre Gradio (generalmente `http://127.0.0.1:7860`). **No habilitar enlaces públicos ni introducir información personal real en esta demo.**

## Linux / Colab

```bash
python -m pip install -e '.[ui,test]'
pytest -q
traveler demo-train
traveler predict --text "El servicio fue excelente y la visita hermosa."
python app.py
```

## Modelo real: cuando se obtenga el dataset con permisos

El CSV debe contener columnas textuales `reseña` y `sentimiento`, este último con etiquetas **positivo, neutral, negativo** (sin números). En caso de que Kaggle proporcione 0/1/2, **se debe comprobar el mapeo contra archivos, descripciones y ejemplos oficiales antes de convertirlo**. La descripción pública del reto contiene inconsistencias. No asumir una correspondencia por intuición.

```bash
traveler evaluate --csv data/mi_dataset_verificado.csv --text-col reseña --label-col sentimiento --out artifacts/holdout.json
traveler train --csv data/mi_dataset_verificado.csv --text-col reseña --label-col sentimiento --out artifacts/real_model.joblib
traveler predict --model artifacts/real_model.joblib --text "La atención fue muy amable."
```

**Precaución metodológica:** primero evaluar una partición independiente; el comando `train` entrena posteriormente con todos los registros válidos para crear el artefacto final. Evite evaluar el mismo modelo en sus propios datos de entrenamiento. Para publicación científica, también exigir validación cruzada, partición por lugar/fecha si aplica, calibración y evaluación externa con muestras de Cartagena.

**Seguridad:** los archivos `.joblib` solamente deben abrirse si se crearon localmente y provienen de una fuente confiable; deserializar un archivo ajeno puede ejecutar código. El entrenamiento no descarga modelos remotos ni llama a servicios de IA.

## Arquitectura

```text
Reseña + ID --> Validación --> Vectorización TF-IDF --> Clasificación 3 clases
                                                |             |
                                Aspectos por reglas             v
                                      |                 Revisión de incertidumbre
                                      v                         |
                               Fragmentos literales --> Política transparente
                                                            |
                                           Etiqueta, evidencia, aspecto, sugerencia, revisión
```

## Qué **no** hace aún

- No utiliza todavía los archivos originales de Kaggle ni tiene etiquetas o métricas reales.
- No incluye un Transformer, LLM, RAG ni un sistema ABSA supervisado; son líneas de experimentación posteriores, no mejoras demostradas.
- El valor mostrado como confianza es **puntuación no calibrada**. No debe tratarse como probabilidad fiable.
- El extractor de aspectos y la selección de fragmentos son heurísticos. Las recomendaciones requieren supervisión humana.
- No se publica automáticamente a GitHub ni se despliega públicamente.

## Investigación / artículo

Véanse `docs/PROTOCOLO_INVESTIGACION.md`, `docs/BITACORA.md`, `docs/DECISIONES.md`, `docs/FUENTES.md` y `docs/PLAN_GITHUB.md`. **No se garantiza publicación**: dependerá de rigor de los experimentos, novedad, calidad de datos, evaluación externa y revisión por pares.

## Fuente del reto

[Traveler Insights 2026 (Kaggle)](https://www.kaggle.com/competitions/travelerinsights2026/overview). La competencia describe tres clases y datos turísticos vinculados a fuentes distintas; su referencia apunta a un dataset de Sucre. Cartagena es un **dominio objetivo futuro**, no se asume como origen de sus reseñas.
