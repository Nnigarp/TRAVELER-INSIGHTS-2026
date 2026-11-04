# Estado verificable — 2026-11-04

**Implementado y probado localmente:** CLI, pipeline TF-IDF + regresión logística, modelo de demostración con datos fabricados y etiquetados explícitamente para integración, detección de aspectos mediante reglas, extracción literal de fragmentos, política de sugerencias, abstención, salida JSON, interfaz Gradio y ocho tests automatizados (8/8 exitosos).

**No incluido:** dataset real de Kaggle, reseñas verificadas de Cartagena, evaluación en un conjunto independiente, Transformers entrenados, publicación de artículo y GitHub remoto.

**Cómo se probó:** `PYTHONPATH=src pytest -q` (8 passed); `PYTHONPATH=src python -m traveler_agent.cli demo-train`; `PYTHONPATH=src python -m traveler_agent.cli predict --text "El guía fue amable pero esperamos demasiado tiempo y el precio estaba mal comunicado."`. En esta frase de demostración el prototipo identificó los aspectos servicio/precio/espera y la derivó a revisión humana; ese comportamiento no prueba precisión generalizable.

**Nota del entorno de creación:** la instalación editable mediante pip no pudo completarse debido a la ausencia de acceso a internet para descargar dependencias de build. Las pruebas se ejecutaron correctamente con las librerías preinstaladas y `PYTHONPATH=src`. En un entorno con acceso a PyPI la instalación indicada en README es la ruta prevista.
