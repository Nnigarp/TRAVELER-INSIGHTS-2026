# Incremento 03 — Perfil de emojis y protocolo de paralelismo (2026-11-04)

## Alcance implementado

- El auditor de CSV incorpora `emoji_profile` con conteos agregados de filas que contienen marcadores Unicode de emojis y emoticonos ASCII.
- El conteo usa rangos Unicode aproximados; **no** segmenta secuencias compuestas ni infiere polaridad. Una reseña con 🙃 puede ser irónica y no siempre negativa.
- El perfil separa los conteos por valor de etiqueta **sin inferir** qué significan 0, 1 o 2.
- Se redactan ejemplos de reseñas y lugares del reporte; `reports/` se excluye de Git.
- Nuevas pruebas unitarias cubren emojis, emoticonos, negación contextual, ausencia de inferencia de etiquetas y redacción.

## Alcance científico pendiente

No disponemos aún de los CSV oficiales en el repositorio. No se ha evaluado rendimiento de clasificación ni determinado si emojis mejoran Macro-F1. Un perfil de Unicode no es modelo de sentimiento y **no reemplaza** un análisis lingüístico contextual.

## Cómo ejecutar tras incorporar la rama

```bash
python -m pip install -e ".[test]"
python -m pytest -q
python scripts/audit_dataset.py data/raw/train.csv --text-col Comentario --target Sentimiento --out reports/audit.json
```

Los nombres de columnas anteriores son **ilustrativos**. Verificar en el CSV de 2026 antes de ejecutar. El informe es local y no debe subirse.

## Paralelismo: plan experimental, no funcionalidad fingida

Se aplicarán los principios de Computación en Paralelo:

1. Primero medir tiempo y memoria del auditor secuencial sobre archivos reales. Evitar paralelizar entradas pequeñas.
2. Solo si el coste justifica la infraestructura, comparar `ProcessPoolExecutor` sobre particiones independientes o archivos separados con 1, 2, 4 y hasta el número real de cores, con orden determinista en los resultados.
3. **Invariante:** resultados agregados idénticos en secuencial/paralelo; no introducir `sleep` ni retrasos artificiales; medir overhead, CPU y pico de RAM. Mantener semilla fijada en aprendizaje automático, no compartir estimadores mutables entre procesos.
4. No hacer entrenamiento de modelos en paralelo de forma que contamine validación o test. Comparaciones deben usar la misma partición y semillas; limitar oversubscription (`n_jobs`, BLAS/OpenMP).
5. Registrar speedup `T1/Tp`, eficiencia `speedup/p`, tamaño de entrada y hardware, e identificar si la versión paralela es más lenta.

## Próximo experimento

Obtener el dataset oficial con licencia y procedencia verificada; auditarlos sin publicarlos. Pre-registrar splits y evaluar: texto con emojis vs. sin emojis vs. representación verbal, siempre con el mismo test independiente. Incluir un subconjunto de negaciones y opiniones mixtas adjudicado manualmente, separado de los ejemplos usados en desarrollo.
