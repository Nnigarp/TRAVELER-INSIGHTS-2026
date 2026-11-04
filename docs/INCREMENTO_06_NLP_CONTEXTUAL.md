# Incremento 06 — señales de NLP y spans verificables

## Implementación (independiente de Kaggle)
- Módulo `linguistic.py`, sin servicios externos, que **conserva la reseña original** y produce una clave normalizada de comparación sin borrar emojis.
- Detección **indicativa y no validada** de español, inglés e italiano por términos frecuentes, con abstención `desconocido` para señales insuficientes o empatadas. No es identificación de idioma fiable para producción.
- Señales de negación y contraste con offsets `[inicio, fin)`; `posible_opinion_mixta` indica solo conectores, no una inferencia de polaridad mixta.
- Spans exactos de palabras de aspecto con `texto_original[inicio:fin] == texto`; no constituyen explicación causal ni ABSA.
- `agent.inspect_review()` añade `perfil_linguistico_exploratorio` sin modificar el resultado anterior del modelo o las recomendaciones.
- Pruebas de Unicode, errores ortográficos, negaciones, idiomas y contratos de integración.

## Límites conocidos
La detección de idioma es heurística; en textos cortos, mezcla de idiomas y dialectos es poco fiable. El vocabulario de aspectos tiene falsos positivos/negativos. Negaciones y sarcasmo requieren evaluación específica. Los offsets se refieren a la cadena después del `strip()` que realiza `inspect_review`, no al texto completo con espacios exteriores; revisar si se necesita preservar la entrada byte por byte y los offsets absolutos.

## Validación local
```cmd
python -m pip install -e ".[test]"
python -m pytest -q
```

## Estrategia de integración
PR independiente contra `main`, con pruebas y CI. Este incremento no depende de #3 ni #4 y no debe fusionarse antes de su revisión. Audio y ASR corresponden al siguiente incremento.
