# Revisión crítica de errores y etiquetas del PR #13 — 5 de noviembre de 2026

**Finalidad:** preparar la explicación del 6 de noviembre sin reentrenar, cambiar datos, abrir el test ni ajustar parámetros. Documento derivado de los informes del PR #13 y del inventario de situaciones en su código. **No se realizó una evaluación adicional**.

## Evidencia y congelación

- Línea base PR #12, integrado: `dfac831aa6050a7242234e95122d97e43fa7a877`. Entrenamiento original de 336 textos inventados, semilla 42 y particiones familiares, artefacto del workflow `42006977003`.
- PR #13 permanece **abierto** en esta revisión: head `35653ce025d87aabb859421b0b7f3f447d363e53`. Desafío 1.260 textos generados de 180 núcleos, semilla 913, workflow `42010214928`. La referencia es documental; no depende de fusionar el PR #13 para consultar el informe.
- SHA-256 original del CSV congelado: `315012f9d553c51906e49bac617ef7ed9107ee2fc3cfd4dfd5a1368c7a4be059`.
- Fuente principal: `docs/GENERALIZACION_FUERA_GENERADOR_2026.md` de la rama `feat/out-of-generator-challenge`, y el artefacto del workflow. Se mantienen las métricas originales; no se publica una evaluación nueva ni un intento de mejorar resultados.
- El workflow inicial **no conservó los modelos serializados originales**; el PR #13 los reconstruyó desde TRAIN original con hiperparámetros y semilla congelados, verificando la puntuación de validación histórica. Esto no equivale a cargar literalmente los pesos antiguos.

## Hallazgo: negativos predichos como positivos

En el desafío hay 420 etiquetas objetivo por clase, **intenciones del generador**, no gold humano. Matrices de confusión con filas = etiqueta objetivo y columnas = predicción, orden [negativo, neutral, positivo]:

| Modelo | Neg→Neg | Neg→Neutral | **Neg→Pos** | Recall negativo | Errores totales |
|---|---:|---:|---:|---:|---:|
| LR | 101 | 87 | **232/420 = 55,24 %** | 24,05 % | 749/1.260 |
| SVM | 90 | 94 | **236/420 = 56,19 %** | 21,43 % | 750/1.260 |
| Dummy | 420 | 0 | 0 | 100 % (pero predice solo negativo) | 840/1.260 |

El recall negativo de Dummy es **engañoso**: al predecir exclusivamente negativo no detecta positivos ni neutrales. El Macro-F1 del Dummy sigue siendo 0,1667.

Matriz completa LR: `[[101,87,232],[74,156,190],[61,105,254]]`.
Matriz completa SVM: `[[90,94,236],[66,161,193],[46,115,259]]`.

| Modelo | F1 negativo | F1 neutral | F1 positivo | Macro-F1 | Accuracy |
|---|---:|---:|---:|---:|---:|
| Dummy | 0,5000 | 0,0000 | 0,0000 | 0,1667 | 33,33 % |
| LR | 0,3079 | 0,4063 | 0,4635 | 0,3926 | 40,56 % |
| SVM | 0,2894 | 0,4076 | 0,4675 | 0,3882 | 40,48 % |

El descenso del 1,0000 al ~0,39 de Macro-F1 **es compatible** con aprendizaje de atajos léxicos en el primer generador, porque compartía expresiones de polaridad entre familias. El nuevo inventario de situaciones cambia vocabulario, estructuras y objetos turísticos. No permite demostrar causalmente qué parte del descenso corresponde a atajos, cambio de dominio textual, frecuencia de clases, neutralidad discutible u otras diferencias. Ambas pruebas usan textos inventados.

## Muestra cualitativa del inventario — NO atribuir predicciones sin evidencia fila a fila

Estos son **fragmentos literales de las situaciones fuente** en `out_of_generator_challenge.py`. Se citan para revisar la *coherencia de la etiqueta objetivo*, **no** porque sepamos que el modelo se equivocó exactamente en estas filas.

| Fragmento de situación inventada | Objetivo | Observación de revisión |
|---|---|---|
| «Nos agregaron consumos que nunca hicimos y discutieron al reclamarlos» | Negativo | Marcador claro de daño/reclamación; revisar si LR/SVM reconocen paráfrasis nuevas de cobro injustificado en un test futuro humano, no reabrir este test |
| «El conductor aceleró demasiado cerca de una zona escolar» | Negativo | Riesgo de seguridad real en el texto ficticio; el término *aceleró* no fue necesariamente visto en entrenamiento original |
| «La rampa estaba clausurada y no existía otra entrada accesible» | Negativo | Describe un fallo específico de accesibilidad; etiqueta objetivo razonable |
| «La página mostraba fotografías y un listado de servicios» | Neutral | Descripción verificable, sin valoración evidente; conservar como intención neutral |
| «La marea cambiaba de color según la hora del día» | Neutral | Descripción física, pero la lectura emocional puede depender del contexto |
| «Los paneles estaban ordenados cronológicamente» | Neutral | Orden puede interpretarse como elogio implícito de organización; **posible ambigüedad** de etiqueta |
| «La visita terminaba a una hora razonable y alcanzamos el regreso» | Positivo | Relación de puntualidad con beneficio explícito; polaridad plausible |
| «No quiero exagerar lo que ocurrió…» seguido de un problema | Negativo en ese núcleo | La negación afecta al acto de *exagerar*, no necesariamente a la polaridad. La marca automática `phenomenon=negation` es **superficial** |
| «Primero pensé otra cosa; al final comprobé que…» | Cualquier clase según el núcleo | Insinúa contraste, pero no especifica la primera opinión: `phenomenon=contrast` **no garantiza contraste semántico** |
| «¿Cómo resumir lo que pasó…?» y siete reformulaciones del mismo núcleo | Según núcleo | No son observaciones independientes; comparten una oración central y pueden inflar la aparente cantidad de evidencia |
| «xq» y «tambien» en sustituciones automáticas | Según núcleo | La variante marcada como `misspelling` **puede carecer de error ortográfico real** si las palabras buscadas no aparecen |

**Control científico:** la cola de ejemplos ambiguos requiere adjudicación humana si se convierte este material en gold; no se corrigen ni reclasifican ahora, porque ello contaminaría la comparación. El script no demuestra cobertura real de sarcasmo contextual ni de opiniones mixtas equilibradas. Los regionalismos son marcadores escritos por el generador y no representan una distribución lingüística real de Colombia o Perú.

## Hipótesis contrastables, no resultados demostrados

1. **Memorización de pistas sintéticas:** el primer generador repetía frases de sentimiento por clase, y los modelos TF-IDF pueden utilizarlas para separar clases con facilidad.
2. **Transferencia léxica limitada:** muchos problemas del desafío usan objetos y consecuencias diferentes —riesgos, cobros, movilidad, acceso—, sin sus expresiones equivalentes en el conjunto original.
3. **Etiquetas de intención imperfectas:** algunas frases neutras implican valoraciones y algunas marcas de fenómeno no describen una construcción semántica real.
4. **Muestras correlacionadas:** siete variaciones por núcleo comparten información. Los 1.260 registros equivalen a 180 núcleos y no a 1.260 participantes.

No se hace inferencia estadística entre grupos ni se pretende medir precisión poblacional. Para validar estas hipótesis, necesitaríamos reseñas auténticas, grupos independientes, anotadores y un nuevo protocolo previamente registrado; no reoptimizar sobre este test.

## Cierre

**Conservar congelados** ambos conjuntos y el primer reporte, sin recortar casos problemáticos ni cambiar etiquetas a posteriori. Ningún ejemplo de este documento debe convertirse en *nuevo entrenamiento* antes de haber separado un test auténtico independiente.

La siguiente prioridad, tras la retroalimentación docente, sigue siendo permiso público verificable, consentimiento/privacidad, etiquetas humanas gold cuando existan y comparación sintético/real/híbrido sobre un **único test humano reservado**.

**Enlaces:** PR #13 https://github.com/Nnigarp/TRAVELER-INSIGHTS-2026/pull/13 ; ejecución de desafío https://github.com/Nnigarp/TRAVELER-INSIGHTS-2026/actions/runs/42010214928 .
