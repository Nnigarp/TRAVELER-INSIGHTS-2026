# Informe generativo local: demostración académica

Complemento de las guías y experimentos anteriores; no modifica sus resultados.
Los criterios de Sabie del 5 de noviembre son los comunicados por el responsable
en este encargo; no se afirma haber accedido a la plataforma docente.

## Caso completo (3–5 minutos)

1. **Problema y usuario.** Un visitante necesita convertir un relato turístico en
   información revisable. La herramienta clasifica, explicita incertidumbre y
   prepara un informe local; el usuario decide qué hacer con él.
2. **Situación e ingreso.** Con LR predeterminado y Qwen desmarcado, introducir
   «El hotel era cómodo, pero el transporte nunca llegó y perdimos la excursión».
   Mostrar el recorrido convencional. Repetir seleccionando BETO experimental y
   habilitando Qwen cuando ambos artefactos estén disponibles.
3. **Percepción y razonamiento.** PerceptionAgent valida la entrada. Whisper Tiny
   transcribe audio y permite edición humana. SemanticsAgent ejecuta el clasificador
   elegido, conserva scores no calibrados y extrae aspectos exploratorios por reglas.
   No afirmar que BETO extrae aspectos ni que Qwen clasifica el sentimiento.
4. **Plan y decisión.** TravelerOrchestrator coordina cuatro especialistas con
   contratos y un plan fijo. ControlAgent conserva su regla histórica de revisión
   y razones visibles. Qwen recibe copias de etiqueta/aspectos/revisión después
   de Control; no puede anularlos. La regla no certifica automatización segura.
5. **Acción verificable.** Qwen3 1.7B compone JSON mediante Ollama local. ActionAgent
   revalida citas completas, catálogo de propuestas y campos bloqueados antes de
   incorporar el informe. Mostrar `generacion_llm=true`, modelo efectivo,
   `validado_por=ActionAgent`, latencia y trazas separadas. El relato no acredita
   que el hotel organizara el transporte.
6. **Resultado y memoria.** Pasar el caso a Lotes, exportar JSON/CSV y corregir una
   etiqueta conservando la predicción. Mostrar aviso de informe anterior a la
   corrección. La acción ejecutada es producir, validar y registrar un informe
   exportable; no se envían reclamaciones ni se reservan servicios.
7. **Falla demostrable.** Desmarcar Qwen y analizar de nuevo: el flujo convencional
   continúa sin consultar Ollama. En una desconexión mostrar código de error,
   `generacion_llm=false` y recomendación por reglas, sin llamarla Qwen.

No imponer una etiqueta para que la demostración «pase»: observar y discutir
la predicción. Es un caso funcional, no una evaluación predictiva independiente
ni una muestra humana etiquetada.

## Correspondencia con documento y sustentación

| Criterio comunicado en Sabie | Comportamiento/evidencia |
|---|---|
| Problema, usuario y arquitectura | Relato turístico, usuario revisor, coordinador + cuatro especialistas |
| Percepción | Validación textual; ASR con procedencia y edición |
| Razonamiento/plan | Inferencia supervisada, aspectos por reglas, plan fijo y contratos |
| Decisión y acción | Control preservado; informe compuesto, validado y exportable |
| Memoria o aprendizaje | Memoria episódica de sesión; aprendizaje supervisado offline |
| Caso y resultado | Reseña solicitada, JSON, diez eventos principales y tres generativos |
| Límites, sesgos y mejoras | Clasificación sintética, revisión insuficiente, pertinencia de citas/propuestas |

BETO es un encoder Transformer ajustado supervisadamente en el experimento
sintético V3. LR/SVM de la DEMO tienen otro corpus/receta: no comparar sus respuestas
como benchmark justo de entrenamiento. Qwen es un modelo generativo; Ollama es su
entorno local de ejecución. Aquí no se ajusta Qwen ni se aprende automáticamente
de reseñas/correcciones. Participan cinco componentes de software; Qwen no se
presenta como un sexto agente autónomo.

El resumen es extractivo y las propuestas proceden de un catálogo auditado.
Qwen compone/selecciona; no prometer redacción libre validada ni razonamiento factual
infalible. Persisten omisiones de contexto, citas poco pertinentes, errores de
clasificación y decisiones de revisión insuficientes. Ninguna prueba demuestra
rendimiento con turistas reales ni seguridad operacional.

## Evidencia para conservar

- JSON del comprobador por callback y HTTP: versión/digest Ollama, protocolo/hash
  del esquema, clasificador efectivo y ausencia de fallback.
- Exportación del lote, corrección y captura de informe/trazas; sin datos privados.
- Latencia de arranque frío y modelo cargado identificadas, más instantáneas de RAM
  con los límites descritos en la guía Windows.
- Transcripción original y editada; señalar ruido/errores de ASR.

El artículo CEUR-WS debe separar el experimento **predictivo de clasificación**
ya congelado de esta **integración generativa funcional**. No añadir Macro-F1,
calidad humana del resumen ni resultados Kaggle no medidos. La comparación de
pertinencia generativa requiere un protocolo y anotación posteriores; la aceptación
actual comprueba contratos, procedencia y continuidad funcional.
