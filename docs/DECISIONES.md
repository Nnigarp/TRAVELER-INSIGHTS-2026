# Registro de decisiones técnicas

**ADR-001 — Línea base reproducible antes de Transformers**

- Decisión: comenzar con TF-IDF de palabras/caracteres + regresión logística.
- Razón: ejecución local, bajo costo, trazabilidad y referencia para comparar mejoras modernas de 2026.
- Alternativa diferida: encoder multilingüe ajustado. No se asume que sea superior.

**ADR-002 — Separar el núcleo del reto de ampliaciones**

- Núcleo obligatorio: clasificación positivo/neutral/negativo.
- Extensiones propuestas: aspectos por reglas, fragmentos citados, política de acción, revisión humana.
- Razón: cubrir requisito académico sin atribuir desempeño experimental a módulos no validados.

**ADR-003 — Confianza y abstención prudentes**

- Puntuaciones de regresión logística son no calibradas; el umbral se usa solo en demo.
- Antes del despliegue: calibrar y validar umbrales contra un conjunto independiente.

**ADR-004 — Datos y privacidad**

- Sin scraping automático ni claves externas; CSV debe ser legítimamente obtenido.
- No subir reseñas privadas o identificadores personales a repositorio público.

**ADR-005 — No usar Kaggle como fuente cartagenera sin evidencia**

- La descripción del reto cita datos de Sucre. Cartagena se estudiará mediante evaluación independiente de transferencia de dominio.

**ADR-006 — Mantener modelo demo y corregir reglas sin usar los cinco casos como entrenamiento**

- Decisión: mejorar cobertura heurística e informar todas las dimensiones detectadas, sin asignar polaridad por aspecto.
- Motivo: los casos manuales evidencian un error de negación, pero ajustar la línea base a cinco entradas sería sobreajuste.
- Validación: pruebas funcionales y `xfail` estricto; falta dataset de test real.
