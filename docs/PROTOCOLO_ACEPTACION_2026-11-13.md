# Registro anterior a la ejecución

Base verificada: main `4df0cebc44ff4130bcba3cd17f12835049ed9953`, PR27 fusionado.

Los tres JSON `academic_acceptance_*` versionan criterios, trece textos funcionales y 36 casos predictivos sintéticos (12 por clase) más ocho ambiguos sin etiqueta de sentimiento. Un único asistente los escribió con conocimiento de limitaciones anteriores: no son opiniones humanas ni una muestra poblacional ciega. Los textos y etiquetas se fijan en Git antes de inferir. Solo el texto entra al clasificador; IDs, regiones, fenómenos y etiquetas son metadatos de auditoría.

Los contratos funcionales no exigen acertar una polaridad: exigen validación, modelo efectivo, trazas, revisión coherente, conservación de datos y exportaciones. Se pueden repetir tras corregir causas raíz. Los casos predictivos sirven para medir y exponer errores; sus resultados no seleccionan modelos, épocas, temperaturas ni umbrales. El primer informe completado queda congelado. Cualquier reproducción posterior se identifica como tal.

LR y SVM son las opciones clásicas de la DEMO y usan su corpus ficticio histórico. BETO es el seleccionado V3, entrenado con 300 textos sintéticos distintos. La comparación describe opciones disponibles, no aísla el efecto de una arquitectura con datos iguales. Si falta el artefacto ajustado, se autoriza una única reconstrucción fija de V3, semilla42, usando solo su TRAIN y la selección publicada; máximo900s, sin búsqueda ni uso de los casos nuevos. No se repiten evaluaciones históricas para seleccionar nada.

Se registrarán Macro-F1, precisión/recall/F1 por clase, matrices, negativos→positivos, errores aceptados por la regla histórica, fenómenos y latencia. Ambigüedad se mide aparte como derivación descriptiva, sin convertirla en neutral ni en cuarta clase. Ni cobertura ni confianza certifican seguridad. No hay criterio de aprobación basado en alcanzar una puntuación predictiva.

Los límites y casos CSV, fallback, voz simulada y Gradio HTTP están enumerados en el protocolo JSON. La voz propia, permisos del navegador y micrófono Windows requieren ensayo personal: no se declaran verificados con un stub. No se amplía a5000 ni se promociona BETO por defecto.
