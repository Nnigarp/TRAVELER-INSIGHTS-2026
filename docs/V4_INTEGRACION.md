# Sustentación de Traveler Insights V4

Desde la raíz del repositorio en el Lenovo, ejecutar en PowerShell:

```powershell
.\scripts\iniciar_demo.ps1
# Desde una segunda consola en la misma carpeta
.\.venv-research\Scripts\python.exe -m pytest tests -q --basetemp=.pytest_cache/revision
.\.venv-research\Scripts\python.exe scripts/audit_v4_integration.py
```

El lanzador académico verifica el manifiesto de BETO V4.2 partial256 y activa las cabezas V4.3 de regresión logística y SVM. Ambas usan el BETO base congelado, pooling medio enmascarado y normalización L2. Sus hashes se verifican antes de cargar joblib. BETO partial256 carga la cabeza supervisada y las adaptaciones de las capas 10 y 11. Los pesos se distribuyen localmente. El arranque histórico y sus pruebas siguen disponibles fuera del perfil. El informe de selección de hiperparámetros y los resultados de TF-IDF y soft-vote están en [el handoff](V4_HANDOFF.md).

Cada corpus local V4.0–V4.3 contiene 5.001 textos distintos, 1.667 por clase. En V4.3 se asignaron 3.360 a entrenamiento, 720 a validación, 720 a prueba y 201 al conjunto difícil. Los cuatro grupos son disjuntos por hash de texto y familia. Entrenamiento, validación y prueba comparten 13 plantillas. Todas las filas de entrenamiento contienen una referencia V4 numerada. La secuencia de puntuación «,.» aparece en 195 positivas, 195 negativas y ninguna neutral. Las frases «saldo quedó del lado favorable» y «saldo quedó del lado desfavorable» aparecen exclusivamente en su clase correspondiente (73 veces cada una). Estos atajos y las referencias únicas limitan lo que demuestra una comprobación por hash. La etiqueta expresa la intención del generador. La validación interna puede sobreestimar la transferencia a redacciones nuevas.

Los tres bancos IE son disjuntos por hash entre sí y respecto a los entrenamientos V4.2 y V4.3. IE-v1 contiene 72 textos distintos. IE-v2 e IE-v3 contienen respectivamente 195 y 227 textos distintos entre sus 300 filas. Las métricas históricas por fila dan más peso a las frases repetidas. El [recálculo por texto único](research/v4/integration_unique.json) conserva ambos denominadores. Estas evaluaciones son sintéticas, observadas durante la selección académica y carecen de anotación humana externa. El siguiente experimento que permitiría juzgar otra adaptación del encoder requiere reseñas anotadas y reservadas antes de entrenar. Se conservaron las variantes TF-IDF y se omitió otra búsqueda sobre los mismos bancos.

La entrada unificada conserva texto original, traducción y procedencia ASR. El agente semántico puntúa el español. Los tres modelos contextuales ven hasta 256 tokens contando especiales. ControlAgent deriva el truncamiento a revisión. Con Qwen habilitado, una negación o contraste activa una consulta independiente sobre el texto completo antes de ControlAgent. Una discrepancia, incertidumbre o fallo del verificador solicita revisión y preserva el resultado supervisado. La generación del informe posterior sigue siendo independiente. Los resultados exportados registran la revisión y su razón.

En el caso archivado de Cartagena (147 palabras), los tres candidatos predijeron negativo y Qwen indicó negativo. En un texto de 2.345 caracteres con doble cobro al final, BETO indicó neutral y Qwen negativo. ControlAgent registró tanto truncamiento como discrepancia. En pruebas de entrada reales, inglés, francés y chino se tradujeron al español; la frase italiana con *camera* quedó retenida para revisión. Whisper Tiny produjo texto en el WAV sintético disponible pero con errores importantes de transcripción. El ensayo comprueba el recorrido técnico, no establece precisión de dictado ni de traducción.
