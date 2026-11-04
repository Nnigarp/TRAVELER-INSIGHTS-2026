# Preparación de GitHub (sin publicación automática)

Repositorio sugerido: `traveler-insights-2026-cartagena` (primero privado). Verificar que no haya secretos, reseñas protegidas o datasets privados.

```bash
git init
git add .
git commit -m "feat: prototipo reproducible agente Traveler Insights 2026"
git branch -M main
# Tras crear un repo vacío bajo tu propia cuenta:
git remote add origin https://github.com/TU_USUARIO/traveler-insights-2026-cartagena.git
git push -u origin main
```

## Propuesta de hitos/ramas

- `main`: base estable con CI verde.
- `feat/dataset-audit`: inventario, mapeo de etiquetas, permisos, hashes.
- `feat/evaluation`: evaluación estratificada y matrices.
- `feat/transformer-baseline`: comparación controlada con encoder multilingüe.
- `feat/aspect-validation`: anotación/validación humana de aspectos.
- `docs/paper`: manuscrito con resultados reproducibles.

**Requisito antes de hacer público:** README, LICENSE compatible, tarjeta de datos, tarjeta de modelo, CI, análisis de privacidad, referencias y artefactos reproducibles. No añadir resultados científicos hasta correr los experimentos.
