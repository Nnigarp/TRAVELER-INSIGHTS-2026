"""Recompute metrics from frozen responses and render a report; no inference."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from traveler_agent.academic_acceptance import registered_cases, require_effective
from traveler_agent.research_metrics import summarize_predictions
from traveler_agent.corpus import file_sha


def analyze(report_path, output, figure=None):
    report = json.loads(report_path.read_text(encoding='utf-8'))
    _, _, rows = registered_cases(report['registration_commit'])
    assert report['status'] == 'COMPLETED' and report['human_gold'] is False
    assert report['model_or_threshold_selection'] is False
    labels = [r for r in rows if r['intended_label'] is not None]
    names = {'lr': 'LR · DEMO36', 'svm': 'SVM · DEMO36', 'beto': 'BETO V3 · TRAIN300'}
    for kind, result in report['models'].items():
        observations = result['observations']
        assert [o['id'] for o in observations] == [r['id'] for r in rows]
        for row, observed in zip(rows, observations):
            assert row['text'] == observed['text'] and not observed['violations']
            require_effective(observed['record'], kind)
        selected = [(r, o) for r, o in zip(rows, observations) if r['intended_label'] is not None]
        recalculated = summarize_predictions([r['intended_label'] for r, _ in selected],
            [o['record']['sentimiento'] for _, o in selected],
            reviewed=[o['record']['revision_humana'] for _, o in selected])
        assert recalculated == result['metrics'], 'Saved metrics differ from frozen responses.'
    lines = ['# Aceptación académica: medición sintética nueva', '',
        'Análisis posterior de respuestas congeladas; este script no entrena, infiere ni selecciona políticas.', '',
        f'Registro: `{report["registration_commit"]}`. Código ejecutado: `{report["execution_commit"]}`. SHA256 del informe: `{file_sha(report_path)}`.', '',
        '36 textos etiquetados por intención de un asistente,12 por clase, más8 ambiguos sin sentimiento gold. Un autor conoce limitaciones anteriores: muestra ad hoc sintética, sin cegamiento humano ni muestreo poblacional. Desde esta ejecución todos los casos están observados. No se editan sus etiquetas ni se utilizan para elegir nuevas configuraciones.', '',
        'LR/SVM usan los36 ejemplos históricos de la DEMO. BETO V3 usa300 TRAIN sintéticos y la selección anterior, semilla42. La diferencia combina datos y arquitectura; no identifica causalmente el efecto del encoder.', '',
        '| Opción | Macro-F1 | Recall negativo | Negativos→positivos /12 | Aceptados /36 | Errores aceptados | Ambiguos derivados /8 |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for kind in ('lr', 'svm', 'beto'):
        item = report['models'][kind]; m = item['metrics']; a = m['abstention']
        accepted_errors = sum(o['record']['sentimiento'] != r['intended_label'] and not o['record']['revision_humana'] for r, o in zip(rows, item['observations']) if r['intended_label'] is not None)
        lines.append(f'| {names[kind]} | {m["macro_f1"]:.4f} | {m["per_class"]["negativo"]["recall"]:.4f} | {m["negative_to_positive_count"]} | {a["accepted"]} | {accepted_errors} | {item["ambiguity"]["referred"]} |')
    lines += ['', 'Los clásicos derivan33/36 y solo aceptan3: la ausencia de errores en ese soporte pequeño no es una solución operativa satisfactoria ni certifica riesgo. BETO acepta35/36, incluye un negativo→positivo sin revisión y deriva0/8 ambiguos. No se promociona BETO ni se modifica la regla histórica. La aceptación funcional PASSED no significa aprobación predictiva o de automatización.', '', '## Matrices y métricas por clase', '', 'Filas: intención sintética prevista. Columnas: predicción. No son etiquetas humanas.', '']
    if figure: lines += ['![Matrices sintéticas](acceptance/' + figure.name + ')', '']
    for kind in ('lr', 'svm', 'beto'):
        m = report['models'][kind]['metrics']
        lines += ['### ' + names[kind], '', '| Prevista / predicha | negativo | neutral | positivo |', '|---|---:|---:|---:|']
        for label, values in zip(m['confusion_matrix_labels'], m['confusion_matrix']):
            lines.append('| ' + label + ' | ' + ' | '.join(map(str, values)) + ' |')
        lines += ['', '| Clase | n | Precisión | Recall | F1 |', '|---|---:|---:|---:|---:|']
        for label in m['confusion_matrix_labels']:
            c = m['per_class'][label]
            lines.append(f'| {label} | {c["support"]:.0f} | {c["precision"]:.4f} | {c["recall"]:.4f} | {c["f1-score"]:.4f} |')
        lines.append('')
    failed = next(o for o in report['models']['beto']['observations'] if o['id'] == 'P002')
    r = failed['record']
    lines += ['## Negativo aceptado como positivo: P002', '', '> ' + failed['text'], '',
        f'Intención registrada: negativo. BETO: **{r["sentimiento"]}**, puntuación máxima no calibrada **{r["confianza_no_calibrada"]:.4f}**, revisión **{r["revision_humana"]}**, motivos `{r["multiagente"]["razones_revision"]}`.', '',
        'El caso combina atención inicial y un resultado adverso; su cierre «no compensa ese descuido» contiene un contraste implícito. La regla histórica solo reconoce «pero», «aunque» o «sin embargo» y el score supera.56. Eso explica la ausencia de derivación en el código, sin justificar el acierto de la etiqueta. No se atribuye causalidad a palabras concretas ni atención sin otro experimento. No se añade una regla para resolver este ejemplo ya observado.', '',
        'La anotación original de P002 conserva los tags negation,mixed. Los tags no son exhaustivos: su análisis cualitativo como contraste implícito no modifica las rebanadas registradas. Un cambio de política necesitará nuevos datos de desarrollo y evaluación.', '',
        '## Fenómenos registrados', '', '| Fenómeno | n | Aciertos LR | Aciertos SVM | Aciertos BETO | N→P LR/SVM/BETO |', '|---|---:|---:|---:|---:|---|']
    for tag in ('negation', 'double_negation', 'contrast', 'implicit_contrast', 'mixed', 'irony', 'sarcasm', 'regionalism', 'typo', 'positive_words_negative'):
        slices = [report['models'][k]['slices'].get(tag) for k in ('lr', 'svm', 'beto')]
        if not all(slices): continue
        n = slices[0]['n']; correct = [sum(m['confusion_matrix'][i][i] for i in range(3)) for m in slices]
        lines.append(f'| {tag} | {n} | {correct[0]}/{n} | {correct[1]}/{n} | {correct[2]}/{n} | ' + '/'.join(str(m['negative_to_positive_count']) for m in slices) + ' |')
    lines += ['', 'Las rebanadas se solapan y algunas no incluyen las tres clases. Son tags de un asistente, sin validación de hablantes. Dos dobles negaciones, una ironía o tres typos no prueban robustez amplia.', '',
        '## Latencia y recursos', '', '| Modelo | p50 ms | p95 ms |', '|---|---:|---:|']
    for kind in ('lr', 'svm', 'beto'):
        latency = report['models'][kind]['latency']
        lines.append(f'| {names[kind]} | {latency["p50_ms"]:.2f} | {latency["p95_ms"]:.2f} |')
    lines += ['', f'Medición completa: {report["wall_seconds"]:.2f}s;44 solicitudes secuenciales por opción. Modelos precargados, llamada de app con análisis/traza; tokenización de auditoría posterior fuera del cronómetro. No mide arranque frío, concurrencia ni SLA. Entradas distintas de las evaluaciones anteriores: no interpretar la diferencia de latencia como optimización del modelo.', '',
        'La reconstrucción necesaria de los pesos V3 tardó213,85s, dentro de900s, una sola vez y sin selección. Los paquetes/decoder ASR se comprueban sin cargar pesos Whisper. Voz real, micrófono y navegador del portátil pendientes.', '',
        '## Errores completos por opción', '']
    for kind in ('lr', 'svm', 'beto'):
        lines += ['### ' + names[kind], '']
        for error in report['models'][kind]['errors']:
            lines.append(f'- `{error["id"]}` {error["intended_label"]}→{error["prediction"]}; revisión={error["human_review"]}. {error["text"]}')
        lines.append('')
    lines += ['El primer informe de predicciones permanece byte a byte congelado. Los ajustes posteriores de aceptación corrigen I/O CSV y presentación, con pruebas de regresión; no cambian modelos, etiquetas ni umbrales ni repiten esta medición como test intacto.', '',
        '[Respuestas y métricas](acceptance/2026-11-13-predictive.json) · [Modelos fijados antes de medir](acceptance/2026-11-13-model-bindings.json) · [Guía Windows y sustentación](SUSTENTACION_Y_ACEPTACION_2026-11-13.md).', '']
    output.write_text('\n'.join(lines), encoding='utf-8')
    if figure:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 3, figsize=(11, 3.9))
        for ax, kind in zip(axes, ('lr', 'svm', 'beto')):
            matrix = report['models'][kind]['metrics']['confusion_matrix']
            ax.imshow(matrix, cmap='Blues', vmin=0, vmax=12)
            ax.set(xticks=range(3), yticks=range(3), xticklabels=['negativo', 'neutral', 'positivo'], yticklabels=['negativo', 'neutral', 'positivo'], title=names[kind], xlabel='Predicción')
            for i in range(3):
                for j in range(3): ax.text(j, i, str(matrix[i][j]), ha='center', va='center', color='white' if matrix[i][j] > 6 else '#172554', fontsize=13)
        axes[0].set_ylabel('Intención sintética prevista')
        fig.suptitle('36 textos sintéticos · 12 por clase · fuentes de entrenamiento distintas', fontsize=12)
        fig.tight_layout(); fig.savefig(figure, dpi=170, bbox_inches='tight'); plt.close(fig)
    return {'status': 'SAVED_RESPONSES_AND_METRICS_VERIFIED', 'prediction_sha256': file_sha(report_path), 'new_predictions_executed': False}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path, default=Path('docs/acceptance/2026-11-13-predictive.json'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--figure', type=Path)
    args = parser.parse_args()
    print(json.dumps(analyze(args.report, args.output, args.figure)))
