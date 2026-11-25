"""Browser acceptance against the REAL local Gradio/LR/SVM app; no model downloads.

Playwright/Chromium are test-only dependencies. Screenshots contain actual DEMO
classifications and no fabricated Qwen reports. Microphone, BETO weights and
real Ollama remain local Windows acceptance tasks.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]
WIDTHS = (390, 768, 1024, 1366, 1536, 1920)
LIGHT = "UTB Claro Premium"
DARK = "Caribe Nocturno"
TEXT = "El hotel era cómodo, pero el transporte nunca llegó y perdimos la excursión."


def read_record(page, element_id):
    # CodeMirror virtualizes long JSON; visible lines are not the full record.
    # Exercise the native download control and inspect its complete bytes.
    page.locator(element_id).hover()
    with page.expect_download(timeout=10000) as download:
        page.locator(element_id + ' a[download]').click()
    return json.loads(Path(download.value.path()).read_text(encoding='utf-8'))


def choose_model(page, label):
    page.locator('#ti-model input').click()
    page.get_by_role('option', name=label, exact=True).click()



def set_theme(page, name):
    page.get_by_role("radio", name=name, exact=True).check()
    expected = "caribe" if name == DARK else "utb"
    page.wait_for_function("name => document.documentElement.dataset.tiTheme === name", arg=expected)


def check_editor(page, check):
    editor = page.locator("#ti-review textarea")
    def measure(theme, state):
        colors = editor.evaluate("""el => {
          const css = getComputedStyle(el), placeholder = getComputedStyle(el, '::placeholder');
          const selection = getComputedStyle(el, '::selection');
          return [css.backgroundColor, css.webkitTextFillColor, css.caretColor,
                  placeholder.webkitTextFillColor, selection.backgroundColor, selection.color];
        }""")
        import re
        def luminance(color):
            channels = [int(v)/255 for v in re.findall(r"[0-9]+", color)[:3]]
            assert len(channels) == 3, color
            linear = [v/12.92 if v <= .04045 else ((v+.055)/1.055)**2.4 for v in channels]
            return sum(v*w for v, w in zip(linear, (.2126, .7152, .0722)))
        bg, fg, caret, placeholder, selected, highlight = map(luminance, colors)
        def contrast(a, b):
            return (max(a,b)+.05)/(min(a,b)+.05)
        check(f"{theme}_{state}_contrast", min(contrast(bg, c) for c in (fg, caret, placeholder)) >= 4.5
              and contrast(selected, highlight) >= 4.5 and ((bg > fg) == (theme == LIGHT)))
    for theme in (LIGHT, DARK):
        set_theme(page, theme)
        editor.evaluate("el => el.blur()")
        measure(theme, "idle")
        editor.hover()
        measure(theme, "hover")
        editor.focus()
        editor.press("a")
        editor.press("Control+a")
        measure(theme, "typing_selection")
        editor.evaluate("el => el.disabled = true")
        measure(theme, "disabled")
        editor.evaluate("el => el.disabled = false")
        editor.fill("")
    set_theme(page, LIGHT)


def check_viewports(page, context, output, report, check):
    """Use the effective CSS width and pixel density of each desktop zoom."""
    cdp = context.new_cdp_session(page)
    report["layout_matrix"] = []
    try:
        for zoom in (1, 1.25):
            for width in WIDTHS:
                css_width = round(width / zoom)
                height = 1000 if width > 390 else 844
                page.set_viewport_size({"width": css_width, "height": round(height / zoom)})
                cdp.send("Emulation.setDeviceMetricsOverride", {
                    "width": css_width, "height": round(height / zoom),
                    "deviceScaleFactor": zoom, "mobile": False,
                })
                page.wait_for_function(
                    "({width, zoom}) => innerWidth === width && devicePixelRatio === zoom",
                    arg={"width": css_width, "zoom": zoom},
                )
                page.evaluate("() => new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve)))")
                dimensions = page.locator("#ti-analysis-row").evaluate("""row => {
                    const rect = row.getBoundingClientRect();
                    const input = row.querySelector('#ti-input-pane').getBoundingClientRect();
                    const result = row.querySelector('#ti-result-pane').getBoundingClientRect();
                    const buttons = [...document.querySelectorAll('#ti-nav [role=tab]')];
                    return {
                        width: innerWidth, left: rect.left, right: document.documentElement.clientWidth - rect.right,
                        overflow: document.documentElement.scrollWidth > innerWidth + 1,
                        stacked: result.top > input.top + 1, inputWidth: input.width, resultWidth: result.width,
                        reportColumns: getComputedStyle(row.querySelector('.ti-report-grid')).gridTemplateColumns.split(' ').length,
                        visibleTabs: buttons.filter(b => b.getBoundingClientRect().width > 0).length,
                    };
                }""")
                key = f"{width}px_{round(zoom * 100)}percent"
                report["layout_matrix"].append({"screen": width, "zoom": zoom, **dimensions})
                check(key + "_symmetric_gutters", 0 <= dimensions["left"] <= max(49, (css_width - 1024) / 2 + 1)
                      and abs(dimensions["left"] - dimensions["right"]) <= 3)
                check(key + "_no_page_overflow", not dimensions["overflow"])
                check(key + "_layout", dimensions["stacked"] and abs(dimensions["resultWidth"] - dimensions["inputWidth"]) <= 2)
                if width == 390 and zoom == 1:
                    check("mobile_tabs_and_single_column", dimensions["visibleTabs"] == 3 and dimensions["reportColumns"] == 1)
                if zoom == 1 and width in (390, 1366, 1920):
                    page.screenshot(path=str(output / f"utb-{width}.png"), full_page=True)
        cdp.send("Emulation.clearDeviceMetricsOverride")
    finally:
        cdp.detach()


def run(output):
    from playwright.sync_api import sync_playwright
    with socket.socket() as probe:
        probe.bind(('127.0.0.1', 0))
        port = probe.getsockname()[1]
    address = f'http://127.0.0.1:{port}'
    env = {**os.environ, 'GRADIO_ANALYTICS_ENABLED': 'False', 'PYTHONPATH': str(ROOT / 'src')}
    command = ('import app; app.app.queue(default_concurrency_limit=1).launch('
               f'server_name="127.0.0.1",server_port={port},share=False,inbrowser=False,quiet=True,'
               '**({"theme":app.THEME,"css":app.UI_CSS} if app._GRADIO_MAJOR>=6 else {}))')
    report = {'scope': 'ACTUAL_GRADIO_BROWSER_AND_DEMO_LR_SVM_NOT_REAL_QWEN_BETO_OR_VOICE',
              'status': 'FAILED', 'checks': [], 'console_errors': [], 'external_requests': []}
    def check(name, condition):
        report['checks'].append({'name': name, 'passed': bool(condition)})
        if not condition:
            raise AssertionError(name)
    with (output / 'server.log').open('w', encoding='utf-8') as log:
        process = subprocess.Popen([sys.executable, '-u', '-c', command], cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        try:
            deadline = time.monotonic() + 45
            while True:
                try:
                    with urlopen(address + '/config', timeout=1) as response:
                        if response.status == 200:
                            break
                except OSError:
                    if process.poll() is not None or time.monotonic() > deadline:
                        raise RuntimeError('Local Gradio did not start; inspect server.log')
                    time.sleep(.2)
            with sync_playwright() as playwright:
                browser = playwright.chromium.launch(headless=True, executable_path=os.getenv('TRAVELER_BROWSER_EXECUTABLE'))
                context = browser.new_context(viewport={'width': 1440, 'height': 1100}, device_scale_factor=1,
                                              reduced_motion='reduce')
                page = context.new_page()
                page.on('pageerror', lambda e: report['console_errors'].append(str(e)))
                def route(request):
                    url = request.request.url
                    if url.startswith(('http://127.0.0.1:', 'blob:', 'data:')):
                        request.continue_()
                    else:
                        report['external_requests'].append(url)
                        request.abort()
                context.route('**/*', route)
                try:
                    # Gradio keeps live event/heartbeat requests; networkidle is
                    # not a readiness signal. Wait for the actual controls.
                    page.goto(address, wait_until='domcontentloaded')
                    page.locator('#ti-review textarea').wait_for(timeout=30000)
                    page.wait_for_function("document.documentElement.dataset.tiTheme === 'utb'")
                    check('three_native_tabs', page.get_by_role('tab').all_text_contents() == ['Análisis', 'Lotes', 'Agentes'])
                    check('default_premium_theme', page.get_by_role('radio', name=LIGHT, exact=True).is_checked())
                    check('empty_real_result', page.locator('#ti-result .ti-placeholder').count() == 1 and page.locator('#ti-result .ti-indicators').count() == 0)
                    check_editor(page, check)
                    page.reload(wait_until="domcontentloaded")
                    page.wait_for_function("document.documentElement.dataset.tiSubmitGuard")
                    page.screenshot(path=str(output / "analysis-desktop.png"), full_page=True)
                    page.locator("#ti-analyze").focus()
                    page.keyboard.press("Tab")
                    check("keyboard_focus_visible_on_clear", page.locator("#ti-clear").evaluate(
                        'el => el === document.activeElement && getComputedStyle(el).outlineStyle === "solid"'))
                    page.get_by_role('button', name='Mixta · transporte', exact=True).click()
                    page.wait_for_function("document.querySelector('#ti-review textarea').value.includes('perdimos la excursión')")
                    page.wait_for_function(f"document.querySelector('#ti-char-count').innerText.includes('{len(TEXT)} /')")
                    check('example_counter', str(len(TEXT)) in page.locator('#ti-char-count').inner_text())
                    config = page.request.get(address + "/config").json()
                    run_id = next(c["id"] for c in config["components"] if c["props"].get("elem_id") == "ti-analyze")
                    submit_id = next(d["id"] for d in config["dependencies"] if [run_id, "click"] in d["targets"])
                    submissions = []
                    def count_submit(request):
                        if request.method == "POST" and "/queue/join" in request.url and request.post_data_json.get("fn_index") == submit_id:
                            submissions.append(True)
                    page.on("request", count_submit)
                    check('example_does_not_infer', page.locator('#ti-result .ti-indicators').count() == 0)
                    page.locator('#ti-analyze').evaluate('button => { button.click(); button.click(); }')
                    page.locator('#ti-result .ti-indicators').wait_for(timeout=30000)
                    page.get_by_text('Detalles técnicos · JSON y trazabilidad', exact=True).click()
                    record = read_record(page, '#ti-detail-json')
                    report['browser_submissions'] = len(submissions)
                    check('duplicate_submit_guard', len(submissions) == 1)
                    check('actual_lr_not_qwen', record['modelo_utilizado'] == 'lr' and 'generacion_qwen' not in record)
                    check('actual_trace_ten_events', len(record['multiagente']['traza']) == 10)
                    page.get_by_text('Detalles técnicos · JSON y trazabilidad', exact=True).click()
                    grid = page.locator('#ti-result .ti-report-grid').evaluate('(el) => getComputedStyle(el).gridTemplateColumns')
                    check('desktop_report_two_columns', len(grid.split()) == 2)
                    check('desktop_no_page_overflow', page.evaluate('document.documentElement.scrollWidth <= innerWidth + 1'))
                    check("light_canvas_matches_shell", page.evaluate("""() => {
                        const backgrounds = ['html', 'body', 'gradio-app', '.gradio-container']
                          .map(selector => getComputedStyle(document.querySelector(selector)).backgroundColor);
                        return backgrounds.every(background => background === 'rgb(243, 247, 252)');
                    }"""))
                    set_theme(page, DARK)
                    check('theme_preserves_result', page.locator('#ti-review textarea').input_value() == TEXT
                          and page.locator('#ti-result .ti-indicators').count() == 1)
                    page.screenshot(path=str(output / 'caribe-desktop.png'), full_page=True)
                    set_theme(page, LIGHT)
                    check_viewports(page, context, output, report, check)
                    page.set_viewport_size({'width': 1366, 'height': 1024})
                    page.get_by_text('Detalles técnicos · JSON y trazabilidad', exact=True).click()
                    check('technical_json_fits', page.locator('.ti-technical').filter(visible=True).first.evaluate(
                        '(el) => el.scrollWidth <= el.clientWidth + 1'))
                    page.get_by_text('Detalles técnicos · JSON y trazabilidad', exact=True).click()
                    page.get_by_role('button', name='Explorar trazabilidad', exact=True).click()
                    page.get_by_role('tab', name='Agentes', exact=True).wait_for()
                    page.wait_for_function("document.querySelector('[role=tab][aria-selected=true]').textContent.trim() === 'Agentes'")
                    check('real_agent_states', page.locator('#ti-agents .ti-flow-event').count() == 10)
                    check('four_agents_and_subordinate_qwen', page.locator('#ti-agents .ti-flow-agent').count() == 4 and page.locator('#ti-agents .ti-qwen-service').count() == 1)
                    page.screenshot(path=str(output / 'agents-desktop.png'), full_page=True)
                    page.get_by_role('tab', name='Análisis', exact=True).click()
                    page.locator("#ti-dictate").click()
                    microphone = page.locator('#ti-audio-input button').filter(has_text="Grabar")
                    microphone.first.wait_for()
                    check('native_microphone_control', microphone.first.is_visible())
                    page.get_by_role('button', name='+ Audio', exact=True).click()
                    page.locator('#ti-audio-input input[type=file]').wait_for(state='attached')
                    check('native_audio_upload', page.locator('#ti-audio-input input[type=file]').count() > 0)
                    page.get_by_text("Idioma y traducción experimental", exact=True).click()
                    for width in (390, 1366):
                        page.set_viewport_size({"width": width, "height": 900})
                        for panel in ("ti-audio-panel", "ti-language-panel"):
                            check(f"{panel}_{width}_fits", page.locator("#"+panel).evaluate(
                                "(el) => el.scrollWidth <= el.clientWidth + 1 && el.getBoundingClientRect().right <= innerWidth"))
                    page.get_by_text("Idioma y traducción experimental", exact=True).click()
                    check('editable_transcript', page.locator('#ti-review textarea').is_editable())
                    page.screenshot(path=str(output / 'audio-desktop.png'), full_page=True)
                    page.get_by_role('tab', name='Análisis', exact=True).click()
                    choose_model(page, 'SVM lineal')
                    page.get_by_role('button', name='Analizar reseña', exact=True).click()
                    page.locator('#ti-result .ti-indicators').get_by_text('SVM', exact=True).wait_for(timeout=30000)
                    page.get_by_text('Detalles técnicos · JSON y trazabilidad', exact=True).click()
                    svm = read_record(page, '#ti-detail-json')
                    check('actual_svm', svm['modelo_utilizado'] == 'svm')
                    page.get_by_text('Detalles técnicos · JSON y trazabilidad', exact=True).click()
                    page.get_by_role('tab', name='Lotes', exact=True).click()
                    page.locator('#ti-batch-config').get_by_text('SVM lineal', exact=False).wait_for()
                    page.locator('#ti-batch-text textarea').fill(TEXT + '\nLa visita fue excelente.')
                    page.get_by_role('button', name='Analizar texto por lotes', exact=True).click()
                    # Gradio tables also keep hidden measurement cells.
                    page.get_by_text('demo-001', exact=True).filter(visible=True).first.wait_for(timeout=30000)
                    page.get_by_role('button', name='Registrar corrección', exact=True).click()
                    page.get_by_text('corregida', exact=True).filter(visible=True).first.wait_for(timeout=10000)
                    page.get_by_role('button', name='Preparar exportaciones', exact=True).click()
                    download_link = page.locator('a[download="analisis_demo.json"]')
                    with page.expect_download() as download:
                        download_link.click()
                    exported = json.loads(Path(download.value.path()).read_text(encoding='utf-8'))
                    check('real_batch_export_and_correction', len(exported['registros']) == 2 and exported['registros'][0]['estado_revision'] == 'corregida' and all(r['modelo_utilizado'] == 'svm' for r in exported['registros']))
                    with page.expect_download() as csv_download:
                        page.locator('a[download="analisis_demo.csv"]').click()
                    csv_rows = list(csv.DictReader(io.StringIO(Path(csv_download.value.path()).read_text(encoding='utf-8-sig'))))
                    check('real_csv_export', len(csv_rows) == 2 and all(r['modelo_utilizado'] == 'svm' for r in csv_rows))
                    check("batch_columns_fit_on_desktop", page.locator("#ti-batch-table .virtual-table-viewport").evaluate(
                        "(el) => el.scrollWidth <= el.clientWidth + 2"))
                    page.screenshot(path=str(output / 'lots-desktop.png'), full_page=True)
                    page.set_viewport_size({"width": 390, "height": 844})
                    check("batch_scroll_is_contained_on_mobile", page.locator("#ti-batch-table .virtual-table-viewport").evaluate(
                        """el => el.scrollWidth > el.clientWidth &&
                              el.getBoundingClientRect().right <= innerWidth &&
                              document.documentElement.scrollWidth <= innerWidth + 1"""))
                    page.set_viewport_size({"width": 1366, "height": 1024})
                    page.locator("#ti-batch-file input[type=file]").set_input_files({
                        "name": "resenas.csv", "mimeType": "text/csv",
                        "buffer": ("reseña\nLa visita en barco fue excelente.\n").encode("utf-8"),
                    })
                    page.locator('#ti-batch-file [aria-label="resenas.csv"]').wait_for()
                    page.get_by_role("button", name="Analizar CSV", exact=True).click()
                    page.get_by_text("La visita en barco fue excelente.", exact=True).filter(visible=True).first.wait_for()
                    check("csv_upload_follows_real_batch_path", page.get_by_text("La visita en barco fue excelente.", exact=True).filter(visible=True).count() >= 1)
                    page.get_by_role('tab', name='Análisis', exact=True).click()
                    # A report's trace button must restore that report, even
                    # after looking at a newer batch in the same session.
                    page.get_by_role('button', name='Explorar trazabilidad', exact=True).click()
                    page.locator('#ti-agents-tab').get_by_text('Análisis individual', exact=True).wait_for()
                    check('trace_button_restores_its_source', page.locator('#ti-agents .ti-flow-event').count() == 10)
                    page.get_by_role('tab', name='Análisis', exact=True).click()
                    page.locator('#ti-clear').click()
                    page.locator('#ti-result .ti-placeholder').wait_for()
                    page.wait_for_function("document.querySelector('#ti-char-count').textContent.includes('0 /')")
                    check('clear_removes_previous_result', page.locator('#ti-review textarea').input_value() == '')
                    page.get_by_role("button", name="Analizar reseña", exact=True).click()
                    page.locator("#ti-result [role=alert]").wait_for()
                    check("validation_alert_is_visible", page.locator("#ti-result [role=alert]").is_visible())
                    page.locator("#ti-review textarea").fill("El guía fue amable.")
                    page.locator("#ti-result .ti-placeholder").wait_for()
                    page.wait_for_function("document.querySelector('#ti-char-count').textContent.includes('19 /')")
                    check("editing_invalidates_report_and_updates_count", page.locator("#ti-result [role=alert]").count() == 0)
                    set_theme(page, DARK)
                    page.reload(wait_until="domcontentloaded")
                    page.wait_for_function("document.documentElement.dataset.tiTheme === 'caribe'")
                    page.wait_for_function("""() => [...document.querySelectorAll('#ti-theme input')].some(
                        input => input.value === 'Caribe Nocturno' && input.checked)""")
                    check("theme_persists_with_radio_synced", page.get_by_role("radio", name=DARK).is_checked())
                    check('no_browser_exceptions', not report['console_errors'])
                    report['status'] = 'PASSED'
                except Exception:
                    page.screenshot(path=str(output / 'failure.png'), full_page=True)
                    (output / 'failure-dom.html').write_text(page.content(), encoding='utf-8')
                    raise
                finally:
                    browser.close()
        except Exception as exc:
            report['failure_type'] = type(exc).__name__
            report['failure'] = str(exc)
        finally:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
    (output / 'browser-check.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    result = run(args.output_dir.resolve())
    print(json.dumps(result, ensure_ascii=False))
    raise SystemExit(0 if result['status'] == 'PASSED' else 1)
