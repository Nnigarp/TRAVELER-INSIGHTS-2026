"""Gradio callback smoke test: no browser/server required."""
import json
from pathlib import Path
import app

def test_batch_ui_callbacks_and_exports():
    records, table, counts, dropdown, message = app.run_batch_lines(
        "El guía fue excelente 😍\nEl precio fue demasiado alto 😡"
    )
    assert len(records) == 2
    assert len(table) == 2
    assert sum(row[1] for row in counts) == 2
    corrected, table2, _, _, status = app.revise(
        records, "demo-001", "positivo", "Corrección verificada"
    )
    assert corrected[0]["etiqueta_corregida_humano"] == "positivo"
    assert records[0]["etiqueta_corregida_humano"] is None
    json_path, csv_path, export_status = app.export_files(corrected)
    assert Path(json_path).is_file()
    assert Path(csv_path).is_file()
    assert json.loads(Path(json_path).read_text(encoding="utf-8"))["tipo"] == "DEMO_NO_VALIDADO"

def test_batch_ui_validation_keeps_empty_state():
    records, _, _, _, message = app.run_batch_lines(" ")
    assert records == []
    assert "Error" in message
