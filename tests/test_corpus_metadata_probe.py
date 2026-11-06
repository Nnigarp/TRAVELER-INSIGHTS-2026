"""Synthetic fixtures only; PILOTO-01 audit never loads external data."""
import csv
import json
import pytest

from traveler_agent.corpus_metadata_probe import ProbeError, inspect_csv, probe


def _csv(path, columns, rows):
    with path.open("w", encoding="utf-8", newline="") as stream:
        out = csv.DictWriter(stream, fieldnames=columns)
        out.writeheader()
        out.writerows(rows)


def test_local_aggregate_values_and_exact_duplicates_without_text_disclosure(tmp_path):
    path = tmp_path / "train.csv"
    cols = ["ID", "Comentario", "Sentimiento", "Sitio", "language"]
    _csv(path, cols, [
        {"ID":"1", "Comentario":"Buen lugar", "Sentimiento":"2",
         "Sitio":"a", "language":"es"},
        {"ID":"2", "Comentario":"  buen   lugar ", "Sentimiento":"0",
         "Sitio":"b", "language":"es"},
        {"ID":"2", "Comentario":"Pésima atención", "Sentimiento":"0",
         "Sitio":"a", "language":"unverified"},
    ])
    report, private = inspect_csv(
        path, text_column="Comentario", label_column="Sentimiento",
        id_column="ID", language_column="language", group_columns=("Sitio",)
    )
    assert report["rows"] == 3
    assert report["label_values_raw_UNMAPPED"] == {"0": 2, "2": 1}
    assert report["normalized_exact_text_duplicate_extra_rows"] == 1
    assert report["conflicting_raw_labels_same_normalized_text"] == 1
    assert report["id_duplicates_extra_rows"] == 1
    assert report["mapping"].startswith("UNRESOLVED")
    assert report["annotation_origin"] == "UNVERIFIED"
    assert "Buen lugar" not in json.dumps(report, ensure_ascii=False)
    assert "Pésima atención" not in json.dumps(report, ensure_ascii=False)
    assert len(private["fingerprints"]) == 2


def test_overlap_across_train_test_does_not_consume_hidden_labels(tmp_path):
    train, test = tmp_path / "train.csv", tmp_path / "test.csv"
    _csv(train, ["ID","Comentario","Sentimiento","Sitio"], [
        {"ID":"1","Comentario":"Tour tranquilo","Sentimiento":"1","Sitio":"A"},
        {"ID":"2","Comentario":"Visita buena","Sentimiento":"2","Sitio":"B"},
    ])
    _csv(test, ["ID","Comentario","Sitio"], [
        {"ID":"1","Comentario":"tour   tranquilo","Sitio":"A"},
        {"ID":"3","Comentario":"Lugar limpio","Sitio":"C"},
    ])
    report = probe(
        train, test=test, text_column="Comentario",
        label_column="Sentimiento", id_column="ID",
        language_column=None, group_columns=("Sitio",)
    )
    assert report["overlap"] == {
        "normalized_exact_text_shared_count":1,
        "shared_nonempty_ids_count":1,
        "shared_group_keys_count":1
    }
    assert report["test"]["label_values_raw_UNMAPPED"] is None
    assert report["admission"].startswith("BLOCKED")


def test_missing_column_and_duplicate_header_fail_closed(tmp_path):
    path = tmp_path / "a.csv"
    _csv(path, ["text"], [{"text":"Hola"}])
    with pytest.raises(ProbeError, match="Columna"):
        inspect_csv(path, text_column="Comentario")
    path.write_text("Comentario,Comentario\nabc,def\n", encoding="utf-8")
    with pytest.raises(ProbeError, match="encabezados"):
        inspect_csv(path, text_column="Comentario")


def test_malformed_row_does_not_echo_user_text(tmp_path):
    path = tmp_path / "a.csv"
    path.write_text("Comentario,Sentimiento\nDATO PRIVADO,0,EXTRA\n",encoding="utf-8")
    with pytest.raises(ProbeError) as err:
        inspect_csv(path, text_column="Comentario", label_column="Sentimiento")
    assert "DATO PRIVADO" not in str(err.value)


def test_no_automatic_language_mapping_or_permission(tmp_path):
    path = tmp_path / "a.csv"
    _csv(path, ["text","target"], [
        {"text":"Lugar excelente","target":"2"},
        {"text":"No me gustó","target":"0"},
    ])
    report = probe(path, test=None, text_column="text",
                   label_column="target", id_column=None,
                   language_column=None, group_columns=())
    assert report["train"]["language_values_source_claims_UNVERIFIED"] is None
    assert report["train"]["label_values_raw_UNMAPPED"] == {"0":1,"2":1}
    assert report["train"]["license_and_redistribution"].startswith("UNVERIFIED")
    assert report["train"]["near_duplicate_semantic_audit"] == "NOT_RUN"
