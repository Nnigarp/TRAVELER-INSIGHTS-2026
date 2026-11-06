"""Synthetic fixtures only; PILOTO-01 audit never loads external data."""
import csv
import json
import pytest

from traveler_agent import corpus_metadata_probe as metadata_probe
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
        id_column="ID", language_column="language", group_columns=("Sitio",),
        allowed_label_values=("0", "2"), allowed_language_values=("es",)
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
        "shared_group_keys_count":1,
        "shared_group_keys_by_column":{"Sitio":1},
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
                   language_column=None, group_columns=(),
                   allowed_label_values=("0", "2"))
    assert report["train"]["language_values_source_claims_UNVERIFIED"] is None
    assert report["train"]["label_values_raw_UNMAPPED"] == {"0":1,"2":1}
    assert report["train"]["license_and_redistribution"].startswith("UNVERIFIED")
    assert report["train"]["near_duplicate_semantic_audit"] == "NOT_RUN"


@pytest.mark.parametrize("option", ["label_column", "language_column"])
def test_review_text_cannot_be_selected_as_a_public_category(tmp_path, option):
    path = tmp_path / "private.csv"
    _csv(path, ["text"], [{"text": "SYNTHETIC PRIVATE REVIEW"}])
    with pytest.raises(ProbeError):
        inspect_csv(path, text_column="text", **{option: "text"})


def test_unapproved_category_cells_are_redacted_even_in_other_columns(tmp_path):
    path = tmp_path / "private.csv"
    secret = "SYNTHETIC_PRIVATE_CELL"
    _csv(path, ["text", "label", "lang"], [
        {"text": "Una visita", "label": secret, "lang": secret},
        {"text": "Otra visita", "label": "2", "lang": "es"},
    ])
    report, _ = inspect_csv(
        path, text_column="text", label_column="label", language_column="lang",
        allowed_label_values=("2",), allowed_language_values=("es",),
    )
    assert secret not in json.dumps(report)
    assert report["label_values_raw_UNMAPPED"] == {"2": 1}
    assert report["redacted_label_value_counts"] == [1]
    assert report["language_values_source_claims_UNVERIFIED"] == {"es": 1}
    assert report["redacted_language_value_counts"] == [1]
    default_report, _ = inspect_csv(path, text_column="text", label_column="label")
    assert default_report["label_values_raw_UNMAPPED"] == {}
    assert default_report["redacted_label_value_counts"] == [1, 1]


@pytest.mark.parametrize("other_author", ["B", ""])
def test_shared_hotel_detected_independently_of_author(tmp_path, other_author):
    columns = ["text", "hotel", "author"]
    train, test = tmp_path / "train.csv", tmp_path / "test.csv"
    _csv(train, columns, [{"text": "Primera visita", "hotel": "H", "author": "A"}])
    _csv(test, columns, [{"text": "Otra visita", "hotel": "H", "author": other_author}])
    result = probe(train, test=test, text_column="text", label_column=None,
                   id_column=None, language_column=None, group_columns=("hotel", "author"))
    assert result["overlap"]["shared_group_keys_count"] == 1
    assert result["overlap"]["shared_group_keys_by_column"] == {"hotel": 1, "author": 0}
    assert result["test"]["blank_group_values_by_column"]["author"] == (not other_author)


def test_equal_values_in_different_group_dimensions_do_not_overlap(tmp_path):
    columns = ["text", "hotel", "author"]
    train, test = tmp_path / "train.csv", tmp_path / "test.csv"
    _csv(train, columns, [{"text": "Una visita", "hotel": "X", "author": "Y"}])
    _csv(test, columns, [{"text": "Otra visita", "hotel": "Y", "author": "X"}])
    result = probe(train, test=test, text_column="text", label_column=None,
                   id_column=None, language_column=None, group_columns=("hotel", "author"))
    assert result["overlap"]["shared_group_keys_count"] == 0


def test_rows_without_text_still_contribute_to_metadata(tmp_path):
    path = tmp_path / "empty_text.csv"
    _csv(path, ["text", "label", "id", "hotel", "lang"], [
        {"text": "", "label": "", "id": "repeat", "hotel": "A", "lang": "es"},
        {"text": "Visita", "label": "2", "id": "repeat", "hotel": "B", "lang": "es"},
        {"text": " ", "label": "0", "id": "", "hotel": "", "lang": "en"},
    ])
    report, _ = inspect_csv(
        path, text_column="text", label_column="label", id_column="id",
        group_columns=("hotel",), language_column="lang",
        allowed_label_values=("0", "2"), allowed_language_values=("es", "en"),
    )
    assert report["rows"] == 3
    assert report["blank_text_rows"] == 2
    assert report["blank_label_rows"] == 1
    assert report["label_values_raw_UNMAPPED"] == {"0": 1, "2": 1}
    assert report["id_duplicates_extra_rows"] == 1
    assert report["blank_id_rows"] == 1
    assert report["distinct_group_keys_by_column"] == {"hotel": 2}
    assert report["blank_group_values_by_column"] == {"hotel": 1}
    assert report["language_values_source_claims_UNVERIFIED"] == {"en": 1, "es": 2}


def test_unterminated_quote_fails_without_echoing_review(tmp_path):
    path = tmp_path / "malformed.csv"
    path.write_text('text\n"SYNTHETIC PRIVATE REVIEW\n', encoding="utf-8")
    with pytest.raises(ProbeError) as error:
        inspect_csv(path, text_column="text")
    assert "SYNTHETIC PRIVATE REVIEW" not in str(error.value)


def test_cli_does_not_echo_private_filesystem_errors(monkeypatch, capsys):
    def failure(*args, **kwargs):
        raise PermissionError("SYNTHETIC_PRIVATE_PATH")
    monkeypatch.setattr(metadata_probe, "probe", failure)
    with pytest.raises(SystemExit) as exit_info:
        metadata_probe.main(["--train", "local.csv", "--text-column", "text"])
    output = capsys.readouterr()
    assert exit_info.value.code == 2
    assert output.out == ""
    assert "PermissionError" in output.err
    assert "SYNTHETIC_PRIVATE_PATH" not in output.err


def test_modified_input_is_not_certified_with_a_different_file_hash(tmp_path, monkeypatch):
    path = tmp_path / "changing.csv"
    _csv(path, ["text"], [{"text": "Visita"}])
    checksums = iter(["before", "after"])
    monkeypatch.setattr(metadata_probe, "_sha", lambda _: next(checksums))
    with pytest.raises(ProbeError, match="cambió"):
        inspect_csv(path, text_column="text")


def test_cli_allowlists_and_valid_multiline_utf8_bom(tmp_path, capsys):
    path = tmp_path / "utf8.csv"
    _csv(path, ["text", "label", "lang"], [
        {"text": 'Una visita\ncon "comillas"', "label": "Good", "lang": "en"},
    ])
    path.write_bytes(b"\xef\xbb\xbf" + path.read_bytes())
    args = ["--train", str(path), "--text-column", "text", "--label-column", "label",
            "--language-column", "lang", "--allow-label-value", "Good",
            "--allow-language-value", "en"]
    metadata_probe.main(args)
    output = capsys.readouterr().out
    report = json.loads(output)
    assert report["train"]["rows"] == 1
    assert report["train"]["label_values_raw_UNMAPPED"] == {"Good": 1}
    assert report["train"]["mapping"].startswith("UNRESOLVED")
    assert "Una visita" not in output
    metadata_probe.main(args)
    assert capsys.readouterr().out == output
