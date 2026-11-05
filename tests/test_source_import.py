"""Offline import and weak-label benchmark tests with fabricated records."""
import json
from pathlib import Path
import pytest
from traveler_agent.corpus import CorpusError
from traveler_agent.source_import import inspect_file,import_file
from traveler_agent.weak_rating_benchmark import run_rating_proxy_benchmark

@pytest.fixture
def approved(tmp_path):
    p=tmp_path/"approval.json"
    p.write_text(json.dumps({
        "source":"argilla_tripadvisor","reuse_rights_reviewed":True,
        "privacy_reviewed":True,"license_reference":"SYNTHETIC_TEST_ONLY",
        "original_content_rights_basis":"SYNTHETIC_TEST_ONLY",
        "reviewer_id":"test"
    }),encoding="utf-8")
    return p

@pytest.fixture
def argilla_fixture(tmp_path):
    p=tmp_path/"tiny_export.jsonl"
    labels=("1","3","5")
    with p.open("w",encoding="utf-8") as f:
        for i in range(180):
            rec={"id":str(i),"text":f"Fabricated hotel experience number {i} synthetic example",
                 "prediction":[{"label":labels[i%3],"score":1.0}],
                 "annotation":None}
            f.write(json.dumps(rec)+"\n")
    return p

def test_inspection_reports_schema_without_private_text(argilla_fixture):
    stats=inspect_file(argilla_fixture,"argilla_tripadvisor")
    assert stats["rows"]==180
    assert stats["schema"]==["annotation","id","prediction","text"]
    assert "Fabricated" not in json.dumps(stats)

def test_real_loader_requires_approval_and_private_output(tmp_path,argilla_fixture,approved):
    with pytest.raises(CorpusError,match="privado"):
        import_file(argilla_fixture,"argilla_tripadvisor",approved,tmp_path/"public"/"out.jsonl")
    output=tmp_path/"private"/"argilla.jsonl"
    stats=import_file(argilla_fixture,"argilla_tripadvisor",approved,output)
    assert stats["n_prepared_rows"]==180
    assert stats["tasks"]=={"global_weak_rating":180}
    assert stats["weak_label_counts"]=={"negativo":60,"neutral":60,"positivo":60}
    assert "Fabricated" not in json.dumps(stats)
    with pytest.raises(CorpusError,match="ya existe"):
        import_file(argilla_fixture,"argilla_tripadvisor",approved,output)

def test_benchmark_is_explicitly_rating_proxy_only(tmp_path,argilla_fixture,approved):
    from traveler_agent.source_import import import_file
    output=tmp_path/"private"/"argilla.jsonl"
    import_file(argilla_fixture,"argilla_tripadvisor",approved,output)
    report=run_rating_proxy_benchmark(output,tmp_path/"artifacts"/"rating")
    assert report["report_type"]=="WEAK_RATING_PROXY_NOT_HUMAN_SENTIMENT"
    assert "HOTEL_UNKNOWN" in report["split_grouping"]
    assert set(report["models"])=={"dummy","lr","svm"}
    assert sum(report["sizes"].values())==180
    assert report["models"]["svm"]["test"]["n"]>0
    assert report["runtime_versions"]["python"].startswith("3.")
    for name in ("dummy","lr","svm"):
        assert (tmp_path/"artifacts"/"rating"/(name+".joblib")).is_file()
        assert len(report["model_hashes"][name])==64
    assert "Fabricated" not in json.dumps(report)
    assert "Spanish evaluation" in " ".join(report["limitations"])
    with pytest.raises(CorpusError,match="sobrescribir"):
        run_rating_proxy_benchmark(output,tmp_path/"artifacts"/"rating")

def test_no_gold_or_cross_source_pooling(tmp_path,argilla_fixture,approved):
    path=tmp_path/"private"/"only_rating.jsonl"
    import_file(argilla_fixture,"argilla_tripadvisor",approved,path)
    lines=path.read_text(encoding="utf-8").splitlines()
    first=json.loads(lines[0])
    first["source"]="crawlfeeds_booking"
    lines[0]=json.dumps(first)
    altered=tmp_path/"private"/"mixed.jsonl"
    altered.write_text("\n".join(lines),encoding="utf-8")
    with pytest.raises(CorpusError,match="exclusivamente"):
        run_rating_proxy_benchmark(altered,tmp_path/"artifacts"/"mixed")
