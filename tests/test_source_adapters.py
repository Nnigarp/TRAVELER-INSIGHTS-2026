"""Source adapters use invented fixtures. No real corpus is downloaded in CI."""
import json
import pytest
from traveler_agent.corpus import CorpusError
from traveler_agent.source_adapters import (
    require_approval,argilla_adapter,crawlfeeds_adapter,hamos_adapter,cross_source_overlap
)

@pytest.fixture
def approve(tmp_path):
    def make(source):
        p=tmp_path / (source+".json")
        p.write_text(json.dumps({
            "source":source,"reuse_rights_reviewed":True,"privacy_reviewed":True,
            "license_reference":"SYNTHETIC_TEST_APPROVAL_ONLY",
            "original_content_rights_basis":"SYNTHETIC_TEST_APPROVAL_ONLY",
            "reviewer_id":"test_reviewer",
        }),encoding="utf-8")
        return p
    return make

def test_requires_explicit_rights_review(tmp_path,approve):
    p=approve("argilla_tripadvisor")
    d=json.loads(p.read_text())
    d["reuse_rights_reviewed"]=False
    p.write_text(json.dumps(d))
    with pytest.raises(CorpusError,match="derechos originales"):
        argilla_adapter([],p)

def test_argilla_weak_rating_not_gold(approve):
    a=approve("argilla_tripadvisor")
    rows=[
        {"id":"1","text":"An invented wonderful hotel stay","prediction":[{"label":"5","score":1.0}],"annotation":None},
        {"id":"2","text":"An invented okay hotel stay","prediction":[{"label":"3","score":1.0}],"annotation":None},
        {"id":"3","text":"An invented awful hotel stay","prediction":[{"label":"1","score":1.0}],"annotation":None},
    ]
    records=argilla_adapter(rows,a)
    assert [r["weak_label"] for r in records]==["positivo","neutral","negativo"]
    assert all(r["group_quality"]=="HOTEL_UNKNOWN" for r in records)
    assert all("NOT_HUMAN_GOLD" in r["label_origin"] for r in records)
    with pytest.raises(CorpusError,match="rating"):
        argilla_adapter([dict(rows[0],prediction=[{"label":"8"}])],a)
    with pytest.raises(CorpusError,match="anotaciones humanas"):
        argilla_adapter([dict(rows[0],annotation={"label":"positive"})],a)

def test_crawlfeeds_preserves_segments_no_global_label(approve):
    a=approve("crawlfeeds_booking")
    rows=[
        {"uniq_id":"1","hotel_name":"Fabricated Hotel","positive_review_text":"Staff was lovely",
         "negative_review_text":"Room was noisy","language":"en",
         "reviewer_name":"Sensitive Person","hotel_address":"Private location"},
    ]
    records=crawlfeeds_adapter(rows,a)
    assert records[0]["weak_label"] is None
    assert records[0]["task"]=="mixed_segments"
    assert records[0]["positive_segment"]=="Staff was lovely"
    assert records[0]["negative_segment"]=="Room was noisy"
    assert "Sensitive Person" not in json.dumps(records)
    assert "Private location" not in json.dumps(records)
    assert crawlfeeds_adapter([dict(rows[0],positive_review_text=r"\N",
                                   negative_review_text=r"\N")],a)==[]

def test_hamos_keeps_native_split_and_blocks_hotel_leakage(approve):
    a=approve("hamos_v4")
    mapping={"text":"sentence","hotel_id":"property","review_id":"uid",
             "quadruples":"opinions","split":"partition"}
    records=hamos_adapter([
        {"sentence":"Invented clean room", "property":"A", "uid":"1",
         "opinions":[{"aspect":"room","polarity":"positive"}],"partition":"train"},
        {"sentence":"Invented dirty bed","property":"B","uid":"2",
         "opinions":[{"aspect":"room","polarity":"negative"}],"partition":"test"},
    ],a,mapping=mapping)
    assert records[0]["official_split"]=="train"
    assert records[0]["task"]=="native_absa_quadruples"
    assert records[0]["weak_label"] is None
    with pytest.raises(CorpusError,match="fuga"):
        hamos_adapter([
            {"sentence":"Invented clean room", "property":"A", "uid":"1",
             "opinions":[],"partition":"train"},
            {"sentence":"Invented dirty bed", "property":"A", "uid":"2",
             "opinions":[],"partition":"test"},
        ],a,mapping=mapping)

def test_cross_source_duplicate_is_reported_without_text(approve):
    ar=argilla_adapter([{"id":"1","text":"An invented wonderful hotel stay",
            "prediction":[{"label":"5","score":1}],"annotation":None}],
        approve("argilla_tripadvisor"))
    cf=crawlfeeds_adapter([{"uniq_id":"2","hotel_name":"Hotel",
            "positive_review_text":"An invented wonderful hotel stay",
            "negative_review_text":r"\N","language":"en"}],
        approve("crawlfeeds_booking"))
    report=cross_source_overlap({"argilla":ar,"crawlfeeds":cf})
    assert len(report["exact_cross_source_pairs"])==1
    assert "wonderful" not in json.dumps(report)
