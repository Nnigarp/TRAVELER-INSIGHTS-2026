"""Bounded allowlisted downloads: mock responses; never access internet in CI."""
import hashlib
import io
import json
from pathlib import Path
import pytest
from traveler_agent.corpus import CorpusError
from traveler_agent import source_download as download

@pytest.fixture
def approval(tmp_path):
    p=tmp_path/"approval.json"
    p.write_text(json.dumps({
        "source":"argilla_tripadvisor",
        "reuse_rights_reviewed":True,"privacy_reviewed":True,
        "license_reference":"SYNTHETIC_TEST_LICENSE",
        "original_content_rights_basis":"SYNTHETIC_TEST_PERMISSION",
        "reviewer_id":"test"
    }))
    return p

def test_explicit_download_authorization_gate(approval,tmp_path):
    with pytest.raises(CorpusError,match="Red deshabilitada"):
        download.download_approved("argilla_tripadvisor",approval,tmp_path/"private")
    with pytest.raises(CorpusError,match="carpeta privada"):
        download.download_approved("argilla_tripadvisor",approval,tmp_path/"public",
                                    explicitly_allow_network=True)

def test_checksum_and_manifest_with_mocked_response(monkeypatch,approval,tmp_path):
    content=b"fictional-only-parquet-test-bytes"
    monkeypatch.setattr(download,"REGISTRY",{
        "argilla_tripadvisor":{
            "url":"https://huggingface.co/fake-pinned-test",
            "filename":"mock.parquet","max_bytes":1000,
            "sha256":hashlib.sha256(content).hexdigest(),
            "license_declared":"SYNTHETIC_TEST",
        },
    })
    calls=[]
    def fake_open(url,timeout):
        calls.append(url)
        return io.BytesIO(content)
    dest=tmp_path/"private"/"raw"
    report=download.download_approved("argilla_tripadvisor",approval,dest,
             explicitly_allow_network=True,opener=fake_open)
    assert calls==["https://huggingface.co/fake-pinned-test"]
    assert (dest/"mock.parquet").read_bytes()==content
    assert report["sha256"]==hashlib.sha256(content).hexdigest()
    assert report["redistribution"]=="NOT_AUTOMATICALLY_AUTHORIZED"
    assert (dest/"mock.parquet.manifest.json").is_file()
    with pytest.raises(CorpusError,match="existente"):
        download.download_approved("argilla_tripadvisor",approval,dest,
             explicitly_allow_network=True,opener=fake_open)

def test_hash_mismatch_removes_partial(monkeypatch,approval,tmp_path):
    monkeypatch.setattr(download,"REGISTRY",{
        "argilla_tripadvisor":{
            "url":"https://huggingface.co/test",
            "filename":"mock.parquet","max_bytes":1000,"sha256":"0"*64,
            "license_declared":"synthetic",
        }
    })
    dest=tmp_path/"private"
    with pytest.raises(CorpusError,match="SHA-256"):
        download.download_approved("argilla_tripadvisor",approval,dest,
            explicitly_allow_network=True,
            opener=lambda *_args,**_kwargs:io.BytesIO(b"fictional"))
    assert not (dest/"mock.parquet").exists()
    assert not (dest/"mock.parquet.partial").exists()

def test_hamos_download_not_claimed_automated(approval,tmp_path):
    with pytest.raises(CorpusError,match="HAMOS"):
        download.download_approved("hamos_v4",approval,tmp_path/"private",
                                   explicitly_allow_network=True)
