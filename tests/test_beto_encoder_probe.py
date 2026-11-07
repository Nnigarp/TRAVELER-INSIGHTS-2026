from pathlib import Path
import json

import pytest

from traveler_agent import beto_encoder_probe as probe


def test_default_cache_miss_does_not_start_network(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(probe, "urlopen", lambda *a, **k: pytest.fail("Unexpected network"))
    with pytest.raises(probe.EncoderProbeError, match="CACHE_MISS"):
        probe.checkpoint(Path("artifacts/empty"), download=False)


def test_reject_unexpected_checkpoint_revision_and_tampered_cache(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    root = tmp_path / "artifacts/cache"
    root.mkdir(parents=True)
    files = {}
    for name in probe.FILES:
        (root / name).write_bytes(b"SYNTHETIC TEST FILE")
        files[name] = {"sha256": probe.file_sha(root / name), "bytes": (root / name).stat().st_size}
    monkeypatch.setattr(probe, "WEIGHT_SHA256", files["pytorch_model.bin"]["sha256"])
    info = {"model": probe.MODEL, "revision": probe.REVISION, "files": files}
    (root / "verified_checkpoint.json").write_text(json.dumps(info), encoding="utf-8")
    assert probe.checkpoint(root)["revision"] == probe.REVISION
    (root / "vocab.txt").write_bytes(b"TAMPERED")
    with pytest.raises(probe.EncoderProbeError, match="modificada"):
        probe.checkpoint(root)


def test_existing_report_and_public_cache_are_not_overwritten(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    existing = tmp_path / "artifacts/existing.json"
    existing.parent.mkdir()
    existing.write_text("previous smoke", encoding="utf-8")
    with pytest.raises(probe.EncoderProbeError):
        probe.run_probe(Path("artifacts/cache"), existing)
    assert existing.read_text() == "previous smoke"
    with pytest.raises(probe.EncoderProbeError):
        probe.checkpoint(Path("public/cache"))


def test_impossible_memory_reading_is_pending_not_a_ram_recommendation():
    pending = probe.choose_memory_measurement(416, 17, 100, [])
    assert pending["peak_rss_verified_mib"] is None
    assert pending["peak_rss_method"] == "PENDING_UNRELIABLE_TELEMETRY"
    checked = probe.choose_memory_measurement(416, 17, 803, [])
    assert checked["peak_rss_sampled_mib"] is None
    assert checked["peak_rss_verified_mib"] == 803
    assert "getrusage" in checked["peak_rss_method"]
