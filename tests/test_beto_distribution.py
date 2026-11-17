"""Tiny I/O fixtures, never represented as trained BETO weights or inference."""
import hashlib
import json
import os
from pathlib import Path
import struct
import subprocess
import sys
from contextlib import nullcontext
import zipfile

import pytest

from traveler_agent import beto_distribution as dist
from traveler_agent.corpus import CorpusError, file_sha
from traveler_agent.labels import LABELS


def tensor_fixture():
    header = json.dumps({'fixture_not_real_beto': {'dtype': 'F32', 'shape': [1], 'data_offsets': [0, 4]}}).encode()
    return struct.pack('<Q', len(header)) + header + b'\x00' * 4


@pytest.fixture
def release(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    root = tmp_path / 'artifacts/source'; (root / 'checkpoint').mkdir(parents=True)
    for name in dist.FILES:
        (root / 'checkpoint' / name).write_bytes(('fixture-not-real-model:' + name).encode())
    base = root / dist.BASE_NAME
    monkeypatch.setattr(dist, 'WEIGHT_SHA256', file_sha(base))
    verified = {'model': dist.MODEL, 'revision': dist.REVISION, 'files': {
        name: {'sha256': file_sha(root / 'checkpoint' / name), 'bytes': (root / 'checkpoint' / name).stat().st_size} for name in dist.FILES}}
    (root / 'checkpoint/verified_checkpoint.json').write_bytes(json.dumps(verified).encode())
    (root / 'head.npz').write_bytes(b'fixture-not-real-head' * 2)
    (root / 'adaptation.safetensors').write_bytes(tensor_fixture())
    meta = {'schema': 'traveler-beto-supervised-v2', 'mode': 'full', 'model': dist.MODEL,
        'revision': dist.REVISION, 'classes': list(LABELS), 'human_validated': False, 'max_length': 256,
        'training': {'supervised': True, 'seed': 42, 'records_sha256': dist.RECORDS_SHA, 'head_epoch': 80, 'encoder_epoch': 1},
        'head_sha256': file_sha(root / 'head.npz'), 'adaptation_sha256': file_sha(root / 'adaptation.safetensors'),
        'checkpoint_manifest': verified}
    (root / 'model_manifest.json').write_bytes(json.dumps(meta).encode())
    monkeypatch.setattr(dist, 'MODEL_MANIFEST_SHA', file_sha(root / 'model_manifest.json'))
    archive = tmp_path / 'fixture.zip'; descriptor = tmp_path / 'fixture.json'
    value = dist.pack(root, archive, descriptor)
    return root, archive, descriptor, value


def test_overlay_excludes_base_and_installs_exact_bytes_without_training(release, monkeypatch):
    root, archive, descriptor, value = release
    with zipfile.ZipFile(archive) as stream:
        assert set(stream.namelist()) == dist.PAYLOAD and dist.BASE_NAME not in stream.namelist()
    monkeypatch.setattr(dist, 'checkpoint', lambda *args, **kwargs: pytest.fail('Unexpected download'))
    output = Path('artifacts/installed')
    report = dist.install(archive, descriptor, output)
    assert report['status'] == 'INSTALLED_VERIFIED_NO_TRAINING'
    assert report['base_source'] == 'REUSED_VERIFIED_LOCAL_BASE'
    for name in [*dist.PAYLOAD, dist.BASE_NAME]: assert (output / name).read_bytes() == (root / name).read_bytes()
    assert dist.artifact_available(output)
    with pytest.raises(CorpusError, match='already exists'): dist.install(archive, descriptor, output)


def test_corrupt_archive_never_creates_deployment(release):
    _, archive, descriptor, _ = release
    archive.write_bytes(archive.read_bytes() + b'tampered')
    with pytest.raises(CorpusError, match='SHA256'): dist.install(archive, descriptor, Path('artifacts/output'))
    assert not Path('artifacts/output').exists()


@pytest.mark.parametrize('extra', ['../escape', '/absolute', 'C:/escape', 'head.npz', 'HEAD.NPZ'])
def test_unsafe_or_duplicate_zip_is_rejected_even_with_matching_archive_hash(release, extra):
    _, archive, descriptor, value = release
    with (pytest.warns(UserWarning, match='Duplicate name') if extra == 'head.npz' else nullcontext()):
        with zipfile.ZipFile(archive, 'a') as stream: stream.writestr(extra, b'bad')
    value.update(archive_sha256=file_sha(archive), archive_bytes=archive.stat().st_size)
    descriptor.write_text(json.dumps(value))
    with pytest.raises(CorpusError, match='ZIP members'): dist.install(archive, descriptor, Path('artifacts/output'))
    assert not Path('artifacts/output').exists()


def test_missing_base_is_explicit_and_never_downloads_silently(release, monkeypatch):
    root, archive, descriptor, _ = release
    (root / dist.BASE_NAME).unlink()
    monkeypatch.setattr(dist, 'checkpoint', lambda *args, **kwargs: pytest.fail('Unexpected download'))
    with pytest.raises(CorpusError, match='BASE_CACHE_MISS'): dist.install(archive, descriptor, Path('artifacts/output'))


def test_install_target_must_remain_under_project_artifacts(release):
    _, archive, descriptor, _ = release
    with pytest.raises(CorpusError, match='inside artifacts'): dist.install(archive, descriptor, Path('outside/model'))


def test_manifest_alone_or_truncated_weights_do_not_enable_ui(release):
    root, _, _, _ = release
    assert dist.artifact_available(root)
    state = root / 'adaptation.safetensors'; original = state.read_bytes(); state.write_bytes(original[:-1])
    assert not dist.artifact_available(root)
    state.write_bytes(original)
    base = root / dist.BASE_NAME; base.write_bytes(b'incomplete')
    assert not dist.artifact_available(root)


def test_malformed_or_absent_model_is_not_available(tmp_path):
    assert not dist.artifact_available(None)
    assert not dist.artifact_available(tmp_path)


@pytest.mark.parametrize('value', [None, [], 'not-a-manifest', {'training': []}])
def test_invalid_json_shape_never_breaks_default_ui(tmp_path, value):
    (tmp_path / 'model_manifest.json').write_text(json.dumps(value))
    assert not dist.artifact_available(tmp_path)
    (tmp_path / 'model_manifest.json').write_text('{}')
    assert not dist.artifact_available(tmp_path)


def test_original_source_hash_and_recipe_cannot_be_relabelled(release):
    root, _, _, _ = release
    (root / 'head.npz').write_bytes(b'modified head')
    with pytest.raises(CorpusError, match='hash mismatch'): dist.validate_source(root)


def test_zip_symlink_is_rejected(release):
    _, archive, descriptor, value = release
    replacement = archive.with_name('symlink.zip')
    with zipfile.ZipFile(archive) as old, zipfile.ZipFile(replacement, 'w') as new:
        for entry in old.infolist():
            data = old.read(entry)
            if entry.filename == 'head.npz':
                entry.create_system = 3; entry.external_attr = 0o120777 << 16
            new.writestr(entry, data)
    value.update(archive_sha256=file_sha(replacement), archive_bytes=replacement.stat().st_size)
    descriptor.write_text(json.dumps(value))
    with pytest.raises(CorpusError, match='Unsafe'): dist.install(replacement, descriptor, Path('artifacts/output'))


def test_failed_payload_hash_leaves_no_partial_deployment_or_lock(release):
    _, archive, descriptor, value = release
    value['files']['head.npz']['sha256'] = '0' * 64
    descriptor.write_text(json.dumps(value))
    output = Path('artifacts/output')
    with pytest.raises(CorpusError, match='Extracted'): dist.install(archive, descriptor, output)
    assert not output.exists() and not output.with_name('output.install.lock').exists()


def test_manifest_only_cannot_offer_beto_in_actual_gradio_config(tmp_path):
    root = Path(__file__).resolve().parents[1]
    incomplete = tmp_path / 'artifacts/incomplete'; incomplete.mkdir(parents=True)
    (incomplete / 'model_manifest.json').write_text('{}')
    env = dict(os.environ, TRAVELER_BETO_ARTIFACT=str(incomplete), GRADIO_ANALYTICS_ENABLED='False',
               PYTHONPATH=os.pathsep.join((str(root), str(root / 'src'))))
    code = "import app,json; print(json.dumps({'choices':[v for _,v in app._MODEL_CHOICES], 'available':app._BETO_AVAILABLE}))"
    result = subprocess.run([sys.executable, '-c', code], cwd=tmp_path, env=env,
                            capture_output=True, text=True, check=True, timeout=45)
    data = json.loads(result.stdout.strip().splitlines()[-1])
    assert data == {'choices': ['lr', 'svm'], 'available': False}
