"""Lossless distribution of the already evaluated PR28 V3 artifact.

No training, recipe changes, inference, network calls or optional encoder imports
on import. The large immutable base file is reused locally or downloaded only
with explicit authorization. The existing supervised V2 loader is unchanged.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import stat
import struct
import tempfile
import zipfile

from .beto_encoder_probe import MODEL, REVISION, WEIGHT_SHA256, FILES, checkpoint
from .corpus import CorpusError, file_sha
from .labels import LABELS

VERSION = 'beto-v3-pr28-seed42-overlay-v1'
MODEL_MANIFEST_SHA = 'c66453d041090bf6d10a9af10fdfbb8ab2d53b70a18f44e4439921a28914a7b7'
RECORDS_SHA = '1ca1521e303325bb314eb6d8c2d8e8b42e1510ac5352a541ab119bd403c16f9d'
BASE_NAME = 'checkpoint/pytorch_model.bin'
PAYLOAD = frozenset({'model_manifest.json', 'head.npz', 'adaptation.safetensors',
    'checkpoint/verified_checkpoint.json'} | {'checkpoint/' + name for name in FILES if name != 'pytorch_model.bin'})
DEFAULT_DESCRIPTOR = Path(__file__).resolve().parents[2] / 'docs/models/beto-v3-pr28-seed42-overlay-v1.json'


def _read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def _write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, ensure_ascii=False, sort_keys=True, indent=2)
        stream.write('\n')


def _child(root, name):
    path = root / name
    if path.is_symlink() or not path.resolve().is_relative_to(root.resolve()) or not path.is_file():
        raise CorpusError('Missing or externally linked model file: ' + name)
    return path


def validate_source(root):
    root = Path(root).resolve()
    if file_sha(_child(root, 'model_manifest.json')) != MODEL_MANIFEST_SHA:
        raise CorpusError('Not the recovered, pre-evaluation-bound PR28 artifact.')
    meta = _read(root / 'model_manifest.json')
    training = meta.get('training', {})
    if (meta.get('schema') != 'traveler-beto-supervised-v2' or meta.get('mode') != 'full'
            or meta.get('model') != MODEL or meta.get('revision') != REVISION
            or meta.get('max_length') != 256 or meta.get('classes') != list(LABELS)
            or meta.get('human_validated') is not False or training.get('seed') != 42
            or training.get('records_sha256') != RECORDS_SHA
            or training.get('head_epoch') != 80 or training.get('encoder_epoch') != 1):
        raise CorpusError('Recovered artifact differs from the registered V3 recipe.')
    expected = {'head.npz': meta['head_sha256'], 'adaptation.safetensors': meta['adaptation_sha256']}
    expected.update({'checkpoint/' + k: v['sha256'] for k, v in meta['checkpoint_manifest']['files'].items()})
    for name, digest in expected.items():
        if file_sha(_child(root, name)) != digest:
            raise CorpusError('Recovered model file hash mismatch: ' + name)
    if _read(_child(root, 'checkpoint/verified_checkpoint.json')) != meta['checkpoint_manifest']:
        raise CorpusError('Base manifest differs from the recovered artifact.')
    if meta['checkpoint_manifest']['files']['pytorch_model.bin']['sha256'] != WEIGHT_SHA256:
        raise CorpusError('Unexpected base checkpoint.')
    return meta


def pack(source, archive, descriptor):
    source, archive, descriptor = Path(source).resolve(), Path(archive), Path(descriptor)
    if archive.exists() or descriptor.exists():
        raise CorpusError('Preserve existing releases; use new output paths.')
    meta = validate_source(source)
    archive.parent.mkdir(parents=True, exist_ok=True)
    records = {name: {'sha256': file_sha(_child(source, name)), 'bytes': (source / name).stat().st_size} for name in sorted(PAYLOAD)}
    try:
        with zipfile.ZipFile(archive, 'x', compression=zipfile.ZIP_DEFLATED, compresslevel=1) as stream:
            for name in sorted(PAYLOAD):
                stream.write(source / name, arcname=name)
        release = {'schema': 'traveler-beto-overlay-v1', 'version': VERSION,
            'source_model_manifest_sha256': MODEL_MANIFEST_SHA, 'archive_name': archive.name,
            'archive_sha256': file_sha(archive), 'archive_bytes': archive.stat().st_size,
            'files': records, 'required_base': dict(meta['checkpoint_manifest']['files']['pytorch_model.bin']),
            'training': meta['training'], 'checkpoint_revision': REVISION,
            'recipe_changed': False, 'weights_quantized_or_retrained': False,
            'notice': 'Upstream model README and its license/corpus caveat are retained in checkpoint/README.md.'}
        _write(descriptor, release)
    except Exception:
        archive.unlink(missing_ok=True)
        raise
    return release


def _descriptor(path):
    release = _read(path)
    if (release.get('schema') != 'traveler-beto-overlay-v1' or release.get('version') != VERSION
            or release.get('source_model_manifest_sha256') != MODEL_MANIFEST_SHA
            or set(release.get('files', {})) != PAYLOAD
            or release.get('required_base', {}).get('sha256') != WEIGHT_SHA256
            or release.get('checkpoint_revision') != REVISION):
        raise CorpusError('Incompatible distribution descriptor.')
    for record in list(release['files'].values()) + [release['required_base']]:
        if (type(record.get('bytes')) is not int or not 0 < record['bytes'] <= 600_000_000
                or len(record.get('sha256', '')) != 64
                or any(c not in '0123456789abcdef' for c in record['sha256'])):
            raise CorpusError('Invalid file size or hash in descriptor.')
    if sum(r['bytes'] for r in release['files'].values()) > 600_000_000:
        raise CorpusError('Unexpected overlay size.')
    return release


def _base_weight(release, base_cache=None, download=False):
    expected = release['required_base']
    candidates = []
    if base_cache is not None:
        candidate = Path(base_cache)
        candidates.append(candidate if candidate.is_file() else candidate / 'pytorch_model.bin')
    root = Path.cwd() / 'artifacts'
    if root.exists():
        candidates += sorted(root.rglob('pytorch_model.bin'))
    for path in dict.fromkeys(candidates):
        if path.is_file() and path.stat().st_size == expected['bytes'] and file_sha(path) == expected['sha256']:
            return path, 'REUSED_VERIFIED_LOCAL_BASE'
    if not download:
        raise CorpusError('BASE_CACHE_MISS: no verified base found under artifacts; specify --base-cache or explicitly --download-base. No training is needed.')
    cache = Path('artifacts/beto-v3-official-base')
    checkpoint(cache, download=True)
    return cache / 'pytorch_model.bin', 'EXPLICIT_PINNED_BASE_DOWNLOAD_NO_TRAINING'


def _archive_entries(stream, release):
    entries = stream.infolist()
    names = [entry.filename for entry in entries]
    if len(names) != len(set(n.casefold() for n in names)) or set(names) != PAYLOAD:
        raise CorpusError('Duplicate, missing or unexpected ZIP members.')
    for entry in entries:
        name = entry.filename
        mode = entry.external_attr >> 16
        if (entry.is_dir() or '\\' in name or PurePosixPath(name).is_absolute()
                or '..' in PurePosixPath(name).parts or ':' in name
                or stat.S_ISLNK(mode) or entry.flag_bits & 1
                or entry.file_size != release['files'][name]['bytes']):
            raise CorpusError('Unsafe or inconsistent ZIP member.')
    return entries


def install(archive, descriptor, output, *, base_cache=None, download=False):
    archive, output = Path(archive).resolve(), Path(output).resolve()
    if not any(output.is_relative_to((Path.cwd() / p).resolve()) for p in ('artifacts', 'data/private')):
        raise CorpusError('Install inside artifacts or data/private of this checkout.')
    if output.exists():
        raise CorpusError('Destination already exists; preserve it or choose a new versioned destination.')
    release = _descriptor(descriptor)
    if archive.stat().st_size != release['archive_bytes'] or file_sha(archive) != release['archive_sha256']:
        raise CorpusError('Archive SHA256/size mismatch; no installation performed.')
    # Validate the entire central directory before selecting any destination.
    with zipfile.ZipFile(archive) as stream:
        _archive_entries(stream, release)
    base, source = _base_weight(release, base_cache, download)
    output.parent.mkdir(parents=True, exist_ok=True)
    required = sum(r['bytes'] for r in release['files'].values()) + release['required_base']['bytes']
    if shutil.disk_usage(output.parent).free < required + 64 * 1024 * 1024:
        raise CorpusError('Insufficient free disk for a complete portable artifact and reserve.')
    lock = output.with_name(output.name + '.install.lock')
    with lock.open('x', encoding='utf-8') as stream:
        stream.write(VERSION)
    try:
        with tempfile.TemporaryDirectory(prefix='beto-install-', dir=output.parent) as temporary:
            staged = Path(temporary) / 'model'; staged.mkdir()
            with zipfile.ZipFile(archive) as stream:
                for entry in _archive_entries(stream, release):
                    target = staged / entry.filename; target.parent.mkdir(parents=True, exist_ok=True)
                    with stream.open(entry) as src, target.open('xb') as dst:
                        shutil.copyfileobj(src, dst, 1024 * 1024)
                    if file_sha(target) != release['files'][entry.filename]['sha256']:
                        raise CorpusError('Extracted model file hash mismatch.')
            shutil.copyfile(base, staged / BASE_NAME)
            validate_source(staged)
            if output.exists(): raise CorpusError('Destination appeared during installation; refusing overwrite.')
            staged.rename(output)
    finally:
        lock.unlink(missing_ok=True)
    return {'status': 'INSTALLED_VERIFIED_NO_TRAINING', 'version': VERSION,
            'archive_sha256': release['archive_sha256'], 'model_manifest_sha256': MODEL_MANIFEST_SHA,
            'base_source': source, 'output': str(output), 'recipe_changed': False,
            'inference_checked': False}


def _complete_safetensors(path):
    try:
        with path.open('rb') as stream:
            size = path.stat().st_size
            raw = stream.read(8)
            if len(raw) != 8: return False
            length = struct.unpack('<Q', raw)[0]
            if not 2 <= length <= min(2 * 1024 * 1024, size - 8): return False
            header = json.loads(stream.read(length))
        offsets = [v['data_offsets'] for k, v in header.items() if k != '__metadata__']
        return bool(offsets) and all(len(v) == 2 and 0 <= v[0] <= v[1] <= size - 8 - length for v in offsets) and max(v[1] for v in offsets) == size - 8 - length
    except (OSError, ValueError, KeyError, TypeError, AttributeError, struct.error):
        return False


def artifact_available(path):
    """Fast startup presence/size gate, not a replacement for loader SHA checks."""
    if path is None: return False
    path = Path(path).resolve()
    try:
        if (path / 'model_manifest.json').stat().st_size > 65536: return False
        meta = _read(_child(path, 'model_manifest.json'))
        if (meta.get('schema') not in ('traveler-beto-supervised-v1', 'traveler-beto-supervised-v2')
                or meta.get('model') != MODEL or meta.get('revision') != REVISION
                or meta.get('classes') != list(LABELS) or meta.get('human_validated') is not False
                or meta.get('training', {}).get('supervised') is not True): return False
        if _child(path, 'head.npz').stat().st_size < 16: return False
        verified = _read(_child(path, 'checkpoint/verified_checkpoint.json'))
        if verified != meta.get('checkpoint_manifest') or set(verified['files']) != set(FILES): return False
        for name, record in verified['files'].items():
            if _child(path, 'checkpoint/' + name).stat().st_size != record['bytes']: return False
        if 'adaptation_sha256' in meta and not _complete_safetensors(_child(path, 'adaptation.safetensors')):
            return False
        return True
    except (OSError, ValueError, KeyError, TypeError, AttributeError, CorpusError):
        return False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    create = sub.add_parser('pack'); create.add_argument('--source', type=Path, required=True)
    create.add_argument('--archive', type=Path, required=True); create.add_argument('--descriptor', type=Path, required=True)
    setup = sub.add_parser('install'); setup.add_argument('--archive', type=Path, required=True)
    setup.add_argument('--descriptor', type=Path, default=DEFAULT_DESCRIPTOR)
    setup.add_argument('--output', type=Path, default=Path('artifacts/beto-v3-pr28-local'))
    setup.add_argument('--base-cache', type=Path); setup.add_argument('--download-base', action='store_true')
    args = parser.parse_args()
    if args.command == 'pack':
        result = pack(args.source, args.archive, args.descriptor)
        result = {k: result[k] for k in ('version', 'archive_name', 'archive_bytes', 'archive_sha256', 'source_model_manifest_sha256')}
    else:
        result = install(args.archive, args.descriptor, args.output, base_cache=args.base_cache, download=args.download_base)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__': main()
