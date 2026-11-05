"""Allowlisted, bounded dataset download after independent rights approval.

NO scraping, and network disabled unless explicitly opted in. Only pre-audited
dataset repository files are eligible. Do not treat approval flags as a legal
determination; a human reviewer must establish the upstream rights basis.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import urllib.request
from pathlib import Path
from .corpus import CorpusError
from .source_adapters import require_approval

REGISTRY = {
    "argilla_tripadvisor": {
        "url": "https://huggingface.co/datasets/argilla/tripadvisor-hotel-reviews/resolve/533be43/data/train-00000-of-00001-0e99e58b23dccc25.parquet",
        "filename": "argilla_tripadvisor_533be43.parquet",
        "max_bytes": 25 * 1024 * 1024,
        "sha256": "3213f73c7f7d4fc3f778daae9f9fb06852339e52aea0be23b6349a93566e5d85",
        "license_declared": "CC-BY-NC-4.0",
    },
    "crawlfeeds_booking": {
        "url": (
            "https://huggingface.co/datasets/crawlfeeds/Booking-Hotel-Reviews-Dataset/"
            "resolve/f39ceeede7b9d8a824909ebe88a7723ad31fa6c5/"
            "Crawlfeeds_booking-italy-hotel-reviews__limit-100000_language-en_after-2025-06_reviewer_country-3-items_20260410_063921.csv"
        ),
        "filename": "crawlfeeds_booking_f39ceee.csv",
        "max_bytes": 6 * 1024 * 1024,
        "sha256": None,  # Hash must be recorded from first legitimate retrieval.
        "license_declared": "CC-BY-NC-4.0",
    },
    # HAMOS is intentionally omitted until official file URLs / mapping verified.
}


def download_approved(source: str, approval_path: Path, destination_dir: Path,
                      *, explicitly_allow_network: bool = False, opener=None) -> dict:
    if source not in REGISTRY:
        raise CorpusError("Fuente no autorizada para descarga fija. HAMOS requiere auditoría de archivos.")
    if not explicitly_allow_network:
        raise CorpusError("Red deshabilitada. Requiere autorización explícita --allow-network.")
    require_approval(approval_path, source)
    dest_dir = Path(destination_dir)
    if "private" not in dest_dir.parts:
        raise CorpusError("Descargas solo bajo una carpeta privada.")
    spec = REGISTRY[source]
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / spec["filename"]
    if dest.exists():
        raise CorpusError("Descarga ya existente: no sobrescribir.")
    partial = dest.with_suffix(dest.suffix + ".partial")
    if partial.exists():
        raise CorpusError("Existe descarga parcial. Revisar antes de reintentar.")
    h = hashlib.sha256()
    n = 0
    open_url = opener or urllib.request.urlopen
    try:
        with open_url(spec["url"], timeout=35) as response, partial.open("xb") as out:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk: break
                n += len(chunk)
                if n > spec["max_bytes"]:
                    raise CorpusError("Archivo excede el límite de tamaño publicado.")
                h.update(chunk)
                out.write(chunk)
        digest = h.hexdigest()
        if spec["sha256"] and digest != spec["sha256"]:
            raise CorpusError("SHA-256 no coincide con el archivo fijado del repositorio.")
        os.replace(partial, dest)
    except Exception:
        if partial.exists(): partial.unlink()
        raise
    manifest = {
        "source": source, "url": spec["url"], "filename": dest.name,
        "sha256": digest, "size_bytes": n,
        "declared_license": spec["license_declared"],
        "rights_reviewed_manually": True,
        "redistribution": "NOT_AUTOMATICALLY_AUTHORIZED",
    }
    dest.with_suffix(dest.suffix + ".manifest.json").write_text(
        json.dumps(manifest,ensure_ascii=False,indent=2),encoding="utf-8"
    )
    return {k:v for k,v in manifest.items() if k!="url"}


def main(argv=None):
    parser=argparse.ArgumentParser(description="Download allowlisted dataset only after verified rights")
    parser.add_argument("--source",required=True,choices=sorted(REGISTRY))
    parser.add_argument("--approval",type=Path,required=True)
    parser.add_argument("--destination",type=Path,default=Path("data/private/originales"))
    parser.add_argument("--allow-network",action="store_true")
    args=parser.parse_args(argv)
    try:
        result=download_approved(args.source,args.approval,args.destination,
                                 explicitly_allow_network=args.allow_network)
    except (CorpusError,OSError,ValueError,urllib.error.URLError):
        parser.exit(2,"Download refused/failed: inspect rights approval, URL and checksum.\n")
    print(json.dumps(result,ensure_ascii=False,indent=2))
    return 0


if __name__=="__main__":
    raise SystemExit(main())
