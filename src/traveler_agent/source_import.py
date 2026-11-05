"""Offline inspect/import for explicitly approved dataset files.

No network, scraping, user identifiers or public dataset artifacts.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import os
from collections import Counter
from pathlib import Path
from .corpus import CorpusError, file_sha
from .source_adapters import (
    argilla_adapter,crawlfeeds_adapter,hamos_adapter
)
# write_private_jsonl intentionally implemented here to avoid spreading IO API.


def _write_private(path, records):
    if "private" not in Path(path).parts:
        raise CorpusError("Salida requiere directorio privado.")
    path=Path(path)
    if path.exists():
        raise CorpusError("Salida ya existe; no sobrescribir datasets auditados.")
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("x",encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record,ensure_ascii=False,sort_keys=True)+"\n")
    return file_sha(path)


def _iter_rows(path,source):
    if source=="argilla_tripadvisor":
        if path.suffix.lower()==".parquet":
            try:
                import pyarrow.parquet as pq
            except ImportError as exc:
                raise CorpusError("Instalar pyarrow opcional para leer Parquet.") from exc
            f=pq.ParquetFile(path)
            for batch in f.iter_batches(batch_size=512):
                yield from batch.to_pylist()
        elif path.suffix.lower()==".jsonl":
            # JSONL is intended for integration tests and explicitly exported local copies.
            with path.open(encoding="utf-8") as f:
                for line in f:
                    if line.strip():yield json.loads(line)
        else:raise CorpusError("Argilla requiere parquet o JSONL exportado.")
    elif source=="crawlfeeds_booking":
        if path.suffix.lower()!=".csv":raise CorpusError("CrawlFeeds requiere CSV.")
        with path.open(encoding="utf-8-sig",newline="") as f:
            yield from csv.DictReader(f)
    elif source=="hamos_v4":
        if path.suffix.lower()!=".jsonl":raise CorpusError("HAMOS requiere JSONL convertido con splits preservados.")
        with path.open(encoding="utf-8") as f:
            for line in f:
                if line.strip():yield json.loads(line)
    else:raise CorpusError("Fuente desconocida.")


def inspect_file(path,source):
    """Audit effective field names and rows WITHOUT printing text or predictions."""
    path=Path(path)
    if not path.is_file() or path.stat().st_size>250*1024*1024:
        raise CorpusError("Archivo ausente o mayor a 250 MiB.")
    n=0; schema=None
    for row in _iter_rows(path,source):
        if not isinstance(row,dict):raise CorpusError("Fila no es un objeto.")
        if schema is None:schema=sorted(row)
        if sorted(row)!=schema:raise CorpusError("El esquema varía entre registros.")
        n+=1
        if n>250000:raise CorpusError("Excede 250k registros.")
    return {"source":source,"rows":n,"schema":schema or [],
            "sha256":file_sha(path),"size_bytes":path.stat().st_size,
            "contains_private_text":True,
            "note":"Schema only; permissions must be reviewed separately."}


def import_file(path,source,approval,output,*,mapping=None):
    inspected=inspect_file(path,source)
    if source=="argilla_tripadvisor":
        data=argilla_adapter(_iter_rows(Path(path),source),Path(approval))
    elif source=="crawlfeeds_booking":
        data=crawlfeeds_adapter(_iter_rows(Path(path),source),Path(approval))
    else:
        if not mapping:raise CorpusError("HAMOS requiere mapping verificado desde archivos originales.")
        data=hamos_adapter(_iter_rows(Path(path),source),Path(approval),mapping=mapping)
    fingerprint=_write_private(output,data)
    return {"source":source,"status":"LOCAL_DATA_ONLY",
        "raw_file_sha256":inspected["sha256"],"processed_file_sha256":fingerprint,
        "n_source_rows":inspected["rows"],"n_prepared_rows":len(data),
        "tasks":dict(Counter(r["task"] for r in data)),
        "weak_label_counts":dict(Counter(str(r["weak_label"]) for r in data)),
        "language_counts":dict(Counter(r["language"] for r in data)),
        "warnings":[
            "Original platform rights must be independently verified.",
            "No annotations are automatically declared human gold.",
            "Non-identical paraphrases and undisclosed reviewer overlap remain.",
        ]}


def main(argv=None):
    parser=argparse.ArgumentParser(description="Offline source inspection/import")
    parser.add_argument("command",choices=("inspect","import"))
    parser.add_argument("--source",required=True,choices=("argilla_tripadvisor","hamos_v4","crawlfeeds_booking"))
    parser.add_argument("--file",required=True,type=Path)
    parser.add_argument("--approval",type=Path)
    parser.add_argument("--output",type=Path)
    parser.add_argument("--mapping",type=Path,help="HAMOS column-name map JSON")
    args=parser.parse_args(argv)
    try:
        if args.command=="inspect":result=inspect_file(args.file,args.source)
        else:
            if not args.approval or not args.output:raise CorpusError("Importación exige approval y output.")
            mapping=json.loads(args.mapping.read_text(encoding="utf-8")) if args.mapping else None
            result=import_file(args.file,args.source,args.approval,args.output,mapping=mapping)
    except (CorpusError, OSError, ValueError, KeyError, UnicodeError):
        parser.exit(2,"Source import blocked: check local format, approval and schema.\n")
    print(json.dumps(result,ensure_ascii=False,indent=2))
    return 0

if __name__=="__main__":
    raise SystemExit(main())
