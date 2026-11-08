"""Private coordinator adapter over the existing CSV corpus manager.

No network, generated reviews or model predictions. Consent/independence are
human declarations, not facts that software can authenticate.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from contextlib import contextmanager
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import random
import shutil
import tempfile
import uuid

from .corpus import (CorpusError, SOURCE_COLUMNS, REVIEW_COLUMNS, ANNOTATION_COLUMNS,
    ADJUDICATION_COLUMNS, POLARITIES, _read_csv, _write_csv, _require_identifier,
    file_sha, text_sha, parse_aspects, validate_sources, validate_reviews,
    validate_annotations, validate_adjudications, record_annotation, record_adjudication)
from .corpus_cli import run as corpus_command
from .corpus_review import (annotator_agreement, near_duplicate_candidates, validate_evidence,
                            validate_span_entries, record_evidence, SPAN_COLUMNS)
from .corpus_splits import prepare_splits

NOTICE_VERSION = "PILOTO01-CONSENT-1"
NOTICE = """Aportación voluntaria — Traveler Insights 2026

Puedes aportar una reseña original en español de una experiencia turística que
hayas vivido. No copies textos de plataformas ni uses generación automática.
Evita nombres de personas, teléfonos, correos y datos personales innecesarios.

Si aceptas, el equipo del proyecto podrá guardar tu texto de forma privada,
permitir su lectura a anotadores humanos para etiquetarlo y usarlo para entrenar
y evaluar clasificadores en este proyecto académico. Se comunicarán resultados
agregados; tu reseña y el registro de autorización no se publicarán por defecto.

El coordinador que te entrega este aviso custodia los archivos locales y gestiona
el acceso de los anotadores. El código seudónimo permite solicitar retirada por
el mismo canal privado, sin pedir nombre o contacto adicionales. Puedes retirar
la aportación antes del congelado del corpus. Después habrá que revisar también
copias y modelos ya producidos: no se promete borrado automático retroactivo.

Participar es voluntario. Puedes no aceptar o no enviar una reseña. Revisa las
finalidades y marca cada aceptación en el formulario; conserva tu código y copia
de este aviso para consultas o retirada. El coordinador debe confirmar contigo
la autorización efectiva; el formulario no verifica tu identidad o autoría.
"""

GUIDE_VERSION = "PILOTO01-ANNOTATION-1"
GUIDE = """Anota solo lo que comunica el texto, sin consultar modelos ni otra anotación.
positivo: evaluación global favorable. negativo: evaluación global desfavorable.
neutral: descripción sin evaluación polar global clara. desconocido: ambigüedad
o balance de opiniones sin predominio justificable; no lo fuerces a neutral.

Mixta es una marca separada: hay valoraciones opuestas sobre uno o varios
aspectos. Puede coexistir con positivo/negativo si hay predominio global; si no
puedes decidir el predominio, marca ambigua y desconocido. Neutral no equivale a
mezcla equilibrada. Negación y contraste se anotan por su sentido, no solo por
palabras clave; si no puedes determinar su presencia, déjala como incierta (null).
Esto no obliga a marcar ambiguo el sentimiento global. Conserva faltas y contexto.
Aspectos y spans son opcionales;
los offsets cuentan caracteres Unicode del texto original, inicio incluido y fin
excluido. No incluyas datos personales en comentarios. Guarda tu trabajo y,
al finalizar, declara que es una anotación humana independiente.
"""

LINK_COLUMNS = ("review_id", "author_id", "place_id", "related_ids_json", "text_sha256", "reviewed_by", "intake_checks")
INTAKE_CHECKS = "privacy;spanish;original_human;real_experience"
CONSENT_KEYS = {"schema", "author_id", "notice_version", "notice_sha256", "accepted_at",
                "annotation", "training", "evaluation", "aggregate_results",
                "original_authorship", "real_experience", "public_raw_text"}
JUDGMENT_KEYS = {"sentiment", "aspects", "spans", "mixed", "ambiguous", "negation", "contrast"}


def _utc():
    return datetime.now(timezone.utc).isoformat()


def _sha_text(text):
    import hashlib
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def private_path(path: Path) -> Path:
    root, target = Path.cwd().resolve() / "data/private", Path(path).resolve()
    if target == root or not target.is_relative_to(root):
        raise CorpusError("Usar una ruta resuelta dentro de data/private, desde la raíz del proyecto.")
    return target


def read_json(path):
    path = private_path(path)
    if path.stat().st_size > 4_000_000:
        raise CorpusError("Archivo privado demasiado grande.")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (UnicodeError, ValueError) as exc:
        raise CorpusError("JSON privado inválido; no se imprime su contenido.") from exc


def write_json(path, value, *, exclusive=True):
    path = private_path(path)
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    raw = (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode("utf-8")
    if exclusive:
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "wb") as stream: stream.write(raw)
    else:
        fd, temporary = tempfile.mkstemp(dir=path.parent, prefix=".human-", suffix=".json")
        try:
            with os.fdopen(fd, "wb") as stream: stream.write(raw)
            os.replace(temporary, path)
        finally:
            if os.path.exists(temporary): os.unlink(temporary)


@contextmanager
def coordinator_lock(root):
    lock = root / ".coordinator-lock"
    try:
        lock.mkdir(mode=0o700)
    except FileExistsError as exc:
        raise CorpusError("Coordinador ocupado o cierre interrumpido; revisar bloqueo local, no escribir en paralelo.") from exc
    try:
        yield
    finally:
        lock.rmdir()


def new_consent_form():
    return {"schema": "human-consent-v1", "author_id": "author_" + uuid.uuid4().hex,
                "notice_version": NOTICE_VERSION, "notice_sha256": _sha_text(NOTICE), "accepted_at": "",
                **{key: False for key in ("annotation", "training", "evaluation", "aggregate_results", "original_authorship", "real_experience", "public_raw_text")}}


def consent_template(output):
    template = new_consent_form()
    write_json(output, template)
    return {"status": "UNSIGNED_TEMPLATE_NOT_CONSENT", "author_id": template["author_id"], "notice_version": NOTICE_VERSION}


def validate_consent(value):
    if not isinstance(value, dict) or set(value) != CONSENT_KEYS or value["schema"] != "human-consent-v1":
        raise CorpusError("Esquema de consentimiento inválido; no añadir identidad o contactos.")
    _require_identifier(value["author_id"], "author_id")
    if value["notice_version"] != NOTICE_VERSION or value["notice_sha256"] != _sha_text(NOTICE):
        raise CorpusError("El consentimiento no corresponde al aviso versionado.")
    for key in ("annotation", "training", "evaluation", "aggregate_results", "original_authorship", "real_experience"):
        if value[key] is not True:
            raise CorpusError("Falta aceptación efectiva de finalidades, autoría original o experiencia real.")
    if value["public_raw_text"] is not False:
        raise CorpusError("Esta recogida no autoriza publicar reseñas o consentimientos.")
    try:
        accepted = datetime.fromisoformat(value["accepted_at"].replace("Z", "+00:00"))
        if accepted.utcoffset() is None: raise ValueError()
        if (accepted-datetime.now(timezone.utc)).total_seconds()>300: raise ValueError()
    except (AttributeError, TypeError, ValueError) as exc:
        raise CorpusError("Fecha de aceptación ISO con zona horaria requerida, sin fecha futura.") from exc
    return value


def validate_judgment(value, text):
    if not isinstance(value, dict) or set(value) != JUDGMENT_KEYS or value["sentiment"] not in POLARITIES:
        raise CorpusError("Dictamen incompleto o sentimiento no admitido.")
    if any(type(value[key]) is not bool for key in ("mixed", "ambiguous")):
        raise CorpusError("Las marcas humanas deben ser booleanos explícitos.")
    if any(value[key] is not None and type(value[key]) is not bool for key in ("negation","contrast")):
        raise CorpusError("Negación/contraste requieren true, false o null si son inciertos.")
    if value["ambiguous"] != (value["sentiment"] == "desconocido"):
        raise CorpusError("Ambigüedad requiere desconocido; desconocido requiere marcar ambigüedad.")
    if value["mixed"] and value["sentiment"] == "neutral":
        raise CorpusError("No convertir mezcla sin predominio en neutral; usar desconocido/ambigua o justificar predominio polar.")
    aspects = parse_aspects(json.dumps(value["aspects"], ensure_ascii=False))
    spans = validate_span_entries(text, value["spans"])
    if any(aspects.get(span["aspect"]) != span["polarity"] for span in spans):
        raise CorpusError("Evidencia y aspecto anotado no coinciden.")
    return value


def load_packet(path):
    packet = read_json(path)
    if (not isinstance(packet, dict) or set(packet) != {"schema", "task_id", "annotator_id", "guide_version", "guide", "items"}
            or packet["schema"] != "blind-human-task-v1" or packet["guide_version"] != GUIDE_VERSION
            or packet["guide"] != GUIDE):
        raise CorpusError("Paquete de anotación inválido o guía distinta.")
    for key in ("task_id", "annotator_id"): _require_identifier(packet[key], key)
    if not isinstance(packet["items"], list) or not 1 <= len(packet["items"]) <= 100:
        raise CorpusError("Paquete vacío o fuera de tamaño.")
    ids = set()
    for item in packet["items"]:
        if (not isinstance(item, dict) or set(item) != {"item_id", "text"}
                or not isinstance(item["text"], str) or not 3 <= len(item["text"]) <= 5000):
            raise CorpusError("El paquete solo admite ID opaco y texto.")
        _require_identifier(item["item_id"], "item_id")
        if item["item_id"] in ids: raise CorpusError("Item duplicado.")
        ids.add(item["item_id"])
    return packet


def _answers(packet_path, output):
    packet = load_packet(packet_path)
    if Path(output).exists():
        response = read_json(output)
        if (set(response) != {"schema", "task_id", "annotator_id", "packet_sha256", "answers", "sealed_at", "independence_attested"}
                or response["schema"] != "blind-human-response-v1" or response["task_id"] != packet["task_id"]
                or response["annotator_id"] != packet["annotator_id"] or response["packet_sha256"] != file_sha(private_path(packet_path))
                or not isinstance(response["answers"], dict)):
            raise CorpusError("Respuestas de otro paquete o esquema inválido.")
        texts = {r["item_id"]: r["text"] for r in packet["items"]}
        for key, value in response["answers"].items():
            if key not in texts: raise CorpusError("Respuesta de item ajeno.")
            validate_judgment(value, texts[key])
        if type(response["independence_attested"]) is not bool:
            raise CorpusError("Declaración de independencia inválida.")
        if response["sealed_at"] is None:
            if response["independence_attested"]: raise CorpusError("Borrador sin finalizar no acredita independencia.")
        else:
            try:
                timestamp=datetime.fromisoformat(response["sealed_at"])
                if timestamp.utcoffset() is None or response["independence_attested"] is not True: raise ValueError()
            except (TypeError,ValueError) as exc:
                raise CorpusError("Finalización de anotación inválida.") from exc
    else:
        response = {"schema": "blind-human-response-v1", "task_id": packet["task_id"], "annotator_id": packet["annotator_id"],
                    "packet_sha256": file_sha(private_path(packet_path)), "answers": {}, "sealed_at": None,
                    "independence_attested": False}
    return packet, response


def save_answer(packet_path, output, item_id, judgment):
    output=private_path(output)
    output.parent.mkdir(parents=True,exist_ok=True,mode=0o700)
    with coordinator_lock(output.parent):
        packet, response = _answers(packet_path, output)
        if response["sealed_at"] is not None: raise CorpusError("Respuestas finalizadas; no sobrescribir.")
        item = next((r for r in packet["items"] if r["item_id"] == item_id), None)
        if item is None: raise CorpusError("Item ajeno al paquete.")
        validate_judgment(judgment, item["text"])
        response["answers"][item_id] = judgment  # own draft is editable until explicit sealing
        write_json(output, response, exclusive=not output.exists())
    return {"status": "OWN_DRAFT_SAVED", "answered": len(response["answers"]), "assigned": len(packet["items"])}


def seal_answers(packet_path, output, *, independent):
    output=private_path(output)
    output.parent.mkdir(parents=True,exist_ok=True,mode=0o700)
    with coordinator_lock(output.parent):
        packet, response = _answers(packet_path, output)
        if not independent: raise CorpusError("Se requiere declaración humana de anotación independiente, sin modelos ni etiquetas ajenas.")
        if response["sealed_at"] is not None: raise CorpusError("Respuestas ya finalizadas.")
        if len(response["answers"]) != len(packet["items"]): raise CorpusError("Completar todos los items antes de finalizar.")
        response.update(sealed_at=_utc(), independence_attested=True)
        write_json(output, response, exclusive=False)
    return {"status": "HUMAN_INDEPENDENCE_DECLARED_NOT_AUTHENTICATED", "answers": len(response["answers"])}


class HumanIntake:
    def __init__(self, root=Path("data/private/piloto01-human-v1")):
        self.root = private_path(root)

    def init(self):
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        with coordinator_lock(self.root):
            corpus_command(["--root", str(self.root), "init"])
            links = self.root / "intake_links.csv"
            if not links.exists(): _write_csv(links, LINK_COLUMNS, [])
            notice = self.root / "participant-notice.json"
            if not notice.exists(): write_json(notice, {"version": NOTICE_VERSION, "text": NOTICE, "sha256": _sha_text(NOTICE)})
            elif read_json(notice)!={"version":NOTICE_VERSION,"text":NOTICE,"sha256":_sha_text(NOTICE)}:
                raise CorpusError("Aviso privado alterado o de otra versión; no reemplazar.")
        return {"status": "EMPTY_TEMPLATES_READY_NOT_CORPUS_ADMISSION", "notice_version": NOTICE_VERSION, "notice_sha256": _sha_text(NOTICE)}

    def _receipt(self, author):
        _require_identifier(author, "author_id")
        receipt_path=private_path(self.root / "permissions" / f"{author}.json")
        if not receipt_path.exists(): raise CorpusError("Consentimiento de autor no registrado.")
        record = read_json(receipt_path)
        _require_identifier(record.get("reviewed_by",""),"reviewer")
        proof = private_path(self.root / "permissions" / f"{_require_identifier(record['consent_id'], 'consent_id')}.proof.json")
        if record["author_id"] != author or record["proof_sha256"] != file_sha(proof):
            raise CorpusError("Prueba de consentimiento ausente o modificada.")
        value=validate_consent(read_json(proof))
        if value["author_id"]!=author: raise CorpusError("Autor de la prueba y registro no coinciden.")
        if record["status"] not in {"active", "withdrawn"}: raise CorpusError("Estado de consentimiento inválido.")
        return record

    def register_consent(self, consent_file, reviewed_by):
        _require_identifier(reviewed_by, "reviewer")
        value = validate_consent(read_json(consent_file))
        author = value["author_id"]
        with coordinator_lock(self.root):
            source_rows = _read_csv(self.root / "sources.csv", SOURCE_COLUMNS)
            receipt_path = self.root / "permissions" / f"{author}.json"
            if receipt_path.exists():
                record = self._receipt(author)
                if read_json(self.root/"permissions"/f"{record['consent_id']}.proof.json")!=value or record["status"] != "active":
                    raise CorpusError("Consentimiento previo distinto o retirado; no sobrescribir.")
            else:
                cid = "consent_" + uuid.uuid4().hex
                proof = private_path(self.root / "permissions" / f"{cid}.proof.json")
                write_json(proof,value)
                record = {"author_id": author, "consent_id": cid, "proof_sha256": file_sha(proof),
                          "reviewed_by": reviewed_by, "recorded_at": _utc(), "status": "active"}
                write_json(receipt_path, record)
            source = {"source_id": author, "source_type": "voluntary", "source_reference": "original_human_real_experience_es",
                      "reuse_basis": "consent", "permission_reference": record["consent_id"], "authorization_reviewed": "yes"}
            existing = next((r for r in source_rows if r["source_id"] == author), None)
            if existing is not None and existing != source: raise CorpusError("Fuente existente incompatible.")
            if existing is None: _write_csv(self.root / "sources.csv", SOURCE_COLUMNS, source_rows + [source])
        return {"status": "CONSENT_RECORDED_HUMAN_REVIEW_DECLARED", "author_id": author, "consent_id": record["consent_id"]}

    def _records(self):
        for name in ("sources.csv","reviews.csv","annotations.csv","adjudications.csv","intake_links.csv"):
            private_path(self.root/name)  # reject redirected files before any CSV read
        sources = _read_csv(self.root / "sources.csv", SOURCE_COLUMNS)
        reviews = _read_csv(self.root / "reviews.csv", REVIEW_COLUMNS)
        if sources: validate_sources(self.root / "sources.csv")
        if reviews: validate_reviews(self.root / "reviews.csv", self.root / "sources.csv")
        receipts = {s["source_id"]: self._receipt(s["source_id"]) for s in sources}
        for source in sources:
            if (source["source_type"]!="voluntary" or source["reuse_basis"]!="consent"
                    or source["permission_reference"]!=receipts[source["source_id"]]["consent_id"]):
                raise CorpusError("Fuente no corresponde al consentimiento original registrado.")
        links = _read_csv(self.root / "intake_links.csv", LINK_COLUMNS)
        if len({r["review_id"] for r in links}) != len(links) or {r["review_id"] for r in links} != {r["review_id"] for r in reviews}:
            raise CorpusError("Reseñas y trazabilidad incompletas; resolver escritura interrumpida antes de continuar.")
        by_id = {r["review_id"]: r for r in reviews}
        for row in links:
            raw = by_id[row["review_id"]]
            if (row["author_id"] != raw["source_id"] or row["place_id"] != raw["group_id"]
                    or row["text_sha256"] != text_sha(raw["text"])):
                raise CorpusError("Trazabilidad de autor/lugar/texto alterada.")
            _require_identifier(row["place_id"], "place_id")
            _require_identifier(row["reviewed_by"], "reviewer")
            if row["intake_checks"]!=INTAKE_CHECKS: raise CorpusError("Revisión de entrada incompleta.")
            related = json.loads(row["related_ids_json"])
            if not isinstance(related, list) or any(x not in by_id for x in related): raise CorpusError("Relaciones de textos inválidas.")
        annotations, _ = validate_annotations(self.root / "annotations.csv", set(by_id))
        adjudications, _ = validate_adjudications(self.root / "adjudications.csv", set(by_id), annotations)
        return sources, reviews, links, annotations, adjudications, receipts

    def contribute(self, text_file, author, place, reviewed_by, related=(), *, privacy_reviewed=False, spanish_verified=False, original_human=False, real_experience=False):
        if not all(x is True for x in (privacy_reviewed, spanish_verified, original_human, real_experience)):
            raise CorpusError("Requiere revisión de privacidad, español, autoría humana original y experiencia real; no inferirlas.")
        _require_identifier(reviewed_by,"reviewer")
        path = private_path(text_file)
        _require_identifier(place, "place_id")
        with coordinator_lock(self.root):
            _, reviews, links, _, _, _ = self._records()
            if self._receipt(author)["status"] != "active": raise CorpusError("Consentimiento retirado.")
            if any(rid not in {r["review_id"] for r in reviews} for rid in related): raise CorpusError("Reseña relacionada inexistente.")
            result = corpus_command(["--root", str(self.root), "contribute", "--text-file", str(path),
                "--source-id", author, "--group-id", place, "--language", "es", "--consent-confirmed"])
            rid = result["review_id"]
            new = next(r for r in _read_csv(self.root / "reviews.csv", REVIEW_COLUMNS) if r["review_id"] == rid)
            _write_csv(self.root / "intake_links.csv", LINK_COLUMNS, links + [{"review_id": rid,
                "author_id": author, "place_id": place, "related_ids_json": json.dumps(sorted(set(related))), "text_sha256": text_sha(new["text"]),
                "reviewed_by":reviewed_by,"intake_checks":INTAKE_CHECKS}])
        return {"status": "ORIGINAL_HUMAN_DECLARED_NOT_AUTHENTICATED", "review_id": rid}

    def link(self, left, right):
        with coordinator_lock(self.root):
            _, reviews, links, *_ = self._records()
            if left == right or not {left,right} <= {r["review_id"] for r in reviews}: raise CorpusError("Enlace de textos inválido.")
            row = next(r for r in links if r["review_id"] == left)
            row["related_ids_json"] = json.dumps(sorted(set(json.loads(row["related_ids_json"]) + [right])))
            _write_csv(self.root / "intake_links.csv", LINK_COLUMNS, links)
        return {"status": "RELATED_TEXTS_LINKED"}

    def withdraw(self, author):
        with coordinator_lock(self.root):
            record = self._receipt(author)
            record.update(status="withdrawn", withdrawn_at=_utc())
            write_json(self.root / "permissions" / f"{author}.json", record, exclusive=False)
        return {"status": "WITHDRAWN_FROM_FUTURE_TASKS_AND_SNAPSHOTS", "historical_copies": "coordinator must recall distributed packets and review existing snapshots/models"}

    def create_task(self, annotator, output, limit=25):
        _require_identifier(annotator, "annotator_id")
        output = private_path(output)
        if not 1 <= limit <= 100 or output.exists(): raise CorpusError("Tamaño de lote inválido (1–100) o destino existente.")
        with coordinator_lock(self.root):
            _, reviews, _, annotations, _, receipts = self._records()
            assignments = defaultdict(set)
            for ann in annotations: assignments[ann["review_id"]].add(ann["annotator_id"])
            for path in (self.root / "tasks").glob("*.index.json"):
                tid=read_json(path)["task_id"]
                _require_identifier(tid,"task_id")
                index=self._task_index(tid,load_packet(self.root/"tasks"/f"{tid}.packet.json"))
                if index["state"] == "active":
                    for entry in index["mapping"].values(): assignments[entry["review_id"]].add(index["annotator_id"])
            eligible = [r for r in reviews if receipts[r["source_id"]]["status"] == "active"
                        and annotator!=r["source_id"] and annotator not in assignments[r["review_id"]] and len(assignments[r["review_id"]]) < 2]
            eligible = sorted(eligible, key=lambda r:r["review_id"])[:limit]
            if not eligible: raise CorpusError("Sin reseñas pendientes para este anotador; dos plazas por reseña.")
            tid = "task_" + uuid.uuid4().hex
            random.Random(tid).shuffle(eligible)
            items, mapping = [], {}
            for row in eligible:
                iid = "item_" + uuid.uuid4().hex
                items.append({"item_id":iid,"text":row["text"]})
                mapping[iid] = {"review_id":row["review_id"], "text_sha256":text_sha(row["text"])}
            packet = {"schema":"blind-human-task-v1","task_id":tid,"annotator_id":annotator,
                      "guide_version":GUIDE_VERSION,"guide":GUIDE,"items":items}
            packet_path = self.root / "tasks" / f"{tid}.packet.json"
            write_json(packet_path, packet)
            index = {"task_id":tid,"annotator_id":annotator,"state":"active","packet_sha256":file_sha(packet_path),"mapping":mapping}
            write_json(self.root / "tasks" / f"{tid}.index.json", index)
            write_json(output, packet)
            if file_sha(output) != index["packet_sha256"]: raise CorpusError("Copia del paquete inconsistente.")
        return {"status":"BLIND_PACKET_READY_PRIVATE_DELIVERY_ONLY","task_id":tid,"items":len(items)}

    def _task_index(self,task_id,packet):
        index=read_json(self.root/"tasks"/f"{task_id}.index.json")
        if (index.get("task_id")!=packet["task_id"] or index.get("annotator_id")!=packet["annotator_id"]
                or index.get("state") not in {"active","cancelled"}
                or index.get("packet_sha256")!=file_sha(self.root/"tasks"/f"{task_id}.packet.json")
                or not isinstance(index.get("mapping"),dict)
                or set(index["mapping"])!={r["item_id"] for r in packet["items"]}):
            raise CorpusError("Asignación y paquete no coinciden; no atribuir dictamen a otra persona.")
        review_ids=[]
        for item in packet["items"]:
            entry=index["mapping"][item["item_id"]]
            if not isinstance(entry,dict) or set(entry)!={"review_id","text_sha256"} or entry["text_sha256"]!=text_sha(item["text"]):
                raise CorpusError("Texto del paquete no corresponde a su reseña asignada.")
            _require_identifier(entry["review_id"],"review_id")
            review_ids.append(entry["review_id"])
        if len(set(review_ids))!=len(review_ids): raise CorpusError("Una reseña repetida dentro del paquete no constituye otro dictamen.")
        return index

    def cancel_task(self, task_id):
        _require_identifier(task_id,"task_id")
        with coordinator_lock(self.root):
            path=self.root/"tasks"/f"{task_id}.index.json"
            index=self._task_index(task_id,load_packet(self.root/"tasks"/f"{task_id}.packet.json"))
            if (self.root/"received"/f"{task_id}.json").exists(): raise CorpusError("Paquete recibido; completar importación en vez de cancelar.")
            index["state"]="cancelled"
            write_json(path,index,exclusive=False)
        return {"status":"TASK_CANCELLED_NOT_RAW_TEXT_ERASURE"}

    def import_answers(self, response_file):
        response=read_json(response_file)
        tid=_require_identifier(response.get("task_id",""),"task_id")
        with coordinator_lock(self.root):
            packet_path=self.root/"tasks"/f"{tid}.packet.json"
            packet,checked=_answers(packet_path,response_file)
            index=self._task_index(tid,packet)
            if (index["state"]!="active" or checked["independence_attested"] is not True or not checked["sealed_at"]
                    or checked["packet_sha256"]!=index["packet_sha256"] or len(checked["answers"])!=len(packet["items"])):
                raise CorpusError("Respuestas incompletas, no finalizadas, canceladas o paquete alterado.")
            _,reviews,_,annotations,_,receipts=self._records()
            by_id={r["review_id"]:r for r in reviews}
            existing={(r["review_id"],r["annotator_id"]):r for r in annotations}
            planned=[]
            for iid,judgment in checked["answers"].items():
                entry=index["mapping"][iid]
                row=by_id[entry["review_id"]]
                if receipts[row["source_id"]]["status"]!="active" or text_sha(row["text"])!=entry["text_sha256"]:
                    raise CorpusError("Consentimiento retirado o texto modificado tras asignación; no importar.")
                canonical={"review_id":row["review_id"],"annotator_id":index["annotator_id"],"sentiment":judgment["sentiment"],
                           "aspects_json":json.dumps(judgment["aspects"],ensure_ascii=False,sort_keys=True)}
                old=existing.get((row["review_id"],index["annotator_id"]))
                if old is not None and old!=canonical: raise CorpusError("No sobrescribir anotación previa distinta.")
                planned.append((row,judgment,old))
            saved=self.root/"received"/f"{tid}.json"
            if saved.exists():
                if read_json(saved)!=checked: raise CorpusError("Otra versión de respuestas ya recibida; no sobrescribir.")
            else:
                write_json(saved,checked)
            # Complete packet validated before any canonical write. Retrying the
            # same sealed file is idempotent/resumes interrupted per-row writes.
            for row,judgment,old in planned:
                if old is None:
                    record_annotation(self.root/"annotations.csv",set(by_id),review_id=row["review_id"],
                                      annotator_id=index["annotator_id"],sentiment=judgment["sentiment"],aspects=judgment["aspects"])
                if judgment["spans"]:
                    evidence=self.root/"evidence_spans.csv"
                    spans=_read_csv(evidence,SPAN_COLUMNS) if evidence.exists() else []
                    if not any(x["review_id"]==row["review_id"] and x["annotator_id"]==index["annotator_id"] for x in spans):
                        record_evidence(evidence,row["review_id"],index["annotator_id"],row["text"],judgment["spans"],judgment["aspects"])
        return {"status":"SEALED_HUMAN_ANNOTATIONS_IMPORTED","annotations":len(planned)}

    def _verified_annotations(self, reviews, annotations):
        by_id={r["review_id"]:r for r in reviews}
        proof={}
        for path in (self.root/"received").glob("*.json"):
            response=read_json(path)
            tid=_require_identifier(response.get("task_id",""),"task_id")
            packet,checked=_answers(self.root/"tasks"/f"{tid}.packet.json",path)
            index=self._task_index(tid,packet)
            if index["state"]!="active" or checked["independence_attested"] is not True or not checked["sealed_at"] or checked["packet_sha256"]!=index["packet_sha256"]:
                raise CorpusError("Prueba de independencia/paquete inválida.")
            for iid,j in checked["answers"].items():
                entry=index["mapping"][iid]; rid=entry["review_id"]
                if text_sha(by_id[rid]["text"])!=entry["text_sha256"]: raise CorpusError("Texto cambiado tras anotación.")
                key=(rid,index["annotator_id"])
                if key in proof: raise CorpusError("Dictamen repetido en paquetes distintos.")
                proof[key]=j
        for row in annotations:
            key=(row["review_id"],row["annotator_id"])
            if key not in proof or row["sentiment"]!=proof[key]["sentiment"] or json.loads(row["aspects_json"])!=proof[key]["aspects"]:
                raise CorpusError("Anotación canónica sin prueba de paquete humano independiente.")
        if set(proof)!={(r["review_id"],r["annotator_id"]) for r in annotations}:
            raise CorpusError("Importación incompleta; reimportar el mismo archivo finalizado.")
        return proof

    def adjudication_view(self, review_id):
        _,reviews,_,annotations,_,receipts=self._records()
        proof=self._verified_annotations(reviews,annotations)
        row=next((r for r in reviews if r["review_id"]==review_id),None)
        persons=sorted(a for rid,a in proof if rid==review_id)
        if row is None or receipts[row["source_id"]]["status"]!="active" or len(persons)!=2:
            raise CorpusError("Adjudicación exige reseña activa y dos dictámenes finalizados distintos.")
        return {"review_id":review_id,"text":row["text"],"annotations":[{"annotator_id":a,**proof[(review_id,a)]} for a in persons]}

    def adjudicate(self, review_id, adjudicator, judgment, reason):
        _require_identifier(adjudicator,"adjudicator_id")
        with coordinator_lock(self.root):
            view=self.adjudication_view(review_id)
            if adjudicator in {r["annotator_id"] for r in view["annotations"]}: raise CorpusError("El adjudicador debe ser una tercera persona/código, no un anotador del par.")
            validate_judgment(judgment,view["text"])
            _,reviews,_,annotations,finals,_=self._records()
            if not isinstance(reason,str) or not 1<=len(reason.strip())<=500: raise CorpusError("Justificación humana de 1–500 caracteres requerida.")
            detail={"review_id":review_id,"adjudicator_id":adjudicator,"judgment":judgment,"reason":reason.strip()}
            path=self.root/"decisions"/f"{review_id}.json"
            if path.exists() and read_json(path)!=detail: raise CorpusError("Adjudicación previa distinta; no sobrescribir.")
            if not path.exists(): write_json(path,detail)
            old=next((r for r in finals if r["review_id"]==review_id),None)
            if old is None:
                record_adjudication(self.root/"adjudications.csv",review_id,judgment["sentiment"],judgment["aspects"],adjudicator,reason,reviews,annotations)
            elif (old["adjudicator_id"]!=adjudicator or old["sentiment"]!=judgment["sentiment"] or json.loads(old["aspects_json"])!=judgment["aspects"] or old["rationale"]!=reason.strip()):
                raise CorpusError("Adjudicación canónica distinta.")
        return {"status":"TWO_HUMAN_ANNOTATIONS_ADJUDICATED","review_id":review_id}

    def _verified_decisions(self,reviews,annotations,adjudications):
        proof=self._verified_annotations(reviews,annotations)
        by_id={r["review_id"]:r for r in reviews}
        result={}
        for row in adjudications:
            rid=row["review_id"]
            persons={person for review,person in proof if review==rid}
            if len(persons)!=2 or row["adjudicator_id"] in persons:
                raise CorpusError("Dos dictámenes y tercera adjudicación son obligatorios.")
            detail=read_json(self.root/"decisions"/f"{rid}.json")
            validate_judgment(detail["judgment"],by_id[rid]["text"])
            if (detail["review_id"]!=rid or detail["adjudicator_id"]!=row["adjudicator_id"]
                    or detail["reason"]!=row["rationale"] or detail["judgment"]["sentiment"]!=row["sentiment"]
                    or detail["judgment"]["aspects"]!=json.loads(row["aspects_json"])):
                raise CorpusError("Adjudicación y metadatos humanos no coinciden.")
            result[rid]=detail
        validate_evidence(self.root/"evidence_spans.csv",reviews,annotations)
        return proof,result

    def status(self):
        with coordinator_lock(self.root):
            _,reviews,links,annotations,adjudications,receipts=self._records()
            proof,decisions=self._verified_decisions(reviews,annotations,adjudications)
            active=[r for r in reviews if receipts[r["source_id"]]["status"]=="active"]
            ids={r["review_id"] for r in active}
            active_annotations=[r for r in annotations if r["review_id"] in ids]
            per_review=Counter(r["review_id"] for r in active_annotations)
            agreement=annotator_agreement(active_annotations)
            people=sorted({r["annotator_id"] for r in active_annotations})
            by_pair={(r["review_id"],r["annotator_id"]):r["sentiment"] for r in active_annotations}
            raw=[]
            for i,left in enumerate(people):
                for right in people[i+1:]:
                    common=[rid for rid in ids if (rid,left) in by_pair and (rid,right) in by_pair]
                    matched=sum(by_pair[(rid,left)]==by_pair[(rid,right)] for rid in common)
                    raw.append({"annotators":[left,right],"shared_reviews":len(common),"agreements":matched,
                                "raw_agreement":matched/len(common) if common else None})
            groups,duplicates=connected_groups(active,links)
            final=[row for rid,row in decisions.items() if rid in ids]
            return {"status":"HUMAN_COLLECTION_WORKFLOW_NOT_MODEL_EVALUATION", "active_reviews":len(active),
                    "withdrawn_reviews":len(reviews)-len(active),"registered_consents":len(receipts),
                    "active_consents":sum(r["status"]=="active" for r in receipts.values()),
                    "annotation_rows":len(active_annotations),"reviews_with_two_verified_annotations":sum(n==2 for n in per_review.values()),
                    "adjudicated_reviews":len(final),"adjudicated_label_counts":dict(Counter(r["judgment"]["sentiment"] for r in final)),
                    "mixed_reviews":sum(r["judgment"]["mixed"] for r in final),"ambiguous_reviews":sum(r["judgment"]["ambiguous"] for r in final),
                    "human_phenomena_counts":{key:sum(r["judgment"][key] is True for r in final) for key in ("negation","contrast")},
                    "human_phenomena_uncertain":{key:sum(r["judgment"][key] is None for r in final) for key in ("negation","contrast")},
                    "agreement":agreement,"raw_agreement":raw,"independence_components":len(set(groups.values())),
                    "near_duplicate_candidate_pairs":len(duplicates["candidates"]),"models_trained_by_this_tool":False,
                    "provenance_limit":"Human authorship, real experience, permission and distinct people are reviewed declarations, not automatically authenticated."}

    def freeze(self,output,reviewed_by,*,links_reviewed=False,seed=42):
        """Export a private component-grouped snapshot compatible with old runner."""
        _require_identifier(reviewed_by,"reviewer")
        output=private_path(output)
        if not links_reviewed: raise CorpusError("Revisar autores, lugares y relaciones semánticas antes del congelado.")
        if output.exists(): raise CorpusError("Snapshot ya existente; no sobrescribir.")
        with coordinator_lock(self.root):
            input_hashes={name:file_sha(private_path(self.root/name)) for name in ("sources.csv","reviews.csv","annotations.csv","adjudications.csv","intake_links.csv")}
            sources,reviews,links,annotations,adjudications,receipts=self._records()
            proof,decisions=self._verified_decisions(reviews,annotations,adjudications)
            active=[r for r in reviews if receipts[r["source_id"]]["status"]=="active"]
            ids={r["review_id"] for r in active}
            if not ids: raise CorpusError("Sin reseñas humanas activas; no preparar benchmark ni inventar resultados.")
            if any(sum(key[0]==rid for key in proof)!=2 or rid not in decisions for rid in ids):
                raise CorpusError("Cada reseña requiere dos anotaciones verificadas y adjudicación explícita, también las ambiguas.")
            groups,duplicates=connected_groups(active,links)
            active_sources=[r for r in sources if r["source_id"] in {r["source_id"] for r in active}]
            output.parent.mkdir(parents=True,exist_ok=True,mode=0o700)
            staging=Path(tempfile.mkdtemp(prefix=".human-freeze-",dir=output.parent))
            try:
                corpus=staging/"corpus"; corpus.mkdir()
                _write_csv(corpus/"sources.csv",SOURCE_COLUMNS,active_sources)
                _write_csv(corpus/"reviews.csv",REVIEW_COLUMNS,[{**row,"group_id":groups[row["review_id"]]} for row in active])
                _write_csv(corpus/"annotations.csv",ANNOTATION_COLUMNS,[r for r in annotations if r["review_id"] in ids])
                _write_csv(corpus/"adjudications.csv",ADJUDICATION_COLUMNS,[r for r in adjudications if r["review_id"] in ids])
                if (self.root/"evidence_spans.csv").exists():
                    _write_csv(corpus/"evidence_spans.csv",SPAN_COLUMNS,[r for r in _read_csv(self.root/"evidence_spans.csv",SPAN_COLUMNS) if r["review_id"] in ids])
                splits=prepare_splits(corpus/"reviews.csv",corpus/"sources.csv",corpus/"annotations.csv",corpus/"adjudications.csv",staging/"splits",seed=seed)
                write_json(staging/"human-phenomena.json",{rid:decisions[rid]["judgment"] for rid in sorted(ids)})
                if any(file_sha(self.root/name)!=digest for name,digest in input_hashes.items()):
                    raise CorpusError("Corpus cambiado durante el congelado; no se crea snapshot.")
                manifest={"schema":"human-intake-snapshot-v1","status":"PARTITIONS_PREPARED_NOT_HUMAN_PERFORMANCE",
                    "origin_root":str(self.root.relative_to(Path.cwd().resolve())),"reviewed_by":reviewed_by,"created_at":_utc(),
                    "seed":seed,"raw_reviews":len(active),"ambiguous_excluded_from_three_class":sum(decisions[rid]["judgment"]["sentiment"]=="desconocido" for rid in ids),
                    "consent_receipt_hashes":{author:file_sha(self.root/"permissions"/f"{author}.json") for author in sorted({r["source_id"] for r in active})},
                    "intake_input_hashes":input_hashes,
                    "annotation_proof_hashes":{str(path.relative_to(self.root)):file_sha(path)
                        for folder,pattern in (("received","*.json"),("tasks","*.json"),("decisions","*.json"))
                        for path in sorted((self.root/folder).glob(pattern))},
                    "component_map":groups,"near_duplicate_screen":duplicates,"split_summary":splits["partitions"],
                    "phenomena_sha256":file_sha(staging/"human-phenomena.json"),
                    "limits":["Author/place codes and declared semantic relations require human review; no guarantee all paraphrases are found.",
                              "All near-duplicate candidates are conservatively grouped, never deleted.",
                              "Nine components is a technical lower bound, not sample-size adequacy.",
                              "No model fitting, test inspection or performance measurement occurred."]}
                write_json(staging/"human-snapshot.json",manifest)
                if output.exists(): raise CorpusError("Destino creado por otro proceso; no sobrescribir.")
                os.rename(staging,output)
            finally:
                if staging.exists(): shutil.rmtree(staging)
        return {"status":"PRIVATE_SNAPSHOT_PREPARED_NO_TRAINING","reviews":len(active),"components":len(set(groups.values())),
                "split_summary":splits["partitions"],"ambiguous_excluded":manifest["ambiguous_excluded_from_three_class"]}

    def verify_snapshot(self,snapshot):
        from .scientific_benchmark import verified_splits
        snapshot=private_path(snapshot)
        manifest=read_json(snapshot/"human-snapshot.json")
        if manifest.get("schema")!="human-intake-snapshot-v1" or manifest["origin_root"]!=str(self.root.relative_to(Path.cwd().resolve())):
            raise CorpusError("Snapshot de otro flujo o versión.")
        with coordinator_lock(self.root):
            _,reviews,_,annotations,adjudications,_=self._records()
            self._verified_decisions(reviews,annotations,adjudications)
            for author,digest in manifest["consent_receipt_hashes"].items():
                if self._receipt(author)["status"]!="active" or file_sha(self.root/"permissions"/f"{author}.json")!=digest:
                    raise CorpusError("Consentimiento retirado o cambiado después del snapshot.")
            if any(file_sha(self.root/name)!=digest for name,digest in manifest["intake_input_hashes"].items()):
                raise CorpusError("Corpus de recogida cambió tras congelarse; revisar una nueva versión antes de entrenar.")
            if any(file_sha(private_path(self.root/name))!=digest for name,digest in manifest["annotation_proof_hashes"].items()):
                raise CorpusError("Pruebas de anotación/adjudicación cambiaron tras congelarse.")
            if file_sha(snapshot/"human-phenomena.json")!=manifest["phenomena_sha256"]: raise CorpusError("Marcas humanas alteradas.")
            verified_splits(snapshot/"corpus",snapshot/"splits")
        return {"status":"SNAPSHOT_AND_CURRENT_CONSENTS_VERIFIED_NOT_PERFORMANCE"}


def connected_groups(reviews,links):
    """OR-union author, place, related IDs, exact texts and literal candidates."""
    ids={r["review_id"] for r in reviews}
    parents={rid:rid for rid in ids}
    def root(rid):
        while parents[rid]!=rid:
            parents[rid]=parents[parents[rid]];rid=parents[rid]
        return rid
    def union(left,right):
        a,b=root(left),root(right)
        if a!=b: parents[max(a,b)]=min(a,b)
    tokens={}
    for link in links:
        rid=link["review_id"]
        if rid not in ids: continue
        keys=[("author",link["author_id"]),("place",link["place_id"]),("related",rid)]
        keys.extend(("related",other) for other in json.loads(link["related_ids_json"]))
        for token in keys:
            if token in tokens: union(rid,tokens[token])
            tokens[token]=rid
    for row in reviews:
        token=("exact",text_sha(row["text"]))
        if token in tokens: union(row["review_id"],tokens[token])
        tokens[token]=row["review_id"]
    duplicates=near_duplicate_candidates(reviews)
    for pair in duplicates["candidates"]: union(*pair["review_ids"])
    members=defaultdict(list)
    for rid in sorted(ids): members[root(rid)].append(rid)
    codes={key:"component_"+_sha_text(json.dumps(group))[:24] for key,group in members.items()}
    return {rid:codes[root(rid)] for rid in sorted(ids)},duplicates
