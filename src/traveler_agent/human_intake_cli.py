"""Local human collection, blind packets and admission checks; never trains."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .corpus import CorpusError
from .human_intake import (HumanIntake, NOTICE, NOTICE_VERSION, GUIDE, consent_template,
    private_path, read_json, write_json, load_packet, save_answer, seal_answers)


def empty_judgment():
    return {"sentiment":None,"aspects":{},"spans":[],"mixed":False,"ambiguous":False,"negation":None,"contrast":None}


def run(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",type=Path,default=Path("data/private/piloto01-human-v1"))
    commands=parser.add_subparsers(dest="command",required=True)
    commands.add_parser("init")
    commands.add_parser("notice")
    commands.add_parser("status")
    form=commands.add_parser("consent-template",help="Plantilla sin aceptar; entregar aviso al participante")
    form.add_argument("--out",type=Path,required=True)
    participant=commands.add_parser("participant-form",help="Formulario HTML local sin red; genera JSON de consentimiento y TXT solo al participante")
    participant.add_argument("--out",type=Path,required=True)
    registration=commands.add_parser("register-consent")
    registration.add_argument("--consent-file",type=Path,required=True)
    registration.add_argument("--reviewed-by",required=True)
    contribute=commands.add_parser("contribute")
    contribute.add_argument("--text-file",type=Path,required=True)
    contribute.add_argument("--author-id",required=True)
    contribute.add_argument("--place-id",required=True)
    contribute.add_argument("--reviewed-by",required=True)
    contribute.add_argument("--related-to",action="append",default=[])
    for flag in ("privacy-reviewed","spanish-verified","original-human","real-experience"):
        contribute.add_argument("--"+flag,action="store_true")
    packet=commands.add_parser("packet",help="Exportar lote privado ciego, sin autores/lugares/etiquetas")
    packet.add_argument("--annotator",required=True)
    packet.add_argument("--out",type=Path,required=True)
    packet.add_argument("--limit",type=int,default=25)
    cancel=commands.add_parser("cancel-task")
    cancel.add_argument("--task-id",required=True)
    link=commands.add_parser("link-related")
    link.add_argument("--left",required=True);link.add_argument("--right",required=True)
    withdraw=commands.add_parser("withdraw")
    withdraw.add_argument("--author-id",required=True)
    template=commands.add_parser("judgment-template")
    template.add_argument("--out",type=Path,required=True)
    answer=commands.add_parser("answer",help="Alternativa CLI a la UI privada de anotación")
    answer.add_argument("--packet",type=Path,required=True)
    answer.add_argument("--responses",type=Path,required=True)
    answer.add_argument("--item-id",required=True)
    answer.add_argument("--judgment-file",type=Path,required=True)
    seal=commands.add_parser("seal")
    seal.add_argument("--packet",type=Path,required=True)
    seal.add_argument("--responses",type=Path,required=True)
    seal.add_argument("--independent-human",action="store_true")
    imported=commands.add_parser("import-answers")
    imported.add_argument("--responses",type=Path,required=True)
    view=commands.add_parser("adjudication-packet",help="Solo después de dos dictámenes, para tercera persona")
    view.add_argument("--review-id",required=True)
    view.add_argument("--out",type=Path,required=True)
    adjudicate=commands.add_parser("adjudicate")
    adjudicate.add_argument("--decision-file",type=Path,required=True)
    freeze=commands.add_parser("freeze",help="Snapshot privado; prepara particiones, no entrena")
    freeze.add_argument("--out",type=Path,required=True)
    freeze.add_argument("--reviewed-by",required=True)
    freeze.add_argument("--links-reviewed",action="store_true")
    verify=commands.add_parser("verify-snapshot")
    verify.add_argument("--snapshot",type=Path,required=True)
    args=parser.parse_args(argv)
    service=HumanIntake(args.root)
    if args.command=="init": return service.init()
    if args.command=="notice": return {"version":NOTICE_VERSION,"notice":NOTICE,"annotation_guide":GUIDE}
    if args.command=="consent-template": return consent_template(args.out)
    if args.command=="participant-form":
        from .participant_form import create_participant_form
        return create_participant_form(args.out)
    if args.command=="register-consent": return service.register_consent(args.consent_file,args.reviewed_by)
    if args.command=="contribute":
        return service.contribute(args.text_file,args.author_id,args.place_id,args.reviewed_by,args.related_to,
            privacy_reviewed=args.privacy_reviewed,spanish_verified=args.spanish_verified,
            original_human=args.original_human,real_experience=args.real_experience)
    if args.command=="packet": return service.create_task(args.annotator,args.out,args.limit)
    if args.command=="cancel-task": return service.cancel_task(args.task_id)
    if args.command=="withdraw": return service.withdraw(args.author_id)
    if args.command=="link-related": return service.link(args.left,args.right)
    if args.command=="judgment-template":
        write_json(args.out,empty_judgment())
        return {"status":"UNLABELED_JUDGMENT_TEMPLATE_NOT_ANNOTATION"}
    if args.command=="answer": return save_answer(args.packet,args.responses,args.item_id,read_json(args.judgment_file))
    if args.command=="seal": return seal_answers(args.packet,args.responses,independent=args.independent_human)
    if args.command=="import-answers": return service.import_answers(args.responses)
    if args.command=="status": return service.status()
    if args.command=="adjudication-packet":
        view=service.adjudication_view(args.review_id)
        write_json(args.out,{**view,"adjudicator_id":None,"judgment":empty_judgment(),"reason":""})
        return {"status":"TWO_ANNOTATIONS_READY_FOR_THIRD_HUMAN_ADJUDICATION"}
    if args.command=="adjudicate":
        value=read_json(args.decision_file)
        if not isinstance(value,dict) or set(value)!={"review_id","text","annotations","adjudicator_id","judgment","reason"}:
            raise CorpusError("Esquema de adjudicación inválido.")
        view=service.adjudication_view(value["review_id"])
        if value["text"]!=view["text"] or value["annotations"]!=view["annotations"]:
            raise CorpusError("Paquete de adjudicación alterado o desactualizado.")
        return service.adjudicate(value["review_id"],value["adjudicator_id"],value["judgment"],value["reason"])
    if args.command=="freeze": return service.freeze(args.out,args.reviewed_by,links_reviewed=args.links_reviewed)
    if args.command=="verify-snapshot": return service.verify_snapshot(args.snapshot)
    raise AssertionError("Comando no implementado")


def main(argv=None):
    try:
        print(json.dumps(run(argv),ensure_ascii=False,indent=2))
    except CorpusError as exc:
        print(json.dumps({"status":"BLOCKED","reason":str(exc)},ensure_ascii=False))
        return 2
    except (OSError,ValueError,TypeError,KeyError):
        print(json.dumps({"status":"BLOCKED","reason":"Archivos o estado incompletos; revisar localmente sin publicar textos, permisos o rutas personales."}))
        return 2
    return 0


if __name__=="__main__":
    raise SystemExit(main())
