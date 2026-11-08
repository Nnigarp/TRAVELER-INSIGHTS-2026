"""Execute real local Gradio HTTP/queue calls, not a visual/microphone check."""
import argparse
import json
import os
from pathlib import Path
import socket
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--beto-artifact',type=Path)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    if args.out.exists():parser.error('Use a new report file')
    os.environ['GRADIO_ANALYTICS_ENABLED']='False'
    if args.beto_artifact:os.environ['TRAVELER_BETO_ARTIFACT']=str(args.beto_artifact)
    import httpx
    from gradio_client import Client
    import app
    from traveler_agent.demo_validation import trace_violations
    with socket.socket() as probe:
        probe.bind(('127.0.0.1',0));port=probe.getsockname()[1]
    address=f'http://127.0.0.1:{port}'
    report={'scope':'REAL_LOCAL_HTTP_QUEUE_NOT_BROWSER_VISUAL_OR_REAL_VOICE','models':{},'errors':[]}
    try:
        app.app.queue(default_concurrency_limit=1).launch(server_name='127.0.0.1',server_port=port,
            share=False,inbrowser=False,prevent_thread_lock=True,quiet=True,
            **({'theme':app.THEME,'css':app.UI_CSS} if app._GRADIO_MAJOR>=6 else {}))
        with httpx.Client(trust_env=False,timeout=30) as connection:
            home=connection.get(address);config=connection.get(address+'/config')
        if home.status_code!=200 or config.status_code!=200:raise RuntimeError('Local frontend/config unavailable')
        report.update(frontend_http_status=home.status_code,config_http_status=config.status_code,
                      configured_models=[value for _,value in app._MODEL_CHOICES])
        client=Client(address,verbose=False,httpx_kwargs={'trust_env':False})
        for kind in (['lr','svm','beto'] if args.beto_artifact else ['lr','svm']):
            started=time.perf_counter()
            summary,payload,view=client.predict('La habitación estaba sucia, pero el personal fue amable.',kind,
                                               api_name='/analyze_with_agents')
            record=json.loads(payload)
            violations=trace_violations(record)
            if record.get('modelo_utilizado')!=kind:violations.append('requested_model_did_not_execute')
            if not view or 'TravelerOrchestrator' not in view:violations.append('agent_view_missing')
            report['models'][kind]={'executed_model':record.get('modelo_utilizado'),
                'sentiment':record.get('sentimiento'),'human_review':record.get('revision_humana'),
                'trace_events':len(record.get('multiagente',{}).get('traza',[])),
                'roundtrip_seconds':time.perf_counter()-started,'violations':violations}
            report['errors'].extend(kind+':'+item for item in violations)
        report['status']='PASSED' if not report['errors'] else 'FAILED'
    finally:
        app.app.close()
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(report,ensure_ascii=False))
    if report['status']!='PASSED':raise SystemExit(1)


if __name__=='__main__':main()
