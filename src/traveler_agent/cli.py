"""CLI entrenar/demo/evaluar/predecir."""
import argparse,json
from pathlib import Path
from .model import train,save_model,load_model,sha256_file
from .dataset import read_labeled_csv
from .experiments import holdout_evaluate
from .demo_data import DEMO
from .agent import inspect_review

def main(argv=None):
    p=argparse.ArgumentParser(description="Traveler Insights 2026: prototipo académico")
    sub=p.add_subparsers(dest='command',required=True)
    d=sub.add_parser('demo-train');d.add_argument('--out',default='artifacts/demo_model.joblib')
    t=sub.add_parser('train');t.add_argument('--csv',required=True);t.add_argument('--text-col',default='reseña');t.add_argument('--label-col',default='sentimiento');t.add_argument('--out',default='artifacts/real_model.joblib')
    e=sub.add_parser('evaluate');e.add_argument('--csv',required=True);e.add_argument('--text-col',default='reseña');e.add_argument('--label-col',default='sentimiento');e.add_argument('--out',default='artifacts/evaluation.json')
    i=sub.add_parser('predict');i.add_argument('--model',default='artifacts/demo_model.joblib');i.add_argument('--text',required=True);i.add_argument('--id',default='');i.add_argument('--threshold',type=float,default=0.56)
    a=p.parse_args(argv)
    if a.command=='demo-train':
        m=train([x[0] for x in DEMO],[x[1] for x in DEMO]);save_model(m,Path(a.out),provenance='SINTETICO_DEMO_NO_PUBLICABLE')
        print(f'Modelo de DEMOSTRACIÓN guardado: {a.out} (NO evidencia científica)')
    elif a.command=='train':
        df=read_labeled_csv(a.csv,a.text_col,a.label_col)
        m=train(df[a.text_col].tolist(),df[a.label_col].tolist())
        save_model(m,Path(a.out),provenance='DATOS_CSV_USUARIO_REVISAR_ETICA_Y_LICENCIA',dataset_sha256=sha256_file(Path(a.csv)))
        print(f'Modelo entrenado con {len(df)} reseñas únicas: {a.out}. Antes de reportar, ejecutar evaluate sobre una partición no vista.')
    elif a.command=='evaluate':
        df=read_labeled_csv(a.csv,a.text_col,a.label_col)
        _,metrics=holdout_evaluate(df,a.text_col,a.label_col)
        path=Path(a.out);path.parent.mkdir(parents=True,exist_ok=True)
        result={"dataset_sha256":sha256_file(Path(a.csv)),"protocol":"holdout estratificado 80/20, seed=42", **metrics}
        path.write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding='utf-8')
        print(json.dumps(result,indent=2,ensure_ascii=False))
    else:
        model=load_model(Path(a.model));print(json.dumps(inspect_review(a.text,model,review_id=a.id,threshold=a.threshold),ensure_ascii=False,indent=2))
if __name__=='__main__':main()
