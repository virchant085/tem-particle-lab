"""Reproducible CLI counterpart of the local learning UI."""
from pathlib import Path
import argparse,json,sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from tem_tracker.learning.data import read_json,build_dataset,AnnotationStore
from tem_tracker.learning.models import load_learning_config,train_model,model_cards
from tem_tracker.learning.tracking import run_learned_tracking
from tem_tracker.learning.workflow import execute

def main():
    parser=argparse.ArgumentParser(description='TEM annotation, training and selected-target tracking')
    parser.add_argument('--project',type=Path,default=ROOT)
    parser.add_argument('--config',type=Path)
    commands=parser.add_subparsers(dest='command',required=True)
    commands.add_parser('status')
    p=commands.add_parser('dataset');p.add_argument('--split-mode',choices=['experiment','temporal'],default='experiment');p.add_argument('--gap',type=int,default=2)
    p=commands.add_parser('train');p.add_argument('--dataset',required=True);p.add_argument('--epochs',type=int);p.add_argument('--initial-model')
    p=commands.add_parser('track');p.add_argument('--input',type=Path,required=True);p.add_argument('--model',required=True);p.add_argument('--seeds',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--association',choices=['bytetrack','trackpy']);p.add_argument('--fps',type=float);p.add_argument('--nm-per-pixel',type=float)
    p=commands.add_parser('evaluate');p.add_argument('--model',required=True);p.add_argument('--dataset',required=True);p.add_argument('--split',choices=['val','test'],default='val')
    a=parser.parse_args();project=a.project.resolve();cfg=load_learning_config(a.config or project/'config/learning.yaml')
    def progress(info):print(json.dumps(info,ensure_ascii=False),flush=True)
    if a.command=='status':result=dict(videos=AnnotationStore(project).summary(),models=model_cards(project))
    elif a.command=='dataset':result=build_dataset(project,a.split_mode,cfg['seed'],cfg['dataset']['validation_fraction'],a.gap)
    elif a.command=='train':
        if a.epochs is not None:
            if not 1<=a.epochs<=2000:raise ValueError('epochs must be 1..2000')
            cfg['training']['epochs']=a.epochs
        result=train_model(project,a.dataset,cfg,progress,a.initial_model)
    elif a.command=='track':
        if a.association:cfg['association']['method']=a.association
        result=run_learned_tracking(project,a.input,a.model,read_json(a.seeds),cfg,a.output,dict(acquisition_fps=a.fps,nm_per_pixel=a.nm_per_pixel),progress)
    else:result=execute(project,'evaluate',dict(model_id=a.model,dataset_id=a.dataset,split=a.split,config=cfg),progress)
    print(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False))

if __name__=='__main__':main()
