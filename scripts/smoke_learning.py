"""Real training on synthetic fixtures, isolated from all TEM data/model registries."""
from pathlib import Path
import argparse,copy,json,shutil,sys,time
import cv2,numpy as np,yaml
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from tem_tracker.io import sha256,save_json
from tem_tracker.learning.data import atomic_json,AnnotationStore,build_dataset
from tem_tracker.learning.models import load_learning_config,train_model,ParticleDetector,model_cards
from tem_tracker.learning.tracking import run_learned_tracking

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);parser.add_argument('--epochs',type=int,default=20)
    a=parser.parse_args();out=a.output.resolve()
    if out.exists() and any(out.iterdir()):raise ValueError('Use a new smoke-test directory')
    out.mkdir(parents=True,exist_ok=True);(out/'config').mkdir()
    shutil.copy2(ROOT/'config/default.yaml',out/'config/default.yaml')
    cfg=load_learning_config(ROOT/'config/learning.yaml')
    cfg['training'].update(epochs=a.epochs,imgsz=128,batch=4,nbs=4,workers=0,amp=False,patience=0,warmup_epochs=0.0)
    # Scores from this very short synthetic training are low. These fixture-only
    # thresholds exercise association; the TEM project configuration is unchanged.
    cfg['inference'].update(imgsz=128,confidence=.05)
    cfg['association'].update(track_low_thresh=.05,track_high_thresh=.10,new_track_thresh=.12)
    (out/'config/learning.yaml').write_text(yaml.safe_dump(cfg),encoding='utf-8')
    pretrained=ROOT/'models/pretrained'/cfg['model']['architecture']
    if pretrained.exists():
        (out/'models/pretrained').mkdir(parents=True);shutil.copy2(pretrained,out/'models/pretrained'/pretrained.name)
    store=AnnotationStore(out);videos=[]
    for group in range(2):
        rng=np.random.default_rng(200+group);path=out/f'synthetic_{group}.avi'
        writer=cv2.VideoWriter(str(path),cv2.VideoWriter_fourcc(*'MJPG'),5,(128,128));labels=[]
        assert writer.isOpened()
        for fi in range(16):
            image=np.clip(175+rng.normal(0,7,(128,128)),0,255).astype(np.uint8)
            boxes=[]
            for ident,(base_x,base_y,r) in enumerate([(30,35,12),(88,85,15)]):
                x=base_x+fi%8;y=base_y+group*3
                cv2.circle(image,(x,y),r,40,-1);cv2.circle(image,(x-3,y-3),r//3,100,-1)
                boxes.append(dict(id=f'p{ident}',x1=x-r,y1=y-r,x2=x+r,y2=y+r,instance_id=ident+1,source='synthetic_known_geometry'))
            writer.write(cv2.cvtColor(image,cv2.COLOR_GRAY2BGR));labels.append(boxes)
        writer.release();v=store.register(path,f'synthetic_independent_{group}');videos.append(v)
        store.update_video(v['id'],v['experiment'],'train' if group==0 else 'val')
        for fi,boxes in enumerate(labels):store.save_annotation(v['id'],fi,boxes,True)
    manifest=build_dataset(out)
    manifest['qualification']='synthetic_workflow_test';manifest['warnings']=['Synthetic software fixture; not human-reviewed TEM labels; no TEM accuracy claim.']
    atomic_json(out/'data/datasets'/manifest['id']/'manifest.json',manifest)
    started=time.perf_counter()
    card=train_model(out,manifest['id'],cfg,lambda p:print(json.dumps(p,ensure_ascii=False),flush=True),role='synthetic_smoke')
    detector=ParticleDetector(out,card['id'],cfg,allow_smoke=True);predictions=detector.predict(store.frame(videos[1]['id'],0))
    assert len(predictions)>0, 'No detections after training; increase epochs to exercise positive inference'
    # Test the entire export path even if the very short training has weak recall.
    seeds=dict(particles=[dict(id=1,frame=0,x=30,y=38,radius=12)])
    metrics=run_learned_tracking(out,store.path(videos[1]['id']),card['id'],seeds,cfg,out/'tracking_output',allow_smoke=True)
    assert metrics['observed_rows']>=12, 'Synthetic target should be observed in most frames; check association'
    assert model_cards(out)==[], 'Synthetic weights must never appear as TEM models'
    report=dict(status='passed',purpose='software_training_inference_export_smoke_only',tem_accuracy_validated=False,
                source_project=str(ROOT),model_id=card['id'],hardware=card['hardware'],epochs_completed=card['epochs_completed'],
                inference_candidates=len(predictions),tracking_metrics=metrics,elapsed_seconds=time.perf_counter()-started)
    save_json(out/'smoke_report.json',report)
    official=out/'models/pretrained'/cfg['model']['architecture']
    if official.exists() and not pretrained.exists():
        pretrained.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(official,pretrained)
    print(json.dumps(report,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
