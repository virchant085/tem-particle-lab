from __future__ import annotations
import time,uuid
from pathlib import Path
import numpy as np
import yaml
from .data import AnnotationStore,build_dataset,validate_snapshot,valid_id,atomic_json
from .models import train_model,ParticleDetector
from .tracking import run_learned_tracking
from ..io import save_json


def suggest_frames(project: Path, video_id: str, cfg: dict, model_id=None, count=12, progress=None):
    store=AnnotationStore(project);video=store.video(video_id)
    reviewed=set(next(v for v in store.summary() if v['id']==video_id)['reviewed_frames'])
    # Bounded pool spans the video. Suggestions are acquisition candidates, never automatic labels.
    pool=[int(i) for i in np.unique(np.linspace(0,video['frames']-1,min(80,video['frames']),dtype=int)) if int(i) not in reviewed]
    if not pool:return {'frames':[],'note':'建议池中的帧已完成标注'}
    detector=ParticleDetector(project,model_id,cfg) if model_id else None
    features=[];scores=[];reasons=[]
    import cv2
    for index,fi in enumerate(pool):
        frame=store.frame(video_id,fi);small=cv2.resize(frame,(24,24)).astype(float)/255
        features.append(small.ravel())
        if detector:
            pred=detector.predict(frame)
            score=float(np.mean([1-abs(2*p['confidence']-1) for p in pred])) if pred else 1.
            reasons.append('当前模型未检出目标：可能是漏检或真实负样本' if not pred else '结合模型分数不确定性与画面差异')
        else:score=0.;reasons.append('覆盖不同时刻和不同画面；首轮人工标注')
        scores.append(score)
        if progress:progress({'stage':'suggesting','frame':index+1,'frames':len(pool),'message':'挑选下一批值得检查的帧'})
    features=np.array(features);chosen=[];distance=np.full(len(pool),1.)
    while len(chosen)<min(count,len(pool)):
        rank=distance+.35*np.array(scores)
        rank[chosen]=-np.inf
        index=int(np.argmax(rank));chosen.append(index)
        new_distance=np.sqrt(np.mean((features-features[index])**2,axis=1))
        distance=np.minimum(distance,new_distance)
        # Avoid filling the batch with neighboring frames when alternatives exist.
        distance*=np.array([.2 if abs(pool[i]-pool[index])<2 and i not in chosen else 1 for i in range(len(pool))])
    return {'frames':[dict(frame=pool[i],score=scores[i],reason=reasons[i]) for i in chosen],
            'method':'diversity + detector uncertainty heuristic','is_reinforcement_learning':False}


def execute(project: Path, action: str, payload: dict, progress):
    project=Path(project).resolve();cfg=payload['config'];store=AnnotationStore(project)
    if action=='dataset':
        progress({'stage':'building_dataset','message':'核对已确认标注，创建固定数据版本'})
        return build_dataset(project,mode=payload.get('split_mode','experiment'),seed=cfg['seed'],
            validation_fraction=cfg['dataset']['validation_fraction'],temporal_gap=int(payload.get('temporal_gap',cfg['dataset']['temporal_gap_frames'])))
    if action=='train':
        return train_model(project,payload['dataset_id'],cfg,progress,initial_model_id=payload.get('initial_model_id'))
    if action=='preview':
        detector=ParticleDetector(project,payload['model_id'],cfg)
        rows=detector.predict(store.frame(payload['video_id'],int(payload['frame'])))
        return {'video_id':payload['video_id'],'frame':int(payload['frame']),'model_id':payload['model_id'],
                'detections':rows,'qualification':detector.card['qualification']}
    if action=='suggest':return suggest_frames(project,payload['video_id'],cfg,payload.get('model_id'),progress=progress)
    if action=='track':
        name='tracking_'+time.strftime('%Y%m%d_%H%M%S')+'_'+uuid.uuid4().hex[:6]
        directory=project/'results/learning'/name
        metrics=run_learned_tracking(project,store.path(payload['video_id']),payload['model_id'],
            {'particles':payload['particles']},cfg,directory,payload.get('calibration'),progress)
        import pandas as pd
        data=pd.read_csv(directory/'tracks.csv')
        cols=['frame','track_id','center_x','center_y','radius_px','observed','status','quality_reason']
        result={'video_id':payload['video_id'],'model_id':payload['model_id'],'metrics':metrics,'base':f'/results/learning/{name}/','rows':__import__('json').loads(data[cols].to_json(orient='records'))}
        atomic_json(store.root/'last_results'/f'{valid_id(payload["video_id"])}_{valid_id(payload["model_id"])}.json',result)
        return result
    if action=='evaluate':
        from .models import model_card
        card=model_card(project,payload['model_id']);dataset=project/'data/datasets'/valid_id(payload['dataset_id'])
        manifest=validate_snapshot(dataset);split=payload.get('split','val')
        if split not in ('val','test') or not manifest['counts'].get(split):raise ValueError('所选数据版本没有该验证/测试集')
        detector=ParticleDetector(project,payload['model_id'],cfg)
        out=project/'results/learning'/('evaluation_'+time.strftime('%Y%m%d_%H%M%S')+'_'+uuid.uuid4().hex[:6]);out.mkdir(parents=True)
        runtime=out/'dataset.yaml';runtime.write_text(yaml.safe_dump(dict(path=str(dataset),train='images/train',val='images/val',test='images/test',names={0:'particle'})),encoding='utf-8')
        metrics=detector.model.val(data=str(runtime),split=split,device=detector.hardware['device'],imgsz=cfg['inference']['imgsz'],
                                   batch=cfg['training']['batch'],workers=0,project=str(out),name='validation',verbose=False)
        report=dict(model_id=card['id'],dataset_id=manifest['id'],split=split,qualification=manifest['qualification'],
                    metrics={str(k):float(v) for k,v in metrics.results_dict.items() if np.isscalar(v) and np.isfinite(v)})
        save_json(out/'metrics.json',report);return report
    raise ValueError('未知任务')
