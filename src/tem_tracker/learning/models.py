from __future__ import annotations

import copy,json,os,platform,time,uuid
from pathlib import Path
from typing import Callable
import numpy as np
import yaml
from ..io import sha256,save_json,environment
from .data import atomic_json,read_json,valid_id,validate_snapshot


def setup_ultralytics(project: Path):
    settings_dir=Path(project)/'work/ultralytics_settings'
    settings_dir.mkdir(parents=True,exist_ok=True)
    os.environ.setdefault('YOLO_CONFIG_DIR',str(settings_dir))
    os.environ.setdefault('YOLO_AUTOINSTALL','false')
    from ultralytics import YOLO,settings
    settings.update({k:False for k in ('sync','hub','wandb','clearml','comet','neptune','mlflow') if k in settings})
    return YOLO


def load_learning_config(path: Path) -> dict:
    cfg=yaml.safe_load(Path(path).read_text(encoding='utf-8'))
    if cfg['model']['class_names']!=['particle']:raise ValueError('当前任务只支持 particle 单类检测')
    if cfg['model']['architecture'] not in ('yolo26n.pt','yolo26s.pt','yolo11n.pt','yolov8n.pt'):
        raise ValueError('请选择受支持的本地迁移学习起始模型')
    t=cfg['training'];a=cfg['association'];inf=cfg['inference']
    if not 1<=int(t['epochs'])<=2000 or not 1<=int(t['batch'])<=64:raise ValueError('训练轮数或批量参数无效')
    if not 64<=int(t['imgsz'])<=2048 or int(t['imgsz'])%32:raise ValueError('图像尺寸需为 64–2048 范围内的 32 倍数')
    if not 0<=a['track_low_thresh']<a['track_high_thresh']<=a['new_track_thresh']<=1:raise ValueError('跟踪置信度门限顺序无效')
    if not 0<=float(inf['confidence'])<=a['track_low_thresh']:raise ValueError('检测门限必须不高于低分恢复门限，否则低分恢复不会生效')
    if a['method'] not in ('bytetrack','trackpy'):raise ValueError('请选择 bytetrack 或 trackpy 关联')
    return cfg


def device_info(requested='auto') -> dict:
    import torch,torchvision
    available=torch.cuda.is_available()
    if requested=='auto':device=0 if available else 'cpu'
    elif str(requested)=='cpu':device='cpu'
    else:
        device=int(requested)
        if not available or not 0<=device<torch.cuda.device_count():raise ValueError('所选 GPU 不可用；可改为 auto 或 cpu')
    return dict(device=device,cuda_used=device!='cpu',cuda_available=available,
                gpu_name=torch.cuda.get_device_name(device) if device!='cpu' else None,
                torch=torch.__version__,torchvision=torchvision.__version__,cuda_runtime=torch.version.cuda,
                python=platform.python_version())


def model_cards(project: Path, include_smoke: bool=False) -> list[dict]:
    cards=[read_json(p) for p in sorted((Path(project)/'models/registry').glob('*/model_card.json'),reverse=True)]
    return [m for m in cards if include_smoke or m.get('role')=='tem_supervised']


def model_card(project: Path, model_id: str, allow_smoke: bool=False) -> dict:
    card=read_json(Path(project)/'models/registry'/valid_id(model_id)/'model_card.json')
    if not card:raise ValueError('找不到已训练的模型版本')
    if card['role']!='tem_supervised' and not allow_smoke:raise ValueError('合成数据检查模型不能用于真实 TEM 追踪')
    p=(Path(project)/card['weights']).resolve()
    if not p.is_relative_to((Path(project)/'models/registry').resolve()) or not p.is_file():raise ValueError('模型权重路径无效')
    if sha256(p)!=card['weights_sha256']:raise ValueError('模型权重已改变，与版本记录不一致')
    return card


def activate_model(project: Path, model_id: str) -> dict:
    card=model_card(project,model_id)
    atomic_json(Path(project)/'models/active.json',{'model_id':model_id,'weights_sha256':card['weights_sha256']})
    return card


def check_lineage_holdout(manifest: dict, ancestors: list[dict]) -> None:
    """A fine-tuned model remembers data seen by its parent, even after a new split."""
    for item in manifest['items']:
        if item['split'] not in ('val','test'):continue
        for old in ancestors:
            seen=[x for x in old['items'] if x['split']=='train' or (item['split']=='test' and x['split']=='val')]
            for trained in seen:
                duplicate=item['image_sha256']==trained['image_sha256']
                group_overlap=manifest['split_mode']=='experiment' and (
                    item['experiment']==trained['experiment'] or item['source_video_sha256']==trained['source_video_sha256'])
                if duplicate or group_overlap:
                    raise ValueError('继续训练的父模型已见过当前验证/测试数据。请保持实验划分一致，或从官方预训练权重重新训练。')


def train_model(project: Path, dataset_id: str, config: dict, progress: Callable|None=None,
                initial_model_id: str|None=None, role: str='tem_supervised') -> dict:
    project=Path(project).resolve();dataset=project/'data/datasets'/valid_id(dataset_id)
    manifest=validate_snapshot(dataset)
    if role not in ('tem_supervised','synthetic_smoke'):raise ValueError('模型用途无效')
    if role=='tem_supervised' and manifest.get('qualification')=='synthetic_workflow_test':
        raise ValueError('合成测试数据不能登记为 TEM 训练结果')
    cfg=copy.deepcopy(config);hardware=device_info(cfg['training']['device'])
    YOLO=setup_ultralytics(project)
    job='train_'+time.strftime('%Y%m%d_%H%M%S')+'_'+uuid.uuid4().hex[:6]
    out=project/'results/learning'/job;out.mkdir(parents=True,exist_ok=False)
    save_json(out/'hardware.json',hardware);save_json(out/'environment.json',dict(environment(),**hardware))
    save_json(out/'config.json',cfg);save_json(out/'dataset_manifest.json',manifest)
    source_root=Path(__file__).resolve().parents[3]
    save_json(out/'source_manifest.json',{p.relative_to(source_root).as_posix():sha256(p)
        for p in list((source_root/'src').rglob('*.py'))+list((source_root/'scripts').glob('*.py'))})
    runtime_data=dict(path=str(dataset),train='images/train',val='images/val',names={0:'particle'})
    if manifest['counts'].get('test',0):runtime_data['test']='images/test'
    yaml_path=out/'runtime_dataset.yaml';yaml_path.write_text(yaml.safe_dump(runtime_data,allow_unicode=True),encoding='utf-8')
    parent_card=None
    if initial_model_id:
        parent_card=model_card(project,initial_model_id,allow_smoke=role=='synthetic_smoke')
        ancestors=[];current=parent_card;visited=set()
        while current:
            if current['id'] in visited:raise ValueError('模型版本链包含循环')
            visited.add(current['id'])
            history_path=project/'data/datasets'/valid_id(current['dataset_id'])/'manifest.json'
            if not history_path.is_file() or sha256(history_path)!=current['dataset_manifest_sha256']:
                raise ValueError('父模型的数据来源记录缺失或已改变，无法核实继续训练的验证划分')
            ancestors.append(read_json(history_path))
            current=model_card(project,current['parent_model_id'],allow_smoke=role=='synthetic_smoke') if current.get('parent_model_id') else None
        check_lineage_holdout(manifest,ancestors)
        initial=project/parent_card['weights']
    else:
        initial=project/'models/pretrained'/cfg['model']['architecture']
        initial.parent.mkdir(parents=True,exist_ok=True)
    if progress:progress({'stage':'loading_model','message':'载入迁移学习初始权重','run':job})
    model=YOLO(str(initial),task='detect')
    params=copy.deepcopy(cfg['training']);params['device']=hardware['device']
    epochs=int(params['epochs'])
    def report_epoch(trainer):
        if progress:
            completed=min(int(trainer.epoch)+1,epochs)
            metrics={str(k):float(v) for k,v in (trainer.metrics or {}).items() if np.isscalar(v) and np.isfinite(v)}
            progress({'stage':'training','epoch':completed,'epochs':epochs,'metrics':metrics,'run':job,
                      'message':f'正在训练 {completed}/{epochs} 轮'})
    model.add_callback('on_fit_epoch_end',report_epoch)
    try:
        model.train(data=str(yaml_path),project=str(out),name='fit',exist_ok=False,seed=int(cfg['seed']),
                    pretrained=True,verbose=False,plots=True,**params)
        best=Path(model.trainer.best)
        if not best.is_file():raise RuntimeError('训练没有生成 best.pt')
        evaluated=YOLO(str(best)).val(data=str(yaml_path),split='val',device=hardware['device'],
                    imgsz=int(params['imgsz']),batch=int(params['batch']),workers=0,
                    project=str(out),name='validation',verbose=False,plots=True)
        metrics={str(k):float(v) for k,v in evaluated.results_dict.items() if np.isscalar(v) and np.isfinite(v)}
        ident='model_'+time.strftime('%Y%m%d_%H%M%S')+'_'+uuid.uuid4().hex[:6]
        registry=project/'models/registry'/ident;registry.mkdir(parents=True,exist_ok=False)
        import shutil
        weights=registry/'best.pt';shutil.copy2(best,weights)
        card=dict(id=ident,role=role,domain='TEM particles' if role=='tem_supervised' else 'synthetic software test only',
                  task='detect',class_names=['particle'],architecture=parent_card['architecture'] if parent_card else cfg['model']['architecture'],
                  weights=weights.relative_to(project).as_posix(),weights_sha256=sha256(weights),
                  dataset_id=dataset_id,dataset_manifest_sha256=sha256(dataset/'manifest.json'),
                  qualification=manifest['qualification'],data_counts=manifest['counts'],warnings=manifest['warnings'],
                  parent_model_id=initial_model_id,pretrained_initialization=str(initial.name),
                  training_run=out.relative_to(project).as_posix(),hardware=hardware,
                  epochs_completed=int(model.trainer.epoch)+1,validation_metrics=metrics,
                  validation_scope='validation set used for model selection; not an untouched test set',
                  created_at=time.strftime('%Y-%m-%dT%H:%M:%S'))
        atomic_json(registry/'model_card.json',card);save_json(out/'model_card.json',card)
        if progress:progress({'stage':'completed','message':'训练及验证完成，已保存新模型版本','model_id':ident})
        return card
    except Exception as exc:
        atomic_json(out/'failure.json',{'error':str(exc),'note':'Partial files retained; no usable model is registered on failure'})
        raise


class ParticleDetector:
    def __init__(self, project: Path, model_id: str, config: dict, allow_smoke: bool=False):
        self.project=Path(project).resolve();self.config=config
        self.card=model_card(self.project,model_id,allow_smoke)
        self.hardware=device_info(config['training']['device'])
        YOLO=setup_ultralytics(self.project);self.model=YOLO(str(self.project/self.card['weights']))
        if list(self.model.names.values())!=['particle']:raise ValueError('模型类别不是单类 particle，拒绝作为 TEM 模型使用')

    def predict(self, frame: np.ndarray) -> list[dict]:
        import cv2
        image=cv2.cvtColor(frame,cv2.COLOR_GRAY2BGR) if frame.ndim==2 else frame
        c=self.config['inference']
        result=self.model.predict(image,conf=c['confidence'],iou=c['iou'],imgsz=c['imgsz'],
            max_det=c['max_det'],device=self.hardware['device'],verbose=False)[0]
        rows=[]
        for i,row in enumerate(result.boxes.data.cpu().numpy()):
            x1,y1,x2,y2,score,cls=row[:6]
            if int(cls)!=0:continue
            rows.append(dict(detection_index=i,x1=float(x1),y1=float(y1),x2=float(x2),y2=float(y2),
                             confidence=float(score),class_id=0))
        return rows
