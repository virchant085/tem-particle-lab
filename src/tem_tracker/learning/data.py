from __future__ import annotations

import hashlib,json,math,re,shutil,threading,time,uuid
from pathlib import Path
from typing import Any
import cv2
import numpy as np
import yaml
from ..io import load_video,sha256


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+'.'+uuid.uuid4().hex+'.tmp')
    temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    temporary.replace(path)


def read_json(path: Path, default=None):
    return json.loads(path.read_text(encoding='utf-8')) if path.exists() else default


def valid_id(value: str) -> str:
    if not isinstance(value,str) or not re.fullmatch(r'[A-Za-z0-9_-]{1,100}',value):
        raise ValueError('无效记录编号')
    return value


def checked_boxes(boxes: list, width: int, height: int) -> list[dict]:
    if not isinstance(boxes,list) or len(boxes)>5000:raise ValueError('标注列表无效')
    result=[];ids=set();instances=set()
    for box in boxes:
        coords=[float(box[k]) for k in ('x1','y1','x2','y2')]
        if not all(math.isfinite(x) for x in coords):raise ValueError('标注坐标必须是有限数值')
        x1,y1,x2,y2=coords
        if not (0<=x1<x2<=width and 0<=y1<y2<=height and x2-x1>=2 and y2-y1>=2):
            raise ValueError('颗粒框必须在图像内，宽高至少 2 像素')
        ident=str(box.get('id') or uuid.uuid4().hex[:10])
        if ident in ids:raise ValueError('同一帧的框编号不能重复')
        ids.add(ident)
        instance=box.get('instance_id')
        if instance in ('',None):instance=None
        elif float(instance)!=int(float(instance)) or int(float(instance))<1:raise ValueError('身份 ID 应为正整数或留空')
        else:instance=int(float(instance))
        if instance is not None:
            if instance in instances:raise ValueError('同一帧的颗粒身份不能重复')
            instances.add(instance)
        result.append(dict(id=ident,class_id=0,x1=x1,y1=y1,x2=x2,y2=y2,instance_id=instance,
                           source=str(box.get('source','manual'))[:80]))
    return result


class AnnotationStore:
    """Raw copies and versioned, explicit human review. Proposals are never training labels."""
    def __init__(self, project: Path):
        self.project=Path(project).resolve();self.root=self.project/'data/learning'
        self.root.mkdir(parents=True,exist_ok=True)
        self.lock=threading.RLock()

    def videos(self) -> list[dict]:
        return read_json(self.root/'videos.json',[])

    def video(self, video_id: str) -> dict:
        valid_id(video_id)
        found=next((v for v in self.videos() if v['id']==video_id),None)
        if not found:raise ValueError('找不到视频')
        return found

    def path(self, video_id: str) -> Path:
        p=(self.project/self.video(video_id)['path']).resolve()
        if not p.is_relative_to(self.root):raise ValueError('视频路径超出数据目录')
        return p

    def register(self, source: Path, experiment: str, name: str|None=None) -> dict:
        experiment=experiment.strip()
        if not experiment or len(experiment)>120:raise ValueError('请填写实验组名称，同一实验的视频使用同一个名称')
        source=Path(source).resolve(strict=True)
        frames,meta=load_video(source)
        ident='v_'+meta['sha256'][:16]
        with self.lock:
            videos=self.videos()
            existing=next((v for v in videos if v['id']==ident),None)
            if existing:return existing
            folder=self.root/'videos';folder.mkdir(exist_ok=True)
            dest=folder/(ident+source.suffix.lower())
            if not dest.exists():shutil.copy2(source,dest)
            if sha256(dest)!=meta['sha256']:raise RuntimeError('原始视频副本校验失败')
            video=dict(id=ident,name=(name or source.name)[:160],experiment=experiment,split='auto',
                       path=dest.relative_to(self.project).as_posix(),sha256=meta['sha256'],
                       frames=len(frames),width=meta['width'],height=meta['height'],container_fps=meta['container_fps'],
                       created_at=time.strftime('%Y-%m-%dT%H:%M:%S'))
            videos.append(video);atomic_json(self.root/'videos.json',videos)
            return video

    def update_video(self, video_id: str, experiment: str, split: str) -> dict:
        if split not in ('auto','train','val','test'):raise ValueError('数据划分无效')
        experiment=experiment.strip()
        if not experiment or len(experiment)>120:raise ValueError('实验组名称无效')
        with self.lock:
            videos=self.videos();video=next((v for v in videos if v['id']==valid_id(video_id)),None)
            if video is None:raise ValueError('找不到视频')
            video.update(experiment=experiment,split=split);atomic_json(self.root/'videos.json',videos)
            return video

    def frame(self, video_id: str, frame: int) -> np.ndarray:
        video=self.video(video_id)
        if not isinstance(frame,int) or not 0<=frame<video['frames']:raise ValueError('帧号越界')
        cap=cv2.VideoCapture(str(self.path(video_id)))
        try:
            cap.set(cv2.CAP_PROP_POS_FRAMES,frame);ok,image=cap.read()
            if not ok:raise ValueError('读取视频帧失败')
            return cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        finally:cap.release()

    def annotation_path(self, video_id: str, frame: int) -> Path:
        video=self.video(video_id)
        if not 0<=int(frame)<video['frames']:raise ValueError('帧号越界')
        return self.root/'annotations'/video_id/f'{int(frame):06d}.json'

    def annotation(self, video_id: str, frame: int) -> dict:
        return read_json(self.annotation_path(video_id,frame),dict(video_id=video_id,frame=frame,
            boxes=[],reviewed=False,negative_confirmed=False,revision=0))

    def save_annotation(self, video_id: str, frame: int, boxes: list, reviewed: bool,
                        negative_confirmed: bool=False, expected_revision: int|None=None) -> dict:
        video=self.video(video_id)
        clean=checked_boxes(boxes,video['width'],video['height'])
        if reviewed and not clean and not negative_confirmed:raise ValueError('空帧需要明确勾选“已确认无颗粒”')
        with self.lock:
            path=self.annotation_path(video_id,frame);old=self.annotation(video_id,frame)
            if expected_revision is not None and old['revision']!=expected_revision:raise ValueError('标注已被其他页面更新，请重新载入后修改')
            result=dict(video_id=video_id,frame=int(frame),boxes=clean,reviewed=bool(reviewed),
                        negative_confirmed=bool(negative_confirmed and not clean),revision=old['revision']+1,
                        review_definition='all visible particles in the full image annotated; ambiguous frames remain draft',
                        saved_at=time.strftime('%Y-%m-%dT%H:%M:%S'))
            if path.exists():
                archive=self.root/'annotation_history'/video_id/f'{frame:06d}_r{old["revision"]:05d}.json'
                archive.parent.mkdir(parents=True,exist_ok=True)
                if not archive.exists():shutil.copy2(path,archive)
            atomic_json(path,result);return result

    def reviewed_records(self) -> list[tuple[dict,dict]]:
        rows=[]
        for video in self.videos():
            directory=self.root/'annotations'/video['id']
            for path in sorted(directory.glob('*.json')):
                annotation=read_json(path)
                if annotation['reviewed']:rows.append((video,annotation))
        return rows

    def summary(self) -> list[dict]:
        result=[]
        for video in self.videos():
            annotations=[read_json(p) for p in (self.root/'annotations'/video['id']).glob('*.json')]
            result.append(dict(video,reviewed_frames=[a['frame'] for a in annotations if a['reviewed']],
                               draft_frames=[a['frame'] for a in annotations if not a['reviewed']],
                               reviewed_boxes=sum(len(a['boxes']) for a in annotations if a['reviewed'])))
        return result


def assign_splits(records: list[tuple[dict,dict]], mode: str, seed: int=17,
                  validation_fraction: float=.25, temporal_gap: int=2) -> tuple[dict,list[str]]:
    if not records:raise ValueError('尚无确认完成的标注；草稿不会用于训练')
    assignments={};warnings=[]
    if mode=='temporal':
        videos={v['id']:v for v,a in records}
        if len(videos)!=1:raise ValueError('单视频试训模式只接受一个有已确认标注的视频；多视频请按实验划分')
        video=next(iter(videos.values()));cut=int((video['frames']-1)*(1-validation_fraction))
        for v,a in records:
            fi=a['frame'];assignments[(v['id'],fi)]='train' if fi<=cut-temporal_gap else 'val' if fi>=cut+temporal_gap+1 else 'excluded_gap'
        warnings.append('同一视频时间分块，仅供流程试训；不能证明对新实验的泛化能力')
    elif mode=='experiment':
        groups=sorted({v['experiment'] for v,a in records})
        if len(groups)<2:raise ValueError('跨实验验证至少需要两个独立实验组；只有一个视频时可选择明确标记的“单视频试训”')
        requested={g:{v['split'] for v,a in records if v['experiment']==g and v['split']!='auto'} for g in groups}
        if any(len(x)>1 for x in requested.values()):raise ValueError('同一实验组不能同时进入训练集与验证/测试集')
        chosen={g:next(iter(s)) for g,s in requested.items() if s}
        automatic=[g for g in groups if g not in chosen]
        automatic.sort(key=lambda g:hashlib.sha256(f'{seed}:{g}'.encode()).hexdigest())
        if 'val' not in chosen.values() and automatic:chosen[automatic.pop(0)]='val'
        if 'train' not in chosen.values() and automatic:chosen[automatic.pop(0)]='train'
        desired=max(1,round(len(groups)*validation_fraction))
        for group in automatic:
            chosen[group]='val' if list(chosen.values()).count('val')<desired else 'train'
        for v,a in records:assignments[(v['id'],a['frame'])]=chosen[v['experiment']]
    else:raise ValueError('数据划分模式应为 experiment 或 temporal')
    counts={s:sum(x==s for x in assignments.values()) for s in ('train','val','test')}
    if not counts['train'] or not counts['val']:raise ValueError('需要至少一帧训练标注和一帧隔离的验证标注；请检查分组或时间间隔')
    if counts['train']<30 or counts['val']<10:warnings.append('当前标注较少，适合首轮试验；模型可靠性仍需更多独立标注验证')
    return assignments,warnings


def build_dataset(project: Path, mode: str='experiment', seed: int=17,
                  validation_fraction: float=.25, temporal_gap: int=2) -> dict:
    store=AnnotationStore(project);records=store.reviewed_records()
    if not .05<=validation_fraction<=.5 or not 0<=temporal_gap<=10000:raise ValueError('划分参数无效')
    assignments,warnings=assign_splits(records,mode,seed,validation_fraction,temporal_gap)
    version='ds_'+time.strftime('%Y%m%d_%H%M%S')+'_'+uuid.uuid4().hex[:6]
    out=store.project/'data/datasets'/version;out.mkdir(parents=True,exist_ok=False)
    manifest=dict(id=version,task='detect',class_names=['particle'],split_mode=mode,seed=seed,
                  temporal_gap_frames=temporal_gap,validation_fraction=validation_fraction,
                  qualification='independent_experiment_holdout' if mode=='experiment' else 'exploratory_single_video',
                  warnings=warnings,items=[],excluded=[],created_at=time.strftime('%Y-%m-%dT%H:%M:%S'))
    seen={}
    try:
        for video,annotation in records:
            split=assignments[(video['id'],annotation['frame'])]
            if split=='excluded_gap':
                manifest['excluded'].append(dict(video_id=video['id'],frame=annotation['frame'],reason='temporal_gap'));continue
            if sha256(store.path(video['id']))!=video['sha256']:raise ValueError('视频副本已经改变，请重新导入')
            frame=store.frame(video['id'],annotation['frame']);content=hashlib.sha256(frame.tobytes()).hexdigest()
            if content in seen and seen[content]!=split:raise ValueError('训练与验证/测试集出现完全重复的图像；请按真实实验重新分组')
            seen[content]=split
            name=f'{video["id"]}_{annotation["frame"]:06d}'
            image_path=out/'images'/split/(name+'.png');label_path=out/'labels'/split/(name+'.txt')
            image_path.parent.mkdir(parents=True,exist_ok=True);label_path.parent.mkdir(parents=True,exist_ok=True)
            success,encoded=cv2.imencode('.png',frame)
            if not success:raise RuntimeError('图像编码失败')
            encoded.tofile(image_path)
            boxes=checked_boxes(annotation['boxes'],video['width'],video['height']);lines=[]
            for b in boxes:
                xc=(b['x1']+b['x2'])/2/video['width'];yc=(b['y1']+b['y2'])/2/video['height']
                w=(b['x2']-b['x1'])/video['width'];h=(b['y2']-b['y1'])/video['height']
                lines.append(f'0 {xc:.9f} {yc:.9f} {w:.9f} {h:.9f}')
            label_path.write_text('\n'.join(lines)+('\n' if lines else ''),encoding='utf-8')
            manifest['items'].append(dict(video_id=video['id'],experiment=video['experiment'],frame=annotation['frame'],
                split=split,annotation_revision=annotation['revision'],boxes=boxes,
                annotation_sha256=hashlib.sha256(json.dumps(annotation,sort_keys=True,ensure_ascii=False).encode('utf-8')).hexdigest(),
                annotation_hash_definition='canonical captured annotation JSON, sorted keys, UTF-8',
                image=image_path.relative_to(out).as_posix(),label=label_path.relative_to(out).as_posix(),
                image_sha256=sha256(image_path),label_sha256=sha256(label_path),source_video_sha256=video['sha256']))
        for split in ('train','val'):
            if not any(i['boxes'] for i in manifest['items'] if i['split']==split):raise ValueError(f'{split} 至少需要一个有颗粒的已确认标注帧')
        dataset=dict(path=str(out.resolve()),train='images/train',val='images/val',names={0:'particle'})
        if any(i['split']=='test' for i in manifest['items']):dataset['test']='images/test'
        (out/'dataset.yaml').write_text(yaml.safe_dump(dataset,allow_unicode=True,sort_keys=False),encoding='utf-8')
        manifest['counts']={split:sum(i['split']==split for i in manifest['items']) for split in ('train','val','test')}
        atomic_json(out/'manifest.json',manifest)
        return manifest
    except Exception as exc:
        atomic_json(out/'failed.json',{'error':str(exc),'note':'Incomplete snapshot retained for diagnosis; not trainable'})
        raise


def validate_snapshot(directory: Path) -> dict:
    directory=Path(directory).resolve();manifest=read_json(directory/'manifest.json')
    if not manifest or (directory/'failed.json').exists():raise ValueError('数据版本未成功创建')
    for item in manifest['items']:
        for key in ('image','label'):
            path=(directory/item[key]).resolve()
            if not path.is_relative_to(directory) or not path.is_file() or sha256(path)!=item[key+'_sha256']:
                raise ValueError('数据版本内容已改变，拒绝复用该版本训练')
    return manifest


def list_datasets(project: Path) -> list[dict]:
    paths=sorted((Path(project)/'data/datasets').glob('*/manifest.json'),reverse=True)
    return [{k:v for k,v in read_json(p).items() if k not in ('items','excluded')} for p in paths if not (p.parent/'failed.json').exists()]
