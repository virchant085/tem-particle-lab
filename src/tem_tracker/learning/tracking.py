from __future__ import annotations

import copy,json,time
from pathlib import Path
from types import SimpleNamespace
from typing import Callable
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
import yaml
from ..io import load_config,load_video,validate_seeds,sha256,save_json,environment
from ..tracking import record
from ..registration import estimate_drift
from ..analysis import enrich,statistics
from ..visualization import render_video,plots
from .models import ParticleDetector


def associate_detections(detections: list[list[dict]], shape: tuple[int,int], config: dict,
                         drift: pd.DataFrame|None=None) -> pd.DataFrame:
    """Reuse upstream association; retain original detector observations for scientific export."""
    ac=config['association'];rows=[]
    offsets=np.zeros((len(detections),2))
    if ac.get('background_compensation',False):
        if drift is None or not drift.cumulative_valid.all():
            raise ValueError('关联漂移补偿需要全程可信配准；请关闭该选项或选择更稳定的背景区域')
        offsets=drift[['drift_x','drift_y']].to_numpy(float)
    if ac['method']=='bytetrack':
        from ultralytics.trackers.byte_tracker import BYTETracker
        from ultralytics.engine.results import Boxes
        args=SimpleNamespace(**{k:ac[k] for k in ('track_high_thresh','track_low_thresh','new_track_thresh','track_buffer','match_thresh','fuse_score')})
        tracker=BYTETracker(args) # locked Ultralytics 8.4.165 uses track_buffer directly in frames
        for fi,items in enumerate(detections):
            data=np.array([[d['x1']-offsets[fi,0],d['y1']-offsets[fi,1],
                            d['x2']-offsets[fi,0],d['y2']-offsets[fi,1],d['confidence'],0] for d in items],dtype=np.float32).reshape(-1,6)
            associated=tracker.update(Boxes(data,shape))
            # New objects after frame zero are tentative for one update upstream.
            # Keep their actual high-confidence observations so a user can select
            # them on their first visible frame. No Kalman-only positions are added.
            present={int(t[4]) for t in associated}
            tentative=[t.result for t in tracker.tracked_stracks if not t.is_activated and t.track_id not in present]
            if tentative:associated=np.concatenate([associated.reshape(-1,8),np.asarray(tentative).reshape(-1,8)])
            for linked in associated:
                index=int(linked[-1])
                if not 0<=index<len(items):raise RuntimeError('ByteTrack 返回了无效检测索引，请检查锁定的依赖版本')
                observation=items[index]
                rows.append(dict(observation,frame=fi,backend_id=int(linked[4]),
                    center_x=(observation['x1']+observation['x2'])/2,
                    center_y=(observation['y1']+observation['y2'])/2,
                    association_center_x=float((linked[0]+linked[2])/2+offsets[fi,0]),
                    association_center_y=float((linked[1]+linked[3])/2+offsets[fi,1])))
    elif ac['method']=='trackpy':
        import trackpy as tp
        tp.quiet();candidates=[]
        for fi,items in enumerate(detections):
            for d in items:
                if d['confidence']<ac['track_high_thresh']:continue
                x=(d['x1']+d['x2'])/2;y=(d['y1']+d['y2'])/2
                candidates.append(dict(d,frame=fi,center_x=x,center_y=y,link_x=x-offsets[fi,0],link_y=y-offsets[fi,1]))
        if candidates:
            linked=tp.link_df(pd.DataFrame(candidates),search_range=ac['point_search_range'],memory=ac['point_memory'],
                              pos_columns=['link_x','link_y'],adaptive_stop=5,adaptive_step=.9)
            for row in linked.to_dict('records'):
                row['backend_id']=int(row.pop('particle'))+1;rows.append(row)
    else:raise ValueError('不支持的关联器')
    columns=['frame','backend_id','detection_index','x1','y1','x2','y2','center_x','center_y','confidence','class_id']
    return pd.DataFrame(rows) if rows else pd.DataFrame(columns=columns)


def selected_trajectories(linked: pd.DataFrame, seeds: list[dict], nframes: int, config: dict) -> pd.DataFrame:
    anchors={(s['frame'],s['id']):s for s in seeds}
    firsts={i:min(s['frame'] for s in seeds if s['id']==i) for i in {s['id'] for s in seeds}}
    radii={};mapping={};last={};misses={};stopped=set();rows=[]
    max_dist=float(config['selection']['max_anchor_distance_px'])
    max_step=float(config['association']['max_link_step_px'])
    for fi in range(nframes):
        candidates=linked[linked.frame==fi].reset_index(drop=True)
        current=[s for s in seeds if s['frame']==fi]
        reserved={bid for i,bid in mapping.items() if (fi,i) not in anchors}
        available=candidates[~candidates.backend_id.isin(reserved)].reset_index(drop=True)
        for s in current:
            mapping.pop(s['id'],None);radii[s['id']]=s['radius'];last.pop(s['id'],None);misses[s['id']]=0;stopped.discard(s['id'])
        if current and len(available):
            costs=np.linalg.norm(np.array([[s['x'],s['y']] for s in current])[:,None]-available[['center_x','center_y']].to_numpy()[None],axis=2)
            # Dummy columns allow unmatched anchors without stealing another anchor's valid match.
            augmented=np.hstack([np.where(costs<=max_dist,costs,1e6),np.full((len(current),len(current)),max_dist+1)])
            rr,cc=linear_sum_assignment(augmented)
            for r,c in zip(rr,cc):
                if c<len(available) and costs[r,c]<=max_dist:mapping[current[r]['id']]=int(available.iloc[c].backend_id)
        for ident in sorted(firsts):
            if fi<firsts[ident]:continue
            anchor=anchors.get((fi,ident));bid=mapping.get(ident)
            choice=candidates[candidates.backend_id==bid] if bid is not None else candidates.iloc[:0]
            reason='selected_anchor_has_no_matching_detection' if bid is None else 'selected_track_not_observed'
            if ident in stopped:choice=candidates.iloc[:0];reason='repeated_untrusted_jump_requires_manual_anchor'
            if len(choice):
                d=choice.iloc[0];point=np.array([d.center_x,d.center_y],float)
                if ident in last and not anchor:
                    old_frame,old_point=last[ident]
                    if np.linalg.norm(point-old_point)>max_step*np.sqrt(fi-old_frame):
                        choice=candidates.iloc[:0];reason='association_displacement_exceeds_gate'
                if len(choice):
                    status=('seed' if fi==firsts[ident] else 'manual_anchor') if anchor else ('reacquired' if misses.get(ident,0) else 'tracked')
                    row=record(fi,ident,radii[ident],point,status,confidence=float(d.confidence),
                        backend_track_id=int(bid),coordinate_type='learned_detection_box_center',quality_reason='')
                    row.update(x1=float(d.x1),y1=float(d.y1),x2=float(d.x2),y2=float(d.y2),
                               width=float(d.x2-d.x1),height=float(d.y2-d.y1),
                               detection_id=f'{fi}:{int(d.detection_index)}',bbox_area_px2=float((d.x2-d.x1)*(d.y2-d.y1)))
                    rows.append(row);last[ident]=(fi,point);misses[ident]=0;continue
            misses[ident]=misses.get(ident,0)+1
            if reason=='association_displacement_exceeds_gate' and misses[ident]>config['association']['track_buffer']:
                stopped.add(ident)
            rows.append(record(fi,ident,radii[ident],status='uncertain' if reason=='association_displacement_exceeds_gate' else 'lost',
                quality_reason=reason,backend_track_id=bid,coordinate_type='learned_detection_box_center'))
    return pd.DataFrame(rows)


def run_learned_tracking(project: Path, input_path: Path, model_id: str, seeds_data: dict,
                         config: dict, output: Path, calibration: dict|None=None,
                         progress: Callable|None=None, allow_smoke: bool=False) -> dict:
    project=Path(project).resolve();out=Path(output).resolve()
    if out.exists() and any(out.iterdir()):raise FileExistsError('结果目录必须为新目录')
    out.mkdir(parents=True,exist_ok=True);start=time.perf_counter()
    base=load_config(project/'config/default.yaml')
    if calibration:
        for key in ('acquisition_fps','nm_per_pixel'):
            value=calibration.get(key)
            if value is not None and (not np.isfinite(value) or value<=0):raise ValueError('标定值必须为有限正数')
            base['calibration'][key]=value
    base['output']['history']=config['selection']['history']
    frames,meta=load_video(input_path,base['output']['max_memory_mb'])
    seeds=validate_seeds(seeds_data,frames[0].shape,len(frames))
    detector=ParticleDetector(project,model_id,config,allow_smoke)
    save_json(out/'model_card.json',detector.card);save_json(out/'input_metadata.json',meta)
    save_json(out/'hardware.json',detector.hardware);save_json(out/'environment.json',dict(environment(),**detector.hardware))
    save_json(out/'seeds.json',{'particles':seeds});save_json(out/'learning_config.json',config)
    (out/'config.yaml').write_text(yaml.safe_dump(dict(base,learning=config),allow_unicode=True,sort_keys=False),encoding='utf-8')
    drift=estimate_drift(frames,base['registration']);drift.to_csv(out/'drift.csv',index=False)
    predictions=[]
    for fi,frame in enumerate(frames):
        predictions.append(detector.predict(frame))
        if progress:progress({'stage':'detecting','frame':fi+1,'frames':len(frames),'message':f'识别颗粒 {fi+1}/{len(frames)} 帧'})
    save_json(out/'detections.json',predictions)
    linked=associate_detections(predictions,frames[0].shape,config,drift);linked.to_csv(out/'all_linked_candidates.csv',index=False)
    tracks=enrich(selected_trajectories(linked,seeds,len(frames),config),drift,meta,base)
    if tracks.duplicated(['frame','track_id']).any():raise RuntimeError('同一帧出现重复身份')
    if tracks.loc[~tracks.observed,['center_x','center_y']].notna().any().any():raise RuntimeError('缺测记录错误地带有测量坐标')
    tracks.to_csv(out/'tracks.csv',index=False,float_format='%.7f')
    tracks[~tracks.observed].to_csv(out/'review_required.csv',index=False)
    statistics(tracks).to_csv(out/'track_statistics.csv',index=False)
    if progress:progress({'stage':'exporting','message':'生成绿色圆圈、黄色轨迹及坐标文件'})
    render_video(frames,tracks,base,meta['container_fps'],out/'annotated.mp4');plots(frames,tracks,out,base)
    if sha256(input_path)!=meta['sha256']:raise RuntimeError('运行期间输入视频发生改变')
    metrics=dict(method='supervised_yolo_'+config['association']['method'],model_id=model_id,
        model_weights_sha256=detector.card['weights_sha256'],frames=len(frames),selected_ids=sorted(tracks.track_id.unique().tolist()),
        observation_rows=len(tracks),observed_rows=int(tracks.observed.sum()),review_rows=int((~tracks.observed).sum()),
        coverage=float(tracks.observed.mean()),coverage_is_accuracy=False,calibration=base['calibration'],
        raw_input_unchanged=True,elapsed_seconds=time.perf_counter()-start,
        coordinate_definition='raw learned detector bounding-box center; Kalman positions retained only as association diagnostics',
        confidence_kind='detector score; not calibrated correctness probability',
        model_qualification=detector.card['qualification'],accuracy_metrics={'HOTA':None,'IDF1':None,'MOTA':None},
        note='Selected trajectory identity accuracy requires independent tracking annotations')
    save_json(out/'metrics.json',metrics)
    source_root=Path(__file__).resolve().parents[3]
    sources=list((source_root/'src').rglob('*.py'))+list((source_root/'scripts').glob('*.py'))
    save_json(out/'source_manifest.json',{p.relative_to(source_root).as_posix():sha256(p) for p in sources})
    (out/'summary.md').write_text(f'# 学习模型追踪结果\n\n模型：{model_id}；关联：{config["association"]["method"]}。\n\n'
        f'观测 {metrics["observed_rows"]}/{len(tracks)}，待复核 {metrics["review_rows"]}；观测率不是准确率。\n\n'
        '轨迹来自检测框中心，缺测不填补坐标；黄色线不跨缺测或人工重设点。标定保持输入状态。\n',encoding='utf-8')
    return metrics
