"""Compare three associations on exactly the same saved learned detections."""
from pathlib import Path
import argparse,copy,json,sys
import numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from tem_tracker.io import load_config,load_video,validate_seeds,save_json
from tem_tracker.learning.data import read_json
from tem_tracker.learning.tracking import associate_detections,selected_trajectories
from tem_tracker.tracking import track_legacy
from tem_tracker.analysis import enrich,statistics,evaluate_reference
from tem_tracker.visualization import render_video,plots

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--input',type=Path,help='Override the recorded original input path');p.add_argument('--reference',type=Path)
    p.add_argument('--tolerance',type=float,default=12.);a=p.parse_args()
    if a.output.exists() and any(a.output.iterdir()):raise ValueError('Use a fresh output directory')
    a.output.mkdir(parents=True,exist_ok=True)
    cfg=read_json(a.run/'learning_config.json');base=load_config(a.run/'config.yaml');meta=read_json(a.run/'input_metadata.json')
    frames,actual=load_video(a.input or Path(meta['input']),base['output']['max_memory_mb'])
    if actual['sha256']!=meta['sha256']:raise ValueError('Input differs from the saved detector run')
    predictions=read_json(a.run/'detections.json');drift=pd.read_csv(a.run/'drift.csv');seeds=validate_seeds(read_json(a.run/'seeds.json'),frames[0].shape,len(frames))
    if len(predictions)!=len(frames):raise ValueError('Detection/frame count mismatch')
    summary=[]
    for method in ['bytetrack','trackpy','legacy_kalman_hungarian']:
        out=a.output/method;out.mkdir();current=copy.deepcopy(cfg)
        if method=='legacy_kalman_hungarian':
            found=[dict(frame=fi,x=(d['x1']+d['x2'])/2,y=(d['y1']+d['y2'])/2,width=d['x2']-d['x1'],height=d['y2']-d['y1'],
                        confidence=d['confidence'],detection_index=d['detection_index'])
                   for fi,items in enumerate(predictions) for d in items if d['confidence']>=cfg['association']['track_high_thresh']]
            obs=pd.DataFrame(found,columns=['frame','x','y','width','height','confidence','detection_index'])
            raw=track_legacy(frames,seeds,base,drift,observations=obs)
        else:
            current['association']['method']=method
            raw=selected_trajectories(associate_detections(predictions,frames[0].shape,current,drift),seeds,len(frames),current)
        tracks=enrich(raw,drift,meta,base);tracks.to_csv(out/'tracks.csv',index=False)
        statistics(tracks).to_csv(out/'track_statistics.csv',index=False)
        render_video(frames,tracks,base,meta['container_fps'],out/'annotated.mp4');plots(frames,tracks,out,base)
        result=dict(method=method,observed_rows=int(tracks.observed.sum()),requested_rows=len(tracks),
                    coverage=float(tracks.observed.mean()),coverage_is_accuracy=False)
        if a.reference:result.update(evaluate_reference(tracks,pd.read_csv(a.reference),a.tolerance))
        save_json(out/'metrics.json',result);summary.append(result)
    pd.DataFrame(summary).to_csv(a.output/'comparison.csv',index=False)
    save_json(a.output/'comparison.json',dict(source_run=str(a.run.resolve()),model_card=read_json(a.run/'model_card.json'),config=cfg,
        legacy_config=base['legacy'],results=summary,reference=str(a.reference) if a.reference else None,
        note='Identical learned detections; coverage alone does not establish identity accuracy or a winning method.'))
    print(json.dumps(summary,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
