from __future__ import annotations

from dataclasses import dataclass
import logging
import cv2
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from .localization import preprocess, patch, locate_template, dark_center, optical_step, masked_ncc

LOG = logging.getLogger(__name__)
ACCEPTED = {"seed", "manual_anchor", "tracked", "reacquired"}


def record(frame: int, ident: int, radius: float, point=None, status="lost", **quality) -> dict:
    x,y = (float(point[0]),float(point[1])) if point is not None else (np.nan,np.nan)
    row = dict(frame=frame, track_id=ident, detection_id=f"{frame}:{ident}" if point is not None else "",
               center_x=x, center_y=y, x1=x-radius, y1=y-radius, x2=x+radius, y2=y+radius,
               width=2*radius, height=2*radius, radius_px=radius, roi_area_px2=np.pi*radius**2,
               mask_area_px2=np.nan, confidence=np.nan, class_id=0, status=status,
               observed=point is not None, coordinate_type="roi_center", quality_reason="")
    row.update(quality)
    return row


@dataclass
class State:
    point: np.ndarray
    radius: int
    template: np.ndarray
    last_frame: int
    misses: int = 0


def track_local(frames: list[np.ndarray], seeds: list[dict], config: dict, drift: pd.DataFrame) -> pd.DataFrame:
    tc = config["tracker"]
    images = [preprocess(f,config["preprocessing"]) for f in frames]
    anchors = {(s["frame"],s["id"]):s for s in seeds}
    states: dict[int,State] = {}
    firsts = {i:min(s["frame"] for s in seeds if s["id"]==i) for i in {s["id"] for s in seeds}}
    rows = []
    for fi,im in enumerate(images):
        proposals = {}
        for ident in sorted(firsts):
            if fi < firsts[ident]: continue
            anchor = anchors.get((fi,ident))
            if anchor:
                r=int(round(anchor["radius"]))
                pos=np.array([anchor["x"],anchor["y"]],float)
                states[ident]=State(pos,r,patch(im,pos,r),fi)
                proposals[ident] = record(fi,ident,r,pos,"seed" if fi==firsts[ident] else "manual_anchor",
                                          confidence=1.,quality_reason="manual_initialization")
                continue
            st=states[ident]
            if st.misses > tc["memory"]:
                proposals[ident]=record(fi,ident,st.radius,status="lost",quality_reason="memory_exhausted_requires_manual_anchor")
                continue
            expected=st.point.copy()
            if config["registration"]["enabled"] and drift.loc[st.last_frame:fi,"valid"].all():
                expected += drift.loc[fi,["drift_x","drift_y"]].to_numpy(float)-drift.loc[st.last_frame,["drift_x","drift_y"]].to_numpy(float)
            gate=float(tc["search_range"])*min(1.5,np.sqrt(fi-st.last_frame))
            ncc_pos,score,margin=locate_template(im,st.template,expected,gate)
            back,_,_=locate_template(images[st.last_frame],patch(im,ncc_pos,st.radius),st.point,gate)
            back_error=float(np.linalg.norm(back-st.point))
            if tc["type"]=="lk":
                candidate,back_error=optical_step(images[st.last_frame],im,st.point,tc)
                score=float(masked_ncc(patch(im,candidate,st.radius),st.template)[0,0])
                margin=np.nan
                valid=(back_error<=tc["lk_max_fb_error"] and np.linalg.norm(candidate-expected)<=gate)
            else:
                candidate=ncc_pos
                valid=back_error<=tc["max_backward_error"] and margin>=tc["min_peak_margin"]
            refined,contrast=dark_center(im,candidate,st.radius,tc["refine_fraction"],tc["max_refine_shift"])
            if tc["refine_center"] and tc["type"]!="lk": candidate=refined
            h,w=im.shape
            inside=0<=candidate[0]<w and 0<=candidate[1]<h
            valid=bool(valid and score>=tc["min_ncc"] and contrast>=tc["min_contrast"] and inside)
            reason=[]
            if score<tc["min_ncc"]: reason.append("low_ncc")
            if np.isfinite(margin) and margin<tc["min_peak_margin"]: reason.append("ambiguous_peak")
            if back_error>(tc["lk_max_fb_error"] if tc["type"]=="lk" else tc["max_backward_error"]): reason.append("backward_inconsistent")
            if contrast<tc["min_contrast"]: reason.append("low_particle_contrast")
            if not inside: reason.append("outside_image")
            proposals[ident]=record(fi,ident,st.radius,candidate if valid else None,
                ("reacquired" if st.misses else "tracked") if valid else "uncertain",
                confidence=score,ncc_peak_margin=margin,backward_error_px=back_error,
                local_contrast=contrast,candidate_x=float(candidate[0]),candidate_y=float(candidate[1]),
                quality_reason=";".join(reason))
        # Reject collisions rather than allowing two selected IDs to silently occupy one object.
        ids=list(proposals)
        collision=set()
        for ix,a in enumerate(ids):
            for b in ids[ix+1:]:
                ra,rb=proposals[a],proposals[b]
                if ra["observed"] and rb["observed"]:
                    dist=np.hypot(ra["center_x"]-rb["center_x"],ra["center_y"]-rb["center_y"])
                    if dist<tc["conflict_fraction"]*(ra["radius_px"]+rb["radius_px"]): collision.update([a,b])
        for ident,row in proposals.items():
            st=states[ident]
            if ident in collision and (fi,ident) not in anchors:
                for k in ("center_x","center_y","x1","y1","x2","y2"): row[k]=np.nan
                row.update(observed=False,status="uncertain",quality_reason="selected_target_collision",detection_id="")
            if (fi,ident) not in anchors:
                if row["observed"]:
                    st.point=np.array([row["center_x"],row["center_y"]])
                    st.template=patch(im,st.point,st.radius)
                    st.last_frame=fi; st.misses=0
                else: st.misses+=1
            rows.append(row)
    return pd.DataFrame(rows)


def detections(frames: list[np.ndarray], config: dict) -> pd.DataFrame:
    dc=config["detector"]
    if dc["type"]=="yolo":
        from pathlib import Path
        if not dc.get("model") or not Path(dc["model"]).is_file():
            raise ValueError("YOLO requires an existing, TEM-trained model file; no automatic generic weights")
        from ultralytics import YOLO
        import torch
        model=YOLO(dc["model"])
        device=0 if torch.cuda.is_available() else "cpu"
        data=[]
        for fi,f in enumerate(frames):
            result=model.predict(cv2.cvtColor(f,cv2.COLOR_GRAY2BGR),conf=dc["confidence"],iou=dc["iou"],device=device,verbose=False)[0]
            for box in result.boxes:
                x1,y1,x2,y2=box.xyxy[0].cpu().numpy()
                data.append(dict(frame=fi,x=(x1+x2)/2,y=(y1+y2)/2,width=x2-x1,height=y2-y1,confidence=float(box.conf[0])))
        return pd.DataFrame(data,columns=["frame","x","y","width","height","confidence"])
    import trackpy as tp
    tp.quiet()
    found=[]
    for fi,f in enumerate(frames):
        p=preprocess(f,config["preprocessing"])
        # Exclude text/scale and most featureless background for the provided TEM configuration.
        p=np.clip(255-p,0,255).astype(np.uint8)
        features=tp.locate(p,int(dc["diameter"]),minmass=dc["minmass"],separation=dc["separation"],
                           percentile=dc["percentile"],engine="python")
        for row in features.itertuples():
            if row.y<dc.get("exclude_top_px",0) or row.y>frames[0].shape[0]-dc.get("exclude_bottom_px",0): continue
            found.append(dict(frame=fi,x=row.x,y=row.y,width=dc["diameter"],height=dc["diameter"],confidence=np.nan,mass=row.mass))
    return pd.DataFrame(found,columns=["frame","x","y","width","height","confidence","mass"])


def track_trackpy(frames: list[np.ndarray], seeds: list[dict], config: dict, drift: pd.DataFrame) -> pd.DataFrame:
    import trackpy as tp
    tp.quiet()
    if len({s["id"] for s in seeds})!=len(seeds):
        raise ValueError("Trackpy comparison accepts one initial anchor per ID; use template for manual corrections")
    det=detections(frames,config)
    if det.empty: return pd.DataFrame([record(f,s["id"],s["radius"]) for s in seeds for f in range(s["frame"],len(frames))])
    det=det.merge(drift[["frame","drift_x","drift_y"]],on="frame")
    det["link_x"]=det.x-det.drift_x;det["link_y"]=det.y-det.drift_y
    linked=tp.link_df(det,search_range=config["tracker"]["search_range"],memory=config["tracker"]["memory"],
                       pos_columns=["link_x","link_y"],adaptive_stop=5,adaptive_step=.9)
    mappings={}
    # Batch seed association per frame, with a one-to-one constraint.
    for fi in sorted({s["frame"] for s in seeds}):
        ss=[s for s in seeds if s["frame"]==fi]; ff=linked[linked.frame==fi]
        if ff.empty: continue
        costs=np.linalg.norm(np.array([[s["x"],s["y"]] for s in ss])[:,None]-ff[["x","y"]].to_numpy()[None],axis=2)
        a,b=linear_sum_assignment(costs)
        for ia,ib in zip(a,b):
            if costs[ia,ib]<=ss[ia]["radius"]: mappings[ss[ia]["id"]]=int(ff.iloc[ib].particle)
    rows=[]
    for s in seeds:
        sub=linked[linked.particle==mappings.get(s["id"],-1)].set_index("frame")
        for fi in range(s["frame"],len(frames)):
            if fi in sub.index:
                p=sub.loc[fi]
                rows.append(record(fi,s["id"],s["radius"],(p.x,p.y),"tracked",confidence=np.nan))
            else: rows.append(record(fi,s["id"],s["radius"],status="lost"))
    return pd.DataFrame(rows)


def box_iou(a: np.ndarray,b: np.ndarray) -> float:
    alo=a[:2]-a[2:4]/2;ahi=a[:2]+a[2:4]/2
    blo=b[:2]-b[2:4]/2;bhi=b[:2]+b[2:4]/2
    inter=float(np.prod(np.maximum(0,np.minimum(ahi,bhi)-np.maximum(alo,blo))))
    return inter/max(1e-9,float(np.prod(a[2:4])+np.prod(b[2:4])-inter))


def track_legacy(frames: list[np.ndarray],seeds: list[dict],config: dict,drift: pd.DataFrame,
                 observations: pd.DataFrame|None=None) -> pd.DataFrame:
    lc=config["legacy"]; det=detections(frames,config) if observations is None else observations
    anchors={(s["frame"],s["id"]):s for s in seeds}
    states={};rows=[]
    for fi in range(len(frames)):
        for (af,ident),s in anchors.items():
            if af!=fi: continue
            k=cv2.KalmanFilter(6,4)
            k.transitionMatrix=np.eye(6,dtype=np.float32);k.transitionMatrix[0,4]=1;k.transitionMatrix[1,5]=1
            k.measurementMatrix=np.eye(4,6,dtype=np.float32)
            k.processNoiseCov=np.eye(6,dtype=np.float32)*lc["process_noise"]
            k.measurementNoiseCov=np.eye(4,dtype=np.float32)*lc["measurement_noise"]
            k.errorCovPost=np.eye(6,dtype=np.float32)
            k.statePost=np.array([s["x"],s["y"],2*s["radius"],2*s["radius"],0,0],np.float32).reshape(-1,1)
            states[ident] = [k,0,s["radius"]]
        ids=[i for i in sorted(states) if states[i][1]<lc["max_age"]]
        dd=det[det.frame==fi]
        ds=dd[["x","y","width","height"]].to_numpy(float)
        predictions=[]
        for ident in ids: predictions.append(states[ident][0].predict().ravel()[:4].astype(float))
        costs=np.ones((len(ids),len(ds)))*1e6
        for i,p in enumerate(predictions):
            for j,d in enumerate(ds):
                iou=1-box_iou(p,d); area=abs(d[2]*d[3]-p[2]*p[3])/max(p[2]*p[3],1e-9)
                distance=np.linalg.norm(p[:2]-d[:2])/config["tracker"]["search_range"]
                mode=lc["cost"]
                cost=iou if mode=="iou" else (.7*iou+.3*distance if mode=="iou_distance" else distance if mode=="motion" else lc["iou_weight"]*iou+lc["area_weight"]*area)
                costs[i,j]=cost
        matched={}
        if costs.size:
            rr,cc=linear_sum_assignment(costs)
            matched={ids[r]:c for r,c in zip(rr,cc) if costs[r,c]<=lc["threshold"]}
        for ident,(k,misses,radius) in states.items():
            anchor=anchors.get((fi,ident))
            if anchor and observations is None:
                rows.append(record(fi,ident,radius,(anchor["x"],anchor["y"]),"seed"));continue
            if ident in matched:
                d=ds[matched[ident]];k.correct(d.astype(np.float32).reshape(4,1));states[ident][1]=0
                # Report actual detection centers, not filtered predictions as measurements.
                row=record(fi,ident,radius,d[:2],('seed' if fi==min(s['frame'] for s in seeds if s['id']==ident) else 'manual_anchor') if anchor else 'tracked')
                if observations is not None:
                    obs=dd.iloc[matched[ident]]
                    row.update(confidence=float(obs.confidence),class_id=0,x1=float(d[0]-d[2]/2),y1=float(d[1]-d[3]/2),
                               x2=float(d[0]+d[2]/2),y2=float(d[1]+d[3]/2),width=float(d[2]),height=float(d[3]),
                               bbox_area_px2=float(d[2]*d[3]),coordinate_type='learned_detection_box_center',
                               detection_id=f'{fi}:{int(obs.detection_index)}',backend_track_id=ident)
                rows.append(row)
            else:
                states[ident][1]+=1;rows.append(record(fi,ident,radius,status="lost"))
    return pd.DataFrame(rows)
