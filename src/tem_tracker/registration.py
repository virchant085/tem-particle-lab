from __future__ import annotations
import cv2
import numpy as np
import pandas as pd


def estimate_drift(frames: list[np.ndarray], config: dict) -> pd.DataFrame:
    h,w = frames[0].shape
    x0,y0,x1,y1 = [int(v*s) for v,s in zip(config["roi_fraction"], [w,h,w,h])]
    if not (0<=x0<x1<=w and 0<=y0<y1<=h) or min(x1-x0,y1-y0)<16:
        raise ValueError("Invalid background registration ROI")
    cumulative = np.zeros(2)
    chain_valid = True
    rows = [dict(frame=0, dx=0., dy=0., drift_x=0., drift_y=0., valid=True,
                 cumulative_valid=True, phase_response=1., ecc_score=1., disagreement=0.)]
    if not config["enabled"]:
        return pd.DataFrame([dict(rows[0], frame=i) for i in range(len(frames))])
    crops = [cv2.GaussianBlur(f[y0:y1,x0:x1].astype(np.float32),(0,0),2) for f in frames]
    window = cv2.createHanningWindow((x1-x0,y1-y0), cv2.CV_32F)
    for i, (a,b) in enumerate(zip(crops[:-1],crops[1:]),1):
        shift,response = cv2.phaseCorrelate(a.copy(),b.copy(),window)
        phase = np.array(shift)
        warp = np.eye(2,3,dtype=np.float32)
        warp[:,2] = np.clip(phase,-config["max_step"],config["max_step"])
        try:
            ecc,warp = cv2.findTransformECC(a,b,warp,cv2.MOTION_TRANSLATION,
                (cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,60,1e-4))
            es = warp[:,2].astype(float)
        except cv2.error:
            ecc,es = -1.,np.full(2,np.nan)
        disagree = float(np.linalg.norm(es-phase))
        if config["method"] not in ("ecc","phase"):
            raise ValueError("registration.method must be ecc or phase")
        step = es if config["method"] == "ecc" else phase
        valid = bool(np.isfinite(step).all() and np.linalg.norm(step)<=config["max_step"]
                     and response>=config["min_phase_response"] and ecc>=config["min_ecc"]
                     and disagree<=config["max_method_disagreement"])
        chain_valid = chain_valid and valid
        if valid: cumulative += step
        rows.append(dict(frame=i, dx=float(step[0]) if valid else np.nan,
                         dy=float(step[1]) if valid else np.nan,
                         drift_x=float(cumulative[0]), drift_y=float(cumulative[1]),
                         valid=valid,cumulative_valid=chain_valid, phase_response=float(response),
                         ecc_score=float(ecc), disagreement=disagree))
    return pd.DataFrame(rows)
