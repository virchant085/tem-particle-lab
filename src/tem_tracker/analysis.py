from __future__ import annotations
import numpy as np
import pandas as pd


def enrich(tracks: pd.DataFrame, drift: pd.DataFrame, metadata: dict, config: dict) -> pd.DataFrame:
    out=tracks.sort_values(["track_id","frame"]).reset_index(drop=True).merge(drift,on="frame",how="left")
    cal=config["calibration"]
    out["video_time_s"]=[metadata["video_timestamps_s"][i] for i in out.frame]
    out["acquisition_time_s"]=out.frame/cal["acquisition_fps"] if cal["acquisition_fps"] else np.nan
    out["time_basis"]="acquisition" if cal["acquisition_fps"] else "video_playback_only"
    out["timestamp"]=out.acquisition_time_s if cal["acquisition_fps"] else out.video_time_s
    out["nm_per_pixel"]=cal["nm_per_pixel"]
    corrected=out.cumulative_valid & out.observed & config["registration"]["enabled"]
    out["x_background_relative_px"]=np.where(corrected,out.center_x-out.drift_x,np.nan)
    out["y_background_relative_px"]=np.where(corrected,out.center_y-out.drift_y,np.nan)
    out["x_nm"]=out.center_x*cal["nm_per_pixel"] if cal["nm_per_pixel"] else np.nan
    out["y_nm"]=out.center_y*cal["nm_per_pixel"] if cal["nm_per_pixel"] else np.nan
    out=out.sort_values(["track_id","frame"]).reset_index(drop=True)
    group=out.groupby("track_id")
    valid=(out.observed & group["observed"].shift(1).fillna(False).astype(bool)
           & group.frame.diff().eq(1) & ~out.status.isin(["manual_anchor", "seed"]))
    step=np.hypot(group.center_x.diff(),group.center_y.diff())
    out["step_px"]=np.where(valid,step,np.nan)
    out["speed_px_per_video_s"]=out.step_px/group.video_time_s.diff()
    out["speed_nm_per_acquisition_s"]=out.step_px*cal["nm_per_pixel"]*cal["acquisition_fps"] if cal["nm_per_pixel"] and cal["acquisition_fps"] else np.nan
    return out.sort_values(["frame","track_id"]).reset_index(drop=True)


def statistics(tracks: pd.DataFrame) -> pd.DataFrame:
    rows=[]
    for ident,g in tracks.groupby("track_id"):
        obs=g[g.observed].sort_values("frame")
        rows.append(dict(track_id=int(ident),frames_requested=len(g),observations=len(obs),missing=len(g)-len(obs),
            coverage=float(len(obs)/len(g)),first_observed_frame=int(obs.frame.min()) if len(obs) else None,
            last_observed_frame=int(obs.frame.max()) if len(obs) else None,
            observed_path_length_px=float(obs.step_px.sum()),
            net_displacement_px=float(np.hypot(obs.center_x.iloc[-1]-obs.center_x.iloc[0],obs.center_y.iloc[-1]-obs.center_y.iloc[0])) if len(obs)>1 else None,
            mean_step_px=float(obs.step_px.mean()),max_step_px=float(obs.step_px.max()),
            uncertain_frames=int((g.status=="uncertain").sum()),
            gap_recoveries=int((g.status=="reacquired").sum())))
    return pd.DataFrame(rows)


def evaluate_reference(tracks: pd.DataFrame, reference: pd.DataFrame, tolerance: float=12.) -> dict:
    """Selected-identity localization evaluation; not a substitute for official MOT metrics."""
    if reference.duplicated(["frame","track_id"]).any(): raise ValueError("Duplicate reference identity/frame")
    if tracks.duplicated(["frame","track_id"]).any(): raise ValueError("Duplicate prediction identity/frame")
    ref=reference.copy()
    if "visible" in ref: ref=ref[ref.visible.astype(bool)]
    merged=ref.merge(tracks[["frame","track_id","center_x","center_y","observed"]],on=["frame","track_id"],how="left",suffixes=("_ref","_pred"))
    err=np.hypot(merged.center_x_ref-merged.center_x_pred,merged.center_y_ref-merged.center_y_pred)
    found=merged.observed.fillna(False).astype(bool)
    return dict(reference_points=len(ref),observed_reference_points=int(found.sum()),
                coverage=float(found.mean()) if len(ref) else None,
                mean_center_error_px=float(err[found].mean()),median_center_error_px=float(err[found].median()),
                p95_center_error_px=float(err[found].quantile(.95)),
                success_within_tolerance=float((found & (err<=tolerance)).mean()) if len(ref) else None,
                tolerance_px=tolerance,HOTA=None,IDF1=None,MOTA=None,
                evaluation_scope="selected identity point localization; human/assistant reference provenance must be supplied")
