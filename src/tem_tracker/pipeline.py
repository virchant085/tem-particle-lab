from __future__ import annotations
import logging
import random
import time
from pathlib import Path
import numpy as np
import pandas as pd
import yaml
from .io import load_video,validate_seeds,environment,save_json,sha256
from .registration import estimate_drift
from .tracking import track_local,track_trackpy,track_legacy
from .analysis import enrich,statistics
from .visualization import render_video,plots


def run(input_path: str|Path,config: dict,seed_data: dict,output: str|Path) -> dict:
    out=Path(output).resolve()
    if out.exists() and any(out.iterdir()): raise FileExistsError(f"Output directory must be new or empty: {out}")
    if out==Path(input_path).resolve().parent: raise ValueError("Output must be separate from the input directory")
    out.mkdir(parents=True,exist_ok=True)
    logger=logging.getLogger("tem_tracker")
    logger.setLevel(logging.INFO)
    handler=logging.FileHandler(out/"logs.txt",encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"));logger.addHandler(handler)
    start=time.perf_counter()
    try:
        random.seed(config["seed"]);np.random.seed(config["seed"]);cv_seed=int(config["seed"])
        import cv2
        cv2.setRNGSeed(cv_seed);cv2.setNumThreads(1)
        logger.info("Loading input read-only: %s",input_path)
        frames,meta=load_video(input_path,config["output"]["max_memory_mb"])
        seeds=validate_seeds(seed_data,frames[0].shape,len(frames))
        (out/"config.yaml").write_text(yaml.safe_dump(config,allow_unicode=True,sort_keys=False),encoding="utf-8")
        save_json(out/"seeds.json",dict(seed_data,particles=seeds));save_json(out/"input_metadata.json",meta)
        save_json(out/"environment.json",environment())
        project=Path(__file__).resolve().parents[2]
        source_files=list((project/"src").rglob("*.py"))+list((project/"scripts").glob("*.py"))+list((project/"ui").glob("*.html"))
        save_json(out/"source_manifest.json",{str(p.relative_to(project)).replace("\\","/"):sha256(p) for p in sorted(source_files)})
        logger.info("Decoded %s frames; selected IDs %s",len(frames),sorted({s["id"] for s in seeds}))
        drift=estimate_drift(frames,config["registration"]);drift.to_csv(out/"drift.csv",index=False)
        method=config["tracker"]["type"]
        runner={"template":track_local,"lk":track_local,"trackpy":track_trackpy,"legacy":track_legacy}.get(method)
        if runner is None: raise ValueError(f"Unknown tracker: {method}")
        raw=runner(frames,seeds,config,drift)
        tracks=enrich(raw,drift,meta,config)
        if tracks.duplicated(["frame","track_id"]).any(): raise RuntimeError("Duplicate identity in a frame")
        if tracks.loc[~tracks.observed,["center_x","center_y"]].notna().any().any(): raise RuntimeError("Missing observations must not have measured coordinates")
        tracks.to_csv(out/"tracks.csv",index=False,float_format="%.7f")
        stats=statistics(tracks);stats.to_csv(out/"track_statistics.csv",index=False,float_format="%.7f")
        tracks[~tracks.observed].to_csv(out/"review_required.csv",index=False)
        tracking_seconds=time.perf_counter()-start
        if config["output"]["video"]:render_video(frames,tracks,config,meta["container_fps"],out/"annotated.mp4")
        plots(frames,tracks,out,config)
        after=sha256(Path(input_path));unchanged=after==meta["sha256"]
        if not unchanged: raise RuntimeError("Input changed during processing")
        metrics=dict(method=method,frames=len(frames),selected_ids=sorted({s["id"] for s in seeds}),
            observation_rows=len(tracks),observed_rows=int(tracks.observed.sum()),
            coverage=float(tracks.observed.mean()),review_rows=int((~tracks.observed).sum()),
            registration_valid_steps=int(drift.valid.sum()-1),registration_total_steps=len(drift)-1,
            tracking_seconds=tracking_seconds,total_seconds=time.perf_counter()-start,
            raw_input_unchanged=unchanged,calibration=config["calibration"],
            accuracy_metrics={"HOTA":None,"IDF1":None,"MOTA":None,"mAP":None},
            accuracy_status="not_evaluated_without_ground_truth",confidence_kind="NCC similarity, not calibrated probability",
            coordinate_definition="manually initialized ROI center; optional local dark-intensity centroid refinement; no mask area")
        save_json(out/"metrics.json",metrics)
        summary=(f"# TEM tracking result\n\nMethod: {method}; frames: {len(frames)}; selected IDs: {metrics['selected_ids']}.\n\n"
                 f"Observed {metrics['observed_rows']}/{len(tracks)} selected target-frames. Coverage is not accuracy. "
                 f"Review {metrics['review_rows']} missing/uncertain target-frames in review_required.csv.\n\n"
                 "Green circles are fixed selected ROIs, yellow paths link only consecutive observed centers. "
                 "No invented coordinates on missing frames. A manual anchor can restart a lost ID.\n\n"
                 "tracks.csv keeps raw image coordinates and separately background-relative coordinates; "
                 "the latter are invalid after an untrusted drift step. This estimates motion relative to the selected background, not proven stage drift.\n\n"
                 "acquisition_fps and nm_per_pixel remain null unless explicitly supplied. Video time is playback time. "
                 "The embedded experimental clock/scale are not automatically interpreted. ROI area is not segmented particle area.\n\n"
                 "No formal GT metrics are asserted. See comparison outputs for explicitly scoped visual-reference checks.\n")
        (out/"summary.md").write_text(summary,encoding="utf-8")
        logger.info("Completed: %s",metrics)
        return metrics
    except Exception:
        logger.exception("Run failed; partial output is retained for diagnosis")
        raise
    finally:
        logger.removeHandler(handler);handler.close()
