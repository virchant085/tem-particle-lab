"""Controlled diagnostics on the same video and same manually selected IDs."""
from pathlib import Path
import argparse,copy,json,sys,time
import pandas as pd
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from tem_tracker.io import load_config,load_video,validate_seeds,save_json,environment,sha256
from tem_tracker.registration import estimate_drift
from tem_tracker.tracking import track_local,track_trackpy,track_legacy
from tem_tracker.analysis import enrich,evaluate_reference,statistics
from tem_tracker.visualization import plots

def main():
    root=Path(__file__).resolve().parents[1]
    p=argparse.ArgumentParser();p.add_argument("--input",type=Path,default=root/"data/raw/sample.mp4")
    p.add_argument("--seeds",type=Path,default=root/"config/selected_particles.json")
    p.add_argument("--config",type=Path,default=root/"config/default.yaml")
    p.add_argument("--reference",type=Path,default=root/"data/annotations/visual_reference.csv")
    p.add_argument("--output",type=Path,required=True);a=p.parse_args()
    if a.output.exists() and any(a.output.iterdir()):raise FileExistsError("Use a new comparison directory")
    a.output.mkdir(parents=True,exist_ok=True)
    cfg=load_config(a.config);frames,meta=load_video(a.input,cfg["output"]["max_memory_mb"])
    seeds=validate_seeds(json.loads(a.seeds.read_text(encoding="utf-8-sig")),frames[0].shape,len(frames))
    save_json(a.output/"input_metadata.json",meta)
    save_json(a.output/"seeds.json",{"particles":seeds})
    save_json(a.output/"environment.json",environment())
    save_json(a.output/"reference_provenance.json",{"path":str(a.reference.resolve()),"sha256":sha256(a.reference),
        "evaluation_frames":"frame > 0 from supplied reference", "note":"Provided example references are approximate and nonblind, not expert GT"})
    sources=list((root/"src").rglob("*.py"))+list((root/"scripts").glob("*.py"))+list((root/"ui").glob("*.html"))
    save_json(a.output/"source_manifest.json",{str(p.relative_to(root)).replace("\\","/"):sha256(p) for p in sorted(sources)})
    reference=pd.read_csv(a.reference);reference=reference[reference.frame>0]
    variants=[("selected",{}),("without_registration",{"registration.enabled":False}),
              ("raw_no_blur",{"preprocessing.gaussian_sigma":0}),
              ("without_centroid_refinement",{"tracker.refine_center":False}),
              ("strict_backward_4px",{"tracker.max_backward_error":4}),
              ("search_12",{"tracker.search_range":12}),("search_20",{"tracker.search_range":20}),
              ("search_40",{"tracker.search_range":40}),
              ("lk",{"tracker.type":"lk"}),("trackpy",{"tracker.type":"trackpy"}),
              ("legacy_iou_area",{"tracker.type":"legacy"}),
              ("legacy_iou",{"tracker.type":"legacy","legacy.cost":"iou"}),
              ("legacy_iou_distance",{"tracker.type":"legacy","legacy.cost":"iou_distance"}),
              ("legacy_motion",{"tracker.type":"legacy","legacy.cost":"motion"})]
    results=[]
    import cv2
    cv2.setNumThreads(1)
    for name,changes in variants:
        c=copy.deepcopy(cfg)
        for key,value in changes.items():section,item=key.split('.');c[section][item]=value
        directory=a.output/name;directory.mkdir()
        save_json(directory/"config.json",c);start=time.perf_counter()
        try:
            d=estimate_drift(frames,c["registration"])
            method=c["tracker"]["type"]
            runner={"template":track_local,"lk":track_local,"trackpy":track_trackpy,"legacy":track_legacy}[method]
            t=enrich(runner(frames,seeds,c,d),d,meta,c)
            elapsed=time.perf_counter()-start
            t.to_csv(directory/"tracks.csv",index=False)
            d.to_csv(directory/"drift.csv",index=False)
            statistics(t).to_csv(directory/"track_statistics.csv",index=False)
            m=evaluate_reference(t,reference,tolerance=12)
            m.update(variant=name,status="completed",all_frame_coverage=float(t.observed.mean()),seconds=elapsed,
                     reference_provenance="assistant approximate visual references; nonblind early frames only")
            results.append(m);save_json(directory/"metrics.json",m)
            if name in ("selected","lk","trackpy","legacy_iou_area"):plots(frames,t,directory,c)
            print(name,"coverage",round(t.observed.mean(),3),"early_error",round(m["mean_center_error_px"],2),flush=True)
        except Exception as exc:
            results.append(dict(variant=name,status="failed",error=str(exc)));print(name,str(exc),flush=True)
    results.append(dict(variant="YOLO_TEM_legacy_full",status="not_run_no_TEM_weights_or_labels"))
    pd.DataFrame(results).to_csv(a.output/"comparison.csv",index=False)
    save_json(a.output/"comparison.json",results)
    if sha256(a.input)!=meta["sha256"]:raise RuntimeError("Input changed during comparison")
    (a.output/"README.md").write_text("# Controlled comparison\n\nAll variants share input and selected IDs. Full-video coverage measures availability, not accuracy.\n\nMean errors use 48 approximate, nonblind assistant visual reference points in frames 1-8; they cannot rank late-frame identity correctness. Formal HOTA/IDF1/MOTA are not asserted.\n\nLegacy comparisons use the same Trackpy localizer, not the unavailable historical YOLO model. Raw/blur and registration toggles apply only to the selected route. Search-range sensitivity is exploratory on this sample, not independent validation.\n",encoding="utf-8")
if __name__=="__main__":main()
