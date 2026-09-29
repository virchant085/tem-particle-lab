"""Read-only YOLO detection dataset integrity checks."""
from pathlib import Path
import argparse,json
import cv2
import numpy as np
import yaml

def validate(path: str|Path) -> dict:
    path=Path(path).resolve();cfg=yaml.safe_load(path.read_text(encoding="utf-8-sig"))
    root=Path(cfg.get("path","."));root=root if root.is_absolute() else path.parent/root
    names=cfg.get("names",[]);nc=len(names)
    errors=[];warnings=[];counts={};seen={}
    if not nc:errors.append("names is empty")
    if "nc" in cfg and cfg["nc"]!=nc:errors.append("nc and names disagree")
    for split in ("train","val"):
        values=cfg.get(split,[]);values=[values] if isinstance(values,str) else values
        images=[]
        for value in values:
            source=Path(value);source=source if source.is_absolute() else root/source
            if source.is_dir(): images.extend(p for p in source.rglob("*") if p.suffix.lower() in {".png",".jpg",".jpeg",".tif",".tiff",".bmp"})
            elif source.suffix==".txt" and source.exists():
                for line in source.read_text(encoding="utf-8-sig").splitlines():
                    p=Path(line.strip());images.append(p if p.is_absolute() else source.parent/p)
            elif source.is_file():images.append(source)
        counts[split]=len(images)
        if not images:errors.append(f"{split}: no images")
        for im in images:
            if str(im.resolve()) in seen:errors.append(f"Image reused across splits: {im}")
            seen[str(im.resolve())]=split
            try:
                frame=cv2.imdecode(np.fromfile(im,np.uint8),cv2.IMREAD_GRAYSCALE)
                if frame is None:raise ValueError("decode failed")
            except Exception as exc:errors.append(f"Unreadable image {im}: {exc}")
            parts=list(im.parts)
            if "images" not in parts:errors.append(f"Image path missing images directory: {im}");continue
            parts[parts.index("images")]="labels";label=Path(*parts).with_suffix(".txt")
            if not label.exists():errors.append(f"Missing label {label}");continue
            lines=label.read_text(encoding="utf-8-sig").splitlines()
            if not lines:warnings.append(f"Empty label (verify intentional negative): {label}")
            for i,line in enumerate(lines,1):
                try:
                    values=np.array([float(x) for x in line.split()])
                    if len(values)!=5 or not np.isfinite(values).all():raise ValueError("expected five finite values")
                    cl,x,y,w,h=values
                    if cl!=int(cl) or not 0<=cl<nc:raise ValueError("invalid class ID")
                    if w<=0 or h<=0 or not (0<=x-w/2<=x+w/2<=1 and 0<=y-h/2<=y+h/2<=1):raise ValueError("box outside image or nonpositive size")
                except ValueError as exc:errors.append(f"{label}:{i}: {exc}")
    return dict(valid=not errors,counts=counts,errors=errors,warnings=warnings,
                note="No files modified. Split by experiment/video to avoid adjacent-frame leakage.")

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("dataset");p.add_argument("--output");a=p.parse_args()
    result=validate(a.dataset);text=json.dumps(result,ensure_ascii=False,indent=2)
    if a.output:Path(a.output).write_text(text,encoding="utf-8")
    print(text);raise SystemExit(0 if result["valid"] else 1)
