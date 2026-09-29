"""Optional TEM YOLO baseline; requires user-supplied annotated dataset."""
from pathlib import Path
import argparse,random
import numpy as np
from validate_dataset import validate

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--data",required=True);p.add_argument("--model",default="yolov8n.pt")
    p.add_argument("--epochs",type=int,default=100);p.add_argument("--imgsz",type=int,default=640)
    p.add_argument("--batch",type=int,default=16);p.add_argument("--output",required=True);p.add_argument("--seed",type=int,default=17)
    a=p.parse_args();v=validate(a.data)
    if not v["valid"]:raise ValueError(v["errors"])
    import torch
    from ultralytics import YOLO
    random.seed(a.seed);np.random.seed(a.seed);torch.manual_seed(a.seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(a.seed)
    model=YOLO(a.model)
    model.train(data=str(Path(a.data).resolve()),epochs=a.epochs,imgsz=a.imgsz,batch=a.batch,
                project=a.output,name="tem_baseline",exist_ok=False,seed=a.seed,
                deterministic=True,device=0 if torch.cuda.is_available() else "cpu")
