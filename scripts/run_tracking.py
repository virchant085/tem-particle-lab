from pathlib import Path
import argparse,json,logging,sys,math
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from tem_tracker.io import load_config
from tem_tracker.pipeline import run

def main():
    p=argparse.ArgumentParser(description="Manually initialized scientific TEM particle tracking")
    p.add_argument("--input",required=True);p.add_argument("--config",default=str(Path(__file__).resolve().parents[1]/"config/default.yaml"))
    p.add_argument("--seeds",required=True);p.add_argument("--output",required=True)
    p.add_argument("--method",choices=["template","lk","trackpy","legacy"])
    p.add_argument("--acquisition-fps",type=float);p.add_argument("--nm-per-pixel",type=float)
    a=p.parse_args();cfg=load_config(a.config)
    if a.method:cfg["tracker"]["type"]=a.method
    for key,value in [("acquisition_fps",a.acquisition_fps),("nm_per_pixel",a.nm_per_pixel)]:
        if value is not None:
            if not math.isfinite(value) or value<=0: p.error(f"{key} must be finite and positive")
            cfg["calibration"][key]=value
    logging.basicConfig(level=logging.INFO,format="%(levelname)s %(message)s")
    result=run(a.input,cfg,json.loads(Path(a.seeds).read_text(encoding="utf-8-sig")),a.output)
    print(json.dumps(result,indent=2,ensure_ascii=False))

if __name__=="__main__":main()
