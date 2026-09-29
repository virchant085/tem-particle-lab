from pathlib import Path
import argparse,sys
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from tem_tracker.analysis import evaluate_reference
from tem_tracker.io import save_json
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--tracks",required=True);p.add_argument("--reference",required=True)
    p.add_argument("--output",required=True);p.add_argument("--tolerance",type=float,default=12.)
    a=p.parse_args();e=evaluate_reference(pd.read_csv(a.tracks),pd.read_csv(a.reference),a.tolerance)
    save_json(Path(a.output),e);print(e)
