"""Isolated, cancellable training/inference task; never runs on an import."""
from pathlib import Path
import argparse,sys,traceback
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from tem_tracker.learning.data import read_json,atomic_json
from tem_tracker.learning.workflow import execute

def main():
    p=argparse.ArgumentParser();p.add_argument('--request',type=Path,required=True);a=p.parse_args()
    request=read_json(a.request);directory=Path(request['job_directory'])
    progress=lambda value:atomic_json(directory/'progress.json',value)
    try:
        progress({'stage':'starting','message':'正在准备任务'})
        result=execute(Path(request['project']),request['action'],request['payload'],progress)
        atomic_json(directory/'result.json',{'status':'completed','result':result})
    except Exception as exc:
        traceback.print_exc()
        atomic_json(directory/'result.json',{'status':'failed','error':str(exc)})
        raise SystemExit(1)

if __name__=='__main__':main()
