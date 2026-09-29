"""Check actual hardware without claiming a TEM model has been trained."""
from pathlib import Path
import json,sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from tem_tracker.learning.models import device_info

if __name__=='__main__':
    import torch,ultralytics,cv2
    info=device_info()
    device='cuda:0' if info['cuda_used'] else 'cpu'
    value=float((torch.ones((32,32),device=device)@torch.ones((32,32),device=device)).sum())
    assert value==32768.0
    print(json.dumps(dict(info,ultralytics=ultralytics.__version__,opencv=cv2.__version__,matrix_check='passed'),ensure_ascii=False,indent=2))
