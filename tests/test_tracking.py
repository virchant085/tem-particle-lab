from pathlib import Path
import copy
import sys
import cv2
import numpy as np
import pandas as pd
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from tem_tracker.io import load_config,validate_seeds
from tem_tracker.localization import masked_ncc,locate_template,patch
from tem_tracker.registration import estimate_drift
from tem_tracker.tracking import track_local,box_iou
from tem_tracker.analysis import enrich,statistics,evaluate_reference
from tem_tracker.visualization import trajectory_segments

@pytest.fixture
def cfg():
    c=load_config(Path(__file__).resolve().parents[1]/"config/default.yaml")
    c["registration"]["enabled"]=False
    c["tracker"].update(search_range=10,memory=1,min_peak_margin=.01,max_backward_error=3)
    return c

def synthetic(positions,noise=1.):
    rng=np.random.default_rng(111)
    yy,xx=np.mgrid[:120,:140]
    return [np.clip(160-80*np.exp(-((xx-x)**2+(yy-y)**2)/100)+rng.normal(0,noise,xx.shape),0,255).astype(np.uint8) for x,y in positions]

def test_known_subpixel_motion(cfg):
    pos=[(40+i*1.3,50+i*.65) for i in range(12)]
    frames=synthetic(pos)
    s=[dict(id=4,frame=0,x=40.,y=50.,radius=16.)]
    t=track_local(frames,s,cfg,estimate_drift(frames,cfg["registration"]))
    assert t.observed.all()
    err=np.linalg.norm(t[["center_x","center_y"]].to_numpy()-np.array(pos),axis=1)
    assert err.max()<1.0
    assert t.track_id.unique().tolist()==[4]

def test_disappearance_outputs_missing_not_prediction(cfg):
    frames=synthetic([(40,50)]*7)
    frames[2:]=[np.full((120,140),160,np.uint8) for _ in frames[2:]]
    t=track_local(frames,[dict(id=1,frame=0,x=40,y=50,radius=16)],cfg,estimate_drift(frames,cfg["registration"]))
    assert not t.loc[t.frame>=2,"observed"].any()
    assert t.loc[t.frame>=2,["center_x","center_y"]].isna().all().all()

def test_manual_anchor_restores_same_identity(cfg):
    frames=synthetic([(40,50)]*3+[(75,70)]*3)
    frames[2]=np.full((120,140),160,np.uint8)
    seeds=[dict(id=9,frame=0,x=40,y=50,radius=16),dict(id=9,frame=4,x=75,y=70,radius=16)]
    t=track_local(frames,seeds,cfg,estimate_drift(frames,cfg["registration"]))
    assert t.loc[t.frame==4,"status"].iloc[0]=="manual_anchor"
    assert t.loc[t.frame==5,"observed"].iloc[0]
    assert t.track_id.unique().tolist()==[9]

def test_delayed_seed_no_earlier_predictions(cfg):
    frames=synthetic([(40,50)]*5)
    t=track_local(frames,[dict(id=1,frame=2,x=40,y=50,radius=16)],cfg,estimate_drift(frames,cfg["registration"]))
    assert t.frame.tolist()==[2,3,4]

def test_ncc_brightness_invariance():
    rng=np.random.default_rng(2);a=rng.normal(100,20,(23,23)).astype(np.float32)
    assert masked_ncc(a*1.2+15,a)[0,0]>.999

def test_constant_template_is_rejected():
    assert masked_ncc(np.ones((31,31),np.float32),np.ones((15,15),np.float32)).max()<0

def test_ncc_border_search_stays_in_image():
    f=synthetic([(4,60)])[0].astype(np.float32)
    p,s,m=locate_template(f,patch(f,np.array([4,60]),15),np.array([4,60]),10)
    assert 0<=p[0]<f.shape[1] and s>.99

def test_registration_sign_and_relative_coordinates(cfg):
    rng=np.random.default_rng(3);base=(rng.random((120,140))*100+60).astype(np.uint8)
    frames=[base,cv2.warpAffine(base,np.float32([[1,0,2],[0,1,-1]]),(140,120),borderMode=cv2.BORDER_REFLECT)]
    rc=copy.deepcopy(cfg["registration"]);rc.update(enabled=True,roi_fraction=[.15,.15,.85,.85],max_method_disagreement=3)
    d=estimate_drift(frames,rc)
    assert d.valid.all()
    assert np.linalg.norm(d.loc[1,["dx","dy"]].to_numpy(float)-[2,-1])<.5

def test_nan_measurement_and_time_calibration(cfg):
    frames=synthetic([(40,50)]*3);d=estimate_drift(frames,cfg["registration"])
    t=track_local(frames,[dict(id=1,frame=0,x=40,y=50,radius=16)],cfg,d)
    t.loc[1,["center_x","center_y"]]=np.nan;t.loc[1,"observed"]=False
    o=enrich(t,d,{"video_timestamps_s":[0,1,2]},cfg)
    assert o.x_nm.isna().all() and o.acquisition_time_s.isna().all()
    assert o.step_px.isna().all() # no bridge across the missing frame
    assert o.x_background_relative_px.isna().all()

def test_invalid_drift_does_not_fabricate_corrected_coords(cfg):
    frames=synthetic([(40,50)]*3);d=estimate_drift(frames,cfg["registration"])
    cfg["registration"]["enabled"]=True
    d.loc[1:,"cumulative_valid"]=False
    t=track_local(frames,[dict(id=1,frame=0,x=40,y=50,radius=16)],cfg,d)
    o=enrich(t,d,{"video_timestamps_s":[0,1,2]},cfg)
    assert o.loc[o.frame>=1,"x_background_relative_px"].isna().all()

@pytest.mark.parametrize("item",[dict(id=0,frame=0,x=1,y=1,radius=10),dict(id=1,frame=100,x=1,y=1,radius=10),dict(id=1,frame=0,x=-2,y=1,radius=10),dict(id=1,frame=0,x=float('nan'),y=1,radius=10)])
def test_invalid_seed(item):
    with pytest.raises(ValueError):validate_seeds({"particles":[item]},(100,100),10)

def test_duplicate_seed():
    s=dict(id=1,frame=0,x=20,y=20,radius=10)
    with pytest.raises(ValueError):validate_seeds({"particles":[s,s]},(100,100),10)

def test_legacy_iou():
    assert box_iou(np.array([10,10,8,8]),np.array([10,10,8,8]))==1
    assert box_iou(np.array([10,10,8,8]),np.array([30,30,8,8]))==0

def test_reference_missing_prediction_penalized():
    ref=pd.DataFrame(dict(frame=[0,1],track_id=[1,1],center_x=[10,11],center_y=[20,20]))
    pred=pd.DataFrame(dict(frame=[0],track_id=[1],center_x=[10],center_y=[20],observed=[True]))
    e=evaluate_reference(pred,ref)
    assert e["coverage"]==.5 and e["success_within_tolerance"]==.5

def test_manual_correction_does_not_create_velocity_or_join_segments(cfg):
    frames=synthetic([(40,50)]*4)
    d=estimate_drift(frames,cfg["registration"])
    t=track_local(frames,[dict(id=1,frame=0,x=40,y=50,radius=16),
                         dict(id=1,frame=2,x=42,y=50,radius=16)],cfg,d)
    o=enrich(t,d,{"video_timestamps_s":[0,1,2,3]},cfg)
    assert o.loc[o.frame==2,"step_px"].isna().all()
    assert [s.frame.tolist() for s in trajectory_segments(o)]==[[0,1],[2,3]]

def test_dataset_validator_detects_invalid_boxes_and_never_modifies_raw(tmp_path):
    import importlib.util,hashlib
    spec=importlib.util.spec_from_file_location("dataset_check",Path(__file__).resolve().parents[1]/"scripts/validate_dataset.py")
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    for split in ("train","val"):
        (tmp_path/"images"/split).mkdir(parents=True)
        (tmp_path/"labels"/split).mkdir(parents=True)
        cv2.imwrite(str(tmp_path/"images"/split/"frame.png"),np.zeros((20,20),np.uint8))
        (tmp_path/"labels"/split/"frame.txt").write_text("0 0.95 0.5 0.2 0.2",encoding="utf-8")
    dataset=tmp_path/"dataset.yaml"
    dataset.write_text("train: images/train\nval: images/val\nnames: [particle]\n",encoding="utf-8")
    before={p:hashlib.sha256(p.read_bytes()).hexdigest() for p in tmp_path.rglob("*") if p.is_file()}
    result=module.validate(dataset)
    assert not result["valid"] and any("outside image" in e for e in result["errors"])
    assert before=={p:hashlib.sha256(p.read_bytes()).hexdigest() for p in before}
