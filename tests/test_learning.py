from pathlib import Path
import copy,sys
import cv2
import numpy as np
import pandas as pd
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from tem_tracker.learning.data import AnnotationStore,assign_splits,build_dataset,validate_snapshot,atomic_json,checked_boxes
from tem_tracker.learning.models import load_learning_config,model_cards,model_card,activate_model,check_lineage_holdout
from tem_tracker.learning.tracking import associate_detections,selected_trajectories
from tem_tracker.io import sha256
from tem_tracker.io import load_config
from tem_tracker.tracking import track_legacy
from tem_tracker.learning.service import Jobs

ROOT=Path(__file__).resolve().parents[1]

@pytest.fixture
def cfg():return load_learning_config(ROOT/'config/learning.yaml')

def make_video(path,seed=1,n=12):
    rng=np.random.default_rng(seed)
    writer=cv2.VideoWriter(str(path),cv2.VideoWriter_fourcc(*'MJPG'),5,(96,96))
    assert writer.isOpened()
    for i in range(n):
        image=rng.integers(80,180,(96,96,3),dtype=np.uint8)
        cv2.circle(image,(30+i,40),8,(30,30,30),-1);writer.write(image)
    writer.release();return path

def box(**kwargs):return dict(id='a',x1=20,y1=30,x2=40,y2=50,**kwargs)

@pytest.fixture
def store_video(tmp_path):
    store=AnnotationStore(tmp_path);v=store.register(make_video(tmp_path/'original.avi'),'experiment_A')
    return store,v

def test_proposals_excluded_until_explicit_review(store_video):
    store,v=store_video
    store.save_annotation(v['id'],0,[box(source='model_proposal')],False)
    assert store.reviewed_records()==[]
    with pytest.raises(ValueError,match='草稿'):build_dataset(store.project,'temporal')
    store.save_annotation(v['id'],0,[box()],True,expected_revision=1)
    assert len(store.reviewed_records())==1
    history=list((store.root/'annotation_history').rglob('*.json'))
    assert len(history)==1
    with pytest.raises(ValueError,match='其他页面'):
        store.save_annotation(v['id'],0,[box()],True,expected_revision=1)
    assert store.annotation(v['id'],0)['revision']==2

def test_empty_frame_requires_explicit_negative(store_video):
    store,v=store_video
    with pytest.raises(ValueError,match='空帧'):store.save_annotation(v['id'],1,[],True)
    record=store.save_annotation(v['id'],1,[],True,negative_confirmed=True)
    assert record['reviewed'] and record['negative_confirmed']

@pytest.mark.parametrize('change',[{'x1':-1},{'x2':100},{'y1':float('nan')},{'x2':21},{'instance_id':1.5}])
def test_bad_annotations_rejected(change):
    b=box();b.update(change)
    with pytest.raises(ValueError):checked_boxes([b],96,96)

def test_conflicting_instance_ids_rejected():
    b=box(instance_id=1);c=dict(b,id='b')
    with pytest.raises(ValueError,match='身份'):checked_boxes([b,c],96,96)

def records_for_groups():
    return [(dict(id=f'v{v}',experiment=group,split='auto',frames=30),dict(frame=i))
            for v,group in enumerate(['A','A','B','C']) for i in [0,4,9,16,25]]

def test_splits_keep_experiment_together():
    records=records_for_groups();assignment,_=assign_splits(records,'experiment')
    for group in ['A','B','C']:
        assert len({assignment[(v['id'],a['frame'])] for v,a in records if v['experiment']==group})==1
    assert set(assignment.values())=={'train','val'}
    assert assign_splits(records,'experiment')[0]==assignment

def test_conflicting_manual_split_rejected():
    records=records_for_groups();records[0][0]['split']='train';records[5][0]['split']='val'
    with pytest.raises(ValueError,match='同一实验'):assign_splits(records,'experiment')

def test_temporal_split_gap():
    v=dict(id='v1',experiment='A',split='auto',frames=26)
    assignments,warnings=assign_splits([(v,dict(frame=i)) for i in range(26)],'temporal',temporal_gap=2)
    train=[fi for (_,fi),s in assignments.items() if s=='train'];val=[fi for (_,fi),s in assignments.items() if s=='val']
    assert max(train)==16 and min(val)==21
    assert len([s for s in assignments.values() if s=='excluded_gap'])==4
    assert warnings

def test_snapshot_immutable_and_raw_unchanged(store_video):
    store,v=store_video;raw_hash=sha256(store.path(v['id']))
    store.save_annotation(v['id'],0,[box()],True)
    store.save_annotation(v['id'],11,[box()],True)
    manifest=build_dataset(store.project,'temporal');directory=store.project/'data/datasets'/manifest['id']
    validate_snapshot(directory)
    assert manifest['counts']==dict(train=1,val=1,test=0)
    assert sha256(store.path(v['id']))==raw_hash
    store.save_annotation(v['id'],0,[],False)
    validate_snapshot(directory)
    (directory/manifest['items'][0]['label']).write_text('0 0 0 0 0',encoding='utf-8')
    with pytest.raises(ValueError,match='改变'):validate_snapshot(directory)

def test_duplicate_image_across_split_rejected(tmp_path,monkeypatch):
    store=AnnotationStore(tmp_path)
    for i in range(2):
        v=store.register(make_video(tmp_path/f'v{i}.avi',seed=i+1),f'experiment_{i}')
        store.save_annotation(v['id'],0,[box()],True)
    monkeypatch.setattr(AnnotationStore,'frame',lambda *args:np.full((96,96),128,np.uint8))
    with pytest.raises(ValueError,match='重复'):build_dataset(tmp_path)

def detection(x,y=40,score=.95):
    return dict(detection_index=0,x1=x-10,y1=y-10,x2=x+10,y2=y+10,confidence=score,class_id=0)

def seed(frame=0,id=5,x=30):return dict(frame=frame,id=id,x=x,y=40,radius=10)

def test_bytetrack_recovers_low_score_and_raw_coordinates(cfg):
    d=[[detection(30)],[detection(34,score=.25)],[detection(38)]]
    linked=associate_detections(d,(96,96),cfg)
    assert linked.backend_id.nunique()==1
    assert linked.center_x.tolist()==[30,34,38]
    assert linked.confidence.tolist()==[.95,.25,.95]
    assert abs(linked.iloc[1].association_center_x-34)>.001
    out=selected_trajectories(linked,[seed()],3,cfg)
    assert out.observed.all() and out.track_id.unique().tolist()==[5]

def test_detection_index_survives_high_low_partition(cfg):
    d=[[dict(detection(30),detection_index=0),dict(detection(70),detection_index=1)],
       [dict(detection(32,score=.2),detection_index=0),dict(detection(72),detection_index=1)]]
    linked=associate_detections(d,(96,96),cfg)
    assert linked.loc[(linked.frame==1)&(linked.detection_index==0),'center_x'].tolist()==[32]
    assert linked.loc[(linked.frame==1)&(linked.detection_index==1),'center_x'].tolist()==[72]

def test_new_particle_can_be_selected_on_appearance(cfg):
    linked=associate_detections([[],[],[detection(30)],[detection(32)]],(96,96),cfg)
    out=selected_trajectories(linked,[seed(frame=2)],4,cfg)
    assert out.frame.tolist()==[2,3] and out.observed.all()

def test_missing_has_no_predicted_measurements(cfg):
    linked=associate_detections([[detection(30)],[detection(32)],[],[detection(34)]],(96,96),cfg)
    out=selected_trajectories(linked,[seed()],4,cfg)
    missing=out[out.frame==2]
    assert not missing.observed.any() and missing[['center_x','center_y']].isna().all().all()
    assert out.loc[out.frame==3,'status'].tolist()==['reacquired']

def test_selected_ids_never_auto_switch_but_manual_anchor_can_resume(cfg):
    linked=pd.DataFrame([dict(detection(x),frame=fi,backend_id=bid,center_x=x,center_y=40)
                         for fi,x,bid in [(0,30,1),(1,32,1),(2,34,2),(3,36,2)]])
    out=selected_trajectories(linked,[seed(),seed(frame=3,x=36)],4,cfg)
    assert out.observed.tolist()==[True,True,False,True]
    assert out.status.tolist()==['seed','tracked','lost','manual_anchor']
    assert out.track_id.nunique()==1

def test_two_anchors_do_not_share_one_detection(cfg):
    linked=associate_detections([[detection(30)]],(96,96),cfg)
    out=selected_trajectories(linked,[seed(id=1),seed(id=2,x=31)],1,cfg)
    assert out.observed.sum()==1

def test_trackpy_handles_gap(cfg):
    cfg['association']['method']='trackpy'
    linked=associate_detections([[detection(30)],[],[detection(34)]],(96,96),cfg)
    assert linked.backend_id.nunique()==1 and linked.center_x.tolist()==[30,34]

def test_smoke_model_cannot_be_used_as_tem(tmp_path):
    weights=tmp_path/'models/registry/smoke/best.pt';weights.parent.mkdir(parents=True);weights.write_bytes(b'test fixture')
    atomic_json(weights.parent/'model_card.json',dict(id='smoke',role='synthetic_smoke',weights=weights.relative_to(tmp_path).as_posix(),weights_sha256=sha256(weights)))
    assert model_cards(tmp_path)==[]
    assert len(model_cards(tmp_path,True))==1
    with pytest.raises(ValueError,match='合成'):model_card(tmp_path,'smoke')
    with pytest.raises(ValueError,match='合成'):activate_model(tmp_path,'smoke')
    model_card(tmp_path,'smoke',True)
    weights.write_bytes(b'changed')
    with pytest.raises(ValueError,match='改变'):model_card(tmp_path,'smoke',True)

def test_fine_tuning_cannot_relabel_seen_experiment_as_holdout():
    a=dict(split='train',experiment='A',source_video_sha256='rawA',image_sha256='frame0')
    b=dict(a,split='val',image_sha256='frame1')
    with pytest.raises(ValueError,match='父模型'):
        check_lineage_holdout(dict(split_mode='experiment',items=[b]),[dict(items=[a])])
    check_lineage_holdout(dict(split_mode='temporal',items=[b]),[dict(items=[a])])
    with pytest.raises(ValueError,match='父模型'):
        check_lineage_holdout(dict(split_mode='temporal',items=[dict(a,split='val')]),[dict(items=[a])])

def test_learned_legacy_baseline_uses_measurements_and_leaves_unmatched_anchors_missing():
    base=load_config(ROOT/'config/default.yaml');frames=[np.zeros((96,96),np.uint8)]*3
    obs=pd.DataFrame([dict(frame=1,x=31,y=40,width=20,height=20,confidence=.9,detection_index=0)])
    out=track_legacy(frames,[seed()],base,pd.DataFrame(),observations=obs)
    assert out.observed.tolist()==[False,True,False]
    assert out.loc[out.frame==1,'center_x'].tolist()==[31]
    assert out.loc[~out.observed,['center_x','center_y']].isna().all().all()

def test_cancel_stops_owned_worker_tree(tmp_path):
    import time,psutil
    scripts=tmp_path/'scripts';scripts.mkdir()
    (scripts/'lab_worker.py').write_text('import time\ntime.sleep(60)\n',encoding='utf-8')
    jobs=Jobs(tmp_path);meta=jobs.start('fixture',{});owned=[]
    try:
        process=jobs.process(meta)
        assert process is not None
        for _ in range(20):
            children=process.children(recursive=True)
            if children:break
            time.sleep(.05)
        owned=children+[process]
        assert jobs.state()['busy']
        result=jobs.cancel()
        assert not result['busy'] and result['status']=='cancelled'
        assert all(not p.is_running() for p in owned)
    finally:
        if jobs.state()['busy']:jobs.cancel()
