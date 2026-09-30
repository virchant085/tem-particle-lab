from __future__ import annotations

import copy,json,subprocess,sys,threading,time,urllib.parse,uuid
from pathlib import Path
import cv2
from .data import AnnotationStore,atomic_json,read_json,valid_id,list_datasets
from .models import load_learning_config,model_cards,model_card,activate_model


class Jobs:
    def __init__(self, project: Path):
        self.project=Path(project);self.root=self.project/'results/learning/jobs';self.root.mkdir(parents=True,exist_ok=True)
        self.lock=threading.Lock()

    def state(self) -> dict:
        meta=read_json(self.root/'latest.json')
        if not meta:return {'busy':False,'status':'idle'}
        directory=self.root/meta['id'];result=read_json(directory/'result.json')
        progress=read_json(directory/'progress.json',{})
        if result:return dict(meta,busy=False,progress=progress,**result)
        process=self.process(meta)
        if process:return dict(meta,busy=True,status='running',progress=progress)
        return dict(meta,busy=False,status='failed',error='任务进程已结束但没有完整结果，请查看任务日志',progress=progress)

    def process(self, meta):
        import psutil
        try:
            process=psutil.Process(meta['pid']);args=process.cmdline()
            if any(str(self.root/meta['id']/'request.json')==x for x in args) and any(x.endswith('lab_worker.py') for x in args):
                return process
        except (psutil.Error,KeyError):pass
        return None

    def start(self, action: str, payload: dict) -> dict:
        with self.lock:
            if self.state()['busy']:raise ValueError('已有训练或推理任务在运行；可继续标注，或先停止当前任务')
            ident='job_'+time.strftime('%Y%m%d_%H%M%S')+'_'+uuid.uuid4().hex[:6]
            directory=self.root/ident;directory.mkdir()
            request=directory/'request.json'
            atomic_json(request,dict(action=action,payload=payload,project=str(self.project),job_directory=str(directory)))
            log=(directory/'job.log').open('w',encoding='utf-8')
            try:
                process=subprocess.Popen([sys.executable,'-X','utf8','-u',str(self.project/'scripts/lab_worker.py'),'--request',str(request)],
                    cwd=self.project,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,
                    creationflags=subprocess.CREATE_NO_WINDOW if sys.platform=='win32' else 0)
            finally:log.close()
            meta=dict(id=ident,action=action,pid=process.pid,started_at=time.strftime('%Y-%m-%dT%H:%M:%S'),
                      log=f'/results/learning/jobs/{ident}/job.log')
            atomic_json(self.root/'latest.json',meta)
            return dict(meta,busy=True,status='running')

    def cancel(self):
        with self.lock:
            meta=read_json(self.root/'latest.json')
            if meta:
                process=self.process(meta)
                if process:
                    import psutil
                    # Windows venv launchers can own the real Python child process.
                    # Stop only this validated job's tree, not just its wrapper.
                    owned=process.children(recursive=True)+[process]
                    for child in reversed(owned):
                        try:child.terminate()
                        except psutil.NoSuchProcess:pass
                    _,alive=psutil.wait_procs(owned,timeout=10)
                    for child in alive:
                        try:child.kill()
                        except psutil.NoSuchProcess:pass
                    atomic_json(self.root/meta['id']/'result.json',{'status':'cancelled','error':'用户停止任务；已保存的标注和之前的模型不受影响'})
            return self.state()


class LearningService:
    def __init__(self, project: Path):
        self.project=Path(project).resolve();self.store=AnnotationStore(self.project);self.jobs=Jobs(self.project)
        self.config=load_learning_config(self.project/'config/learning.yaml')
        if not self.store.videos():
            sample=self.project/'data/raw/sample.mp4'
            if sample.exists():
                video=self.store.register(sample,'sample_experiment_01')
                seeds=read_json(self.project/'config/selected_particles.json',{}).get('particles',[])
                boxes=[]
                for s in seeds:
                    if s.get('frame',0)!=0:continue
                    boxes.append(dict(id=f'draft_{s["id"]}',x1=max(0,s['x']-s['radius']),y1=max(0,s['y']-s['radius']),
                        x2=min(video['width'],s['x']+s['radius']),y2=min(video['height'],s['y']+s['radius']),
                        instance_id=None,source='previous_ROI_proposal_not_ground_truth'))
                self.store.save_annotation(video['id'],0,boxes,reviewed=False)

    def get(self, path: str, query: dict):
        value=lambda key,default=None:query.get(key,[default])[0]
        if path=='/api/lab/state':
            return dict(app_version='ml_v2',videos=self.store.summary(),datasets=list_datasets(self.project),
                        models=model_cards(self.project),active_model=read_json(self.project/'models/active.json'),
                        config=self.config,job=self.jobs.state()),'application/json'
        if path=='/api/lab/job':return self.jobs.state(),'application/json'
        if path=='/api/lab/annotation':return self.store.annotation(value('video'),int(value('frame',0))),'application/json'
        if path=='/api/lab/frame':
            frame=self.store.frame(value('video'),int(value('frame',0)));ok,data=cv2.imencode('.png',frame)
            if not ok:raise ValueError('图像编码失败')
            return data.tobytes(),'image/png'
        if path=='/api/lab/selections':
            video=valid_id(value('video'));self.store.video(video)
            return read_json(self.store.root/'selections'/f'{video}.json',{'particles':[],'model_id':None}),'application/json'
        if path=='/api/lab/result':
            video=valid_id(value('video'));model=valid_id(value('model'));self.store.video(video)
            return read_json(self.store.root/'last_results'/f'{video}_{model}.json'),'application/json'
        if path=='/api/lab/trace-results':
            from .trace_results import list_trace_results
            return list_trace_results(self.project,self.store,value('video')),'application/json'
        raise ValueError('未知学习接口')

    def post(self, path: str, query: dict, body: bytes):
        if path=='/api/lab/import-video':
            name=Path(query.get('name',['video.mp4'])[0]).name
            suffix=Path(name).suffix.lower()
            if suffix not in ('.mp4','.avi','.mov','.mkv','.m4v'):raise ValueError('请选择 MP4、AVI、MOV 或 MKV 视频')
            directory=self.store.root/'incoming';directory.mkdir(exist_ok=True)
            temporary=directory/(uuid.uuid4().hex+suffix);temporary.write_bytes(body)
            return self.store.register(temporary,query.get('experiment',['new_experiment'])[0],name)
        data=json.loads(body)
        if path=='/api/lab/time-traces':
            from .trace_results import export_time_traces
            return export_time_traces(self.project,self.store,data)
        if path=='/api/lab/annotation':
            return self.store.save_annotation(data['video_id'],int(data['frame']),data['boxes'],bool(data.get('reviewed')),
                                              bool(data.get('negative_confirmed')),data.get('revision'))
        if path=='/api/lab/video':return self.store.update_video(data['video_id'],data['experiment'],data['split'])
        if path=='/api/lab/selections':
            from ..io import validate_seeds
            video=self.store.video(data['video_id']);model_card(self.project,data['model_id'])
            seeds=validate_seeds(data,(video['height'],video['width']),video['frames']) if data.get('particles') else []
            saved=dict(video_id=video['id'],model_id=data['model_id'],particles=seeds)
            atomic_json(self.store.root/'selections'/f'{video["id"]}.json',saved);return saved
        if path=='/api/lab/activate':return activate_model(self.project,data['model_id'])
        if path=='/api/lab/cancel':return self.jobs.cancel()
        if path=='/api/lab/job':
            action=data.get('action');cfg=copy.deepcopy(self.config)
            if action not in ('dataset','train','preview','track','suggest','evaluate'):raise ValueError('不支持的任务')
            if action in ('preview','track','suggest'):
                self.store.video(data['video_id'])
                if action!='suggest' or data.get('model_id'):model_card(self.project,data['model_id'])
            if action=='train':
                valid_id(data['dataset_id'])
                for key,lower,upper in [('epochs',1,1000),('batch',1,32),('imgsz',128,1280)]:
                    val=int(data.get(key,cfg['training'][key]))
                    if not lower<=val<=upper:raise ValueError('训练参数超出允许范围')
                    cfg['training'][key]=val
                if cfg['training']['imgsz']%32:raise ValueError('输入尺寸必须为 32 的倍数')
                if data.get('initial_model_id'):model_card(self.project,data['initial_model_id'])
            if action=='track':
                association=data.get('association','bytetrack')
                if association not in ('bytetrack','trackpy'):raise ValueError('不支持的关联器')
                cfg['association']['method']=association
                threshold=float(data.get('new_track_thresh',cfg['association']['new_track_thresh']))
                if not cfg['association']['track_low_thresh']<threshold<=1:raise ValueError('新目标最低分必须大于低分恢复门限且不超过 1')
                cfg['association']['new_track_thresh']=threshold
                cfg['association']['track_high_thresh']=min(cfg['association']['track_high_thresh'],threshold)
                saved=self.post('/api/lab/selections',{},json.dumps(data).encode())
                data['particles']=saved['particles']
                if not saved['particles']:raise ValueError('请先选择需要追踪的颗粒')
            data['config']=cfg
            return self.jobs.start(action,data)
        raise ValueError('未知学习操作')
