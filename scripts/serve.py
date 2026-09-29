"""Loopback-only manual particle selection and result review. No external service."""
from __future__ import annotations
from pathlib import Path
import argparse,copy,json,secrets,sys,threading,time,urllib.parse,webbrowser
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
import mimetypes
import cv2
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from tem_tracker.io import load_config,load_video,validate_seeds,save_json
from tem_tracker.pipeline import run
from tem_tracker.learning.service import LearningService

ROOT=Path(__file__).resolve().parents[1]

def serve(input_path: Path,config_path: Path,seeds_path: Path|None,port: int,no_browser: bool):
    config=load_config(config_path)
    frames,meta=load_video(input_path)
    initial=json.loads(seeds_path.read_text(encoding="utf-8-sig")) if seeds_path else {"particles":[]}
    state={"input":str(input_path.resolve()),"frames":frames,"meta":meta,"seeds":initial,
           "busy":False,"error":None,"result":None,"result_rows":[]}
    token=secrets.token_urlsafe(32)
    guard=threading.Lock()
    learning=LearningService(ROOT)

    class Handler(BaseHTTPRequestHandler):
        def send(self,body: bytes,kind="application/json",code=200):
            self.send_response(code);self.send_header("Content-Type",kind)
            self.send_header("Content-Length",str(len(body)));self.send_header("Cache-Control","no-store")
            self.send_header("X-Content-Type-Options","nosniff");self.end_headers();self.wfile.write(body)
        def obj(self,data,code=200): self.send(json.dumps(data,ensure_ascii=False,allow_nan=False).encode("utf-8"),code=code)
        def do_GET(self):
            url=urllib.parse.urlparse(self.path);path=url.path
            try:
                if path in ("/","/learn"): self.send((ROOT/"ui/learning.html").read_bytes(),"text/html; charset=utf-8")
                elif path=="/manual":self.send((ROOT/"ui/index.html").read_bytes(),"text/html; charset=utf-8")
                elif path in ("/learning.js","/learning.css"):
                    self.send((ROOT/"ui"/path.lstrip('/')).read_bytes(),"text/javascript; charset=utf-8" if path.endswith('.js') else "text/css; charset=utf-8")
                elif path.startswith('/api/lab/'):
                    data,kind=learning.get(path,urllib.parse.parse_qs(url.query))
                    if path=='/api/lab/state':data['token']=token
                    if isinstance(data,bytes):self.send(data,kind)
                    else:self.obj(data)
                elif path=="/api/meta":
                    self.obj(dict(metadata=state["meta"],seeds=state["seeds"],config=config,token=token,project_root=str(ROOT),app_version='ml_v2'))
                elif path.startswith("/api/frame/"):
                    fi=int(path.rsplit("/",1)[-1])
                    if not 0<=fi<len(state["frames"]):raise ValueError("Frame outside video")
                    ok,encoded=cv2.imencode(".png",state["frames"][fi])
                    if not ok:raise ValueError("Frame encoding failed")
                    self.send(encoded.tobytes(),"image/png")
                elif path=="/api/status":
                    self.obj({k:state[k] for k in ["busy","error","result","result_rows"]})
                elif path.startswith("/results/"):
                    root=(ROOT/"results").resolve();target=(ROOT/urllib.parse.unquote(path.lstrip("/"))).resolve()
                    if not target.is_relative_to(root) or not target.is_file():raise ValueError("Result file unavailable")
                    self.send(target.read_bytes(),mimetypes.guess_type(target.name)[0] or "application/octet-stream")
                else:self.obj({"error":"Not found"},404)
            except (ValueError,OSError,KeyError) as exc:self.obj({"error":str(exc)},400)
        def do_POST(self):
            if self.headers.get("X-Session-Token")!=token:
                self.obj({"error":"Invalid local session"},403);return
            try:
                size=int(self.headers.get("Content-Length","0"))
                if size<=0 or size>512*1024*1024:raise ValueError("Upload must be between 1 byte and 512 MB")
                if self.path.startswith('/api/lab/'):
                    parsed=urllib.parse.urlparse(self.path)
                    self.obj(learning.post(parsed.path,urllib.parse.parse_qs(parsed.query),self.rfile.read(size)));return
                with guard:
                    if state["busy"]:self.obj({"error":"A run is already active"},409);return
                    state["busy"]=True
                body=self.rfile.read(size)
                if self.path=="/api/upload":
                    dest=ROOT/"data/imported";dest.mkdir(parents=True,exist_ok=True)
                    dest=dest/(time.strftime("video_%Y%m%d_%H%M%S_")+secrets.token_hex(3)+".mp4")
                    dest.write_bytes(body)
                    ff,mm=load_video(dest,config["output"]["max_memory_mb"])
                    state.update(input=str(dest),frames=ff,meta=mm,seeds={"particles":[]},busy=False,result=None,result_rows=[])
                    self.obj({"ok":True});return
                if self.path!="/api/run":raise ValueError("Unknown operation")
                payload=json.loads(body);seeds={"schema_version":1,"particles":payload["particles"],"selection_note":"User points/anchors in local selector"}
                validate_seeds(seeds,state["frames"][0].shape,len(state["frames"]))
                cfg=copy.deepcopy(config)
                for k in ["acquisition_fps","nm_per_pixel"]:
                    val=payload.get(k)
                    if val is not None:
                        val=float(val)
                        if not 0<val<1e9:raise ValueError("Calibration must be positive")
                    cfg["calibration"][k]=val
                search=float(payload.get("search_range",cfg["tracker"]["search_range"]))
                if not 2<=search<=150:raise ValueError("Search range must be 2-150 px")
                cfg["tracker"]["search_range"]=search
                name=time.strftime("manual_%Y%m%d_%H%M%S_")+secrets.token_hex(2)
                state.update(error=None,result=None,result_rows=[],seeds=seeds)
                input_snapshot=state["input"]
                def work():
                    try:
                        metrics=run(input_snapshot,cfg,seeds,ROOT/"results"/name)
                        data=pd.read_csv(ROOT/"results"/name/"tracks.csv")
                        cols=["frame","track_id","center_x","center_y","radius_px","observed","status","quality_reason"]
                        state["result_rows"]=json.loads(data[cols].to_json(orient="records"))
                        state["result"]={"name":name,"metrics":metrics,"base":f"/results/{name}/"}
                    except Exception as exc: state["error"]=str(exc)
                    finally: state["busy"]=False
                threading.Thread(target=work,daemon=True).start()
                self.obj({"ok":True,"name":name})
            except Exception as exc:
                state["busy"]=False;self.obj({"error":str(exc)},400)
        def log_message(self,fmt,*args):
            if not self.path.startswith(("/api/status","/api/frame")):super().log_message(fmt,*args)

    server=ThreadingHTTPServer(("127.0.0.1",port),Handler)
    url=f"http://127.0.0.1:{server.server_port}"
    print(url,flush=True)
    if not no_browser:webbrowser.open(url)
    try:server.serve_forever()
    except KeyboardInterrupt:pass
    finally:server.server_close()

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--input",type=Path,default=ROOT/"data/raw/sample.mp4")
    p.add_argument("--config",type=Path,default=ROOT/"config/default.yaml")
    p.add_argument("--seeds",type=Path,default=ROOT/"config/selected_particles.json")
    p.add_argument("--port",type=int,default=8765);p.add_argument("--no-browser",action="store_true")
    a=p.parse_args();serve(a.input,a.config,a.seeds,a.port,a.no_browser)
