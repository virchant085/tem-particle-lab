"""Start or reuse this project's local server without a terminal window."""
from pathlib import Path
import ctypes,json,os,socket,subprocess,sys,time,urllib.error,urllib.request,webbrowser

ROOT=Path(__file__).resolve().parents[1]

def existing(port: int) -> bool:
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/api/meta",timeout=.5) as response:
            data=json.load(response)
        return data.get('app_version')=='ml_v2' and os.path.normcase(data.get("project_root",""))==os.path.normcase(str(ROOT))
    except (OSError,ValueError):return False

def launch() -> str:
    port=None
    for candidate in range(8765,8785):
        if existing(candidate):return f"http://127.0.0.1:{candidate}/"
        with socket.socket() as sock:
            try:sock.bind(("127.0.0.1",candidate));port=candidate;break
            except OSError:pass
    if port is None:raise RuntimeError("No free local port in 8765-8784")
    (ROOT/"results").mkdir(exist_ok=True)
    python=Path(sys.executable).with_name("python.exe") if os.name=="nt" else Path(sys.executable)
    with (ROOT/"results/server.log").open("a",encoding="utf-8") as log:
        proc=subprocess.Popen([str(python),"-X","utf8","-u",str(ROOT/"scripts/serve.py"),"--port",str(port),"--no-browser"],
            cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name=="nt" else 0)
    for _ in range(120):
        if existing(port):return f"http://127.0.0.1:{port}/"
        if proc.poll() is not None:break
        time.sleep(.25)
    raise RuntimeError(f"Server did not start. See {ROOT/'results/server.log'}")

if __name__=="__main__":
    try:
        url=launch()
        if "--no-browser" in sys.argv:print(url)
        else:webbrowser.open(url)
    except Exception as exc:
        if os.name=="nt":ctypes.windll.user32.MessageBoxW(0,str(exc),"TEM Tracker",0x10)
        raise
