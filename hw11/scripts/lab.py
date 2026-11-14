"""Shared orchestration. Uses a separate kubeconfig and only the hw11 namespace."""
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import gzip
import json
import os
from pathlib import Path
import subprocess
import threading
import time
import urllib.parse
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
WORK = Path(os.environ.get("HW11_WORKSPACE", ROOT.parents[1] / "hw11/workspace"))
KUBE = str(WORK / "bin/kubectl")
HELM = str(WORK / "bin/helm")
ENV = {**os.environ, "KUBECONFIG": str(WORK / "kubeconfig"),
       "DOCKER_HOST": os.environ.get("DOCKER_HOST", "unix://"+str(Path.home()/".colima/mlops-hw11/docker.sock"))}
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)
PROM = "http://127.0.0.1:19090"


def command(args, stdin=None, check=True, timeout=600):
    p = subprocess.run([str(x) for x in args], input=stdin, capture_output=True, text=True,
                       env=ENV, timeout=timeout)
    if check and p.returncode:
        raise RuntimeError(f"{args}: {p.stderr[-2500:]} {p.stdout[-1000:]}")
    return p


def k(*args, stdin=None, check=True):
    return command([KUBE, "-n", "hw11", *args], stdin=stdin, check=check).stdout


def get(kind, selector=None):
    args = ["get", kind, "-o", "json"]
    if selector:
        args += ["-l", selector]
    return json.loads(k(*args))


def pods():
    return [p for p in get("pods", "app=hw11-api")["items"] if not p["metadata"].get("deletionTimestamp")]


def upgrade(**values):
    args = [HELM, "upgrade", "hw11", str(ROOT/"chart"), "-n", "hw11", "--reuse-values"]
    for key, value in values.items():
        args += ["--set", f"{key}={str(value).lower() if isinstance(value,bool) else value}"]
    began = time.time()
    result = command(args)
    print(result.stdout, flush=True)
    return began


def wait_ready(timeout=180):
    k("rollout", "status", "deployment/hw11-api", f"--timeout={timeout}s")


def save(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2))


def prom(query, at=None):
    params = {"query":query}
    if at is not None:
        params["time"]=str(at)
    with urllib.request.urlopen(PROM+"/api/v1/query?"+urllib.parse.urlencode(params), timeout=15) as r:
        return json.load(r)


def prom_get(endpoint):
    with urllib.request.urlopen(PROM+endpoint, timeout=15) as r:
        return json.load(r)


def in_pod(url, method="GET", body=None, client="client-0", timeout=30):
    code = "import urllib.request,json; u=urllib.request.Request("+repr(url)+",method="+repr(method)+",data="+repr(None if body is None else json.dumps(body).encode())+",headers={'Content-Type':'application/json'}); r=urllib.request.urlopen(u,timeout="+str(timeout)+"); print(r.read().decode())"
    return k("exec", client, "--", "python", "-c", code)


def reset_caches():
    for p in pods():
        if p["status"].get("podIP") and not p["metadata"].get("deletionTimestamp"):
            in_pod("http://"+p["status"]["podIP"]+":8080/reset-cache","POST")


class Collector:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.stop_event = threading.Event()
        self.logs = {}
        self.thread = None

    def __enter__(self):
        self.thread = threading.Thread(target=self.collect, daemon=True)
        self.thread.start()
        return self

    def collect(self):
        raw_path=self.directory/"kubernetes.jsonl.gz"
        with gzip.open(raw_path,"at") as raw, (self.directory/"timeline.jsonl").open("a") as small:
            while not self.stop_event.is_set():
                began=time.monotonic()
                try:
                    now=time.time()
                    data=get("pods,endpointslices,hpa")
                    raw.write(json.dumps({"wall":now,"data":data})+"\n");raw.flush()
                    row={"wall":now,"pods":[],"endpoints":[],"hpa":[]}
                    for x in data["items"]:
                        kind=x["kind"]
                        name=x["metadata"]["name"]
                        if kind=="Pod" and x["metadata"].get("labels",{}).get("app")=="hw11-api":
                            status=x["status"]
                            ready=any(c["type"]=="Ready" and c["status"]=="True" for c in status.get("conditions",[]))
                            row["pods"].append({"name":name,"uid":x["metadata"]["uid"],
                              "created":x["metadata"]["creationTimestamp"],"ip":status.get("podIP"),
                              "deleting":x["metadata"].get("deletionTimestamp"),"ready":ready,
                              "conditions":status.get("conditions",[]),
                              "restarts":sum(c["restartCount"] for c in status.get("containerStatuses",[]))})
                            for c in status.get("containerStatuses",[]):
                                if "running" in c.get("state",{}):
                                    key=(name,c["restartCount"])
                                    if key not in self.logs:
                                        f=(self.directory/f"{name}-restart{c['restartCount']}.log").open("w")
                                        process=subprocess.Popen([KUBE,"-n","hw11","logs","-f",name,"--timestamps"],
                                          stdout=f,stderr=subprocess.STDOUT,env=ENV,text=True)
                                        self.logs[key]=(process,f)
                        elif kind=="EndpointSlice" and x["metadata"].get("labels",{}).get("kubernetes.io/service-name") in ("hw11-api","hw11-sticky"):
                            row["endpoints"].append({"service":x["metadata"]["labels"]["kubernetes.io/service-name"],
                              "endpoints":x.get("endpoints",[])})
                        elif kind=="HorizontalPodAutoscaler":
                            row["hpa"].append({"name":name,"status":x.get("status",{}),"spec":x["spec"]})
                    try:
                        row["metrics"]=prom('{__name__=~"hw11_running|hw11_waiting|hw11_backend_ready"} or sum by (pod) (rate(container_cpu_usage_seconds_total{namespace="hw11",container="api"}[30s]))')["data"]["result"]
                        row["alerts"]=prom_get("/api/v1/alerts")["data"]["alerts"]
                    except Exception as exc:
                        row["metrics_error"]=str(exc)
                    small.write(json.dumps(row)+"\n");small.flush()
                except Exception as exc:
                    small.write(json.dumps({"wall":time.time(),"error":str(exc)})+"\n");small.flush()
                self.stop_event.wait(max(.1,2-(time.monotonic()-began)))

    def __exit__(self,*args):
        self.stop_event.set()
        self.thread.join(timeout=25)
        for proc,f in self.logs.values():
            if proc.poll() is None:
                proc.terminate()
            try:proc.wait(timeout=3)
            except subprocess.TimeoutExpired:proc.kill();proc.wait()
            f.close()


def schedule(duration, rate, traffic="uniform", prefix="load", body_overrides=None):
    n=round(duration*rate)
    rows=[]
    for i in range(n):
        if traffic=="uniform":
            client=i%4
        else:
            client=0 if i%15<12 else 1+(i%15-12)
        body={"prompt":"Classify this GitHub issue: how do I configure the application? ".ljust(128,"x"),
              "history":[],"prefix":f"project-{client}-{(i//4)%8:02}",
              "request_id":f"{prefix}-{i:04}","session_id":f"dialog-{client}",
              "max_chunks":5,"cpu_scale":1.0}
        if body_overrides:body.update(body_overrides)
        rows.append({"client":client,"offset":i/rate,"body":body})
    return rows


def load(name, rows, service="hw11-api", directory=None, start_delay=3, timeout=30):
    directory=Path(directory or RESULTS/name)
    directory.mkdir(parents=True,exist_ok=True)
    start=time.time()+start_delay
    configs=[]
    for i in range(4):
        config={"experiment":name,"url":f"http://{service}:8080","start_wall":start,"timeout":timeout,
                "schedule":[r for r in rows if r["client"]==i]}
        configs.append(config)
        save(directory/f"client-{i}-config.json",config)
    def client_run(i):
        if not configs[i]["schedule"]:return []
        result=k("exec","-i",f"client-{i}","--","python","/app/client.py",stdin=json.dumps(configs[i]))
        (directory/f"client-{i}-raw.jsonl").write_text(result)
        return [json.loads(line) for line in result.splitlines() if line.strip()]
    with ThreadPoolExecutor(max_workers=4) as pool:
        data=[x for group in pool.map(client_run,range(4)) for x in group]
    data.sort(key=lambda x:x["planned_wall"])
    (directory/"requests.jsonl").write_text("".join(json.dumps(x,ensure_ascii=False)+"\n" for x in data))
    save(directory/"counts.json",counts(data,planned=len(rows)))
    return data


def counts(rows,planned=None):
    from collections import Counter
    return {"planned":len(rows) if planned is None else planned,"sent":len(rows),
            **dict(Counter(r["outcome"] for r in rows)),
            "http_5xx":sum(r["http_status"] is not None and r["http_status"]>=500 for r in rows)}


@contextmanager
def background_load(*args,**kwargs):
    pool=ThreadPoolExecutor(max_workers=1)
    future=pool.submit(load,*args,**kwargs)
    try:yield future
    finally:pool.shutdown(wait=True)


def checkpoint(directory,label):
    # Quiet boundaries allow a complete scrape before taking cumulative histogram snapshots.
    time.sleep(6)
    now=time.time()
    result={"wall":now,"series":prom('{__name__=~"hw11_ttfc_seconds_bucket|hw11_ttfc_seconds_count|hw11_total_seconds_bucket|hw11_total_seconds_count|hw11_cache_total"}',now)}
    save(Path(directory)/f"metrics-{label}.json",result)
    return result
