"""Export retained Prometheus series with a correct selector; keep original snapshots untouched."""
import bisect
from lab import *
QUERY='{__name__=~"hw11_running|hw11_waiting|hw11_backend_ready"} or sum by (pod) (rate(container_cpu_usage_seconds_total{namespace="hw11",container="api"}[30s]))'
COUNTERS='{__name__=~"hw11_ttfc_seconds_bucket|hw11_ttfc_seconds_count|hw11_total_seconds_bucket|hw11_total_seconds_count|hw11_cache_total"}'


def recover(folder):
    source=folder/"timeline.jsonl"
    if not source.exists() or not (folder/"metrics-after.json").exists():
        return
    lines=[json.loads(x) for x in source.read_text().splitlines()]
    valid=[x for x in lines if "pods" in x]
    if not valid:return
    start=valid[0]["wall"];end=valid[-1]["wall"]+2
    params={"query":QUERY,"start":str(start),"end":str(end),"step":"2"}
    url=PROM+"/api/v1/query_range?"+urllib.parse.urlencode(params)
    with urllib.request.urlopen(url,timeout=30) as response:
        result=json.load(response)
    save(folder/"prometheus-range-complete.json",{"query":QUERY,"start":start,"end":end,"step":2,
        "retrieved_wall":time.time(),"reason":"Plain OR ignores metric name when matching; use one __name__ selector.",
        "result":result})
    by_time={}
    for series in result["data"]["result"]:
        for timestamp,value in series["values"]:
            by_time.setdefault(timestamp,[]).append({"metric":series["metric"],"value":[timestamp,value]})
    times=sorted(by_time)
    for line in lines:
        i=bisect.bisect_right(times,line["wall"])-1
        if i>=0:
            timestamp=times[i]
            line["metrics_original_query"]=line.get("metrics",[])
            line["metrics"]=by_time[timestamp]
            line["metrics_eval_wall"]=timestamp
            line["metrics_source"]="prometheus-range-complete.json, preceding evaluation (within 2 s)"
    (folder/"timeline-complete.jsonl").write_text("".join(json.dumps(x)+"\n" for x in lines))
    for label in ("before","after"):
        original=json.loads((folder/f"metrics-{label}.json").read_text())
        at=original["wall"]
        save(folder/f"metrics-{label}-complete.json",{"wall":at,"query":COUNTERS,
             "retrieved_wall":time.time(),"series":prom(COUNTERS,at)})
    print("EXPORTED",folder.name,len(times),"historical samples",flush=True)


if __name__=="__main__":
    for folder in sorted(RESULTS.glob("B-*")):
        if folder.is_dir():recover(folder)
