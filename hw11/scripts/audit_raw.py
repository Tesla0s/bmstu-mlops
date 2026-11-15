"""Independent cross-check of source schedules, client results, server logs and histograms."""
from pathlib import Path
from collections import Counter, OrderedDict
import gzip, hashlib, json, math, statistics, sys

ROOT=Path(__file__).resolve().parents[1]
R=ROOT/'results'
failures=[]; checks=0
evidence={}
def check(ok,label):
    global checks
    checks+=1
    if not ok:failures.append(label)
def read(p):return json.loads(p.read_text())
def lines(p):return [json.loads(x) for x in p.read_text().splitlines()]
def close(a,b):return math.isclose(a,b,abs_tol=1e-6,rel_tol=1e-9)
def quantile(values,q):
    if not values:return None
    v=sorted(values);p=(len(v)-1)*q;i=int(p)
    return v[i]+(v[min(i+1,len(v)-1)]-v[i])*(p-i)
def logs(folder):
    result=[]
    for p in folder.glob('*.log'):
        for line in p.read_text().splitlines():
            if '{' not in line:continue
            try:row=json.loads(line[line.index('{'):])
            except ValueError:continue
            if 'wall' in row:result.append(row)
    return result

all_count=0; folders=0
for p in sorted(R.rglob('requests.jsonl')):
    if p.parent==R:continue
    data=lines(p);d=p.parent;folders+=1;all_count+=len(data)
    configs=[read(c) for c in sorted(d.glob('client-*-config.json'))]
    expected={}
    for c in configs:
        for item in c['schedule']:
            expected[item['body']['request_id']]=(c,item)
    check(len(data)==len(expected)==read(d/'counts.json')['planned'],f'{d.name}: planned/sent cardinality')
    check(len({x['request_id'] for x in data})==len(data),f'{d.name}: unique request IDs')
    for x in data:
        label=d.name+'/'+x['request_id']
        c,item=expected[x['request_id']]
        check(x['request']==item['body'],label+': original request payload')
        check(close(x['planned_wall'],c['start_wall']+item['offset']),label+': planned wall time')
        check(close(x['send_lag_seconds'],x['sent_mono']-x['planned_mono']),label+': send lag')
        check(x['planned_mono']<=x['sent_mono']<=x['end_mono'],label+': monotonic order')
        check(close(x['total_seconds'],x['end_mono']-x['sent_mono']),label+': total duration')
        check(len(x['inter_chunk_seconds'])==max(0,x['chunks']-1),label+': N-1 intervals')
        check(all(t>=0 for t in x['inter_chunk_seconds']),label+': nonnegative intervals')
        if x['chunks']:
            check(x['sent_mono']<=x['first_mono']<=x['end_mono'],label+': first chunk timestamp')
            check(close(x['ttfc_seconds'],x['first_mono']-x['sent_mono']),label+': TTFC formula')
            check(x['first_mono']+sum(x['inter_chunk_seconds'])<=x['end_mono']+1e-6,label+': last chunk before EOF')
        else:check(x['first_mono'] is None and x['ttfc_seconds'] is None,label+': no fabricated TTFC')
        if x['outcome']=='completed':
            terminal=x['terminal']
            check(x['http_status']==200 and x['chunks']==x['expected_chunks']==terminal['chunks'],label+': completed accounting')
            check(all(terminal[k]==x[k] for k in ['request_id','pod','model_id','artifact_sha256','backend_mode']),label+': terminal identity')
            check(terminal['history_echo']==x['request'].get('history',[]),label+': history isolation')
            check(terminal['prompt_echo']==x['request']['prompt'],label+': prompt echo')
        elif x['outcome']=='rejected':
            check(x['http_status']==429 and not x['chunks'] and x['terminal'] is None,label+': rejection accounting')
        elif x['outcome']=='broken':
            check(x['http_status']==200 and x['terminal'] is None,label+': broken accounting')
        elif x['outcome']=='http_error':check(x['http_status']>=400,label+': HTTP error accounting')
    c=read(d/'counts.json')
    observed=Counter(x['outcome'] for x in data)
    check(all(c.get(k,0)==observed[k] for k in ['completed','rejected','broken','expired','http_error','connection_error']),d.name+': outcome totals')
    check(c['http_5xx']==sum(x['http_status'] is not None and x['http_status']>=500 for x in data),d.name+': 5xx total')

summary=read(R/'summary.json')
bs={}
for folder in sorted(R.glob('B-*')):
    if not folder.is_dir():continue
    data=lines(folder/'requests.jsonl');config=read(folder/'conditions.json')
    schedules=[]
    for c in sorted(folder.glob('client-*-config.json')):
        schedules.extend(read(c)['schedule'])
    schedules.sort(key=lambda x:x['offset'])
    digest=hashlib.sha256(json.dumps(schedules,sort_keys=True).encode()).hexdigest()
    check(digest==config['schedule_sha256'],folder.name+': schedule hash independently rebuilt')
    check(all(len(x['request']['prompt'])==128 and x['expected_chunks']==5 for x in data),folder.name+': input/output lengths')
    client_counts=Counter(x['client'] for x in data)
    check(sorted(client_counts.values())==([72]*4 if config['traffic']=='uniform' else [19,19,19,231]),folder.name+': client traffic mix')
    check(close(schedules[-1]['offset']+1/config['arrival_rate'],120),folder.name+': 120-second arrival window')
    start=read(folder/'metrics-before-complete.json')['wall'];end=read(folder/'metrics-after-complete.json')['wall']
    check(start<min(x['sent_wall'] for x in data) and max(x['end_wall'] for x in data)<end,folder.name+': same quiet comparison window')
    events=logs(folder)
    completed=[e for e in events if e['event']=='completed' and start<=e['wall']<=end]
    good=[x for x in data if x['outcome']=='completed']
    check(len(completed)==len(good),folder.name+': exact client/server cohort size')
    check({(e['request_id'],e['pod']) for e in completed}=={(x['request_id'],x['pod']) for x in good},folder.name+': exact client/server cohort identity')
    by_id={x['request_id']:x for x in data}
    per_pod={}
    for e in sorted(completed,key=lambda x:x['wall']):
        cache=per_pod.setdefault(e['pod'],OrderedDict())
        key=by_id[e['request_id']]['request']['prefix']
        check(e['cache_hit']==(key in cache),folder.name+': cache trace '+e['request_id'])
        check(e['cache_hit']==by_id[e['request_id']]['cache_hit'],folder.name+': client/server cache result')
        cache[key]=True;cache.move_to_end(key)
        while len(cache)>16:cache.popitem(last=False)
    for pod in per_pod:
        resets=[e['wall'] for e in events if e['event']=='cache_reset' and e['pod']==pod and e['wall']<start]
        check(bool(resets),folder.name+': reset logged '+pod)
        last_reset=max(resets) if resets else 0
        check(not any(e['event']=='received' and e['pod']==pod and last_reset<e['wall']<start for e in events),folder.name+': no traffic between reset and boundary')
    # Rebuild every classic-histogram bucket directly from the server's per-request values.
    before=read(folder/'metrics-before-complete.json')['series']['data']['result']
    after=read(folder/'metrics-after-complete.json')['series']['data']['result']
    def bucket_map(series,metric):
        return {(e['metric']['pod'],e['metric'].get('le')):float(e['value'][1]) for e in series if e['metric']['__name__']==metric}
    bucket_checks=0
    for name,field in [('hw11_ttfc_seconds_bucket','ttfc_seconds'),('hw11_total_seconds_bucket','total_seconds')]:
        a=bucket_map(before,name);b=bucket_map(after,name)
        active={x["pod"] for x in data}
        check({k for k in a if k[0] in active}=={k for k in b if k[0] in active},folder.name+': stable active histogram series '+name)
        check(all(k[0] not in active for k in set(a)^set(b)),folder.name+': excluded stale series served no requests')
        for (pod,le),v in b.items():
            delta=v-a[(pod,le)]
            actual=sum(e[field]<=float(le) for e in completed if e['pod']==pod)
            check(delta==actual,folder.name+': bucket '+name+'/'+pod+'/'+le)
            bucket_checks+=1
    cumulative=summary['B'][folder.name]['histogram_buckets']
    target=len(completed)*.95;last_boundary=0;last_count=0;derived=None
    for bound,count in sorted((float(k),v) for k,v in cumulative.items()):
        if count>=target:
            derived=last_boundary if math.isinf(bound) else last_boundary+(bound-last_boundary)*(target-last_count)/(count-last_count)
            break
        last_boundary,last_count=bound,count
    check(close(derived,summary['B'][folder.name]['prometheus_p95']),folder.name+': independent histogram quantile')
    bs[folder.name]={'cohort_n':len(completed),'buckets_verified':bucket_checks,'cache_replayed':len(completed),'prometheus_p95_recomputed':derived}
    for key,q in [('ttfc_p50',.5),('ttfc_p95',.95)]:
        check(close(quantile([x['ttfc_seconds'] for x in good],q),summary['B'][folder.name][key]),folder.name+': '+key)
for traffic in ['uniform','hot80']:
    for repeat in [1,2,3]:
        a=read(R/f'B-{traffic}-sticky-r{repeat}/conditions.json')
        b=read(R/f'B-{traffic}-ordinary-r{repeat}/conditions.json')
        check(a['schedule_sha256']==b['schedule_sha256'],f'{traffic} {repeat}: identical policies workload')

# Every Kubernetes sample must come from a single ordered capture, without an earlier run appended.
captures={}
for p in R.glob('*/timeline.jsonl'):
    t=lines(p);wall=[x['wall'] for x in t]
    errors=[x for x in t if 'error' in x or 'metrics_error' in x]
    check(wall==sorted(wall) and len(set(wall))==len(wall),p.parent.name+': timeline monotonic')
    check(not errors,p.parent.name+': no collector errors')
    max_gap=max((b-a for a,b in zip(wall,wall[1:])),default=0)
    check(max_gap<10,p.parent.name+': no missing monitoring interval')
    captures[p.parent.name]={'samples':len(t),'span_seconds':wall[-1]-wall[0],'max_gap_seconds':max_gap}
# Recalculate all reported A/C latency statistics without importing analyze.py.
main_dirs=[p.parent for p in sorted(R.glob("B-*/requests.jsonl"))]
main_dirs += [R/name for name in ("A-warmup-broken","A-warmup-fixed","A-rollout","A-rollback","A-crash","A-long-0","A-long-1","A-killed-stream","C-hpa-uniform","C-hpa-hot80")]
main_dirs += [R/"C-alerts"/name for name in ("burst","healthy")]
source_main=[x for d in main_dirs for x in lines(d/"requests.jsonl")]
master=lines(R/"requests.jsonl")
source_map={(x["experiment"],x["request_id"]):x for x in source_main}
check(len(source_map)==len(source_main)==len(master),"Master journal: no dropped or duplicated main attempts")
check(all({k:v for k,v in x.items() if k!="attempt_id"}==source_map[(x["experiment"],x["request_id"])] for x in master),"Master journal preserves source records")
for d in main_dirs:
    if d.name.startswith("B-"):continue
    records=lines(d/"requests.jsonl")
    name=d.name if d.parent.name!="C-alerts" else "alerts-"+d.name
    group="A" if name.startswith("A-") else "C"
    good=[x for x in records if x["outcome"]=="completed"]
    published=summary[group][name]
    for key,field,q in [("ttfc_p50","ttfc_seconds",.5),("ttfc_p95","ttfc_seconds",.95),("total_p95","total_seconds",.95),("queue_p95","queue_seconds",.95)]:
        value=quantile([x[field] for x in good],q)
        check(published[key] is None if value is None else close(published[key],value),name+": independently recalculated "+key)
    check(close(published["error_fraction"],1-len(good)/len(records)),name+": failure fraction")

warm={}
for name in ("A-warmup-broken","A-warmup-fixed"):
    d=R/name;records=lines(d/"requests.jsonl");events=logs(d)
    low=min(x["sent_wall"] for x in records);high=max(x["end_wall"] for x in records)
    received=[e for e in events if e["event"]=="received" and low-1<=e["wall"]<=high+1]
    check({e["request_id"] for e in received}=={x["request_id"] for x in records},name+": server received every scheduled request")
    early=[e for e in received if not e["warm"]]
    check(len(early)>0 if name.endswith("broken") else len(early)==0,name+": actual early requests")
    check(len(early)==sum(x["http_status"]==503 for x in records),name+": early requests match actual 503 responses")
    trace=lines(d/"timeline.jsonl")
    early_admitted=0
    for e in early:
        candidates=[x for x in trace if x["wall"]<=e["wall"]]
        if not candidates:continue
        point=candidates[-1]
        pod=next((p for p in point["pods"] if p["name"]==e["pod"]),None)
        admitted={ip for es in point["endpoints"] if es["service"]=="hw11-api" for ep in es["endpoints"] if ep["conditions"].get("ready") for ip in ep["addresses"]}
        if pod and pod["ready"] and pod["ip"] in admitted:early_admitted+=1
    if early:check(early_admitted>0,name+": premature Ready and EndpointSlice match a real early request")
    warm[name]={"requests_before_warmup":len(early),"early_requests_with_ready_endpoint":early_admitted}

normal=lines(R/"A-rollout/requests.jsonl")
check({x["model_id"] for x in normal}=={"A","B"},"Rolling update actually serves both A and B")
roll_events=logs(R/"A-rollout")
for x in normal:
    if x["model_id"]!="B":continue
    warmed=[e["wall"] for e in roll_events if e["event"]=="warmup_finished" and e["pod"]==x["pod"]]
    check(warmed and min(warmed)<x["sent_wall"],"B only receives traffic after its logged warmup")
# All old-Pod responses, including ordinary queued requests, must finish before the grace deadline.
old_deadlines={}
with gzip.open(R/"A-rollout/kubernetes.jsonl.gz","rt") as raw:
    for line in raw:
        for obj in json.loads(line)["data"]["items"]:
            meta=obj["metadata"]
            if obj["kind"]=="Pod" and meta.get("deletionTimestamp"):
                old_deadlines.setdefault(meta["name"],meta["deletionTimestamp"])
from datetime import datetime
for x in normal+lines(R/"A-long-0/requests.jsonl")+lines(R/"A-long-1/requests.jsonl"):
    if x["pod"] in old_deadlines:
        deadline=datetime.fromisoformat(old_deadlines[x["pod"]].replace("Z","+00:00")).timestamp()
        check(x["end_wall"]<deadline,"Every old-Pod response completed within grace: "+x["request_id"])
terminated=read(R/"A-crash/recovered-pod.json")["status"]["containerStatuses"][0]
check(terminated["restartCount"]>=1 and terminated["lastState"]["terminated"]["exitCode"]==137,"Forced kill independently confirmed by container exit 137 and restart")
rollback=read(R/"A-rollback/events.json")
cleanup=rollback.get("rollback_end",rollback.get("rollback_cleanup_observed_wall"))
after=[x for x in lines(R/"A-rollback/requests.jsonl") if x["sent_wall"]>cleanup]
check(after and all(x["outcome"]=="completed" and x["model_id"]=="A" for x in after),"Confirmed A responses after rollback completed/cleanup observed")

hpa_details={}
from datetime import datetime
epoch=lambda x:datetime.fromisoformat(x.replace("Z","+00:00")).timestamp()
for name in ("C-hpa-uniform","C-hpa-hot80"):
    d=R/name;trace=lines(d/"timeline.jsonl");ev=read(d/"events.json");requests=lines(d/"requests.jsonl")
    active_start={p["metadata"]["name"] for p in ev["initial_pods"]}
    new=[p for p in ev["three_ready"]["pods"] if p["metadata"]["name"] not in active_start]
    check(len(new)==1,name+": exactly one added replica")
    pod=new[0];created=epoch(pod["metadata"]["creationTimestamp"])
    ready=epoch(next(c for c in pod["status"]["conditions"] if c["type"]=="Ready")["lastTransitionTime"])
    check(ready-created>=20,name+": new replica waited for configured warmup")
    active_names=active_start|{pod["metadata"]["name"]}
    every_spec=[h["spec"] for x in trace for h in x.get("hpa",[])]
    check(all(h["minReplicas"]==2 and h["maxReplicas"]==3 and h["metrics"][0]["resource"]["name"]=="cpu" and h["metrics"][0]["resource"]["target"]["averageUtilization"]==60 for h in every_spec),name+": exact HPA configuration")
    # Inspect actual requested CPU and consumption used by the HPA, not just the plotted curve.
    resource_cpu=[m["resource"]["current"]["averageUtilization"] for x in trace for h in x.get("hpa",[]) for m in h["status"].get("currentMetrics",[]) if m.get("type")=="Resource" and "averageUtilization" in m["resource"]["current"]]
    check(resource_cpu and max(resource_cpu)>60,name+": HPA observed CPU above target")
    stabilizing=[x["wall"] for x in trace if any(c.get("reason")=="ScaleDownStabilized" for h in x.get("hpa",[]) for c in h["status"].get("conditions",[]))]
    check(bool(stabilizing),name+": actual downscale stabilization observed")
    scale_down=epoch(ev["two_ready"]["hpa"]["lastScaleTime"])
    check(scale_down>ev["load_end"] and ev["two_ready"]["wall"]>=scale_down,name+": delayed downscale after load")
    served=sum(x["pod"]==pod["metadata"]["name"] for x in requests)
    if name.endswith("hot80"):check(served==0,name+": added replica served no bound clients")
    else:check(served>0,name+": added replica handled ordinary traffic")
    hpa_details[name]={"created_utc":pod["metadata"]["creationTimestamp"],"ready_utc":next(c for c in pod["status"]["conditions"] if c["type"]=="Ready")["lastTransitionTime"],
        "creation_to_ready_seconds":ready-created,"peak_average_cpu_percent":max(resource_cpu),
        "new_replica_requests":served,"last_response_wall":max(x["end_wall"] for x in requests),
        "load_runner_finished_wall":ev["load_end"],"downscale_decision_wall":scale_down,
        "two_ready_observed_wall":ev["two_ready"]["wall"],"return_after_runner_seconds":ev["two_ready"]["wall"]-ev["load_end"],
        "return_after_last_response_seconds":ev["two_ready"]["wall"]-max(x["end_wall"] for x in requests),
        "first_stabilization_observation_wall":min(stabilizing)}

alerts=lines(R/"C-alerts/timeline.jsonl")
alert_info={}
for name in ("HighTTFC","HighFailureFraction"):
    observed=[]
    for x in alerts:
        state=next((a["state"] for a in x.get("alerts",[]) if a["labels"]["alertname"]==name),"inactive")
        if not observed or observed[-1][1]!=state:observed.append((x["wall"],state))
    check([x[1] for x in observed]==["inactive","pending","firing","inactive"],name+": exact state lifecycle")
    firing=next(a for x in alerts for a in x.get("alerts",[]) if a["labels"]["alertname"]==name and a["state"]=="firing")
    first_firing=next(x[0] for x in observed if x[1]=="firing")
    check(first_firing-epoch(firing["activeAt"])>=15,name+": alert held long enough before firing")
    alert_info[name]=observed
healthy=lines(R/"C-alerts/healthy/requests.jsonl")
end=read(R/"C-alerts/events.json")["final_wall"]
recent=[x for x in healthy if end-60<=x["end_wall"]<=end]
check(len(recent)>=20 and all(x["outcome"]=="completed" and x["ttfc_seconds"]<1 for x in recent),"Alerts return to normal with sufficient healthy samples")


evidence={'request_folders':folders,'all_source_attempts':all_count,'B':bs,'captures':captures,'warmup':warm,'hpa':hpa_details,'alerts':alert_info,'healthy_final_minute':len(recent)}
out={'passed':not failures,'assertions':checks,'failures':failures,'evidence':evidence}
(R/'independent-audit.json').write_text(json.dumps(out,ensure_ascii=False,indent=2))
print(json.dumps({'passed':out['passed'],'assertions':checks,'failures':failures,'folders':folders,'all_source_attempts':all_count},ensure_ascii=False,indent=2))
if failures:sys.exit(1)
