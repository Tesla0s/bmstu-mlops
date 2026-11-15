"""Derive all statistics and the four required figures from saved raw observations."""
from collections import Counter, defaultdict
from datetime import datetime
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
R=ROOT/"results"
FIG=ROOT/"figures"
FIG.mkdir(exist_ok=True)
plt.rcParams.update({"font.family":"DejaVu Sans","font.size":10,"axes.grid":True,
                     "grid.alpha":.22,"figure.dpi":130,"savefig.dpi":170})


def read(path):
    return json.loads(Path(path).read_text())


def rows(path):
    p=Path(path)
    if p.name=="timeline.jsonl" and p.with_name("timeline-complete.jsonl").exists():p=p.with_name("timeline-complete.jsonl")
    return [json.loads(x) for x in p.read_text().splitlines()] if p.exists() else []


def percentile(values,q):
    return float(np.percentile(values,q)) if values else None


def summarize(data):
    good=[r for r in data if r["outcome"]=="completed"]
    return {"sent":len(data),"outcomes":dict(Counter(r["outcome"] for r in data)),
     "success_n":len(good),"error_fraction":1-len(good)/len(data) if data else 0,
     "ttfc_p50":percentile([r["ttfc_seconds"] for r in good],50),
     "ttfc_p95":percentile([r["ttfc_seconds"] for r in good],95),
     "total_p95":percentile([r["total_seconds"] for r in good],95),
     "queue_p95":percentile([r["queue_seconds"] for r in good],95),
     "inter_chunk_p95":percentile([v for r in good for v in r["inter_chunk_seconds"]],95),
     "send_lag_p95":percentile([r["send_lag_seconds"] for r in data],95),
     "send_lag_max":max((r["send_lag_seconds"] for r in data),default=0),
     "pod_distribution":dict(Counter(r["pod"] for r in data)),
     "cache_hit_fraction_success":sum(r["cache_hit"] for r in good)/len(good) if good else None,
     "client_ips":sorted({r["client_ip"] for r in data}),
     "server_client_ips":sorted({r["server_observed_client_ip"] for r in good}),
     "observed_span_seconds":max(r["end_wall"] for r in data)-min(r["planned_wall"] for r in data) if data else 0}


def metrics_series(timeline,name):
    out=defaultdict(list)
    for s in timeline:
        for m in s.get("metrics",[]):
            if m["metric"].get("__name__")==name:
                out[m["metric"]["pod"]].append((s["wall"],float(m["value"][1])))
    return out


def idle_capacity(timeline):
    valid=0;skew=0
    for s in timeline:
        running={m["metric"]["pod"]:float(m["value"][1]) for m in s.get("metrics",[]) if m["metric"].get("__name__")=="hw11_running"}
        waiting={m["metric"]["pod"]:float(m["value"][1]) for m in s.get("metrics",[]) if m["metric"].get("__name__")=="hw11_waiting"}
        if len(running)!=2:continue
        valid+=1
        if any(waiting.get(a,0)>0 and running[b]==0 for a in running for b in running if a!=b):
            skew+=1
    return {"samples":valid,"hot_waiting_other_idle_samples":skew,"fraction":skew/valid if valid else 0}


def bucket_delta(folder,metric):
    def extract(label):
        path=folder/f"metrics-{label}-complete.json"
        if not path.exists():path=folder/f"metrics-{label}.json"
        series=read(path)["series"]["data"]["result"]
        return {(x["metric"]["pod"],x["metric"].get("le")):float(x["value"][1])
                for x in series if x["metric"].get("__name__")==metric}
    before=extract("before");after=extract("after")
    result=defaultdict(float)
    for key,value in after.items():
        if key in before:
            result[key[1]]+=value-before[key]
    return dict(result)


def server_events(folder,start,end):
    events=[]
    for path in folder.glob("*.log"):
        for line in path.read_text().splitlines():
            if "{" not in line:continue
            try:event=json.loads(line[line.index("{"):])
            except ValueError:continue
            if start<=event.get("wall",0)<=end:events.append(event)
    return events


def audit_and_summary():
    summary={"B":{},"A":{},"C":{}}
    all_requests=[]
    for p in sorted(R.glob("B-*/requests.jsonl")):
        folder=p.parent;data=rows(p);conditions=read(folder/"conditions.json")
        result=summarize(data)
        result["conditions"]=conditions
        result["idle_capacity"]=idle_capacity(rows(folder/"timeline.jsonl"))
        buckets=bucket_delta(folder,"hw11_ttfc_seconds_bucket")
        result["server_completed_count"]=buckets.get("+Inf",0)
        vector=read(folder/"prometheus-p95.json")["result"]["data"]["result"]
        result["prometheus_p95"]=float(vector[0]["value"][1]) if vector else None
        result["histogram_buckets"]=buckets
        assert result["server_completed_count"]==result["success_n"],(folder,result)
        assert result["client_ips"]==result["server_client_ips"],folder
        assert len(result["client_ips"])==4,folder
        assert result["observed_span_seconds"]>=120,folder
        assert len(data)==288,folder
        assert all(r["backend_mode"]=="simulated" for r in data)
        summary["B"][folder.name]=result
        all_requests.extend(data)
    assert len(summary["B"])==12
    for traffic in ("uniform","hot80"):
        for repeat in (1,2,3):
            a=summary["B"][f"B-{traffic}-sticky-r{repeat}"]["conditions"]
            b=summary["B"][f"B-{traffic}-ordinary-r{repeat}"]["conditions"]
            assert a["schedule_sha256"]==b["schedule_sha256"]
    for name in ("A-warmup-broken","A-warmup-fixed","A-rollout","A-rollback","A-crash","A-long-0","A-long-1","A-killed-stream"):
        data=rows(R/name/"requests.jsonl")
        summary["A"][name]=summarize(data);all_requests.extend(data)
        if name in ("A-warmup-broken","A-warmup-fixed"):
            events=server_events(R/name,min(r["sent_wall"] for r in data)-1,max(r["end_wall"] for r in data)+1)
            early=[e for e in events if e.get("event")=="received" and not e.get("warm")]
            summary["A"][name]["requests_before_warmup"]=len(early)
            summary["A"][name]["early_request_ids"]=[e["request_id"] for e in early]
            assert bool(early)==(name=="A-warmup-broken"),(name,early)
    for name in ("C-hpa-uniform","C-hpa-hot80"):
        data=rows(R/name/"requests.jsonl")
        summary["C"][name]={**summarize(data),"events":read(R/name/"events.json")}
        all_requests.extend(data)
    for name in ("burst","healthy"):
        data=rows(R/"C-alerts"/name/"requests.jsonl")
        summary["C"]["alerts-"+name]=summarize(data);all_requests.extend(data)
    alert_states=defaultdict(list)
    for s in rows(R/"C-alerts/timeline.jsonl"):
        by_name={a["labels"]["alertname"]:a["state"] for a in s.get("alerts",[])}
        for name in ("HighTTFC","HighFailureFraction"):
            state=by_name.get(name,"inactive")
            if not alert_states[name] or alert_states[name][-1]["state"]!=state:
                alert_states[name].append({"wall":s["wall"],"state":state})
    summary["C"]["alert_transitions"]=dict(alert_states)
    for name,states in alert_states.items():
        labels=[x["state"] for x in states]
        assert "pending" in labels and "firing" in labels and labels[-1]=="inactive",(name,labels)
    assert summary["A"]["A-warmup-broken"]["outcomes"].get("http_error",0)>0
    assert summary["A"]["A-warmup-fixed"]["success_n"]==48
    for name in ("A-rollout","A-long-0","A-long-1","A-rollback"):
        s=summary["A"][name];assert s["sent"]==s["success_n"],name
    rollout=read(R/"A-rollout/events.json")
    for name in ("A-long-0","A-long-1"):
        long=rows(R/name/"requests.jsonl")[0]
        assert long["sent_wall"]<rollout["rollout_start"]<long["end_wall"]
    for row in all_requests:
        row["attempt_id"]=row["experiment"]+"/"+row["request_id"]
    assert len({r["attempt_id"] for r in all_requests})==len(all_requests)
    all_requests.sort(key=lambda x:(x["planned_wall"],x["request_id"]))
    (R/"requests.jsonl").write_text("".join(json.dumps(x,ensure_ascii=False)+"\n" for x in all_requests))
    (R/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2))
    (R/"B-verified-summary.json").write_text(json.dumps(summary["B"],ensure_ascii=False,indent=2))
    return summary


def graph_a():
    names=("A-warmup-broken","A-warmup-fixed","A-rollout")
    titles=("Дефект: HTTP ошибочно означает готовность","Исправление: ждём завершения прогрева","Штатный выпуск A → B")
    fig,axes=plt.subplots(2,3,figsize=(15,8),sharex="col")
    for i,(name,title) in enumerate(zip(names,titles)):
        timeline=rows(R/name/"timeline.jsonl");req=rows(R/name/"requests.jsonl")
        start=min(s["wall"] for s in timeline)
        t=[s["wall"]-start for s in timeline]
        axes[0,i].step(t,[sum(p["ready"] and not p["deleting"] for p in s.get("pods",[])) for s in timeline],where="post",label="Ready",color="#315ca5")
        axes[0,i].step(t,[sum(e["conditions"].get("ready",False) for es in s.get("endpoints",[]) if es["service"]=="hw11-api" for e in es["endpoints"]) for s in timeline],where="post",ls="--",label="Готовые адреса Service",color="#dd8b32")
        axes[0,i].step(t,[sum(not p["deleting"] for p in snap.get("pods",[])) for snap in timeline],where="post",ls=":",color="#6b6d75",label="Все Pod, включая прогрев")
        backend=[]
        for snap in timeline:
            live={p["name"] for p in snap.get("pods",[]) if not p["deleting"]}
            backend.append(sum(float(m["value"][1]) for m in snap.get("metrics",[]) if m["metric"].get("__name__")=="hw11_backend_ready" and m["metric"].get("pod") in live))
        axes[0,i].step(t,backend,where="post",color="#299472",alpha=.8,label="Прогретые backend, шаг 5 с")
        axes[0,i].set(title=title,ylabel="Реплики / адреса, шт.",ylim=(-.2,3.5))
        event=read(R/name/"events.json")["rollout_start"]-start
        for ax in axes[:,i]:ax.axvline(event,color="black",ls=":",label="Начало обновления")
        for outcome,color,marker in (("completed","#2c8c72","o"),("http_error","#d55050","x"),("rejected","#e09b35","x"),("connection_error","#9b56b6","x")):
            selected=[r for r in req if r["outcome"]==outcome]
            if selected:axes[1,i].scatter([r["sent_wall"]-start for r in selected],[r["ttfc_seconds"] or 0 for r in selected],c=color,marker=marker,s=20,label={"completed":"Завершён","http_error":"HTTP-ошибка","rejected":"Отклонён","connection_error":"Нет соединения"}[outcome])
        if name=="A-rollout":
            for n in (0,1):
                d=rows(R/f"A-long-{n}/requests.jsonl")[0]
                axes[1,i].hlines(3+n*2,d["sent_wall"]-start,d["end_wall"]-start,color="#9b56b6",lw=4,label="Длительный поток" if n==0 else None)
            axes[1,i].text(.98,.98,"Полосы: время жизни двух потоков\n(их высота условна)",transform=axes[1,i].transAxes,va="top",ha="right",fontsize=9)
        axes[1,i].set(xlabel="Время с начала опыта, с",ylabel="TTFC успешных ответов, с")
        axes[0,i].legend(fontsize=8);axes[1,i].legend(fontsize=8,loc="center right" if name=="A-rollout" else "lower right")
    fig.suptitle("A · Настоящие Ready / EndpointSlice и запросы; прогрев 20 с, 2 реплики\nПрогрев: 0,8 запроса/с; штатный выпуск: 0,2 запроса/с + 2 длительных потока",fontsize=13)
    fig.tight_layout(rect=(0,0,1,.93));fig.savefig(FIG/"01-warmup-rollout.png");plt.close(fig)


def graph_b_queues():
    fig,axes=plt.subplots(2,2,figsize=(13,8),sharex="col")
    for i,policy in enumerate(("sticky","ordinary")):
        name=f"B-hot80-{policy}-r1";req=rows(R/name/"requests.jsonl");timeline=rows(R/name/"timeline.jsonl")
        start=min(r["planned_wall"] for r in req)
        names=sorted({r["pod"] for r in req})
        for j,pod in enumerate(names):
            data=[r for r in req if r["pod"]==pod and r["outcome"]=="completed"]
            label="Pod "+pod.rsplit("-",1)[-1]
            axes[0,i].scatter([r["sent_wall"]-start for r in data],[r["ttfc_seconds"] for r in data],s=14,label=label)
            points=metrics_series(timeline,"hw11_waiting").get(pod,[])
            if points:axes[1,i].step([x-start for x,y in points],[y for x,y in points],where="post",label=label)
        axes[0,i].set(title="ClientIP" if policy=="sticky" else "Обычный Service",ylabel="TTFC успешных ответов, с")
        axes[1,i].set(xlabel="Время измерения, с",ylabel="Ожидающие в очереди, запросов")
        axes[0,i].legend();axes[1,i].legend()
    fig.suptitle("B · 80% трафика от одного клиента · повтор 1\n2,4 запроса/с, 2 реплики, один вычислитель на Pod, ёмкость 8 запросов",fontsize=13)
    fig.tight_layout(rect=(0,0,1,.92));fig.savefig(FIG/"02-pod-latency-queue.png");plt.close(fig)


def graph_b_repeats(summary):
    fig,axes=plt.subplots(2,2,figsize=(12,8),sharex=True)
    for col,traffic in enumerate(("uniform","hot80")):
        for policy,color,offset in (("sticky","#d27a39",-.13),("ordinary","#347caa",.13)):
            ss=[summary["B"][f"B-{traffic}-{policy}-r{r}"] for r in (1,2,3)]
            x=np.arange(1,4)+offset
            axes[0,col].bar(x,[s["ttfc_p95"] for s in ss],.25,label=("ClientIP" if policy=="sticky" else "Обычный Service"),color=color)
            axes[1,col].bar(x,[100*s["error_fraction"] for s in ss],.25,label=("ClientIP" if policy=="sticky" else "Обычный Service"),color=color)
        axes[0,col].set(title="Равномерный трафик" if traffic=="uniform" else "80% через одного клиента",ylabel="Клиентский p95 TTFC, с")
        axes[1,col].set(ylabel="Неуспешные попытки, %",xlabel="Независимый повтор",xticks=[1,2,3])
    handles,labels=axes[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,loc="upper center",bbox_to_anchor=(.5,.9),ncol=2)
    fig.suptitle("B · Две политики × два сценария × три повтора\nКаждая фаза 120 с, 288 запланированных запросов; кеш пуст перед фазой",fontsize=13)
    fig.tight_layout(rect=(0,0,1,.86));fig.savefig(FIG/"03-policy-comparison.png");plt.close(fig)


def graph_c():
    fig,axes=plt.subplots(2,2,figsize=(14,8),sharex="col")
    for col,traffic in enumerate(("uniform","hot80")):
        name="C-hpa-"+traffic
        timeline=rows(R/name/"timeline.jsonl");req=rows(R/name/"requests.jsonl");events=read(R/name/"events.json")
        start=min(s["wall"] for s in timeline)
        t=[s["wall"]-start for s in timeline]
        axes[0,col].step(t,[sum(p["ready"] and not p["deleting"] for p in s.get("pods",[])) for s in timeline],where="post",label="Готовые Pod")
        axes[0,col].step(t,[s.get("hpa",[{}])[0].get("status",{}).get("desiredReplicas",np.nan) if s.get("hpa") else np.nan for s in timeline],where="post",ls="--",label="Желаемые HPA")
        twin=axes[0,col].twinx()
        bins=np.arange(0,max(t)+10,10)
        counts,_=np.histogram([r["planned_wall"]-start for r in req],bins=bins)
        twin.step(bins[:-1],counts/10,where="post",color="#ba6b3f",alpha=.7,label="Плановая нагрузка")
        twin.set_ylabel("Запросов/с");twin.set_ylim(0,3.2)
        axes[0,col].set(title="Обычный Service" if traffic=="uniform" else "ClientIP · горячий клиент",ylabel="Число реплик",ylim=(0,3.5))
        axes[0,col].legend(loc="lower left",fontsize=9);twin.legend(loc="upper right",fontsize=9)
        series=defaultdict(list)
        active_names={p["name"] for point in timeline for p in point.get("pods",[])}
        for s in timeline:
            for m in s.get("metrics",[]):
                if "__name__" not in m["metric"] and m["metric"].get("pod") in active_names:
                    series[m["metric"]["pod"]].append((s["wall"]-start,float(m["value"][1])))
        for pod,points in series.items():
            axes[1,col].plot([x for x,y in points],[y for x,y in points],label=pod.rsplit("-",1)[-1])
        axes[1,col].axhline(.15,ls=":",color="black",label="Цель: 60% от 0,25 CPU")
        axes[1,col].axvline(events["load_end"]-start,color="#ba6b3f",ls="--")
        axes[1,col].set(xlabel="Время опыта, с",ylabel="Используемые ядра CPU")
        axes[1,col].legend(fontsize=8)
    fig.suptitle("C · Реальное масштабирование 2 → 3 → 2 по вычислительной нагрузке\nCPU ×6, паузы между частями отключены; окно уменьшения 60 с",fontsize=13)
    fig.tight_layout(rect=(0,0,1,.92));fig.savefig(FIG/"04-hpa-load.png");plt.close(fig)


if __name__=="__main__":
    summary=audit_and_summary()
    graph_a();graph_b_queues();graph_b_repeats(summary);graph_c()
    print("Raw data audited; summary and four figures generated.")
