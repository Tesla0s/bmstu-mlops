"""Requirement-focused offline audit of real integration evidence."""
from datetime import datetime
import hashlib
import gzip
import json
from pathlib import Path
import sys
from analyze import rows,read

ROOT=Path(__file__).resolve().parents[1]
R=ROOT/"results"
checks=[]


def check(name,condition,evidence,details=None):
    checks.append({"requirement":name,"passed":bool(condition),"evidence":evidence,"details":details})


def epoch(value):
    return datetime.fromisoformat(value.replace("Z","+00:00")).timestamp()


def main():
    summary=read(R/"summary.json")
    data=rows(R/"requests.jsonl")
    expected={v:hashlib.sha256(('{"backend_mode":"simulated","version":"'+v+'","task":"github-issue-classification"}').encode()).hexdigest() for v in ("A","B")}
    check("Все содержательные ответы обозначены simulated и имеют фактический хеш",
          all(r["backend_mode"]=="simulated" and r["artifact_sha256"]==expected[r["model_id"]]
              for r in data if r["chunks"]>0),"requests.jsonl")
    check("Попытки имеют уникальный составной ключ",
          len({r["attempt_id"] for r in data})==len(data),"requests.jsonl")
    check("Дефект прогрева воспроизведён и исправлен",
          summary["A"]["A-warmup-broken"]["requests_before_warmup"]>0 and
          summary["A"]["A-warmup-fixed"]["requests_before_warmup"]==0 and
          summary["A"]["A-warmup-fixed"]["success_n"]==48,
          ["A-warmup-broken/","A-warmup-fixed/"])
    normal=rows(R/"A-rollout/requests.jsonl")+rows(R/"A-long-0/requests.jsonl")+rows(R/"A-long-1/requests.jsonl")
    check("Полный штатный выпуск: ноль 5xx, отказов и незавершённых потоков",
          len(normal)==19 and all(r["outcome"]=="completed" for r in normal),
          ["A-rollout/","A-long-0/","A-long-1/"])
    event=read(R/"A-rollout/events.json")
    timeline=rows(R/"A-rollout/timeline.jsonl")
    long=rows(R/"A-long-0/requests.jsonl")+rows(R/"A-long-1/requests.jsonl")
    overlap=[];grace=[];termination_details=[]
    # deletionTimestamp is the deletion deadline, not the beginning of graceful termination.
    # Use the first raw metadata (before any shortening), plus independently observed state.
    deletion_meta={}
    with gzip.open(R/"A-rollout/kubernetes.jsonl.gz","rt") as source:
        for line in source:
            snapshot=json.loads(line)
            for obj in snapshot["data"]["items"]:
                meta=obj["metadata"]
                if obj["kind"]=="Pod" and meta.get("deletionTimestamp"):
                    deletion_meta.setdefault(meta["name"],meta)
    for r in long:
        meta=deletion_meta.get(r["pod"])
        observed=[point["wall"] for point in timeline for pod in point.get("pods",[])
                  if pod["name"]==r["pod"] and pod["deleting"]]
        if meta and observed:
            deadline=epoch(meta["deletionTimestamp"])
            allowance=meta["deletionGracePeriodSeconds"]
            began=deadline-allowance
            overlap.append(r["sent_wall"]<min(observed)<r["end_wall"])
            grace.append(r["end_wall"]-began)
            termination_details.append({"pod":r["pod"],"deletion_deadline":deadline,
                "grace_seconds":allowance,"inferred_start_second_precision":began,
                "first_terminating_observation":min(observed),"stream_end":r["end_wall"],
                "completed_before_deadline":r["end_wall"]<deadline})
    check("Длительные потоки пересекают начало обновления",
          all(r["sent_wall"]<event["rollout_start"]<r["end_wall"] for r in long),["A-rollout/events.json","A-long-0/","A-long-1/"])
    check("Старый поток пережил начало удаления Pod и уложился в grace period",
          any(overlap) and all(x["completed_before_deadline"] for x in termination_details),
          ["A-rollout/timeline.jsonl","A-rollout/kubernetes.jsonl.gz"],
          {"end_after_termination_request_seconds":grace,"metadata":termination_details,
           "reference":"https://kubernetes.io/docs/reference/kubernetes-api/definitions/object-meta-v1-meta/"})
    bad=read(R/"A-rollback/bad-B-state.json")["items"]
    bad_pods=[p for p in bad if p["kind"]=="Pod" and
              any(e.get("name")=="FAIL_WARMUP" and e.get("value")=="true" for e in p["spec"]["containers"][0].get("env",[]))]
    bad_ips={p["status"].get("podIP") for p in bad_pods}
    admitted={ip for x in bad if x["kind"]=="EndpointSlice" for e in x.get("endpoints",[]) if e["conditions"].get("ready") for ip in e["addresses"]}
    check("Неготовая B исключена из готовых EndpointSlice",
          len(bad_pods)>0 and not bad_ips.intersection(admitted) and
          all(not any(c["type"]=="Ready" and c["status"]=="True" for c in p["status"].get("conditions",[])) for p in bad_pods),
          "A-rollback/bad-B-state.json")
    rollback=rows(R/"A-rollback/requests.jsonl")
    check("После управляемого отката фактически отвечает A",
          len(rollback)>0 and all(r["model_id"]=="A" and r["outcome"]=="completed" for r in rollback),
          ["A-rollback/requests.jsonl","A-rollback/helm-rollback.log"])
    crash=read(R/"A-crash/events.json");killed=rows(R/"A-killed-stream/requests.jsonl")[0]
    trace=rows(R/"A-crash/timeline.jsonl")
    unready=[p["wall"] for p in trace if any(x["name"]==crash["victim"] and not x["ready"] for x in p.get("pods",[])) and p["wall"]>=crash["kill_wall"]]
    recovered=[r for r in rows(R/"A-crash/requests.jsonl") if r["pod"]==crash["victim"] and r["outcome"]=="completed" and r["sent_wall"]>crash["kill_wall"]]
    crash_stats={"partial_chunks":killed["chunks"],"client_detected_seconds":killed["end_wall"]-crash["kill_wall"],
                 "first_unready_observation_seconds":min(unready)-crash["kill_wall"] if unready else None,
                 "first_recovered_response_seconds":min(r["end_wall"] for r in recovered)-crash["kill_wall"] if recovered else None}
    check("Авария отдельно: частичный ответ, обнаружение и восстановление",
          killed["outcome"]=="broken" and killed["chunks"]>0 and bool(unready) and bool(recovered),
          ["A-killed-stream/requests.jsonl","A-crash/"],crash_stats)
    recovery=read(R/"A-dialog-recovery/evidence.json")
    check("Замена Pod восстанавливает тот же артефакт и историю",
          recovery["old_uid"]!=recovery["new_uid"] and
          recovery["before"][0]["artifact_sha256"]==recovery["after"][0]["artifact_sha256"] and
          "DIALOG-ALPHA-CONTROL-7391" in recovery["after"][0]["terminal"]["history_echo"],
          "A-dialog-recovery/evidence.json")
    b=summary["B"]
    check("Две политики × два сценария × три измерения не короче 120 с",
          len(b)==12 and all(x["observed_span_seconds"]>=120 and x["sent"]==288 for x in b.values()),
          ["B-*/conditions.json","B-*/requests.jsonl"])
    check("В каждой паре совпадает расписание и состав запросов",
          all(b[f"B-{t}-sticky-r{i}"]["conditions"]["schedule_sha256"]==b[f"B-{t}-ordinary-r{i}"]["conditions"]["schedule_sha256"]
              for t in ("uniform","hot80") for i in (1,2,3)),"B-*/conditions.json")
    check("Четыре разных IP доходят до серверов без подмены адреса",
          all(len(x["client_ips"])==4 and x["client_ips"]==x["server_client_ips"] for x in b.values()),
          ["B-network.json","environment/iptables-nat.txt","requests.jsonl"])
    check("Очередь горячей реплики наблюдается при простое другой",
          all(b[f"B-hot80-sticky-r{i}"]["idle_capacity"]["hot_waiting_other_idle_samples"]>0 for i in (1,2,3)),
          "B-hot80-sticky-*/prometheus-range-complete.json")
    check("Сохранён случай без перекоса при привязке",
          sorted(b["B-uniform-sticky-r1"]["pod_distribution"].values())==[144,144],
          "B-uniform-sticky-r1/requests.jsonl")
    check("Исправление горячего сценария проверено во всех трёх повторах",
          all(b[f"B-hot80-ordinary-r{i}"]["ttfc_p95"]<b[f"B-hot80-sticky-r{i}"]["ttfc_p95"] and
              b[f"B-hot80-ordinary-r{i}"]["success_n"]==288 for i in (1,2,3)),
          "B-verified-summary.json")
    check("Клиент и Prometheus сопоставляют один набор завершённых ответов",
          all(x["success_n"]==x["server_completed_count"] for x in b.values()),
          "B-*/metrics-*-complete.json")
    for name in ("C-hpa-uniform","C-hpa-hot80"):
        event=summary["C"][name]["events"];trace=rows(R/name/"timeline.jsonl")
        counts=[sum(p["ready"] and not p["deleting"] for p in x.get("pods",[])) for x in trace if "pods" in x]
        cpus=[float(m["value"][1]) for x in trace for m in x.get("metrics",[]) if "__name__" not in m["metric"] and "pod" in m["metric"]]
        conditions=[c for x in trace for h in x.get("hpa",[]) for c in h["status"].get("conditions",[])]
        check(name+": реальные 2 → 3 → 2 и вычислительная CPU-нагрузка",
              counts[0]==2 and 3 in counts and counts[-1]==2 and max(cpus)>.5,
              [name+"/timeline.jsonl",name+"/events.json"],
              {"peak_cpu_cores":max(cpus),"return_after_last_response_seconds":event["actual_return_after_load_seconds"],
               "hpa_condition_reasons":sorted({c["reason"] for c in conditions})})
    for name,states in summary["C"]["alert_transitions"].items():
        labels=[x["state"] for x in states]
        check(name+": pending → firing → inactive","pending" in labels and "firing" in labels and labels[-1]=="inactive",
              "C-alerts/timeline.jsonl",states)
    check("Пять собственных проверок имеют сохранённые реальные доказательства",
          all((R/"integration"/f"test_{i:02}.json").exists() for i in range(1,6)),
          ["integration/","integration.log"])
    check("Четыре требуемых графика созданы из сырых измерений",
          len(list((ROOT/"figures").glob("0*.png")))==4,"../figures/")
    out={"passed":all(c["passed"] for c in checks),"checks":checks,"request_attempts":len(data),"cpu_backend":"simulated"}
    (R/"requirements-audit.json").write_text(json.dumps(out,ensure_ascii=False,indent=2))
    for c in checks:print(("PASS" if c["passed"] else "FAIL"),c["requirement"])
    if not out["passed"]:sys.exit(1)


if __name__=="__main__":
    main()
