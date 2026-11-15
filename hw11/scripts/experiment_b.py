"""Twelve measured runs: 2 policies × 2 traffic patterns × 3 repetitions × 120 s."""
from lab import *
from collections import Counter


def main():
    prepare_outputs("B-network.json", *[f"B-{traffic}-{policy}-r{repeat}"
        for traffic in ("uniform","hot80") for policy in ("sticky","ordinary") for repeat in (1,2,3)])
    upgrade(image="hw11-api:1.0.1",modelVersion="A",warmupSeconds=20,brokenProbes=False,failWarmup=False,revision="B-baseline",**{"hpa.enabled":False})
    wait_ready()
    # HPA removal does not implicitly restore replicas in all Helm histories.
    k("scale","deployment/hw11-api","--replicas=2")
    wait_ready()
    save(RESULTS/"B-network.json",{"pods":get("pods"),"services":get("services"),
      "kube_proxy":json.loads(command([KUBE,"-n","kube-system","get","configmap","kube-proxy","-o","json"]).stdout),
      "keep_alive":False,"connection_header":"close","network":"kindnet; kube-proxy iptables; in-cluster IPv4"})
    for repeat in (1,2,3):
        for traffic in ("uniform","hot80"):
            # Alternate order between repetitions to reduce time/order confounding.
            order=("sticky","ordinary") if repeat%2 else ("ordinary","sticky")
            rows=schedule(120,2.4,traffic,f"B-{traffic}-r{repeat}")
            for policy in order:
                name=f"B-{traffic}-{policy}-r{repeat}"
                folder=RESULTS/name
                folder.mkdir(exist_ok=True)
                print("START",name,time.strftime("%H:%M:%S"),flush=True)
                reset_caches()
                save(folder/"conditions.json",{"duration_seconds":120,"arrival_rate":2.4,"cache_initial":"empty",
                  "warmup":"both Pods warmed for 20 seconds before all trials; only prefix cache reset before each trial",
                  "policy":policy,"traffic":traffic,"repeat":repeat,"cpu_hit":.02,"cpu_miss":.16,
                  "cpu_chunk":.025,"delay_chunk":.1,"chunks":5,"concurrency":1,"queue_capacity":8,"cache_capacity":16,
                  "schedule_sha256":__import__("hashlib").sha256(json.dumps(rows,sort_keys=True).encode()).hexdigest()})
                before=checkpoint(folder,"before")
                with Collector(folder):
                    data=load(name,rows,"hw11-sticky" if policy=="sticky" else "hw11-api",folder,timeout=20)
                after=checkpoint(folder,"after")
                expression=f'histogram_quantile(0.95, sum by (le) (hw11_ttfc_seconds_bucket @ {after["wall"]:.3f} - hw11_ttfc_seconds_bucket @ {before["wall"]:.3f}))'
                save(folder/"prometheus-p95.json",{"query":expression,"result":prom(expression)})
                print("DONE",name,counts(data),"pods",dict(Counter(r["pod"] for r in data)),flush=True)
    print("B COMPLETE",flush=True)


if __name__=="__main__":
    main()
