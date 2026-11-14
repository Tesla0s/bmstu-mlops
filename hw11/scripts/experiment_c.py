"""Actual CPU-driven HPA, a repeated hot-client scenario, and alert state transitions."""
from lab import *


def wait_replicas(count,timeout=240):
    began=time.time()
    deadline=time.monotonic()+timeout
    while time.monotonic()<deadline:
        items=pods()
        live=[p for p in items if not p["metadata"].get("deletionTimestamp")]
        ready=[p for p in live if any(c["type"]=="Ready" and c["status"]=="True" for c in p["status"].get("conditions",[]))]
        h=get("hpa/hw11-api")["status"]
        if len(live)==count and len(ready)==count and h.get("desiredReplicas")==count:
            return {"wall":time.time(),"wait_seconds":time.time()-began,"pods":items,"hpa":h}
        time.sleep(3)
    raise RuntimeError(f"HPA did not reach {count} ready replicas")


def main():
    upgrade(image="hw11-api:1.0.1",modelVersion="A",brokenProbes=False,failWarmup=False,
            revision="C-baseline",**{"hpa.enabled":True})
    wait_ready()
    k("get","--raw","/apis/metrics.k8s.io/v1beta1/namespaces/hw11/pods")
    save(RESULTS/"C-resource-metrics-before.json",json.loads(k("get","--raw","/apis/metrics.k8s.io/v1beta1/namespaces/hw11/pods")))
    wait_replicas(2)
    for traffic in ("uniform","hot80"):
        label="C-hpa-"+traffic;directory=RESULTS/label
        directory.mkdir(exist_ok=True)
        service="hw11-api" if traffic=="uniform" else "hw11-sticky"
        reset_caches()
        if traffic=="hot80":
            # Establish each client's affinity while only the original two Pods are available.
            load("C-affinity-prime",schedule(8,.5,prefix="prime"),service,RESULTS/"C-affinity-prime")
        initial=pods()
        with Collector(directory):
            with background_load(label,schedule(150,2.4,traffic,label,
                                 {"cpu_scale":6.0,"chunk_delay":0.0}),service,directory,timeout=25) as future:
                up=wait_replicas(3)
                data=future.result()
                load_end=time.time()
            down=wait_replicas(2)
            save(directory/"events.json",{"initial_pods":initial,"three_ready":up,"load_end":load_end,
                "two_ready":down,"actual_return_after_load_seconds":down["wall"]-load_end,
                "configured_downscale_stabilization_seconds":60,
                "measurement_seconds":150,"arrival_rate":2.4,"cpu_scale":6.0,"chunk_delay":0.0})
        print(label,counts(data),"cooldown",down["wall"]-load_end,flush=True)
    upgrade(**{"hpa.enabled":False})
    wait_ready()
    label="C-alerts";directory=RESULTS/label
    with Collector(directory):
        time.sleep(10)
        began=time.time()
        burst=load("C-alert-burst",schedule(45,18,prefix="alert-burst",body_overrides={"chunk_delay":.2}),
                   directory=directory/"burst",timeout=30)
        healthy_begin=time.time()
        healthy=load("C-alert-healthy",schedule(100,.5,prefix="alert-healthy"),
                     directory=directory/"healthy",timeout=30)
        time.sleep(10)
        final=prom_get("/api/v1/alerts")
        save(directory/"events.json",{"burst_begin":began,"healthy_begin":healthy_begin,
            "final_wall":time.time(),"final_alerts":final,"burst_counts":counts(burst),
            "healthy_counts":counts(healthy),"latency_threshold_seconds":1,
            "failure_fraction_threshold":.05,"minimum_observations":20,"window_seconds":60,"for_seconds":15})
    print("C COMPLETE",flush=True)


if __name__=="__main__":
    main()
