"""Warmup defect/fix, graceful rollout, bad release rollback, crash and restoration."""
from lab import *
import sys
import shutil


def one_request(name,pod_ip,body):
    row={"client":0,"offset":0,"body":{"prompt":"How do I report an issue?","request_id":name,
        "session_id":"dialog-A","max_chunks":5,"cpu_scale":1.0,**body}}
    return load(name,[row],pod_ip,timeout=70)


def main():
    if "--only-recovery" not in sys.argv:
        upgrade(image="hw11-api:1.0.1",modelVersion="A",warmupSeconds=20,brokenProbes=False,failWarmup=False,
                revision="A-baseline",**{"hpa.enabled":False})
        wait_ready()
        # Identical 60s/0.8 rps schedules on two already warmed replicas; only probes change.
        for broken,label in ((True,"A-warmup-broken"),(False,"A-warmup-fixed")):
            directory=RESULTS/label
            with Collector(directory):
                rows=schedule(60,.8,prefix="warmup-identical")
                with background_load(label,rows,directory=directory) as future:
                    time.sleep(6)
                    began=upgrade(brokenProbes=broken,revision=label)
                    save(directory/"events.json",{"rollout_start":began,"broken_probes":broken,
                          "warmup_seconds":20,"same_schedule":"warmup-identical","rate":.8})
                    wait_ready()
                    data=future.result()
                time.sleep(2)
            print(label,counts(data),flush=True)
        # Two old-Pod streams guarantee that a stream also crosses termination of an old Pod.
        label="A-rollout";directory=RESULTS/label
        old=pods()
        with Collector(directory):
            with background_load(label,schedule(85,.2,prefix="normal-rollout"),directory=directory,timeout=70) as requests:
                pool=ThreadPoolExecutor(max_workers=2)
                long_futures=[pool.submit(one_request,f"A-long-{i}",p["status"]["podIP"],
                             {"max_chunks":90,"chunk_delay":.4,"prefix":"long-stream"}) for i,p in enumerate(old)]
                time.sleep(7)
                began=upgrade(modelVersion="B",revision="normal-B")
                wait_ready()
                ready_at=time.time()
                long_data=[r for f in long_futures for r in f.result()]
                pool.shutdown()
                data=requests.result()
                save(directory/"events.json",{"rollout_start":began,"rollout_ready":ready_at,
                     "old_pods":[p["metadata"]["name"] for p in old],"grace_seconds":60,"prestop_seconds":5,
                     "uvicorn_grace_seconds":45,"long_stream_request_ids":[r["request_id"] for r in long_data]})
                save(directory/"full-run-counts.json",counts(data+long_data))
            time.sleep(2)
        print(label,counts(data+long_data),flush=True)
        assert counts(data+long_data).get("http_5xx")==0
        assert all(r["outcome"]=="completed" for r in data+long_data)

        # Establish A, deploy deliberately unready B, and use Helm's stored revision to roll back.
        upgrade(modelVersion="A",failWarmup=False,revision="pre-bad-A");wait_ready()
        history=json.loads(command([HELM,"history","hw11","-n","hw11","-o","json"]).stdout)
        prior_revision=history[-1]["revision"]
        label="A-rollback";directory=RESULTS/label
        with Collector(directory):
            with background_load(label,schedule(95,.7,prefix="bad-B"),directory=directory) as future:
                time.sleep(6)
                bad_begin=upgrade(modelVersion="B",failWarmup=True,revision="unready-B")
                time.sleep(32)
                bad_state=get("pods,endpointslices")
                save(directory/"bad-B-state.json",bad_state)
                rollback_begin=time.time()
                rollback=command([HELM,"rollback","hw11",str(prior_revision),"-n","hw11","--wait","--timeout","180s"])
                (directory/"helm-rollback.log").write_text(rollback.stdout+rollback.stderr)
                wait_ready()
                rollback_end=time.time()
                data=future.result()
                save(directory/"events.json",{"bad_B_begin":bad_begin,"rollback_begin":rollback_begin,
                     "rollback_end":rollback_end,"helm_revision_restored":prior_revision})
        print(label,counts(data),flush=True)
        assert all(r["model_id"]=="A" and r["outcome"]=="completed" for r in data)

    # Abrupt SIGKILL inside a container: keep distinct attempts and record the transport break.
    label="A-crash";directory=RESULTS/label
    victim=pods()[0]
    with Collector(directory):
        with background_load(label,schedule(85,.5,prefix="crash-background"),directory=directory) as requests:
            pool=ThreadPoolExecutor(max_workers=1)
            stream=pool.submit(one_request,"A-killed-stream",victim["status"]["podIP"],
                              {"max_chunks":90,"chunk_delay":.4})
            time.sleep(7)
            crash_begin=time.time()
            container_id=victim["status"]["containerStatuses"][0]["containerID"].split("://",1)[1]
            kill_command=[shutil.which("docker") or "/opt/homebrew/bin/docker","exec","mlops-hw11-control-plane",
                          "ctr","-n","k8s.io","tasks","kill","--signal","SIGKILL",container_id]
            result=command(kill_command)
            save(directory/"events.json",{"kill_wall":crash_begin,"victim":victim["metadata"]["name"],
                 "container_id":container_id,"command":kill_command,"exit_code":result.returncode,
                 "output":result.stdout,"stderr":result.stderr})
            killed=stream.result();pool.shutdown()
            data=requests.result()
            k("wait","--for=condition=Ready","pod/"+victim["metadata"]["name"],"--timeout=180s")
            (directory/"previous-container.log").write_text(k("logs",victim["metadata"]["name"],"--previous",check=False))
            save(directory/"recovered-pod.json",get("pod/"+victim["metadata"]["name"]))
            save(directory/"full-run-counts.json",counts(data+killed))
    print(label,counts(data+killed),flush=True)
    assert any(r["outcome"]=="broken" for r in killed)

    # Delete a Pod, keep full dialogue outside it, then ask the replacement directly.
    label="A-dialog-recovery";directory=RESULTS/label
    victim=pods()[0]
    old_uid=victim["metadata"]["uid"]
    marker="DIALOG-ALPHA-CONTROL-7391"
    history=[marker,"How can I configure issue labels?"]
    with Collector(directory):
        before=one_request("dialog-before",victim["status"]["podIP"],{"history":history})
        old_names={p["metadata"]["name"] for p in pods()}
        deletion=time.time()
        k("delete","pod",victim["metadata"]["name"],"--wait=false")
        deadline=time.monotonic()+120
        replacement=None
        while time.monotonic()<deadline:
            current=pods()
            matches=[p for p in current if p["metadata"]["uid"]!=old_uid and
                     p["metadata"]["name"] not in old_names and
                     any(c["type"]=="Ready" and c["status"]=="True" for c in p["status"].get("conditions",[]))]
            if matches:
                replacement=matches[0];break
            time.sleep(2)
        if replacement is None:raise RuntimeError("No ready replacement")
        after=one_request("dialog-after",replacement["status"]["podIP"],{"history":history+[before[0]["terminal"]["answer"]]})
        other=one_request("dialog-beta",replacement["status"]["podIP"],
                          {"history":["DIALOG-BETA-CONTROL-1826"],"session_id":"dialog-B"})
        save(directory/"evidence.json",{"deleted":deletion,"old_uid":old_uid,"new_uid":replacement["metadata"]["uid"],
            "old_pod":victim["metadata"]["name"],"new_pod":replacement["metadata"]["name"],
            "before":before,"after":after,"other":other})
    print("A COMPLETE",flush=True)


if __name__=="__main__":
    main()
