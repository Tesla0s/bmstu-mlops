"""Five own checks; first three perform real HTTP calls to Pods in Kubernetes."""
import unittest
from lab import *

TEST_DIR=RESULTS/"integration"
TEST_DIR.mkdir(exist_ok=True)


def inspect_url(url):
    code="""import urllib.request,urllib.error,json
try:
 r=urllib.request.urlopen(URL,timeout=5)
except urllib.error.HTTPError as e:
 r=e
print(json.dumps({"status":r.code,"body":r.read().decode()}))
""".replace("URL",repr(url))
    return json.loads(k("exec","client-0","--","python","-c",code))


def ask(name,service="hw11-api",history=None,session="integration"):
    row={"client":0,"offset":0,"body":{"request_id":name,"session_id":session,"prompt":"How to add an issue label?",
         "history":history or [],"prefix":"integration-prefix","max_chunks":3,"cpu_scale":1.0}}
    return load(name,[row],service,TEST_DIR/name,start_delay=.5)[0]


class KubernetesIntegration(unittest.TestCase):
    def test_01_warmup_is_not_ready_or_an_endpoint(self):
        old_names={p["metadata"]["name"] for p in pods()}
        began=time.time()
        upgrade(revision="integration-warmup",modelVersion="A",brokenProbes=False,failWarmup=False,**{"hpa.enabled":False})
        deadline=time.monotonic()+90
        new=None
        while time.monotonic()<deadline:
            candidates=[p for p in pods() if p["metadata"]["name"] not in old_names
               and p["status"].get("podIP") and any("running" in c.get("state",{}) for c in p["status"].get("containerStatuses",[]))]
            if candidates:
                candidate=candidates[0]
                try:
                    probe=inspect_url("http://"+candidate["status"]["podIP"]+":8080/ready")
                    new=candidate;break
                except RuntimeError:
                    pass
            time.sleep(.5)
        self.assertIsNotNone(new,"No running new Pod observed during warmup")
        ip=new["status"]["podIP"]
        endpoints=get("endpointslices")
        admitted=[ip for x in endpoints["items"] if x["metadata"].get("labels",{}).get("kubernetes.io/service-name")=="hw11-api"
                  for e in x.get("endpoints",[]) if e["conditions"].get("ready") for ip in e["addresses"]]
        self.assertEqual(probe["status"],503)
        self.assertNotIn(ip,admitted)
        wait_ready()
        reply=ask("warmup-ready",ip)
        self.assertEqual(reply["outcome"],"completed")
        self.assertEqual(reply["pod"],new["metadata"]["name"])
        save(TEST_DIR/"test_01.json",{"began":began,"new_pod":new,"unready_probe":probe,"endpoints":endpoints,"reply":reply})

    def test_02_actual_loaded_version_after_rollback(self):
        rollback=json.loads((RESULTS/"A-rollback/events.json").read_text())
        rows=[json.loads(x) for x in (RESULTS/"A-rollback/requests.jsonl").read_text().splitlines()]
        self.assertTrue(any(r["sent_wall"]>rollback["rollback_begin"] for r in rows))
        self.assertTrue(all(r["model_id"]=="A" and r["outcome"]=="completed" for r in rows))
        reply=ask("rollback-live")
        self.assertEqual(reply["model_id"],"A")
        self.assertEqual(reply["outcome"],"completed")
        self.assertEqual(reply["artifact_sha256"],"73ee23f9d56e4263a3194bf0bb250236a0512a7c1f0763f141b86c516dda4a87")
        save(TEST_DIR/"test_02.json",{"rollback":rollback,"live_reply":reply})

    def test_03_replacement_restores_artifact_and_dialogue(self):
        current=pods();victim=current[0]
        names={p["metadata"]["name"] for p in current}
        history=["RECOVERY-CONTROL-4062"]
        before=ask("replacement-before",victim["status"]["podIP"],history)
        k("delete","pod",victim["metadata"]["name"],"--wait=false")
        deadline=time.monotonic()+120
        replacement=None
        while time.monotonic()<deadline:
            matches=[p for p in pods() if p["metadata"]["name"] not in names
                     and any(c["type"]=="Ready" and c["status"]=="True" for c in p["status"].get("conditions",[]))]
            if matches:replacement=matches[0];break
            time.sleep(2)
        self.assertIsNotNone(replacement)
        after=ask("replacement-after",replacement["status"]["podIP"],history+[before["terminal"]["answer"]])
        self.assertEqual(before["artifact_sha256"],after["artifact_sha256"])
        self.assertNotEqual(before["pod"],after["pod"])
        self.assertEqual(after["terminal"]["history_echo"][0],history[0])
        save(TEST_DIR/"test_03.json",{"old_uid":victim["metadata"]["uid"],
             "new_uid":replacement["metadata"]["uid"],"before":before,"after":after})

    def test_04_accounting_keeps_rejections_and_broken_streams(self):
        dirs=list(RESULTS.glob("B-*/requests.jsonl"))+[RESULTS/"A-killed-stream/requests.jsonl"]
        rows=[json.loads(line) for p in dirs for line in p.read_text().splitlines()]
        rejected=[r for r in rows if r["outcome"]=="rejected"]
        broken=[r for r in rows if r["outcome"]=="broken"]
        self.assertTrue(rejected)
        self.assertTrue(broken)
        self.assertTrue(all(r["http_status"]==429 and r["first_mono"] is None for r in rejected))
        self.assertTrue(all(r["http_status"]==200 and r["terminal"] is None for r in broken))
        self.assertTrue(any(r["chunks"]>0 for r in broken))
        self.assertTrue(all(r["end_mono"]>=r["sent_mono"] for r in rows))
        for p in RESULTS.glob("B-*/counts.json"):
            c=json.loads(p.read_text());self.assertEqual(c["planned"],c["sent"])
            self.assertEqual(c["sent"],sum(c.get(k,0) for k in
                 ("completed","rejected","broken","expired","http_error","connection_error")))
        save(TEST_DIR/"test_04.json",{"rejected":len(rejected),"broken":len(broken),
             "partial_chunks":[r["chunks"] for r in broken],"total_rows":len(rows)})

    def test_05_two_dialogues_never_share_history(self):
        with ThreadPoolExecutor(max_workers=2) as pool:
            a=pool.submit(ask,"isolation-alpha","hw11-api",["ALPHA-PRIVATE-CONTROL-9917"],"alpha")
            b=pool.submit(ask,"isolation-beta","hw11-api",["BETA-PRIVATE-CONTROL-2143"],"beta")
            first,second=a.result(),b.result()
        self.assertEqual(first["terminal"]["history_echo"],["ALPHA-PRIVATE-CONTROL-9917"])
        self.assertEqual(second["terminal"]["history_echo"],["BETA-PRIVATE-CONTROL-2143"])
        self.assertNotIn("BETA-PRIVATE",json.dumps(first["terminal"]))
        self.assertNotIn("ALPHA-PRIVATE",json.dumps(second["terminal"]))
        save(TEST_DIR/"test_05.json",{"alpha":first,"beta":second})


if __name__=="__main__":
    unittest.main(verbosity=2)
