"""Generate reviewable, versioned monitoring manifests without external Helm dependencies."""
from pathlib import Path
import json
import yaml

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "monitoring"
OUT.mkdir(exist_ok=True)
prom = {
 "global":{"scrape_interval":"5s","evaluation_interval":"5s"},
 "rule_files":["/etc/prometheus/alerts.yaml"],
 "scrape_configs":[
  {"job_name":"hw11-api", "kubernetes_sd_configs":[{"role":"pod","namespaces":{"names":["hw11"]}}],
   "relabel_configs":[
    {"source_labels":["__meta_kubernetes_pod_label_app"],"action":"keep","regex":"hw11-api"},
    {"source_labels":["__meta_kubernetes_pod_container_port_name"],"action":"keep","regex":"http"},
    {"source_labels":["__meta_kubernetes_pod_name"],"target_label":"pod"},
    {"source_labels":["__meta_kubernetes_namespace"],"target_label":"namespace"}]},
  {"job_name":"kube-state-metrics","static_configs":[{"targets":["kube-state-metrics:8080"]}]},
  {"job_name":"cadvisor","scheme":"https",
   "authorization":{"credentials_file":"/var/run/secrets/kubernetes.io/serviceaccount/token"},
   "tls_config":{"ca_file":"/var/run/secrets/kubernetes.io/serviceaccount/ca.crt"},
   "kubernetes_sd_configs":[{"role":"node"}],
   "relabel_configs":[
    {"target_label":"__address__","replacement":"kubernetes.default.svc:443"},
    {"source_labels":["__meta_kubernetes_node_name"],"target_label":"__metrics_path__",
     "replacement":"/api/v1/nodes/$1/proxy/metrics/cadvisor"}],
   "metric_relabel_configs":[
    {"source_labels":["namespace"],"action":"keep","regex":"hw11"},
    {"source_labels":["__name__"],"action":"keep",
     "regex":"container_cpu_usage_seconds_total|container_memory_working_set_bytes"}]}
 ]}
alerts={"groups":[{"name":"hw11","interval":"5s","rules":[
 {"alert":"HighTTFC",
  "expr":'histogram_quantile(0.95, sum by (le) (rate(hw11_ttfc_seconds_bucket[1m]))) > 1 and sum(increase(hw11_ttfc_seconds_count[1m])) >= 20',
  "for":"15s","labels":{"severity":"warning"},
  "annotations":{"summary":"TTFC p95 > 1 second, at least 20 completed requests in 1m"}},
 {"alert":"HighFailureFraction",
  "expr":'(sum(rate(hw11_results_total{outcome!="completed"}[1m])) / sum(rate(hw11_results_total[1m]))) > 0.05 and sum(increase(hw11_results_total[1m])) >= 20',
  "for":"15s","labels":{"severity":"warning"},
  "annotations":{"summary":"Failure fraction > 5%, at least 20 outcomes in 1m"}}
 ]}]}
queries=[
 ("Время до первой части · p95", 'histogram_quantile(0.95, sum by (le,pod) (rate(hw11_ttfc_seconds_bucket[1m])))',"s"),
 ("Длительность ответа · p95",'histogram_quantile(0.95, sum by (le,pod) (rate(hw11_total_seconds_bucket[1m])))',"s"),
 ("В работе и в очереди",'{__name__=~"hw11_running|hw11_waiting"}',"short"),
 ("Завершённые ответы в секунду",'sum by (pod) (rate(hw11_results_total{outcome="completed"}[1m]))',"reqps"),
 ("Отказы и обрывы в секунду",'sum by (pod,outcome) (rate(hw11_results_total{outcome!="completed"}[1m]))',"reqps"),
 ("Использование процессора",'sum by (pod) (rate(container_cpu_usage_seconds_total{namespace="hw11",container="api"}[1m]))',"cores"),
 ("Оперативная память",'sum by (pod) (container_memory_working_set_bytes{namespace="hw11",container="api"})',"bytes"),
 ("Готовые реплики",'sum(kube_pod_status_ready{namespace="hw11",pod=~"hw11-api-.*",condition="true"})',"short"),
 ("Попадания в имитацию кеша",'sum by (pod,result) (rate(hw11_cache_total[1m]))',"ops"),
 ("Ожидание очереди · p95",'histogram_quantile(0.95, sum by (le,pod) (rate(hw11_queue_seconds_bucket[1m])))',"s")]
dashboard={"id":None,"uid":"hw11","title":"ДЗ 11 · сервис под нагрузкой","schemaVersion":41,
 "version":1,"tags":["hw11","simulated"],"timezone":"browser","refresh":"5s",
 "time":{"from":"now-15m","to":"now"},"panels":[]}
for i,(title,query,unit) in enumerate(queries):
 dashboard["panels"].append({"id":i+1,"type":"timeseries","title":title,
   "gridPos":{"x":(i%2)*12,"y":(i//2)*8,"w":12,"h":8},
   "datasource":{"type":"prometheus","uid":"prom"},
   "targets":[{"refId":"A","expr":query,"legendFormat":("{{__name__}} · {{pod}}" if i==2 else "{{pod}} · {{outcome}}" if i==4 else "{{pod}} · {{result}}" if i==8 else "Ready" if i==7 else "{{pod}}")}],
   "fieldConfig":{"defaults":{"unit":unit,"min":0},"overrides":[]}})
(OUT/"prometheus.yaml").write_text(yaml.safe_dump(prom,sort_keys=False))
(OUT/"alerts.yaml").write_text(yaml.safe_dump(alerts,sort_keys=False))
(OUT/"dashboard.json").write_text(json.dumps(dashboard,ensure_ascii=False,indent=2))
(OUT/"queries.txt").write_text("\n\n".join(title+"\n"+query for title,query,_ in queries))
objects=[]
def obj(kind,name,**kw):
 d={"apiVersion":"v1","kind":kind,"metadata":{"name":name,"namespace":"hw11"},**kw}
 objects.append(d);return d
obj("ServiceAccount","monitoring")
objects.extend([
 {"apiVersion":"rbac.authorization.k8s.io/v1","kind":"ClusterRole","metadata":{"name":"hw11-monitoring"},
  "rules":[{"apiGroups":[""],"resources":["nodes","nodes/proxy","pods","services","endpoints","namespaces"],"verbs":["get","list","watch"]},
           {"apiGroups":["apps"],"resources":["deployments","replicasets"],"verbs":["get","list","watch"]},
           {"apiGroups":["autoscaling"],"resources":["horizontalpodautoscalers"],"verbs":["get","list","watch"]}]},
 {"apiVersion":"rbac.authorization.k8s.io/v1","kind":"ClusterRoleBinding","metadata":{"name":"hw11-monitoring"},
  "roleRef":{"apiGroup":"rbac.authorization.k8s.io","kind":"ClusterRole","name":"hw11-monitoring"},
  "subjects":[{"kind":"ServiceAccount","name":"monitoring","namespace":"hw11"}]}])
obj("ConfigMap","prometheus-config",data={"prometheus.yml":yaml.safe_dump(prom),"alerts.yaml":yaml.safe_dump(alerts)})
obj("ConfigMap","grafana-provision",data={
 "datasources.yaml":yaml.safe_dump({"apiVersion":1,"datasources":[{"name":"Prometheus","uid":"prom","type":"prometheus","url":"http://prometheus:9090","access":"proxy","isDefault":True}]}),
 "dashboards.yaml":yaml.safe_dump({"apiVersion":1,"providers":[{"name":"hw11","type":"file","options":{"path":"/var/lib/grafana/dashboards"}}]})})
obj("ConfigMap","grafana-dashboard",data={"hw11.json":json.dumps(dashboard,ensure_ascii=False)})
def deployment(name,image,port,request_mem,limit_mem,args=None,volumes=None,mounts=None,env=None):
 container={"name":name,"image":image,"ports":[{"containerPort":port}],
  "resources":{"requests":{"cpu":"50m","memory":request_mem},"limits":{"cpu":"500m","memory":limit_mem}},
  "readinessProbe":{"httpGet":{"path":"/-/ready" if name=="prometheus" else "/api/health" if name=="grafana" else "/healthz","port":port},"initialDelaySeconds":3,"periodSeconds":5}}
 if args:container["args"]=args
 if mounts:container["volumeMounts"]=mounts
 if env:container["env"]=[{"name":k,"value":v} for k,v in env.items()]
 objects.append({"apiVersion":"apps/v1","kind":"Deployment","metadata":{"name":name,"namespace":"hw11"},
  "spec":{"replicas":1,"selector":{"matchLabels":{"app":name}},"template":{"metadata":{"labels":{"app":name}},
          "spec":{"serviceAccountName":"monitoring","containers":[container],"volumes":volumes or []}}}})
 obj("Service",name,spec={"selector":{"app":name},"ports":[{"port":port,"targetPort":port}]})
deployment("prometheus","prom/prometheus:v3.6.0",9090,"128Mi","512Mi",
 ["--config.file=/etc/prometheus/prometheus.yml","--storage.tsdb.path=/prometheus","--storage.tsdb.retention.time=3h","--storage.tsdb.retention.size=256MB"],
 [{"name":"config","configMap":{"name":"prometheus-config"}},{"name":"data","emptyDir":{"sizeLimit":"512Mi"}}],
 [{"name":"config","mountPath":"/etc/prometheus"},{"name":"data","mountPath":"/prometheus"}])
deployment("kube-state-metrics","registry.k8s.io/kube-state-metrics/kube-state-metrics:v2.17.0",8080,"32Mi","128Mi",
 ["--resources=pods,nodes,deployments,horizontalpodautoscalers"])
deployment("grafana","grafana/grafana:12.2.0",3000,"96Mi","256Mi",volumes=[
 {"name":"data","emptyDir":{"sizeLimit":"256Mi"}},{"name":"provision","configMap":{"name":"grafana-provision"}},
 {"name":"dashboard","configMap":{"name":"grafana-dashboard"}}],mounts=[
 {"name":"data","mountPath":"/var/lib/grafana"},
 {"name":"provision","mountPath":"/etc/grafana/provisioning/datasources/datasources.yaml","subPath":"datasources.yaml"},
 {"name":"provision","mountPath":"/etc/grafana/provisioning/dashboards/dashboards.yaml","subPath":"dashboards.yaml"},
 {"name":"dashboard","mountPath":"/var/lib/grafana/dashboards"}],env={
 "GF_AUTH_ANONYMOUS_ENABLED":"true","GF_AUTH_ANONYMOUS_ORG_ROLE":"Viewer",
 "GF_AUTH_DISABLE_LOGIN_FORM":"true","GF_ANALYTICS_REPORTING_ENABLED":"false",
 "GF_ANALYTICS_CHECK_FOR_UPDATES":"false","GF_ANALYTICS_CHECK_FOR_PLUGIN_UPDATES":"false"})
(OUT/"resources.yaml").write_text(yaml.safe_dump_all(objects,sort_keys=False))
print("Rendered monitoring resources and ten-panel dashboard")
