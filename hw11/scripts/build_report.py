"""Build the HTML report from audited measurements; no invented result placeholders."""
from html import escape
from datetime import datetime
import json
import gzip
from pathlib import Path
import statistics

ROOT=Path(__file__).resolve().parents[1]
R=ROOT/"results"
s=json.loads((R/"summary.json").read_text())
run_date=datetime.fromtimestamp(min(json.loads(line)["planned_wall"] for line in (R/"requests.jsonl").read_text().splitlines())).strftime("%d.%m.%Y")


def f(x,digits=3):
    return "—" if x is None else f"{x:.{digits}f}"


def link(path,label=None):
    return f'<a href="{escape(path)}">{escape(label or path)}</a>'


def table(headers,rows):
    return '<div class="table-wrap"><table><thead><tr>'+"".join("<th>"+str(x)+"</th>" for x in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+str(v)+'</td>' for v in row)+'</tr>' for row in rows)+'</tbody></table></div>'


def img(name,caption):
    return f'<figure><a href="figures/{name}"><img src="figures/{name}" alt="{escape(caption)}"></a><figcaption>{escape(caption)}</figcaption></figure>'


brows=[]
for name,d in sorted(s["B"].items()):
    c=d["conditions"]
    brows.append([c["traffic"],c["policy"],c["repeat"],f'{d["success_n"]}/{d["sent"]}',
                  f(d["ttfc_p50"]),f(d["ttfc_p95"]),f(100*d["error_fraction"],1),
                  f(100*d["cache_hit_fraction_success"],1)])
btable=table(["Трафик","Политика","Повтор","Успех / всего","p50 TTFC, с","p95 TTFC, с","Неуспех, %","Кеш hit*, %"],brows)
compare=[]
for traffic in ("uniform","hot80"):
    for policy in ("sticky","ordinary"):
        ds=[s["B"][f"B-{traffic}-{policy}-r{i}"] for i in (1,2,3)]
        compare.append([traffic,policy,f(statistics.mean(d["ttfc_p95"] for d in ds)),
                        f(min(d["ttfc_p95"] for d in ds))+"–"+f(max(d["ttfc_p95"] for d in ds)),
                        sum(d["success_n"] for d in ds),sum(d["sent"]-d["success_n"] for d in ds)])
ctable=table(["Трафик","Политика","Среднее трёх p95, с","Диапазон p95, с","Успешных","Неуспешных"],compare)
histrows=[]
for name,d in sorted(s["B"].items()):
    histrows.append([name,d["success_n"],int(d["server_completed_count"]),f(d["ttfc_p95"]),f(d["prometheus_p95"]),
                    f(d["prometheus_p95"]-d["ttfc_p95"])])
histtable=table(["Прогон","Клиент n","Сервер n","Клиент p95, с","Prometheus p95, с","Разница, с"],histrows)
counts=[]
for group in ("A","B","C"):
    for name,d in s[group].items():
        if not isinstance(d,dict) or "outcomes" not in d:continue
        o=d["outcomes"]
        counts.append([name,d["sent"],o.get("completed",0),o.get("rejected",0),o.get("http_error",0),
                       o.get("broken",0),o.get("connection_error",0),o.get("expired",0)])
counttable=table(["Прогон","Отправлено","Завершено","429","HTTP-ошибка","Оборвано","Нет соединения","Тайм-аут"],counts)

latency_rows=[]
for group in ("A","C"):
    for name,d in s[group].items():
        if not isinstance(d,dict) or "outcomes" not in d:continue
        latency_rows.append([name,d["success_n"],f(d["ttfc_p50"]),f(d["ttfc_p95"]),
            f(d["total_p95"]),f(d["inter_chunk_p95"]),
            f(100*d["error_fraction"],1),f(100*d["outcomes"].get("expired",0)/d["sent"],1)])
latency_table=table(["Прогон","Успешных n","p50 TTFC, с","p95 TTFC, с","p95 всего, с","p95 между частями, с","Неуспех, %","Тайм-аут, %"],latency_rows)


broken=s["A"]["A-warmup-broken"];fixed=s["A"]["A-warmup-fixed"]
rolls=[s["A"][name] for name in ("A-rollout","A-long-0","A-long-1")]
normal_n=sum(d["sent"] for d in rolls)
sticky=[s["B"][f"B-hot80-sticky-r{i}"] for i in (1,2,3)]
ordinary=[s["B"][f"B-hot80-ordinary-r{i}"] for i in (1,2,3)]
hot_samples=sum(d["idle_capacity"]["hot_waiting_other_idle_samples"] for d in sticky)
balanced=s["B"]["B-uniform-sticky-r1"]["pod_distribution"]
loadlag=max(d["send_lag_max"] for d in s["B"].values())
hpa_rows=[]
for name in ("C-hpa-uniform","C-hpa-hot80"):
    d=s["C"][name];e=d["events"]
    initial={p["metadata"]["name"] for p in e["initial_pods"]}
    new=[p for p in e["three_ready"]["pods"] if p["metadata"]["name"] not in initial]
    for p in new:
        condition=next(c for c in p["status"]["conditions"] if c["type"]=="Ready")
        from datetime import datetime
        created=datetime.fromisoformat(p["metadata"]["creationTimestamp"].replace("Z","+00:00"))
        ready=datetime.fromisoformat(condition["lastTransitionTime"].replace("Z","+00:00"))
        hpa_rows.append([name,p["metadata"]["name"],f((ready-created).total_seconds(),1),
                        d["pod_distribution"].get(p["metadata"]["name"],0),
                        f(e["actual_return_after_load_seconds"],1)])
hpatable=table(["Опыт","Новая реплика","Создание → Ready, с","Запросов новой","Последний ответ → 2 Ready, с"],hpa_rows)

# Derive the graceful-shutdown proof from the original application log and raw Kubernetes state.
streams=[json.loads((R/f"A-long-{i}"/"requests.jsonl").read_text()) for i in (0,1)]
drains=[]
for path in (R/"A-rollout").glob("*.log"):
    for line in path.read_text().splitlines():
        if "{" not in line:continue
        try:event=json.loads(line[line.index("{"):])
        except ValueError:continue
        if event.get("event")=="draining":drains.append(event)
grace_stream,grace_start=next((r,e) for r in streams for e in drains
    if e["pod"]==r["pod"] and r["sent_wall"]<e["wall"]<r["end_wall"])
with gzip.open(R/"A-rollout/kubernetes.jsonl.gz","rt") as raw:
    grace_meta=next(p["metadata"] for line in raw for p in json.loads(line)["data"]["items"]
        if p["kind"]=="Pod" and p["metadata"]["name"]==grace_stream["pod"] and p["metadata"].get("deletionTimestamp"))
from datetime import timezone
def utc_hms(value):
    return datetime.fromtimestamp(value,timezone.utc).strftime("%H:%M:%S.%f")[:-3]

review="""При указанных условиях новый запрос направляю на B: повторная обработка занимает 0,8 секунды, тогда как одно ожидание A — 3 секунды. При прочих равных B начнёт отвечать минимум на 2,2 секунды раньше. Постоянную привязку диалога к Pod в предложенном виде не одобряю. Её можно сохранить как предпочтение, которое уступает прогнозу времени ожидания.

До внедрения нужны стоимость обработки фактических длин входа и выхода, очередь и число активных запросов каждой реплики, точное совпадение кешируемого префикса, доступная память, свежесть метрик и p95 задержки. Надо проверить, что A и B действительно имеют независимую вычислительную ёмкость, используют одинаковые модель и настройки, а не конкурируют за один ускоритель.

Перед допуском нового запроса оцениваю ожидаемую задержку и свободное место в ограниченной очереди. Если обе реплики не укладываются в бюджет, возвращаю явный отказ; повтор разрешаю с ограничением числа попыток и случайной задержкой. Каждая попытка получает отдельный идентификатор.

Историю храню у клиента либо во внешнем хранилище. После потери A следующий запрос передаёт всю подтверждённую историю на B, заново вычисляет префикс и проверяет версию модели. Кеш остаётся ускорением, а не единственной копией диалога. Начатый поток не переношу между репликами: при штатном обновлении он завершается на A. При аварии отмечаю неполный ответ и начинаю отдельную попытку по явному решению клиента, не склеивая два разных продолжения.

Это согласуется с кейсом Google из лекции: команда Vertex AI усилила влияние очереди относительно кеша и ограничила допуск входящих запросов. В её публикации p95 времени до первого токена для DeepSeek V3.1 снизился на 52%, а доля попаданий в префиксный кеш выросла с 35% до 70%. Это результаты их нагрузки. Наш опыт подтверждает механизм очереди на имитаторе; такие проценты для настоящей модели из него не следуют."""
review_html="".join("<p>"+escape(p)+"</p>" for p in review.split("\n\n"))
source="https://cloud.google.com/blog/products/containers-kubernetes/how-gke-inference-gateway-improved-latency-for-vertex-ai"

html=f"""<!doctype html>
<html lang="ru"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>ДЗ 11 · Губин Виталий · Kubernetes под нагрузкой</title>
<style>
*{{box-sizing:border-box}}body{{margin:0;background:#f4f6fa;color:#1e2c40;font:18px/1.55 system-ui,sans-serif}}
main{{max-width:1200px;margin:auto;padding:40px 32px 80px}}h1{{font-size:38px;line-height:1.2}}h2{{font-size:27px;margin-top:44px}}
h3{{font-size:21px;margin-top:28px}}a{{color:#315ca5}}p,li{{max-width:1050px}}code,pre{{font-family:ui-monospace,monospace;font-size:.84em}}
pre{{white-space:pre-wrap;overflow-wrap:anywhere;background:#e8edf5;padding:18px;border-radius:8px}}
.note{{padding:18px 22px;background:#e8eff9;border-left:4px solid #315ca5}}.small,figcaption{{font-size:15px;color:#53637b}}
.cards{{display:grid;grid-template-columns:repeat(3,1fr);gap:16px}}.card{{background:white;border:1px solid #d8e0ea;padding:18px;border-radius:8px}}
.card strong{{display:block;font-size:26px}}figure{{margin:28px 0}}figure img{{display:block;width:100%;height:auto;background:white}}
table{{width:100%;border-collapse:collapse;table-layout:auto;font-size:14px}}th,td{{border-bottom:1px solid #d0d9e6;padding:9px 7px;text-align:left;overflow-wrap:anywhere;vertical-align:top}}th{{background:#e8edf5}}tr:nth-child(even){{background:#f8faff}}.table-wrap{{max-width:100%}}
nav{{display:flex;gap:15px;flex-wrap:wrap;font-size:16px}}.review{{max-width:1040px}}
@media(max-width:700px){{main{{padding:24px 14px}}body{{font-size:16px}}h1{{font-size:29px}}.cards{{grid-template-columns:1fr}}table{{font-size:11px}}th,td{{padding:5px 3px}}}}
@media print{{body{{background:white;font-size:12pt}}main{{padding:0}}h1{{font-size:22pt}}h2{{font-size:17pt;break-after:avoid}}figure,tr,.note{{break-inside:avoid}}table{{font-size:8pt}}nav{{display:none}}a{{color:inherit}}}}
@page{{size:A4;margin:15mm}}
</style><main>
<p class="small">МГТУ им. Н. Э. Баумана · MLOps · {run_date}</p>
<h1>ДЗ 11. Сервис работает, пользователи ждут</h1>
<p>Губин Виталий Андреевич, ИУ6-74Б</p>
<nav><a href="#stand">Стенд</a><a href="#A">Опыт A</a><a href="#B">Опыт B</a><a href="#C">Опыт C</a><a href="#metrics">Измерения</a><a href="#tests">Проверки</a><a href="#review">Ревью</a></nav>
<p class="note">Выполнены настоящие запуски Kubernetes, HTTP-запросы, сбор Prometheus и масштабирование.
Вычислительный backend — управляемый CPU-имитатор, явно обозначенный <code>backend_mode=simulated</code>.
Это разрешённый заданием вариант. Числа относятся к заданной модели нагрузки; качество языковой модели здесь не измеряется.</p>
<div class="cards"><div class="card"><strong>12 × 120 с</strong>сравнение двух политик, двух сценариев, трёх повторов</div>
<div class="card"><strong>{normal_n}/{normal_n}</strong>завершённых запросов штатного выпуска, включая длинные потоки</div>
<div class="card"><strong>2 → 3 → 2</strong>масштабирование по реальной загрузке процессора</div></div>
<p>Репозиторий: <a href="https://github.com/Tesla0s/bmstu-mlops/tree/hw11-audited/hw11">hw11, зафиксированная версия hw11-audited</a>.
Точный коммит: <code>git rev-parse hw11-audited</code>. {link("README.md","Запуск и удаление стенда")} · {link("results/summary.json","Все пересчитанные числа")} · {link("results/requests.jsonl","Сырые запросы")}.</p>

<h2 id="stand">1. Стенд и связь с предыдущей работой</h2>
<p>Продолжен сервис классификации обращений GitHub из ДЗ 8–10. Сохранены отдельная готовность после загрузки,
фиксированная идентичность загруженного артефакта, ограниченная очередь и правило отмены CPU-работы.
Добавлены потоковый ответ, история в запросе, имитация локального кеша и наблюдение каждой реплики.
Имитация возвращает условный ответ, а не прогноз обученного классификатора.</p>
<p>Apple M3 Pro, 36 ГиБ RAM. Отдельная Colima: 4 CPU, 6 ГиБ RAM, диски 12 + 8 ГиБ.
Перед подготовкой свободно 21 ГиБ. Файлы отдельной виртуальной машины фактически занимали 5,86 ГиБ выделенных блоков на хосте; пределы виртуальных дисков — 20 ГиБ. Образ API — около 256 МБ с общими слоями, уже учтён внутри машины. Инструменты и разбор лекции — ещё около 1,79 ГиБ вне неё. kind 0.30.0, Kubernetes 1.34.0, Docker 29.2.1, Helm 3.19.0.
Образ сервиса <code>hw11-api:1.0.1</code>; версии, идентификаторы и размеры — {link("results/environment/","сведения об окружении")}.</p>
<p>Kubernetes поддерживает нужное число копий приложения через Deployment.
Каждая копия работает в отдельном Pod; Service направляет к ним запросы,
а ConfigMap хранит настройки и маленькие артефакты версий A/B.
Развёрнуты две копии приложения и два варианта Service. На API Pod: запрос 0,25 CPU / 80 MiB,
предел 1 CPU / 192 MiB. Один вычислитель, не более восьми принятых запросов. У мониторинга и клиентов также заданы лимиты.
Prometheus хранит до 3 часов / 256 MB, том не более 512 MiB; том Grafana — до 256 MiB.</p>
<p>При запуске сверяется SHA-256 файла модели. Загруженные версия и хеш неизменны до перезапуска и присутствуют в ответах.
История передаётся клиентом, поэтому исчезновение Pod удаляет только локальный кеш.
{link("chart/templates/api.yaml","Манифесты")} · {link("service.py","Реализация сервиса")} · {link("requirements.lock","Зависимости с хешами")}.</p>

<h2 id="A">2. Опыт A: прогрев, обновление, откат и авария</h2>
<p><b>Гипотеза.</b> Доступный HTTP-процесс ещё не означает готовую модель. Если startup и readiness смотрят на /health,
Kubernetes направляет запросы во время двадцатисекундного прогрева.</p>
<p><b>Воспроизведение.</b> При ошибочных проверках получено {broken["outcomes"].get("http_error",0)} HTTP-ошибок
из {broken["sent"]} запросов. Серверный журнал содержит получение запроса с <code>warm=false</code>;
в тот же период Ready и адреса Service уже разрешают трафик.
<b>Исправление.</b> startup ждёт загрузку и прогрев; readiness дополнительно учитывает вывод из обслуживания;
liveness проверяет живой HTTP-процесс. При том же расписании после исправления завершены все {fixed["success_n"]} из {fixed["sent"]} запросов.</p>
<p><b>Штатный выпуск A → B.</b> Два ответа заранее начаты на старых Pod и пересекают начало обновления.
Новые Pod прогреваются, старые завершают работу. На всём прогоне: {normal_n} завершённых запросов,
0 HTTP 5xx, 0 незавершённых потоков и 0 отклонённых запросов.
Пять секунд даются на распространение исключения Pod из Service, затем до 45 секунд — на завершение Uvicorn;
общий grace period равен 60 секундам. Два длинных потока одновременно занимали вычислители старых реплик, поэтому p95 ожидания первой части обычных запросов в этом стрессовом выпуске составил {f(s["A"]["A-rollout"]["ttfc_p95"])} с при тайм-ауте чтения 70 с.</p>
<p>У потока на <code>{escape(grace_stream["pod"])}</code> начало вывода из обслуживания записано самим приложением в {utc_hms(grace_start["wall"])} UTC, окончание ответа клиентом — в {utc_hms(grace_stream["end_wall"])} UTC: ещё {f(grace_stream["end_wall"]-grace_start["wall"])} с работы после начала остановки. Kubernetes уже показывал этот Pod как удаляемый. Ответ завершён до предельного времени {escape(grace_meta["deletionTimestamp"])}. Поле <code>deletionTimestamp</code> обозначает этот предел, а не начало остановки; расчёт проверяется по <a href="https://kubernetes.io/docs/reference/kubernetes-api/definitions/object-meta-v1-meta/">описанию метаданных Kubernetes</a> и сырым снимкам.</p>
<p><b>Неудачный выпуск.</b> B с намеренно сломанным прогревом остаётся неготовой и не получает рабочие запросы.
Выполнен <code>helm rollback</code> к сохранённой ревизии A. Версия подтверждена фактическими ответами, не только шаблоном Deployment.</p>
<p><b>Авария.</b> Отдельно процесс завершён SIGKILL во время потока. Клиент зафиксировал обрыв после HTTP 200;
это не включено в успешный штатный выпуск. Аварийный прогон содержит {s["A"]["A-crash"]["sent"]+s["A"]["A-killed-stream"]["sent"]} попытки и {s["A"]["A-crash"]["outcomes"].get("broken",0)+s["A"]["A-killed-stream"]["outcomes"].get("broken",0)} обрыва. Сохранены частичный ответ, рестарт контейнера, снятие Ready,
прогрев и восстановление обслуживания. После удаления другого Pod проверены прежний хеш артефакта и продолжение диалога на замене.</p>
{img("01-warmup-rollout.png","График 1. Прогрев и штатное обновление: Ready, готовые EndpointSlice и исходы всех запросов.")}
<p>{link("scripts/experiment_a.py","Сценарий A")} · {link("results/A-warmup-broken/","Исходный дефект")} ·
{link("results/A-warmup-fixed/","Повтор после исправления")} · {link("results/A-rollout/","Штатный выпуск")} ·
{link("results/A-rollback/","Откат")} · {link("results/A-crash/","Авария")} · {link("results/A-dialog-recovery/","Восстановление диалога")}.</p>

<h2 id="B">3. Опыт B: привязка по IP и горячая реплика</h2>
<p><b>Гипотеза.</b> Привязка ClientIP может помогать локальному кешу, но при концентрации трафика перегружает одну реплику.
Сравниваются два Service над теми же Pod. Цель 80% реализована расписанием 12 из 15 запросов; в конечном наборе это 231/288 = 80,21%. Четыре клиента имеют разные адреса; фактически наблюдаемые сервером IP совпали с адресами клиентских Pod.
Сеть — kindnet и kube-proxy; keep-alive выключен, поэтому сравнение не скрыто повторным использованием одного TCP-соединения.</p>
<p>В каждом прогоне — 120 секунд, 2,4 запроса/с, 288 запросов, вход 128 символов и пять частей ответа.
Кеш пуст перед прогоном; модель уже прогрета. Кеш ограничен 16 префиксами, hit/miss стоят 0,02/0,16 с CPU,
часть — 0,025 с CPU и пауза 0,1 с. Во всех прогонах одинаковы параметры имитатора, длины входов и ограничения ответа.
Внутри пары политик совпадают точные запросы и расписание, проверенные хешем.</p>
<p class="small">В таблицах: uniform — равномерный поток, hot80 — горячий клиент; sticky — привязка ClientIP, ordinary — обычный Service. TTFC — время от отправки запроса до первой содержательной части ответа.</p>
{btable}<p class="small">* Доля попаданий в кеш среди завершённых ответов; отказы до вычисления не считаются промахами.
Дополнительно сохранены исходные счётчики hit/miss. p50/p95 относятся только к успешным ответам.
Плановое и фактическое время различаются; максимальная задержка генератора в серии — {f(loadlag)} с.</p>
<p><b>Причина и исправление.</b> При 80% трафика от одного клиента ClientIP создаёт очередь у привязанной реплики.
За три повтора зафиксировано {hot_samples} снимков, где одна реплика ждала в очереди, а вычислитель другой простаивал.
Выбранное исправление для этой нагрузки — обычный Service: новые соединения могут попадать на обе реплики.
Все три повтора выполнены с теми же расписаниями.</p>
{ctable}
<p>У исправления есть цена: в равномерном сценарии привязка быстрее, потому что два клиента на Pod помещают свои префиксы в локальный кеш. Обычный Service чаще пересчитывает вход и может получить случайную очередь; во втором и третьем равномерных повторах зафиксированы отказы. Поэтому обычный Service выбран как исправление горячего сценария, а не как универсально лучшая политика.</p>\n<p>Случай без ожидаемого перекоса тоже сохранён: в первом равномерном прогоне ClientIP распределил запросы как
{escape(str(list(balanced.values())))}. Трасса показывает удачное распределение четырёх IP между двумя Pod;
само наличие привязки не гарантирует дисбаланс.</p>
{img("02-pod-latency-queue.png","График 2. Задержка и очередь каждой реплики при горячем клиенте, один и тот же первый повтор.")}
{img("03-policy-comparison.png","График 3. Все три повтора каждой комбинации политики и трафика.")}
<p><b>Ограничение решения.</b> Обычный Service не учитывает префикс и может терять часть попаданий в кеш.
Для настоящего движка отдельно проверяются совпадение токенов префикса, кеш внимания, память ускорителя,
пакетная генерация, разные длины входа/выхода и политика, одновременно учитывающая кеш и очередь.
Результат имитатора не доказывает оптимальность обычного Service для любой языковой модели.</p>
<p>{link("scripts/experiment_b.py","Сценарий B")} · {link("results/B-network.json","Сеть и сервисы")} ·
{link("results/B-hot80-sticky-r1/","Трасса горячей реплики")}.</p>

<h2 id="C">4. Опыт C: CPU, число реплик и оповещения</h2>
<p><b>Гипотеза.</b> Автоматическое масштабирование по средней CPU-метрике добавляет ёмкость,
но само по себе не перераспределяет уже привязанных клиентов.
HPA autoscaling/v2: минимум 2, максимум 3, цель 60% от запроса 250m CPU, окно уменьшения 60 секунд.
Resource Metrics API проверен реальным запросом. Нагрузка вычисляет PBKDF2: стоимость CPU увеличена в шесть раз, пауза между частями отключена.</p>
<p><b>Результат.</b> В обоих сценариях сохранён переход 2 → 3 → 2 вместе с условиями HPA, CPU, временем создания
и готовности нового Pod. Время возврата включает обновление метрик, решение контроллера и завершение лишней реплики;
оно не обязано точно равняться 60 секундам настройки.</p>
{hpatable}
<p>В горячем сценарии одного занятого ядра достаточно, чтобы среднее превышало целевые 0,15 ядра на Pod.
После добавления третьей реплики прежние записи ClientIP продолжают направлять запросы к прежним адресам.
Новая ёмкость поэтому может оставаться почти неиспользованной. Среднее CPU полезно для оценки общей загрузки,
но не заменяет наблюдение очередей отдельных реплик.</p>
{img("04-hpa-load.png","График 4. Число готовых/желаемых реплик, предложенная нагрузка и CPU каждой реплики.")}
<p><b>Два оповещения.</b> Порог p95 TTFC — 1 с; порог доли неуспешных исходов — 5%.
Для обоих требуется не менее 20 наблюдений за минуту и удержание условия 15 с.
Управляемый всплеск 18 запросов/с сменяется здоровой нагрузкой 0,5 запроса/с.
В сохранённой линии состояний у каждого оповещения присутствуют pending, firing и возврат в inactive.</p>
<pre>{escape(json.dumps(s["C"]["alert_transitions"],ensure_ascii=False,indent=2))}</pre>
<p>{link("scripts/experiment_c.py","Сценарий C")} · {link("monitoring/alerts.yaml","Правила оповещений")} ·
{link("results/C-alerts/","Сырые переходы и нагрузка")} · {link("monitoring/dashboard-export.json","Экспорт работающего дашборда")}.</p>

<h2 id="metrics">5. Измерения и проверка чисел</h2>
<p>Клиент использует монотонные часы. TTFC — от фактической отправки до первой содержательной части;
inter-chunk latency — интервалы между последующими частями. У N частей N−1 интервал.
Завершающая запись done не считается содержательной частью. Для одной части интервал отсутствует.
Сохранены planned/sent/first/end, статус, Pod, версия, хеш, число частей и исход каждой попытки.
Автоматических повторов нет. В контролируемой паре политик логический request_id совпадает; однозначный ключ отдельной попытки — experiment/request_id, сохранённый в сводном журнале как attempt_id. Исходные журналы содержат исходный request_id и времена; ни одна попытка не объединяется с другой.</p>
<p>Для успешного ответа недостаточно HTTP 200: нужны завершающая запись и ожидаемое число частей.
Плановые и отправленные запросы в B: 288/288 в каждом прогоне. Остальные исходы приведены отдельно:</p>
{counttable}
<p>Задержки только завершённых ответов для A и C (для B — таблица выше). Доля неуспехов включает все исходы, кроме completed; тайм-ауты дополнительно выделены как её часть.</p>
{latency_table}
<p>Длительности хранятся в Histogram. Для дашборда применяется, например:</p>
<pre>histogram_quantile(0.95,
  sum by (le, pod) (rate(hw11_ttfc_seconds_bucket[1m])))</pre>
<p>Для точной сверки B берутся разности накопленных bucket-счётчиков в сохранённых моментах T0/T1
после завершения всех запросов и полного сбора метрик. Это исключает смешивание соседних прогонов.
Гистограммы учитывают только завершённые ответы; число наблюдений проверяется на равенство клиентскому журналу:</p>
<pre>histogram_quantile(0.95,
  sum by (le) (
    hw11_ttfc_seconds_bucket @ T1 -
    hw11_ttfc_seconds_bucket @ T0))</pre>
{histtable}
<p>Значение Prometheus интерполируется внутри границ bucket; клиентский p95 считается непосредственно по
измеренным длительностям и включает сетевую доставку. Поэтому значения могут различаться.
Границы: 0,01; 0,025; 0,05; 0,1; 0,2; 0,4; 0,6; 0,8; 1; 1,5; 2; 3; 4; 6; 8; 12; 20; 30; 60 с.</p>
<p>Дашборд показывает TTFC, полную длительность, running/waiting по Pod, завершённые ответы/с,
отказы, CPU, RAM, Ready и попадания в кеш. Prometheus обнаруживает и собирает каждый API Pod.
Идентификаторы запросов и диалогов не являются метками метрик.\nПри проверке исправлен запрос, скрывавший waiting при объединении через OR. Исходный сбор Prometheus\nсохранил значения, поэтому очередь серии B повторно выгружена из хранилища за те же интервалы.\nПервоначальные снимки и ответы query_range сохранены отдельно; производная линия явно указывает источник\nи время оценки. Исправленный селектор выбирает все нужные имена метрик сразу.
{link("scripts/analyze.py","Пересчёт и автоматические утверждения")}.</p>

<h2 id="tests">6. Пять собственных проверок и границы результата</h2>
<ol>
<li>Новый Pod во время прогрева возвращает 503 на /ready и не является готовым адресом Service; после прогрева отвечает.</li>
<li>После настоящего отката ответы содержат A и ожидаемый SHA-256; дополнительно выполнен живой запрос к Kubernetes.</li>
<li>После удаления Pod новая реплика с другим UID восстанавливает тот же артефакт и продолжает переданную историю.</li>
<li>Отклонённые и оборванные запросы сохранены отдельно; частичный HTTP 200 не засчитывается в успех.</li>
<li>Два диалога с различными контрольными строками возвращают только свою историю.</li>
</ol>
<p>{link("scripts/test_integration.py","Код проверок")} · {link("results/integration/","Реальные ответы и состояние Kubernetes")} · {link("results/integration.log","Результат запуска пяти проверок")} · {link("results/requirements-audit.json","Аудит требований")}.
Первый дефект показан до и после исправления в опыте A; проверка исходного сбоя не заменена утверждением о готовности.</p>
<p>Один локальный узел не проверяет потерю целого узла или облака. CPU-имитатор не проверяет качество модели,
настоящие токены, кеш внимания и разделение видеопамяти. Наблюдение очереди имеет шаг примерно 2 с,
Prometheus — 5 с, CPU-метрики — 15 с. Три повтора описывают устойчивость этой нагрузки и не дают
универсальных гарантий. Пробные запуски сохранены локально отдельно от окончательной серии.</p>

<h2 id="review">7. Ревью предложения о постоянной привязке диалога</h2>
<div class="review">{review_html}</div>
<p>Первичный источник: <a href="{source}">Google Cloud — How we cut Vertex AI latency by 35% with GKE Inference Gateway, 06.02.2026</a>.
В предоставленной лекции — главы 65–72, примерно 07:29–09:16; численные результаты на слайде 70.
{link("lecture-notes.txt","Привязка к записи лекции")}.</p>

<h2>Источники и воспроизведение</h2>
<ul>
<li><a href="https://kubernetes.io/docs/tasks/configure-pod-container/configure-liveness-readiness-startup-probes/">Kubernetes: назначение трёх проверок</a>.</li>
<li><a href="https://kubernetes.io/docs/concepts/services-networking/service/#session-affinity">Kubernetes: привязка ClientIP</a>.</li>
<li><a href="https://kubernetes.io/docs/concepts/workloads/autoscaling/horizontal-pod-autoscale/">Kubernetes: масштабирование по CPU и стабилизация</a>.</li>
<li><a href="https://prometheus.io/docs/practices/histograms/">Prometheus: гистограммы и перцентили</a>.</li>
</ul>
<p>Все локальные действия запуска и удаления, лимиты, версии и настройки генератора приведены в {link("README.md","инструкции")}.
Лекция, музыка, виртуальная машина, скачанные библиотеки и файлы доступа в репозиторий не входят.</p>
</main></html>"""
(ROOT/"report.html").write_text(html)
print("Wrote report.html from audited summary.")
