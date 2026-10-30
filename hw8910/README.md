# ДЗ 8+9+10: выпуск модели, проверки и API

Губин Виталий Андреевич, ИУ6-74Б.

Продолжение классификации обращений GitHub: `bug / feature_request / question`.
Исправлены семь причин ошибок прототипа. Процесс на Prefect проверяет данные,
файлы модели и полный отчёт качества, затем атомарно меняет принятую версию.
FastAPI загружает именно эту версию и сообщает её фактический идентификатор.

[Подробный отчёт](REPORT.txt), [семь дефектов](defects.csv),
[исходный прототип](https://github.com/Tesla0s/bmstu-mlops/commit/686c0237195630aee0018ae4d4be326468832f82),
[исправленное решение](https://github.com/Tesla0s/bmstu-mlops/tree/hw8910-ready/hw8910).

## Проверка без модели

Нужны Python 3.12 и `uv`. Из каталога `hw8910`:

```bash
make install
make check
make check-extra lint
make check-negative
```

`make check` сам запускает временный локальный Prefect и завершает его после
13 исходных и 7 собственных тестов. HTTP-запросы и отдельные процессы настоящие;
только вычислитель заменён явно помеченным `STUB ONLY`. PyTorch и веса не нужны.

`make check-negative` отключает проверку допуска во временной копии, запускает
тот же набор и проверяет ненулевой код. Рабочая версия остаётся исправной.

- [20 зелёных тестов без PyTorch](reports/fast-green.log).
- [7 ошибок на исходном прототипе](reports/integration-original.log);
  [каждый запрос и subprocess](reports/integration-original).
- [Исправленная версия: подробные HTTP/процессные доказательства](reports/integration).
- [Намеренно сломанная копия](reports/fast-red.log), [контроль восстановления](reports/mutation-summary.json).
- [11 дополнительных отрицательных случаев](reports/negative-contracts.json).
- [37 прежних тестов ДЗ 6+7](reports/hw67-regression.log).
- Отпечатки тестов преподавателя неизменны: `09bd65e6b886` и `c42ac7788026`.

Для повторения красных тестов на исходнике:

```bash
mkdir -p .local/original
git archive 686c023 hw8910 | tar -x -C .local/original
HW8910_TARGET="$PWD/.local/original/hw8910" \
HW8910_EVIDENCE="$PWD/.local/original-evidence" \
.venv/bin/python -m unittest discover -s checks -v
# Ожидается exit=1, семь FAIL по исходным причинам.
```

## Реальная модель и исходные данные

Новый поиск параметров и обучение не выполнялись. Использован точный адаптер
ДЗ 6+7: training run `c21ef017fc9a475ebdb8ab9e279fd2fb`,
evaluation run `b11b03d74c2746bcb2c7ff4b070823e6`.

[Исходные ответы и отчёты](../hw67/reports/evaluation) уже в репозитории.
Данные оценки доступны через [релиз ДЗ 6+7](https://github.com/Tesla0s/bmstu-mlops/releases/tag/hw67-data-v1):

```bash
cd ../hw67
uv sync --frozen
make fetch-eval verify-reports
cd ../hw8910
make install-model
make fetch-model
make export-model
```

`make fetch-model` получает [отдельный артефакт адаптера](https://github.com/Tesla0s/bmstu-mlops/releases/tag/hw8910-model-v1):
22 447 554 байта, SHA-256 закреплён в [model-artifact.json](model-artifact.json).
После проверки архива проверяются все файлы. [Публичное скачивание и выпуск в чистом каталоге проверены](reports/public-artifact-check.json). В Git нет весов, среды Python и кешей.
База скачивается отдельно из `Qwen/Qwen3-0.6B` по точной ревизии
`c1899de289a04d12100db370d81485cdf75e47ca`; её реальные файлы тоже проверяются.

Вместо публичного артефакта можно экспортировать тот же run из своего MLflow:

```bash
MLFLOW_TRACKING_URI=http://127.0.0.1:5006 \
.venv/bin/python export_existing.py --workspace .local/real-from-mlflow
```

`export_existing.py` читает готовый результат обучения, сверяет происхождение и
создаёт `manifest-real.json`. Это не повторное обучение.
Существующий опубликованный рабочий каталог не перезаписывается.

Проверенный режим генерации: Apple M3 Pro, MPS (вычисления на графическом процессоре Apple), `bfloat16`, Transformers 4.57.6,
PEFT 0.17.1, PyTorch 2.8.0. Для CPU/CUDA или других условий нужна новая оценка;
сервер не подменяет устройство или настройки молча. Быстрые тесты переносимы.

## Процесс и сервер

В первом терминале:

```bash
make prefect
```

Во втором:

```bash
make release
make serve
```

Рабочий каталог по умолчанию — `.local/real`; можно передать
`WORKSPACE=/полный/путь`. Prefect: `http://127.0.0.1:4208`;
API: `http://127.0.0.1:8098`, описание запросов — `/docs`.

```bash
curl http://127.0.0.1:8098/health
curl http://127.0.0.1:8098/ready
curl -s http://127.0.0.1:8098/generate \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"Title: Editor crashes on save. Description: Ctrl+S closes the editor.","max_new_tokens":32}'
```

До загрузки и прогрева `/ready` и генерация возвращают 503.
Пустой ввод, превышение 8192 символов/2048 токенов с системным сообщением,
нецелый или отличный от 32 лимит получают 4xx до генерации.
Тело запроса ограничено 32768 байтами, вычислитель — одним запросом,
число одновременно обслуживаемых/ожидающих генераций — восемью.
Вход не обрезается. `/debug` — локальная диагностика числа вызовов.

В ответе есть текст, `request_id`, загруженный `model_id`, `runtime_sha256`,
число выходных токенов, ожидание в очереди, время генерации и полное серверное время.
Изменение `accepted.json` не переключает модель в памяти; нужен перезапуск.

```bash
make benchmark     # работающий прогретый API на 8098
make smoke         # сам запускает и перезапускает настоящую модель
make release-audit # реальный допуск: кеш, изменение данных, несовместимый режим
make stub-experiment # отдельный НОВЫЙ .local/stub-experiment; явный STUB ONLY
```

Нельзя одновременно запускать несколько писателей в одном рабочем каталоге.
Для нового прогона `stub-experiment` задайте новый каталог через его Python-команду.
`make smoke` использует отдельные случайные порты и завершает свои процессы.

## Доказательства

| Требование | Файл |
|---|---|
| Четыре задачи и кеш между процессами | [pipeline.py](pipeline.py), [журнал реальных попыток](reports/real-release/attempts.jsonl) |
| Источник весов и их суммы | [model-origin.json](reports/model-origin.json) |
| Связанный полный допуск | [release-report.json](reports/real-release/release-report.json) |
| Принятый manifest | [accepted.json](reports/real-release/accepted.json) |
| Отказ и прежний SHA-256 указателя | [real-release/evidence.json](reports/real-release/evidence.json) |
| Реальная готовность, 4xx без вычислений, ответы, два запуска | [real-api/evidence.json](reports/real-api/evidence.json) |
| A → B → A и восстановление события после аварии | [stub-experiment/evidence.json](reports/stub-experiment/evidence.json) |
| Все 16 попыток | [attempts.csv](reports/benchmark/attempts.csv), [JSONL](reports/benchmark/attempts.jsonl) |
| Сводка измерений | [summary.json](reports/benchmark/summary.json) |

Локальные журналы сохраняют реальные абсолютные пути и идентификаторы запусков.
Для воспроизведения используются команды выше, а не временные пути из журналов.

## Когда что запускать

| Изменение | Конкретные действия |
|---|---|
| Код процесса/сервера | `make check check-extra lint`. Если изменились генерация, шаблон или обработка входа: в `../hw67` повторить `make evaluate eval export-evidence`, затем собрать новый manifest и выпуск. Документация не требует обучения. |
| Данные | `make release-audit` проверяет инвалидацию. Для новых данных оценки в `../hw67`: `make evaluate eval export-evidence`; старые отчёты выпуск не откроют. Новые train-данные требуют решения о переобучении; только тогда `make sweep candidate evaluate eval export-evidence`. |
| Новый адаптер | В `../hw67`: `make candidate evaluate eval export-evidence`; сохранить прежний `reports/model-origin.json` отдельной версией, экспортировать новый run в новый рабочий каталог, проверить полную оценку, затем выполнить команды нового выпуска ниже. Старую историю и файлы оставить для отката. Если меняется режим API, его контракт и полная оценка обновляются вместе. |

Пример для нового адаптера после завершения его полной оценки в ДЗ 6+7:

```bash
mv reports/model-origin.json reports/model-origin.previous.json
MLFLOW_TRACKING_URI=http://127.0.0.1:5006 \
.venv/bin/python export_existing.py --workspace .local/next
.venv/bin/python lab.py --workspace .local/next run \
  --manifest manifest-real.json --report release-report.json
.venv/bin/python service.py --workspace .local/next --port 8099
```

Это проверочный запуск новой версии на другом порту. Старый сервер продолжает
обслуживать принятую версию; переключение выполняется после проверки нового
`/ready` и контрольного ответа. При изменении поведения генерации сначала
приведите генератор оценки ДЗ 6+7 к тем же условиям и выполните полную оценку.

Для смены кандидата проверка происхождения намеренно требует явно обновить
`reports/model-origin.json` после новой оценки. Нельзя просто подставить другие веса
под прежние контрольные суммы. Текущий публичный артефакт соответствует только версии 1.

`ci.example.yml` — пример автоматизации. Выполнены локальные проверки;
удалённый GitHub Actions не запускался, защита слияния не заявляется.

Итоговая проверка согласованности: `.venv/bin/python scripts/audit_submission.py`; [46 проверок](reports/submission-audit.json).

Видео будет записано отдельно после проверки содержательной части.
