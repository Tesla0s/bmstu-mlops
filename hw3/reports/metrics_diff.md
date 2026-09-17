# Разница метрик v1 → v2

Реальный вывод DVC: сравнение исходной v1 с коммитом v2 `f044b295ccdc35de87c97c52274d2f7d726bda4a`.
Команда: `dvc metrics diff hw3-v1 HEAD --targets metrics/clean.json --md`.
Данные совпадают с окончательными метками; из текущей версии повторить через `make diff`.

| Path               | Metric                   | hw3-v1   | HEAD   | Change   |
|--------------------|--------------------------|----------|--------|----------|
| metrics/clean.json | dropped_exact_dup        | 0        | 1      | 1        |
| metrics/clean.json | dropped_length           | 61       | 72     | 11       |
| metrics/clean.json | groups                   | 1436     | 1876   | 440      |
| metrics/clean.json | pii_hits.email           | 17       | 18     | 1        |
| metrics/clean.json | pii_hits.home_path       | 118      | 236    | 118      |
| metrics/clean.json | pii_hits.mention         | 72       | 108    | 36       |
| metrics/clean.json | pii_hits.query_secret    | 0        | 1      | 1        |
| metrics/clean.json | pii_hits.signature_name  | 14       | 17     | 3        |
| metrics/clean.json | pii_hits.url_credentials | 1        | 2      | 1        |
| metrics/clean.json | pii_hits.windows_home    | 30       | 44     | 14       |
| metrics/clean.json | pii_rows_masked          | 105      | 141    | 36       |
| metrics/clean.json | rows_in                  | 1500     | 1953   | 453      |
| metrics/clean.json | rows_out                 | 1437     | 1878   | 441      |
| metrics/clean.json | user_chars.cv            | 0.787    | 0.82   | 0.033    |
| metrics/clean.json | user_chars.p10           | 228      | 216    | -12      |
| metrics/clean.json | user_chars.p50           | 694      | 636    | -58      |
| metrics/clean.json | user_chars.p90           | 1886     | 1842   | -44      |
| metrics/clean.json | user_chars.ratio_p90_p10 | 8.27     | 8.53   | 0.26     |
| metrics/clean.json | version                  | v1       | v2     | -        |
