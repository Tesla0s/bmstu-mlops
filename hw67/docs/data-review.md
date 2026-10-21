# Проверка набора вопросов

60 обращений из val версии hw3-v2. По 20 на класс, по 30 на репозиторий.
Тексты не переписаны; expected сверены по содержанию до генерации ответов.
Разбор выполнен с помощью ИИ, без заявления о независимой экспертной разметке.
Полные тексты, исходные метки и ссылки доступны в DVC-архиве.

Пограничные случаи: вопрос pandas#51062 содержит сообщение о несовместимости,
но сформулирован как запрос помощи с кодом; сохранена исходная метка question.
vscode#282117 допускает влияние сохранённых настроек; метка bug отражает
заявленное неожиданное поведение, а не установленную причину дефекта.
Проверяем намерение обращения, не подтверждаем реальность каждого программного дефекта.

| Обращение | Метка | Основание |
|---|---|---|
| [microsoft/vscode#232527](https://github.com/microsoft/vscode/issues/232527) | question | Вопрос о поиске учебной таблицы в своей среде. |
| [microsoft/vscode#233467](https://github.com/microsoft/vscode/issues/233467) | question | Просьба объяснить и вернуть прежний вид приглашения терминала. |
| [microsoft/vscode#233541](https://github.com/microsoft/vscode/issues/233541) | question | Просьба помочь с конфигурацией среды восстановления. |
| [microsoft/vscode#234324](https://github.com/microsoft/vscode/issues/234324) | question | Запрос помощи с переменными окружения при удалённом подключении. |
| [microsoft/vscode#236985](https://github.com/microsoft/vscode/issues/236985) | question | Вопрос о запуске sqlite3 в учебной среде. |
| [microsoft/vscode#237223](https://github.com/microsoft/vscode/issues/237223) | question | Запрос инструкции по изменению сочетаний клавиш. |
| [microsoft/vscode#241447](https://github.com/microsoft/vscode/issues/241447) | question | Просьба настроить открытие терминала, а не добавить новый механизм. |
| [microsoft/vscode#241531](https://github.com/microsoft/vscode/issues/241531) | question | Просьба помочь восстановить удалённые пользовательские файлы. |
| [microsoft/vscode#243627](https://github.com/microsoft/vscode/issues/243627) | question | Вопрос о способе удаления настроек удалённых расширений. |
| [microsoft/vscode#243633](https://github.com/microsoft/vscode/issues/243633) | question | Вопрос о назначении файлов локальной истории. |
| [microsoft/vscode#276636](https://github.com/microsoft/vscode/issues/276636) | feature_request | Предлагается поддержка инструкций организации. |
| [microsoft/vscode#276658](https://github.com/microsoft/vscode/issues/276658) | feature_request | Предлагается многострочная строка состояния. |
| [microsoft/vscode#276955](https://github.com/microsoft/vscode/issues/276955) | feature_request | Предлагается новое правило подтверждения URL. |
| [microsoft/vscode#277055](https://github.com/microsoft/vscode/issues/277055) | feature_request | Предлагается выбор пользовательской модели для сообщения коммита. |
| [microsoft/vscode#277286](https://github.com/microsoft/vscode/issues/277286) | feature_request | Предлагается изолированное выполнение команд агента. |
| [microsoft/vscode#277511](https://github.com/microsoft/vscode/issues/277511) | feature_request | Предлагается механизм прочитанности сессий. |
| [microsoft/vscode#277799](https://github.com/microsoft/vscode/issues/277799) | feature_request | Предлагается история удалённых подключений. |
| [microsoft/vscode#277905](https://github.com/microsoft/vscode/issues/277905) | feature_request | Предлагается новый пункт Save All в контекстном меню. |
| [microsoft/vscode#278404](https://github.com/microsoft/vscode/issues/278404) | feature_request | Предлагается показывать ветку при переключении окон. |
| [microsoft/vscode#280206](https://github.com/microsoft/vscode/issues/280206) | feature_request | Предлагается новый вариант в выборе версии TypeScript. |
| [microsoft/vscode#281844](https://github.com/microsoft/vscode/issues/281844) | bug | Сообщается о несовпадении существующего поведения createDirectory и тестов. |
| [microsoft/vscode#281915](https://github.com/microsoft/vscode/issues/281915) | bug | Описаны шаги, при которых индикатор изменений не появляется. |
| [microsoft/vscode#281924](https://github.com/microsoft/vscode/issues/281924) | bug | У существующей облачной сессии не отображается кнопка изменений. |
| [microsoft/vscode#282035](https://github.com/microsoft/vscode/issues/282035) | bug | Закрытая сессия не освобождается: описана проверка точки останова. |
| [microsoft/vscode#282081](https://github.com/microsoft/vscode/issues/282081) | bug | Команда проверки обновлений не показывает существующий диалог. |
| [microsoft/vscode#282113](https://github.com/microsoft/vscode/issues/282113) | bug | Закрытие отдельного окна неожиданно открывает сессию в редакторе. |
| [microsoft/vscode#282117](https://github.com/microsoft/vscode/issues/282117) | bug | Сообщается о неожиданном выборе режима существующей команды; допускается влияние сохранённого состояния. |
| [microsoft/vscode#282118](https://github.com/microsoft/vscode/issues/282118) | bug | Одна секция результата неожиданно отображается дважды. |
| [microsoft/vscode#282812](https://github.com/microsoft/vscode/issues/282812) | bug | Описана регрессия выделения перенесённой строки после обновления. |
| [microsoft/vscode#283220](https://github.com/microsoft/vscode/issues/283220) | bug | Кнопка Pause не останавливает работу. |
| [pandas-dev/pandas#46088](https://github.com/pandas-dev/pandas/issues/46088) | question | Вопрос о поведении resample при отсутствующих наблюдениях. |
| [pandas-dev/pandas#46237](https://github.com/pandas-dev/pandas/issues/46237) | question | Вопрос о фильтрации данных по месяцу. |
| [pandas-dev/pandas#47484](https://github.com/pandas-dev/pandas/issues/47484) | question | Вопрос о настройке обязательности полей Parquet. |
| [pandas-dev/pandas#47593](https://github.com/pandas-dev/pandas/issues/47593) | question | Автор просит способ обработки нулевых символов и не утверждает дефект библиотеки. |
| [pandas-dev/pandas#47969](https://github.com/pandas-dev/pandas/issues/47969) | question | Запрос помощи с выборкой строк и типами после чтения CSV. |
| [pandas-dev/pandas#51062](https://github.com/pandas-dev/pandas/issues/51062) | question | Вопрос о совместимом способе использования SQLAlchemy 2; пограничный с bug, но цель текста — помощь с кодом. |
| [pandas-dev/pandas#51270](https://github.com/pandas-dev/pandas/issues/51270) | question | Просьба объяснить различие трёх классов группировки. |
| [pandas-dev/pandas#51633](https://github.com/pandas-dev/pandas/issues/51633) | question | Вопрос о способе использования асинхронного SQLAlchemy. |
| [pandas-dev/pandas#52622](https://github.com/pandas-dev/pandas/issues/52622) | question | Запрос объяснения границ интервалов qcut. |
| [pandas-dev/pandas#52663](https://github.com/pandas-dev/pandas/issues/52663) | question | Вопрос о преобразовании типов даты. |
| [pandas-dev/pandas#55645](https://github.com/pandas-dev/pandas/issues/55645) | feature_request | Запрос нового целевого варианта сборки musllinux/aarch64. |
| [pandas-dev/pandas#55659](https://github.com/pandas-dev/pandas/issues/55659) | feature_request | Просьба вернуть удалённый метод, то есть изменить предоставляемый интерфейс. |
| [pandas-dev/pandas#56575](https://github.com/pandas-dev/pandas/issues/56575) | feature_request | Запрос поддержки upsert для Oracle. |
| [pandas-dev/pandas#56917](https://github.com/pandas-dev/pandas/issues/56917) | feature_request | Запрос поддержки списка уровней вместо одного. |
| [pandas-dev/pandas#57268](https://github.com/pandas-dev/pandas/issues/57268) | feature_request | Предлагается улучшение сохранения исходного типа интервала. |
| [pandas-dev/pandas#57299](https://github.com/pandas-dev/pandas/issues/57299) | feature_request | Предлагаются новые методы range. |
| [pandas-dev/pandas#57731](https://github.com/pandas-dev/pandas/issues/57731) | feature_request | Предлагается новый метод чередования столбцов. |
| [pandas-dev/pandas#58108](https://github.com/pandas-dev/pandas/issues/58108) | feature_request | Предлагается дополнительный вариант отображения словаря. |
| [pandas-dev/pandas#58220](https://github.com/pandas-dev/pandas/issues/58220) | feature_request | Предлагается прямой выбор Arrow-типов в преобразовании дат. |
| [pandas-dev/pandas#59324](https://github.com/pandas-dev/pandas/issues/59324) | feature_request | Предлагается расширение оформления имён индекса в LaTeX. |
| [pandas-dev/pandas#60646](https://github.com/pandas-dev/pandas/issues/60646) | bug | Приведён пример потери наносекунд при вычислении max. |
| [pandas-dev/pandas#60647](https://github.com/pandas-dev/pandas/issues/60647) | bug | Текущий расчёт рабочих дней включает праздники вопреки ожидаемому календарю. |
| [pandas-dev/pandas#60740](https://github.com/pandas-dev/pandas/issues/60740) | bug | Новый вариант stack ошибается на ранее допустимом пустом списке. |
| [pandas-dev/pandas#60773](https://github.com/pandas-dev/pandas/issues/60773) | bug | Пример показывает неожиданные различия единиц даты. |
| [pandas-dev/pandas#60831](https://github.com/pandas-dev/pandas/issues/60831) | bug | Существующая группировка выдаёт лишнее предупреждение из-за индексирования. |
| [pandas-dev/pandas#60885](https://github.com/pandas-dev/pandas/issues/60885) | bug | Некорректная календарная неделя принимается вместо ошибки. |
| [pandas-dev/pandas#60980](https://github.com/pandas-dev/pandas/issues/60980) | bug | Пример показывает неверное значение года после unstack/reindex. |
| [pandas-dev/pandas#61182](https://github.com/pandas-dev/pandas/issues/61182) | bug | При отрицании булевы значения неожиданно становятся целыми. |
| [pandas-dev/pandas#61188](https://github.com/pandas-dev/pandas/issues/61188) | bug | Сравнение дат падает только при полностью пустой серии. |
| [pandas-dev/pandas#61309](https://github.com/pandas-dev/pandas/issues/61309) | bug | Параметр экранирования не применяется к имени столбцов. |

## Пересечения с обучением

Индекс содержит все 1315 исходных вопросов, а не только 32 строки короткого поиска.
Проверены совпадение SHA-256 JSONL с метаданными тензоров и идентичность всех id в исходном порядке.
Точных нормализованных совпадений и пересечений group_id с train не обнаружено.
Восемь val-примеров, использованных для выбора кандидата, тоже не входят в golden.
Оба исходных файла тензоров побайтово совпадают с ДЗ 4.
Дополнительно просмотрены 14 ближайших пар по символьным n-граммам; результаты ниже.

| Golden / ближайший train | Сходство | Разбор |
|---|---:|---|
| pandas-dev/pandas#47484 / pandas-dev/pandas#38827 | 0.860 | Сходство 0,86 вызвано длинными линиями дефисов таблиц; запись Parquet и отображение таблицы в терминале — разные задачи. |
| microsoft/vscode#282118 / microsoft/vscode#281362 | 0.544 | Общий контекст фоновых сессий и ссылка на обзор; дублирование результата отличается от неработающей кнопки. |
| microsoft/vscode#282113 / microsoft/vscode#281436 | 0.542 | Общая обзорная ссылка и отметки вложений; закрытие окна отличается от отображения прогресса. |
| microsoft/vscode#232527 / microsoft/vscode#212483 | 0.529 | Совпадают служебные строки Codespaces; поиск таблицы и открытие файла — разные действия. |
| microsoft/vscode#233541 / microsoft/vscode#212483 | 0.488 | Совпадают служебные строки Codespaces; режим восстановления и потерянный файл — разные вопросы. |
| pandas-dev/pandas#60773 / pandas-dev/pandas#60341 | 0.480 | Общий словарь datetime64; единицы Timestamp и неправильная интерпретация пикосекунд — разные дефекты. |
| microsoft/vscode#282117 / microsoft/vscode#281653 | 0.476 | Общая обзорная ссылка; выбор режима при создании и запись не в тот каталог — разные симптомы. |
| microsoft/vscode#234324 / microsoft/vscode#192946 | 0.466 | Общие слова tunnel/server; переменные окружения и несколько туннелей — разные запросы. |
| microsoft/vscode#233467 / microsoft/vscode#188255 | 0.464 | Совпадает браузерная информация; вид терминала и подключение MongoDB — разные задачи. |
| microsoft/vscode#276636 / microsoft/vscode#277642 | 0.450 | Общие ссылки документации инструкций; поддержка организации и валидация excludeAgent — разные обращения. |
| microsoft/vscode#277905 / microsoft/vscode#187989 | 0.448 | Общий Explorer и клавиши; добавление Save All и восстановление пропавшего Explorer — разные задачи. |
| microsoft/vscode#282035 / microsoft/vscode#281350 | 0.442 | Общий объект сессии; освобождение памяти и сохранение статистики после перезагрузки — разные дефекты. |
| microsoft/vscode#281924 / microsoft/vscode#281362 | 0.442 | Общий экран облачных сессий; отсутствующая кнопка и неработающая кнопка — разные наблюдения; контекст коррелирует. |
| pandas-dev/pandas#51270 / pandas-dev/pandas#43011 | 0.433 | Общий GroupBy; объяснение классов и предложение ускорения группировки — разные намерения. |

Это не доказательство отсутствия всех смысловых утечек. Общие подсистемы,
шаблоны GitHub и обзорные ссылки создают корреляции; исходная группировка — по семействам issue.
Golden взят из прежнего val и предназначен для повторной проверки регрессий,
а не для заявления независимой итоговой точности. Test-набор в этой работе не использовался.
