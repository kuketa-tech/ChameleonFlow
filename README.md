# ChameleonFlow

ChameleonFlow это прототип адаптивной клиент-серверной системы, которая должна:

- измерять качество канала связи;
- выбирать транспорт под текущие условия;
- при необходимости морфить тайминги трафика;
- собирать агрегаты и передавать их на control plane.

Сейчас репозиторий находится на стадии инженерного каркаса и offline-ML pipelines. В нём уже есть:

- клиентские и серверные заглушки;
- единые контракты между компонентами;
- pipelines подготовки данных;
- обучение `sensor` и `morpher`;
- CLI-команды для обучения, оценки и инференса;
- synthetic multi-domain датасет для честной leave-one-domain-out проверки.

README написан как onboarding для нового человека, который впервые открывает этот репозиторий. Если твой друг будет заниматься именно `morpher`, ему достаточно сначала прочитать разделы:

1. `Быстрый старт`
2. `Карта репозитория`
3. `Датасеты`
4. `Morpher: обучение и инференс`

## Быстрый старт

### Требования

- Linux/macOS shell
- `uv`
- Python `3.11+`
- для `morpher` нужен `torch`
- для подготовки `browser_iat.csv` из PCAP нужен `tcpdump`

### Установка зависимостей

```bash
uv sync --dev --extra train
```

Если `torch` ещё не установлен, он подтянется из `extra train`.

### Проверка проекта

```bash
.venv/bin/pytest
```

### Главный CLI

Все основные действия выполняются через:

```bash
python main.py --help
```

или через локальное окружение:

```bash
.venv/bin/python main.py --help
```

## Что уже реализовано, а что пока заглушка

Реально работает уже сейчас:

- подготовка датасетов для `sensor` и `morpher`;
- обучение моделей;
- экспорт metadata;
- `sensor` inference;
- `morpher` inference;
- transfer evaluation;
- leave-one-domain-out evaluation;
- controlled synthetic sensor domains;
- минимальный `FastAPI` сервер с health/model/aggregate endpoint'ами.

Пока остаются заглушками:

- runtime-интеграция реальной модели `sensor` в клиент;
- runtime-интеграция `morpher` в реальный сетевой стек;
- реальный bandit вместо in-memory stub;
- реальные транспортные плагины вместо loopback-stub реализаций;
- продовый control plane и хранилище моделей.

## Архитектура верхнего уровня

Поток данных в текущем виде такой:

1. Сырые сетевые данные приводятся к табличному формату.
2. Для `sensor` строятся оконные признаки качества канала.
3. Для `morpher` строятся последовательности `IAT` (`inter-arrival time`).
4. Модели обучаются offline.
5. Артефакты модели и metadata сохраняются локально.
6. Через отдельные CLI-команды можно запускать инференс на новых данных.

## Карта репозитория

Ниже перечислены основные директории и файлы, которые реально важны для работы.

### Корень репозитория

- [`TASK.txt`](/home/senamorsin/Desktop/ChameleonFlow/TASK.txt)  
  Исходное описание задачи.

- [`IMPLEMENTATION_PLAN.md`](/home/senamorsin/Desktop/ChameleonFlow/IMPLEMENTATION_PLAN.md)  
  План реализации.

- [`pyproject.toml`](/home/senamorsin/Desktop/ChameleonFlow/pyproject.toml)  
  Зависимости, конфиги `pytest`, `ruff`, `mypy`, sources для `torch`.

- [`uv.lock`](/home/senamorsin/Desktop/ChameleonFlow/uv.lock)  
  Зафиксированный lock-файл окружения.

- [`main.py`](/home/senamorsin/Desktop/ChameleonFlow/main.py)  
  Главный CLI проекта. Отсюда вызываются почти все сценарии: подготовка данных, обучение, инференс, evaluation.

- [`index.html`](/home/senamorsin/Desktop/ChameleonFlow/index.html)  
  Служебный файл, не связан с ML pipelines.

- [`.env.example`](/home/senamorsin/Desktop/ChameleonFlow/.env.example)  
  Пример переменных окружения.

- [`.gitignore`](/home/senamorsin/Desktop/ChameleonFlow/.gitignore)  
  Игнорирование локальных датасетов и артефактов; synthetic controlled dataset специально разрешён для коммита.

### `shared/`

Общие типы между клиентом и сервером.

- [`shared/contracts.py`](/home/senamorsin/Desktop/ChameleonFlow/shared/contracts.py)  
  Базовые dataclass/enum-контракты:
  - `TransportKind`
  - `SensorResult`
  - `BanditContext`
  - `BanditDecision`
  - `SessionAggregate`

### `client/`

Скелет клиентского рантайма.

- [`client/app/core/models.py`](/home/senamorsin/Desktop/ChameleonFlow/client/app/core/models.py)  
  Реэкспорт общих контрактов из `shared`.

- [`client/app/core/state.py`](/home/senamorsin/Desktop/ChameleonFlow/client/app/core/state.py)  
  Локальное состояние клиента:
  - активный транспорт;
  - активные версии моделей;
  - неотправленные агрегаты.

- [`client/app/sensor/service.py`](/home/senamorsin/Desktop/ChameleonFlow/client/app/sensor/service.py)  
  Простая stub-реализация `sensor`:
  - `ChannelMetricsWindow`
  - `ThresholdSensor`
  Сейчас это не ML-модель, а ручной пороговый baseline.

- [`client/app/bandit/service.py`](/home/senamorsin/Desktop/ChameleonFlow/client/app/bandit/service.py)  
  In-memory bandit stub:
  - выбирает транспорт по score;
  - умеет принимать reward обновлением score.

- [`client/app/transports/base.py`](/home/senamorsin/Desktop/ChameleonFlow/client/app/transports/base.py)  
  Абстрактный интерфейс транспортного плагина.

- [`client/app/transports/stub_base.py`](/home/senamorsin/Desktop/ChameleonFlow/client/app/transports/stub_base.py)  
  Базовая заглушка транспорта на локальной очереди.

- [`client/app/transports/doh.py`](/home/senamorsin/Desktop/ChameleonFlow/client/app/transports/doh.py)  
  Заглушка `DoH` транспорта.

- [`client/app/transports/webrtc.py`](/home/senamorsin/Desktop/ChameleonFlow/client/app/transports/webrtc.py)  
  Заглушка `WebRTC` транспорта.

- [`client/app/transports/quic.py`](/home/senamorsin/Desktop/ChameleonFlow/client/app/transports/quic.py)  
  Заглушка `QUIC` транспорта.

- [`client/app/transports/registry.py`](/home/senamorsin/Desktop/ChameleonFlow/client/app/transports/registry.py)  
  Реестр транспортов, который собирает доступные плагины.

- [`client/app/config/client.example.yaml`](/home/senamorsin/Desktop/ChameleonFlow/client/app/config/client.example.yaml)  
  Пример клиентского конфига.

- [`client/tests/`](/home/senamorsin/Desktop/ChameleonFlow/client/tests)  
  Тесты клиентских заглушек:
  - `test_sensor.py`
  - `test_bandit.py`
  - `test_state.py`
  - `test_transports.py`

### `server/`

Скелет control server.

- [`server/app/main.py`](/home/senamorsin/Desktop/ChameleonFlow/server/app/main.py)  
  `FastAPI` application factory и app instance.

- [`server/app/api/routes.py`](/home/senamorsin/Desktop/ChameleonFlow/server/app/api/routes.py)  
  Текущие endpoint'ы:
  - `GET /health`
  - `GET /models/latest`
  - `POST /metrics/aggregates`

- [`server/app/api/schemas.py`](/home/senamorsin/Desktop/ChameleonFlow/server/app/api/schemas.py)  
  Pydantic-схемы API.

- [`server/app/config.example.yaml`](/home/senamorsin/Desktop/ChameleonFlow/server/app/config.example.yaml)  
  Пример серверного конфига.

- [`server/tests/test_api_routes.py`](/home/senamorsin/Desktop/ChameleonFlow/server/tests/test_api_routes.py)  
  Тесты API endpoints.

### `ml/`

Главная ML-часть проекта: датасеты, признаки, обучение, инференс, оценка.

#### `ml/training/`

Ниже перечислены все важные модули.

- [`ml/training/prepare_sensor_metrics.py`](/home/senamorsin/Desktop/ChameleonFlow/ml/training/prepare_sensor_metrics.py)  
  Нормализует внешний tabular dataset к внутреннему формату `sensor`.

- [`ml/training/sensor_pipeline.py`](/home/senamorsin/Desktop/ChameleonFlow/ml/training/sensor_pipeline.py)  
  Строит оконные признаки для `sensor`.  
  Здесь формируется основной feature set, включая session-relative baseline features:
  - `packet_loss_ratio_delta`
  - `rtt_mean_ratio_to_session_start`
  - `packets_per_second_ratio_to_session_start`
  - и другие.

- [`ml/training/sensor_models.py`](/home/senamorsin/Desktop/ChameleonFlow/ml/training/sensor_models.py)  
  Реестр алгоритмов для `sensor`:
  - `lightgbm`
  - `lightgbm_sigmoid`
  - `hist_gradient_boosting`
  - `random_forest`
  - `extra_trees`
  - `logistic_regression`

- [`ml/training/sensor_metrics.py`](/home/senamorsin/Desktop/ChameleonFlow/ml/training/sensor_metrics.py)  
  Универсальные метрики:
  - бинарная классификация;
  - threshold sweep;
  - probability summary;
  - regression metrics для `morpher`.

- [`ml/training/train_sensor.py`](/home/senamorsin/Desktop/ChameleonFlow/ml/training/train_sensor.py)  
  Обучение single-dataset `sensor`.

- [`ml/training/train_sensor_multidomain.py`](/home/senamorsin/Desktop/ChameleonFlow/ml/training/train_sensor_multidomain.py)  
  Обучение `sensor` на нескольких доменах сразу с pooled/per-domain/macro метриками.

- [`ml/training/sensor_multidomain.py`](/home/senamorsin/Desktop/ChameleonFlow/ml/training/sensor_multidomain.py)  
  Вспомогательная логика multi-domain split/balancing/metrics.

- [`ml/training/evaluate_sensor_transfer.py`](/home/senamorsin/Desktop/ChameleonFlow/ml/training/evaluate_sensor_transfer.py)  
  Transfer evaluation: train на одном датасете, eval на другом.

- [`ml/training/evaluate_sensor_loo.py`](/home/senamorsin/Desktop/ChameleonFlow/ml/training/evaluate_sensor_loo.py)  
  Leave-one-domain-out evaluation на нескольких доменах.

- [`ml/training/infer_sensor.py`](/home/senamorsin/Desktop/ChameleonFlow/ml/training/infer_sensor.py)  
  Offline инференс `sensor` по raw metrics.

- [`ml/training/compare_sensor_models.py`](/home/senamorsin/Desktop/ChameleonFlow/ml/training/compare_sensor_models.py)  
  Сравнение нескольких алгоритмов `sensor` на одном и том же split.

- [`ml/training/generate_synthetic_sensor_metrics.py`](/home/senamorsin/Desktop/ChameleonFlow/ml/training/generate_synthetic_sensor_metrics.py)  
  Простой synthetic baseline dataset для smoke/prototype.

- [`ml/training/generate_controlled_sensor_domains.py`](/home/senamorsin/Desktop/ChameleonFlow/ml/training/generate_controlled_sensor_domains.py)  
  Основной synthetic multi-domain generator.  
  Генерирует несколько доменов с разными baseline-сетями, приложениями и impairment-профилями.

- [`ml/training/prepare_cicids2017_sensor_metrics.py`](/home/senamorsin/Desktop/ChameleonFlow/ml/training/prepare_cicids2017_sensor_metrics.py)  
  Конвертация `CIC-IDS2017` в формат `sensor`.

- [`ml/training/prepare_iscxvpn2016_sensor_metrics.py`](/home/senamorsin/Desktop/ChameleonFlow/ml/training/prepare_iscxvpn2016_sensor_metrics.py)  
  Конвертация `ISCXVPN2016` в формат `sensor`.

- [`ml/training/sensor_experiment_dataset.py`](/home/senamorsin/Desktop/ChameleonFlow/ml/training/sensor_experiment_dataset.py)  
  Manifest-driven pipeline для своих экспериментов:
  - структура run manifest;
  - label assignment по фазам;
  - сбор processed dataset из нескольких run directories.

- [`ml/training/run_sensor_ping_experiment.py`](/home/senamorsin/Desktop/ChameleonFlow/ml/training/run_sensor_ping_experiment.py)  
  Минимальный host-side collector:
  - читает manifest;
  - накладывает `tc netem`;
  - гоняет `ping`;
  - пишет `sensor_metrics_raw.csv`.

- [`ml/training/prepare_browser_iat.py`](/home/senamorsin/Desktop/ChameleonFlow/ml/training/prepare_browser_iat.py)  
  Превращает таблицу packet timestamps в `browser_iat.csv`.

- [`ml/training/prepare_browser_iat_from_pcap.py`](/home/senamorsin/Desktop/ChameleonFlow/ml/training/prepare_browser_iat_from_pcap.py)  
  Готовит `browser_iat.csv` прямо из `.pcap/.pcapng/.zip`.

- [`ml/training/morpher_pipeline.py`](/home/senamorsin/Desktop/ChameleonFlow/ml/training/morpher_pipeline.py)  
  Строит sequence dataset для `morpher`:
  - нормализация;
  - sliding windows по `iat_ms`.

- [`ml/training/morpher_model.py`](/home/senamorsin/Desktop/ChameleonFlow/ml/training/morpher_model.py)  
  Архитектура `morpher`:
  - `MorpherConfig`
  - `LSTM -> Linear`
  - checkpoint payload builder

- [`ml/training/train_morpher.py`](/home/senamorsin/Desktop/ChameleonFlow/ml/training/train_morpher.py)  
  Обучение `morpher` на `IAT` последовательностях.

- [`ml/training/infer_morpher.py`](/home/senamorsin/Desktop/ChameleonFlow/ml/training/infer_morpher.py)  
  Offline инференс `morpher` по `browser_iat.csv` и чекпоинту `.pt`.

- [`ml/training/dataset_registry.py`](/home/senamorsin/Desktop/ChameleonFlow/ml/training/dataset_registry.py)  
  Работа с реестром датасетов.

#### `ml/datasets/`

- [`ml/datasets/registry.yaml`](/home/senamorsin/Desktop/ChameleonFlow/ml/datasets/registry.yaml)  
  Реестр датасетов и notes.

- `ml/datasets/raw/`  
  Сырые внешние датасеты. В git не хранятся, кроме `README` для `iscxvpn2016`.

- `ml/datasets/processed/`  
  Подготовленные локальные таблицы для обучения.

- [`ml/datasets/processed/controlled_sensor_domains/summary.json`](/home/senamorsin/Desktop/ChameleonFlow/ml/datasets/processed/controlled_sensor_domains/summary.json)  
  Summary для закоммиченного synthetic multi-domain sensor dataset.

- [`ml/datasets/processed/controlled_sensor_domains/fiber_lab.csv`](/home/senamorsin/Desktop/ChameleonFlow/ml/datasets/processed/controlled_sensor_domains/fiber_lab.csv)  
  Synthetic домен `fiber_lab`.

- [`ml/datasets/processed/controlled_sensor_domains/home_wifi.csv`](/home/senamorsin/Desktop/ChameleonFlow/ml/datasets/processed/controlled_sensor_domains/home_wifi.csv)  
  Synthetic домен `home_wifi`.

- [`ml/datasets/processed/controlled_sensor_domains/lte_edge.csv`](/home/senamorsin/Desktop/ChameleonFlow/ml/datasets/processed/controlled_sensor_domains/lte_edge.csv)  
  Synthetic домен `lte_edge`.

- [`ml/datasets/processed/controlled_sensor_domains/public_hotspot.csv`](/home/senamorsin/Desktop/ChameleonFlow/ml/datasets/processed/controlled_sensor_domains/public_hotspot.csv)  
  Synthetic домен `public_hotspot`.

- [`ml/datasets/processed/controlled_sensor_domains/satellite_emulated.csv`](/home/senamorsin/Desktop/ChameleonFlow/ml/datasets/processed/controlled_sensor_domains/satellite_emulated.csv)  
  Synthetic домен `satellite_emulated`.

#### `ml/tests/`

Тесты всей ML-части:

- подготовка датасетов;
- обучение `sensor` и `morpher`;
- инференс;
- transfer eval;
- LOO eval;
- synthetic generators;
- experiment dataset workflow.

### `scripts/`

Shell-обёртки над CLI.

- [`scripts/train_sensor.sh`](/home/senamorsin/Desktop/ChameleonFlow/scripts/train_sensor.sh)
- [`scripts/train_sensor_multidomain.sh`](/home/senamorsin/Desktop/ChameleonFlow/scripts/train_sensor_multidomain.sh)
- [`scripts/train_morpher.sh`](/home/senamorsin/Desktop/ChameleonFlow/scripts/train_morpher.sh)
- [`scripts/infer_sensor.sh`](/home/senamorsin/Desktop/ChameleonFlow/scripts/infer_sensor.sh)
- [`scripts/infer_morpher.sh`](/home/senamorsin/Desktop/ChameleonFlow/scripts/infer_morpher.sh)
- [`scripts/evaluate_sensor_transfer.sh`](/home/senamorsin/Desktop/ChameleonFlow/scripts/evaluate_sensor_transfer.sh)
- [`scripts/prepare_iscxvpn2016_sensor.sh`](/home/senamorsin/Desktop/ChameleonFlow/scripts/prepare_iscxvpn2016_sensor.sh)
- [`scripts/init_sensor_experiment.sh`](/home/senamorsin/Desktop/ChameleonFlow/scripts/init_sensor_experiment.sh)
- [`scripts/run_sensor_ping_experiment.sh`](/home/senamorsin/Desktop/ChameleonFlow/scripts/run_sensor_ping_experiment.sh)
- [`scripts/build_sensor_experiment_dataset.sh`](/home/senamorsin/Desktop/ChameleonFlow/scripts/build_sensor_experiment_dataset.sh)
- [`scripts/generate_sensor_sample.sh`](/home/senamorsin/Desktop/ChameleonFlow/scripts/generate_sensor_sample.sh)

### `docs/`

- [`docs/training.md`](/home/senamorsin/Desktop/ChameleonFlow/docs/training.md)  
  Расширенные инструкции по обучению и подготовке данных.

- [`docs/datasets.md`](/home/senamorsin/Desktop/ChameleonFlow/docs/datasets.md)  
  Список датасетов и reference notes.

### `infra/`

- [`infra/compose/docker-compose.yml`](/home/senamorsin/Desktop/ChameleonFlow/infra/compose/docker-compose.yml)  
  Минимальный compose skeleton.

### `tests/`

- [`tests/test_cli.py`](/home/senamorsin/Desktop/ChameleonFlow/tests/test_cli.py)  
  Smoke/help tests на главный CLI.

## Датасеты

Большие raw datasets и model artifacts по умолчанию не хранятся в git. Исключение: synthetic controlled dataset для `sensor`, который уже закоммичен в репозиторий.

### Уже лежит в репозитории

Synthetic датасет для `sensor` уже находится в:

- `ml/datasets/processed/controlled_sensor_domains/`

Он нужен для:

- быстрой отладки признаков;
- LOO evaluation;
- честной unified-задачи `healthy vs degraded`;
- bootstrap до сбора реальных данных.

### Нужно скачать отдельно

#### 1. CIC-IDS2017

Источник:

- `https://www.unb.ca/cic/datasets/ids-2017.html`

Нужные файлы:

- `Monday-WorkingHours.pcap_ISCX.csv`
- `Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv`

Положить сюда:

- `ml/datasets/raw/cicids2017/`

Подготовка:

```bash
python main.py prepare-cicids2017-sensor \
  -i ml/datasets/raw/cicids2017/Monday-WorkingHours.pcap_ISCX.csv \
  -i ml/datasets/raw/cicids2017/Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv \
  -o ml/datasets/processed/sensor_metrics.csv
```

#### 2. ISCXVPN2016 / VPN-nonVPN

Источник:

- `https://www.kaggle.com/datasets/noobbcoder2/vpn-and-non-vpn-application-traffic-cic-vpn2016`

Скачивание:

```bash
mkdir -p ml/datasets/raw/iscxvpn2016
curl -L -o ml/datasets/raw/iscxvpn2016/vpn-and-non-vpn-application-traffic-cic-vpn2016.zip \
  https://www.kaggle.com/api/v1/datasets/download/noobbcoder2/vpn-and-non-vpn-application-traffic-cic-vpn2016
unzip -o ml/datasets/raw/iscxvpn2016/vpn-and-non-vpn-application-traffic-cic-vpn2016.zip \
  -d ml/datasets/raw/iscxvpn2016
```

Ожидаемый файл:

- `ml/datasets/raw/iscxvpn2016/consolidated_traffic_data.csv`

Подготовка:

```bash
python main.py prepare-iscxvpn2016-sensor \
  -i ml/datasets/raw/iscxvpn2016/consolidated_traffic_data.csv \
  -o ml/datasets/processed/iscxvpn2016_sensor_metrics.csv \
  --positive-pattern vpn
```

Важно:

- это proxy dataset для проверки domain shift;
- это не финальная ground truth разметка деградации канала.

#### 3. Westermo sample для `morpher`

Источник:

- `https://github.com/westermo/network-traffic-dataset`

Нужный архив:

- `right.zip`

Положить сюда:

- `ml/datasets/raw/westermo/right.zip`

Подготовка:

```bash
python main.py prepare-browser-iat-from-pcap \
  ml/datasets/raw/westermo/right.zip \
  ml/datasets/processed/browser_iat.csv
```

## Sensor: обучение, оценка и инференс

### Минимальный synthetic smoke-run

```bash
python main.py generate-sensor-sample ml/datasets/processed/sensor_metrics.csv
python main.py train-sensor \
  ml/datasets/processed/sensor_metrics.csv \
  ml/exported/sensor.txt \
  ml/exported/sensor.metadata.json
```

### Обучение single-dataset sensor

```bash
python main.py train-sensor \
  ml/datasets/processed/sensor_metrics.csv \
  ml/exported/sensor.txt \
  ml/exported/sensor.metadata.json \
  --output-onnx ml/exported/sensor.onnx
```

Что получается:

- `sensor.txt` или `.joblib` модель;
- `sensor.metadata.json`;
- опционально `sensor.onnx`.

### Сравнение алгоритмов

```bash
python main.py compare-sensor-models \
  ml/datasets/processed/sensor_metrics.csv \
  ml/exported/sensor_benchmark.json
```

### Mixed-domain sensor

```bash
python main.py train-sensor-multidomain \
  -i ml/datasets/processed/sensor_metrics.csv \
  -i ml/datasets/processed/iscxvpn2016_sensor_metrics.csv \
  ml/exported/sensor_multidomain.txt \
  ml/exported/sensor_multidomain.metadata.json \
  --output-onnx ml/exported/sensor_multidomain.onnx \
  --algorithm lightgbm \
  --balance-domains
```

### Transfer evaluation

```bash
python main.py evaluate-sensor-transfer \
  ml/datasets/processed/sensor_metrics.csv \
  ml/datasets/processed/iscxvpn2016_sensor_metrics.csv \
  ml/exported/sensor_transfer.json \
  --algorithm lightgbm
```

### Honest LOO на synthetic controlled domains

```bash
python main.py evaluate-sensor-loo \
  -i ml/datasets/processed/controlled_sensor_domains/fiber_lab.csv \
  -i ml/datasets/processed/controlled_sensor_domains/home_wifi.csv \
  -i ml/datasets/processed/controlled_sensor_domains/lte_edge.csv \
  -i ml/datasets/processed/controlled_sensor_domains/public_hotspot.csv \
  -i ml/datasets/processed/controlled_sensor_domains/satellite_emulated.csv \
  ml/exported/controlled_sensor_loo.json \
  --algorithm lightgbm \
  --balance-domains
```

### Sensor inference

Инференс запускается по raw metrics table.

Пример:

```bash
python main.py infer-sensor \
  ml/datasets/processed/sensor_metrics.csv \
  ml/exported/sensor.txt \
  ml/exported/sensor_predictions.csv \
  --metadata-path ml/exported/sensor.metadata.json
```

Аналог через shell wrapper:

```bash
./scripts/infer_sensor.sh \
  ml/datasets/processed/sensor_metrics.csv \
  ml/exported/sensor.txt \
  ml/exported/sensor_predictions.csv \
  --metadata-path ml/exported/sensor.metadata.json
```

Что пишет команда:

- `session_id`
- `window_start`
- `label`
- `probability`
- `degraded`

Если `--threshold` не указан, берётся threshold из metadata.

## Morpher: обучение и инференс

Этот раздел самый важный для человека, который будет работать с `morpher`.

### Что такое morpher в этом репозитории

`Morpher` здесь это offline-модель, которая по последовательности прошлых `IAT` предсказывает следующий `IAT`.

Текущая постановка:

- вход: последовательность длины `sequence_length`;
- модель: `LSTM(hidden_size) -> Linear(1)`;
- target: следующий `iat_ms`;
- loss: `HuberLoss`.

### Формат входных данных

Нужен файл `browser_iat.csv` или `.parquet` со столбцами:

- `trace_id`
- `iat_ms`

Опционально:

- `packet_index`

### Как получить `browser_iat.csv`

Из packet table:

```bash
python main.py prepare-browser-iat \
  raw_packets.csv \
  ml/datasets/processed/browser_iat.csv
```

Из PCAP/ZIP:

```bash
python main.py prepare-browser-iat-from-pcap \
  ml/datasets/raw/westermo/right.zip \
  ml/datasets/processed/browser_iat.csv
```

### Обучение morpher

```bash
python main.py train-morpher \
  ml/datasets/processed/browser_iat.csv \
  ml/exported/morpher.pt \
  ml/exported/morpher.metadata.json \
  --output-onnx ml/exported/morpher.onnx \
  --device auto
```

Аналог через wrapper:

```bash
./scripts/train_morpher.sh \
  ml/datasets/processed/browser_iat.csv \
  ml/exported/morpher.pt \
  ml/exported/morpher.metadata.json \
  --output-onnx ml/exported/morpher.onnx \
  --device auto
```

Что сохраняется:

- `morpher.pt`  
  Torch checkpoint, который содержит:
  - `config`
  - `state_dict`
  - `normalization.mean`
  - `normalization.std`

- `morpher.metadata.json`  
  Training metadata:
  - `sequence_length`
  - `hidden_size`
  - `epochs`
  - `device`
  - `validation_metrics`
  - `normalization_mean`
  - `normalization_std`

- `morpher.onnx`  
  Опциональный экспорт ONNX.

### Morpher inference

Инференс запускается по `browser_iat.csv` и чекпоинту `.pt`.

```bash
python main.py infer-morpher \
  ml/datasets/processed/browser_iat.csv \
  ml/exported/morpher.pt \
  ml/exported/morpher_predictions.csv \
  --device auto
```

Аналог через wrapper:

```bash
./scripts/infer_morpher.sh \
  ml/datasets/processed/browser_iat.csv \
  ml/exported/morpher.pt \
  ml/exported/morpher_predictions.csv \
  --device auto
```

Что пишет команда:

- `trace_id`
- `target_packet_index`
- `actual_iat_ms`
- `predicted_iat_ms`
- `abs_error_ms`

То есть друг может сразу обучить модель, потом прогнать инференс на том же или новом `browser_iat.csv`, и посмотреть ошибку по каждой точке.

### Какие файлы читать, если работаешь именно над morpher

В таком порядке:

1. [`ml/training/prepare_browser_iat.py`](/home/senamorsin/Desktop/ChameleonFlow/ml/training/prepare_browser_iat.py)
2. [`ml/training/prepare_browser_iat_from_pcap.py`](/home/senamorsin/Desktop/ChameleonFlow/ml/training/prepare_browser_iat_from_pcap.py)
3. [`ml/training/morpher_pipeline.py`](/home/senamorsin/Desktop/ChameleonFlow/ml/training/morpher_pipeline.py)
4. [`ml/training/morpher_model.py`](/home/senamorsin/Desktop/ChameleonFlow/ml/training/morpher_model.py)
5. [`ml/training/train_morpher.py`](/home/senamorsin/Desktop/ChameleonFlow/ml/training/train_morpher.py)
6. [`ml/training/infer_morpher.py`](/home/senamorsin/Desktop/ChameleonFlow/ml/training/infer_morpher.py)
7. [`ml/tests/test_train_morpher.py`](/home/senamorsin/Desktop/ChameleonFlow/ml/tests/test_train_morpher.py)
8. [`ml/tests/test_infer_morpher.py`](/home/senamorsin/Desktop/ChameleonFlow/ml/tests/test_infer_morpher.py)

## Как собрать свой реальный датасет для sensor

Если цель это финальная модель, а не synthetic benchmark, нужно собирать свои run'ы.

### 1. Инициализировать run

```bash
python main.py init-sensor-experiment \
  ml/datasets/raw/experiments/run-001 \
  run-001 \
  lab_wifi \
  browsing \
  delay-jitter
```

### 2. Собрать raw metrics

Минимальный bootstrap collector:

```bash
python main.py run-sensor-ping-experiment \
  ml/datasets/raw/experiments/run-001 \
  1.1.1.1 \
  eth0 \
  --sudo
```

Это создаст `sensor_metrics_raw.csv`.

### 3. Собрать processed dataset из run directories

```bash
python main.py build-sensor-experiment-dataset \
  -i ml/datasets/raw/experiments/run-001 \
  -i ml/datasets/raw/experiments/run-002 \
  ml/datasets/processed/experiments_sensor_metrics.csv
```

### 4. Обучить sensor уже на своих данных

```bash
python main.py train-sensor \
  ml/datasets/processed/experiments_sensor_metrics.csv \
  ml/exported/sensor_experiments.txt \
  ml/exported/sensor_experiments.metadata.json
```

## Сервер и локальные заглушки

### Запуск FastAPI server

```bash
python main.py serve
```

После запуска доступны:

- `GET /health`
- `GET /models/latest`
- `POST /metrics/aggregates`

### Посмотреть доступные транспортные заглушки

```bash
python main.py list-transports
```

Ожидаемый вывод сейчас:

- `doh`
- `webrtc`
- `quic`

## Полезные команды

### Полный тестовый прогон

```bash
.venv/bin/pytest
```

### Только help по CLI

```bash
python main.py --help
```

### Help по morpher

```bash
python main.py train-morpher --help
python main.py infer-morpher --help
```

### Help по sensor

```bash
python main.py train-sensor --help
python main.py train-sensor-multidomain --help
python main.py infer-sensor --help
python main.py evaluate-sensor-transfer --help
python main.py evaluate-sensor-loo --help
```

## Краткое резюме по артефактам

### Для sensor

- вход: raw metrics table
- обучение: `train-sensor` или `train-sensor-multidomain`
- инференс: `infer-sensor`
- оценки:
  - `compare-sensor-models`
  - `evaluate-sensor-transfer`
  - `evaluate-sensor-loo`

### Для morpher

- вход: `browser_iat.csv`
- обучение: `train-morpher`
- инференс: `infer-morpher`

## Что читать дальше

Если нужен более подробный разбор команд и схем входных данных, смотри:

- [`docs/training.md`](/home/senamorsin/Desktop/ChameleonFlow/docs/training.md)
- [`docs/datasets.md`](/home/senamorsin/Desktop/ChameleonFlow/docs/datasets.md)
