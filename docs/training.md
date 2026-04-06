# Training Pipelines

## Environment

For training extras, install the optional group:

```bash
uv sync --dev --extra train
```

`torch` is only required for the morpher training script. The preprocessing code and dataset registry work without it.

## Building your own sensor dataset

The current repo now supports a controlled-run workflow:

1. initialize a run directory with a manifest
2. collect raw metric rows into `sensor_metrics_raw.csv`
3. label those rows from the manifest phases
4. merge multiple runs into one processed training table

Initialize a run:

```bash
python main.py init-sensor-experiment \
  ml/datasets/raw/experiments/run-001 \
  run-001 \
  lab_wifi \
  browsing \
  delay-jitter
```

or:

```bash
./scripts/init_sensor_experiment.sh \
  ml/datasets/raw/experiments/run-001 \
  run-001 \
  lab_wifi \
  browsing \
  delay-jitter
```

This creates:

- `manifest.json`
- `sensor_metrics_raw.csv` template

For a minimal host-side collector, you can run the built-in ping/netem workflow:

```bash
python main.py run-sensor-ping-experiment \
  ml/datasets/raw/experiments/run-001 \
  1.1.1.1 \
  eth0 \
  --sudo
```

or:

```bash
./scripts/run_sensor_ping_experiment.sh \
  ml/datasets/raw/experiments/run-001 \
  1.1.1.1 \
  eth0 \
  --sudo
```

What it does:

- loads `manifest.json`
- applies each phase to the selected interface via `tc netem`
- runs continuous `ping -D -O`
- writes a minimal `sensor_metrics_raw.csv`

This collector is intentionally minimal:

- it gives you RTT/loss-driven runs quickly
- it sets `retransmissions=0` and `resets=0`
- it is a bootstrap path, not the final high-fidelity transport collector

The raw metrics file should contain at least:

- `session_id`
- `timestamp`
- `packets_sent`
- `packets_lost`
- `rtt_ms`
- `retransmissions`
- `resets`

The manifest defines baseline / impairment / recovery phases with offsets in seconds.
The builder assigns labels from those phases, so the raw metrics file does not need its own label column.

Merge one or more runs into a processed dataset:

```bash
python main.py build-sensor-experiment-dataset \
  -i ml/datasets/raw/experiments/run-001 \
  -i ml/datasets/raw/experiments/run-002 \
  ml/datasets/processed/experiments_sensor_metrics.csv
```

or:

```bash
./scripts/build_sensor_experiment_dataset.sh \
  ml/datasets/raw/experiments/run-001 \
  ml/datasets/raw/experiments/run-002 \
  ml/datasets/processed/experiments_sensor_metrics.csv
```

The combined dataset keeps extra metadata columns such as:

- `run_id`
- `domain`
- `application`
- `scenario`
- `phase_name`
- `impairment_type`
- `severity`
- `capture_offset_seconds`

These columns are ignored by the current feature pipeline, but they are useful for honest train/validation splits and later diagnostics.

## Controlled synthetic domains

If you want a reproducible multi-domain impairment benchmark before collecting host traffic, generate the built-in controlled domains:

```bash
python main.py generate-controlled-sensor-domains \
  ml/datasets/processed/controlled_sensor_domains \
  --summary-path ml/datasets/processed/controlled_sensor_domains/summary.json
```

This writes one raw-metrics CSV per domain, for example:

- `fiber_lab.csv`
- `home_wifi.csv`
- `lte_edge.csv`
- `public_hotspot.csv`
- `satellite_emulated.csv`

Each domain mixes multiple applications and multiple impairment types with baseline, impairment, and recovery phases.
The current feature pipeline also computes session-relative baseline features so the model is less tied to absolute RTT or throughput scales.

Run honest leave-one-domain-out evaluation on those generated domains:

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

The LOO summary includes:

- `macro_metrics` at the configured threshold
- `macro_best_by_f1`
- per-domain holdout metrics
- per-domain threshold sweeps
- probability summaries for each holdout domain

## Sensor training

There are now two useful modes:

- `train-sensor` for single-dataset baselines
- `train-sensor-multidomain` for a more honest proxy setup with per-domain and macro metrics

If your goal is adaptation across datasets, prefer the multidomain path.

If you do not have a prepared dataset yet, generate a synthetic one first:

```bash
python main.py generate-sensor-sample ml/datasets/processed/sensor_metrics.csv
```

or:

```bash
./scripts/generate_sensor_sample.sh ml/datasets/processed/sensor_metrics.csv
```

If you already have tabular network metrics under different column names, normalize them first:

```bash
python main.py prepare-sensor-metrics raw_metrics.csv ml/datasets/processed/sensor_metrics.csv
```

For the downloaded `CIC-IDS2017` sample files in this repo:

```bash
python main.py prepare-cicids2017-sensor \
  -i ml/datasets/raw/cicids2017/Monday-WorkingHours.pcap_ISCX.csv \
  -i ml/datasets/raw/cicids2017/Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv \
  -o ml/datasets/processed/sensor_metrics.csv
```

For a locally downloaded `ISCXVPN2016` flow dump in `CSV` or `ARFF`, convert it the same way:

```bash
python main.py prepare-iscxvpn2016-sensor \
  -i ml/datasets/raw/iscxvpn2016/vpn_dataset.csv \
  -o ml/datasets/processed/iscxvpn2016_sensor_metrics.csv \
  --positive-pattern vpn
```

or:

```bash
./scripts/prepare_iscxvpn2016_sensor.sh \
  ml/datasets/raw/iscxvpn2016/vpn_dataset.csv \
  ml/datasets/processed/iscxvpn2016_sensor_metrics.csv \
  --positive-pattern vpn
```

This mapping is only a transfer-validation proxy:

- rows whose label or filename contains `vpn` become positive by default
- rows without `vpn` become negative
- use it as an independent holdout or stress-test, not as the final ground truth for channel degradation

Direct Python entry point:

```bash
python -m ml.training.train_sensor \
  --input ml/datasets/processed/sensor_metrics.csv \
  --output-model ml/exported/sensor.txt \
  --output-metadata ml/exported/sensor.metadata.json \
  --output-onnx ml/exported/sensor.onnx
```

Shell wrapper:

```bash
./scripts/train_sensor.sh \
  ml/datasets/processed/sensor_metrics.csv \
  ml/exported/sensor.txt \
  ml/exported/sensor.metadata.json \
  --output-onnx ml/exported/sensor.onnx
```

This is why `FileNotFoundError` happened earlier: `ml/datasets/processed/sensor_metrics.csv` was an example output path, not a file that already existed in the repo.

Expected input columns:

- `session_id`
- `timestamp`
- `packets_sent`
- `packets_lost`
- `rtt_ms`
- `retransmissions`
- `resets`
- `label`

The script aggregates rows into 5-second windows and computes:

- `packet_loss_ratio`
- `rtt_cv`
- `retransmission_ratio`
- `reset_ratio`

Validation output includes:

- `roc_auc`
- `average_precision`
- `accuracy`
- `precision`
- `recall`
- `f1`
- thresholded confusion matrix at the configured detection threshold

## Multidomain sensor training

This is the recommended proxy setup when you want less misleading metrics.

Why:

- `CIC-IDS2017` and `ISCXVPN2016` have different native labels
- a model trained on only one of them can report inflated in-domain metrics and collapse on the other
- the multidomain trainer keeps dataset-native positives, reports pooled metrics, per-domain metrics, and macro averages across domains

Run it like this:

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

or:

```bash
./scripts/train_sensor_multidomain.sh \
  ml/datasets/processed/sensor_metrics.csv \
  ml/datasets/processed/iscxvpn2016_sensor_metrics.csv \
  ml/exported/sensor_multidomain.txt \
  ml/exported/sensor_multidomain.metadata.json \
  --output-onnx ml/exported/sensor_multidomain.onnx \
  --algorithm lightgbm \
  --balance-domains
```

Important interpretation note:

- `task_semantics=proxy_nonbaseline_traffic_across_domains` is still a proxy task
- these are more honest adaptation metrics than single-domain validation
- they are still not the final ground truth for channel impairment detection

The multidomain metadata includes:

- `validation_metrics` for pooled holdout rows
- `validation_metrics_by_domain`
- `validation_metrics_macro`
- `training_rows_by_domain`
- `validation_rows_by_domain`
- `domain_notes`

## Sensor transfer evaluation

To check whether the detector generalizes across datasets instead of memorizing one source, train on one processed file and evaluate on another:

```bash
python main.py evaluate-sensor-transfer \
  ml/datasets/processed/sensor_metrics.csv \
  ml/datasets/processed/iscxvpn2016_sensor_metrics.csv \
  ml/exported/sensor_transfer.json \
  --algorithm lightgbm
```

or:

```bash
./scripts/evaluate_sensor_transfer.sh \
  ml/datasets/processed/sensor_metrics.csv \
  ml/datasets/processed/iscxvpn2016_sensor_metrics.csv \
  ml/exported/sensor_transfer.json \
  --algorithm lightgbm
```

The transfer summary includes:

- `eval_metrics` at the configured threshold
- `threshold_sweep.best_by_f1`
- `eval_probability_summary`
- train/eval class balance and row counts

## Morpher training

Prepare `IAT` rows from a packet-timestamp table first:

```bash
python main.py prepare-browser-iat \
  ml/datasets/processed/browser_packets.csv \
  ml/datasets/processed/browser_iat.csv
```

For the downloaded `westermo` PCAP sample in this repo:

```bash
python main.py prepare-browser-iat-from-pcap \
  ml/datasets/raw/westermo/right.zip \
  ml/datasets/processed/browser_iat.csv
```

Packet-table input columns:

- `trace_id`
- `timestamp`

Optional packet-table input column:

- `packet_index`

Then run training:

```bash
python main.py train-morpher \
  ml/datasets/processed/browser_iat.csv \
  ml/exported/morpher.pt \
  ml/exported/morpher.metadata.json \
  --output-onnx ml/exported/morpher.onnx \
  --device auto
```

Direct Python entry point:

```bash
python -m ml.training.train_morpher \
  --input ml/datasets/processed/browser_iat.csv \
  --output-model ml/exported/morpher.pt \
  --output-metadata ml/exported/morpher.metadata.json \
  --output-onnx ml/exported/morpher.onnx \
  --device auto
```

Shell wrapper:

```bash
./scripts/train_morpher.sh \
  ml/datasets/processed/browser_iat.csv \
  ml/exported/morpher.pt \
  ml/exported/morpher.metadata.json \
  --output-onnx ml/exported/morpher.onnx \
  --device auto
```

Expected input columns:

- `trace_id`
- `iat_ms`

Optional input column:

- `packet_index`

The script builds rolling sequences of length `20` and trains an `LSTM(32) -> Linear(1)` predictor with `HuberLoss`.
If CUDA is visible in your terminal, `--device auto` will use it; otherwise the trainer falls back to CPU.
Validation output includes:

- `huber_loss`
- `mae_ms`
- `rmse_ms`

## Dataset source of truth

Use `ml/datasets/registry.yaml` to keep track of official dataset URLs, local paths, and preprocessing notes.
