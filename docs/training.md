# Training Pipelines

## Environment

For training extras, install the optional group:

```bash
uv sync --dev --extra train
```

`torch` is only required for the morpher training script. The preprocessing code and dataset registry work without it.

## Sensor training

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
