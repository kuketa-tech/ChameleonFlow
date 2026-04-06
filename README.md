# ChameleonFlow

Prototype of an adaptive client-server transport system that combines:

- channel quality sensing
- traffic timing morphing
- contextual transport selection

## Current state

The repository already contains:

- client/server project skeleton
- shared contracts for sensor, bandit, transport, and aggregates
- FastAPI server stub with core endpoints
- transport, sensor, and bandit stubs
- dataset registry and first offline training pipelines

## Repository layout

- `client/` client runtime skeleton
- `server/` control server skeleton
- `ml/` dataset registry and training scripts
- `docs/datasets.md` selected public datasets
- `docs/training.md` training commands and input schemas

## Environment

Install dependencies:

```bash
uv sync --dev --extra train
```

`torch` is only required for the morpher training path.

## Datasets

Large raw datasets and generated training artifacts are intentionally not stored in git.

Local directories used by the project:

- `ml/datasets/raw/cicids2017/`
- `ml/datasets/raw/iscxvpn2016/`
- `ml/datasets/raw/westermo/`
- `ml/datasets/processed/`

Machine-readable dataset registry:

- `ml/datasets/registry.yaml`

### 1. CIC-IDS2017 for sensor training

Source:

- official page: `https://www.unb.ca/cic/datasets/ids-2017.html`

Minimum files used by the current prototype:

- `Monday-WorkingHours.pcap_ISCX.csv`
- `Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv`

Place them here:

- `ml/datasets/raw/cicids2017/`

Build the processed sensor dataset:

```bash
python main.py prepare-cicids2017-sensor \
  -i ml/datasets/raw/cicids2017/Monday-WorkingHours.pcap_ISCX.csv \
  -i ml/datasets/raw/cicids2017/Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv \
  -o ml/datasets/processed/sensor_metrics.csv
```

### 2. ISCXVPN2016 / VPN-nonVPN for transfer validation

Source:

- Kaggle mirror: `https://www.kaggle.com/datasets/noobbcoder2/vpn-and-non-vpn-application-traffic-cic-vpn2016`

Download:

```bash
mkdir -p ml/datasets/raw/iscxvpn2016
curl -L -o ml/datasets/raw/iscxvpn2016/vpn-and-non-vpn-application-traffic-cic-vpn2016.zip \
  https://www.kaggle.com/api/v1/datasets/download/noobbcoder2/vpn-and-non-vpn-application-traffic-cic-vpn2016
unzip -o ml/datasets/raw/iscxvpn2016/vpn-and-non-vpn-application-traffic-cic-vpn2016.zip \
  -d ml/datasets/raw/iscxvpn2016
```

This currently gives:

- `ml/datasets/raw/iscxvpn2016/consolidated_traffic_data.csv`

Convert it to the project training format:

```bash
python main.py prepare-iscxvpn2016-sensor \
  -i ml/datasets/raw/iscxvpn2016/consolidated_traffic_data.csv \
  -o ml/datasets/processed/iscxvpn2016_sensor_metrics.csv \
  --positive-pattern vpn
```

Important:

- this mapping is only a transfer-validation proxy
- rows with labels like `VPN-BROWSING` become positive
- rows like `BROWSING`, `CHAT`, `MAIL` become negative
- this is useful for domain-shift testing, not as the final ground truth for channel degradation

### 3. Westermo sample for morpher preprocessing

Source:

- repository: `https://github.com/westermo/network-traffic-dataset`

The current prototype expects the reduced sample archive:

- `right.zip`

Place it here:

- `ml/datasets/raw/westermo/right.zip`

Then build `IAT` rows:

```bash
python main.py prepare-browser-iat-from-pcap \
  ml/datasets/raw/westermo/right.zip \
  ml/datasets/processed/browser_iat.csv
```

## How the datasets are used

### Sensor model

Train the current sensor model on processed `CIC-IDS2017` features:

```bash
python main.py train-sensor \
  ml/datasets/processed/sensor_metrics.csv \
  ml/exported/sensor.txt \
  ml/exported/sensor.metadata.json \
  --output-onnx ml/exported/sensor.onnx
```

Compare several tabular models on the same split:

```bash
python main.py compare-sensor-models \
  ml/datasets/processed/sensor_metrics.csv \
  ml/exported/sensor_benchmark.json
```

Check transfer from `CIC-IDS2017` to `ISCXVPN2016`:

```bash
python main.py evaluate-sensor-transfer \
  ml/datasets/processed/sensor_metrics.csv \
  ml/datasets/processed/iscxvpn2016_sensor_metrics.csv \
  ml/exported/sensor_transfer.json \
  --algorithm lightgbm
```

### Morpher model

Train the current timing morpher on `browser_iat.csv`:

```bash
python main.py train-morpher \
  ml/datasets/processed/browser_iat.csv \
  ml/exported/morpher.pt \
  ml/exported/morpher.metadata.json \
  --output-onnx ml/exported/morpher.onnx \
  --device auto
```

## Quick checks

List supported transport stubs:

```bash
python main.py list-transports
```

Run the current test suite:

```bash
.venv/bin/pytest
```

More detailed notes live in:

- `docs/datasets.md`
- `docs/training.md`
