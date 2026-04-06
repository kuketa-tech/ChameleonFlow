# Dataset Selection

## Chosen starting point

The prototype uses a mixed dataset strategy instead of betting on one corpus:

- `ISCXIDS2012` for labeled anomaly scenarios and replay-based degradation cases.
- `CIC-IDS2017` for modern benign and attack diversity plus ready-made flow CSVs.
- `ISCXVPN2016` for encrypted traffic baselines and, critically, browser-like `IAT` extraction from web browsing sessions.
- `MAWI` as optional supplemental validation for timing distributions, not as the primary browser corpus.

The machine-readable source of truth lives in `ml/datasets/registry.yaml`.

## Why `VPN-nonVPN` stays in scope

`ISCXVPN2016` is not excluded. It is one of the main datasets:

- For the `morpher`, it is the best public starting point in this repo because the dataset explicitly includes web browsing traffic from `Firefox and Chrome`.
- For the `sensor`, it is useful as realistic encrypted background traffic, but it should be combined with `CIC-IDS2017`, `ISCXIDS2012`, and synthetic `tc netem` impairments so the detector does not overfit to one capture style.

## Practical preprocessing plan

### Sensor

1. Convert packet or flow captures into a tabular stream with timestamps and per-interval metrics.
2. Aggregate metrics into 5-second windows.
3. Compute:
   - packet loss ratio
   - RTT coefficient of variation
   - retransmission ratio
   - reset ratio
4. Attach labels from dataset metadata or replay schedule.

### Morpher

1. Start from `ISCXVPN2016` browsing captures.
2. Extract packet timestamps per session.
3. Convert timestamps to inter-arrival times in milliseconds.
4. Build rolling sequences of length `20` and predict the next `IAT`.
5. Keep `MAWI` only as a secondary validation source for comparing timing distributions.
