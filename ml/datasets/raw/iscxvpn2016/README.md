# ISCXVPN2016 Download Status

This directory is reserved for the `ISCXVPN2016` / `VPN-nonVPN` dataset.

As of 2026-04-06, automatic download from the known public sources was not possible:

- the official CIC mirror redirected dataset file URLs back to the dataset index page
- the public Kaggle dataset page exposed metadata, but the download endpoint returned HTML instead of the archive without an authenticated Kaggle session

Attempted sources:

- `https://www.unb.ca/cic/datasets/vpn.html`
- `https://cicresearch.ca/CICDataset/ISCX-VPN-NonVPN-2016/`
- `https://www.kaggle.com/datasets/noobbcoder2/vpn-and-non-vpn-application-traffic-cic-vpn2016`

Once you place a downloaded `CSV` or `ARFF` dump here, convert it with:

```bash
python main.py prepare-iscxvpn2016-sensor \
  -i ml/datasets/raw/iscxvpn2016/<your-file>.csv \
  -o ml/datasets/processed/iscxvpn2016_sensor_metrics.csv \
  --positive-pattern vpn
```

Then run cross-dataset evaluation with:

```bash
python main.py evaluate-sensor-transfer \
  ml/datasets/processed/sensor_metrics.csv \
  ml/datasets/processed/iscxvpn2016_sensor_metrics.csv \
  ml/exported/sensor_transfer.json \
  --algorithm lightgbm
```
