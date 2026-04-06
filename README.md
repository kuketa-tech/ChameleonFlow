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

## Main locations

- `client/` client runtime skeleton
- `server/` control server skeleton
- `ml/` dataset registry and training scripts
- `docs/datasets.md` selected public datasets
- `docs/training.md` training commands and input schemas

## Quick checks

List supported transport stubs:

```bash
python main.py list-transports
```

Run the current test suite:

```bash
.venv/bin/pytest
```
