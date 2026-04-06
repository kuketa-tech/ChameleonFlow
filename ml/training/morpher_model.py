from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(slots=True)
class MorpherConfig:
    sequence_length: int = 20
    hidden_size: int = 32


def build_morpher_model(nn: Any, config: MorpherConfig) -> Any:
    class LSTMIATPredictor(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.lstm = nn.LSTM(
                input_size=1,
                hidden_size=config.hidden_size,
                batch_first=True,
            )
            self.output = nn.Linear(config.hidden_size, 1)

        def forward(self, inputs: Any) -> Any:
            sequence_outputs, _ = self.lstm(inputs)
            return self.output(sequence_outputs[:, -1, :])

    return LSTMIATPredictor()


def build_checkpoint_payload(
    *,
    config: MorpherConfig,
    state_dict: dict[str, Any],
    normalization_mean: float,
    normalization_std: float,
) -> dict[str, Any]:
    return {
        "config": asdict(config),
        "state_dict": state_dict,
        "normalization": {
            "mean": normalization_mean,
            "std": normalization_std,
        },
    }
