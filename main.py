from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Annotated

import typer
import uvicorn

from client.app.core.agent import run_client_agent
from client.app.transports.registry import build_transport_registry
from ml.training.compare_sensor_models import compare_sensor_models
from ml.training.infer_morpher import infer_morpher
from ml.training.infer_sensor import infer_sensor
from ml.training.evaluate_sensor_loo import evaluate_sensor_loo
from ml.training.evaluate_sensor_transfer import evaluate_sensor_transfer
from ml.training.generate_controlled_sensor_domains import (
    controlled_domain_names,
    generate_controlled_sensor_domains,
)
from ml.training.generate_synthetic_sensor_metrics import (
    build_synthetic_sensor_metrics_frame,
    save_synthetic_sensor_metrics,
)
from ml.training.prepare_browser_iat_from_pcap import build_iat_frame_from_archive
from ml.training.prepare_browser_iat import build_browser_iat_frame, load_packet_table, save_iat_frame
from ml.training.prepare_cicids2017_sensor_metrics import build_cicids2017_sensor_metrics
from ml.training.prepare_iscxvpn2016_sensor_metrics import build_iscxvpn2016_sensor_metrics
from ml.training.prepare_sensor_metrics import build_sensor_metrics_frame, save_sensor_metrics_frame
from ml.training.sensor_experiment_dataset import (
    build_sensor_experiment_dataset,
    initialize_sensor_experiment_run,
)
from ml.training.run_sensor_ping_experiment import run_sensor_ping_experiment
from ml.training.train_morpher import train_morpher
from ml.training.train_sensor_multidomain import train_sensor_multidomain
from ml.training.train_sensor import train_sensor

app = typer.Typer(help="ChameleonFlow developer CLI.")


@app.command("serve")
def serve(host: str = "127.0.0.1", port: int = 8000) -> None:
    uvicorn.run("server.app.main:app", host=host, port=port, reload=False)


@app.command("list-transports")
def list_transports() -> None:
    for transport_kind in build_transport_registry():
        typer.echo(transport_kind.value)


@app.command("client-run")
def client_run(
    server_base_url: str = "http://127.0.0.1:8000",
    isp_id: str = "isp-a",
    traffic_type: str = "web",
    sessions: int = 10,
    seed: int = 42,
) -> None:
    result = asyncio.run(
        run_client_agent(
            server_base_url=server_base_url,
            isp_id=isp_id,
            traffic_type=traffic_type,
            sessions=sessions,
            seed=seed,
        ),
    )
    typer.echo(f"sessions={result.sessions_attempted} aggregates_sent={result.aggregates_sent}")


@app.command("prepare-browser-iat")
def prepare_browser_iat(
    input_path: Path,
    output_path: Path,
    trace_column: str = "trace_id",
    timestamp_column: str = "timestamp",
    packet_index_column: str = "packet_index",
    max_iat_ms: float = 5_000.0,
) -> None:
    packet_table = load_packet_table(input_path)
    iat_frame = build_browser_iat_frame(
        packet_table,
        trace_column=trace_column,
        timestamp_column=timestamp_column,
        packet_index_column=packet_index_column,
        max_iat_ms=max_iat_ms,
    )
    save_iat_frame(iat_frame, output_path)
    typer.echo(f"prepared {len(iat_frame)} rows")


@app.command("prepare-sensor-metrics")
def prepare_sensor_metrics(
    input_path: Path,
    output_path: Path,
) -> None:
    metric_table = load_packet_table(input_path)
    sensor_frame = build_sensor_metrics_frame(metric_table)
    save_sensor_metrics_frame(sensor_frame, output_path)
    typer.echo(f"prepared {len(sensor_frame)} sensor rows")


@app.command("prepare-cicids2017-sensor")
def prepare_cicids2017_sensor(
    input_paths: Annotated[list[Path], typer.Option("--input", "-i")],
    output_path: Annotated[Path, typer.Option("--output", "-o")],
) -> None:
    sensor_frame = build_cicids2017_sensor_metrics(input_paths)
    save_sensor_metrics_frame(sensor_frame, output_path)
    typer.echo(f"prepared {len(sensor_frame)} CIC-IDS2017-derived sensor rows")


@app.command("prepare-iscxvpn2016-sensor")
def prepare_iscxvpn2016_sensor(
    input_paths: Annotated[list[Path], typer.Option("--input", "-i")],
    output_path: Annotated[Path, typer.Option("--output", "-o")],
    positive_pattern: str = "vpn",
) -> None:
    sensor_frame = build_iscxvpn2016_sensor_metrics(
        input_paths,
        positive_pattern=positive_pattern,
    )
    save_sensor_metrics_frame(sensor_frame, output_path)
    typer.echo(f"prepared {len(sensor_frame)} ISCXVPN2016-derived sensor rows")


@app.command("init-sensor-experiment")
def init_sensor_experiment(
    run_dir: Path,
    run_id: str,
    domain: str,
    application: str,
    scenario: str,
    operator: str | None = None,
    host: str | None = None,
    interface: str | None = None,
    window_seconds: int = 5,
    baseline_seconds: float = 60.0,
    impairment_seconds: float = 120.0,
    recovery_seconds: float = 60.0,
) -> None:
    manifest = initialize_sensor_experiment_run(
        run_dir=run_dir,
        run_id=run_id,
        domain=domain,
        application=application,
        scenario=scenario,
        operator=operator,
        host=host,
        interface=interface,
        window_seconds=window_seconds,
        baseline_seconds=baseline_seconds,
        impairment_seconds=impairment_seconds,
        recovery_seconds=recovery_seconds,
    )
    typer.echo(
        f"initialized run={manifest.run_id} "
        f"manifest={run_dir / 'manifest.json'} "
        f"raw_metrics={run_dir / manifest.files.raw_metrics}"
    )


@app.command("build-sensor-experiment-dataset")
def build_sensor_experiment_dataset_command(
    input_dirs: Annotated[list[Path], typer.Option("--input", "-i")],
    output_path: Path,
    strict_coverage: bool = True,
) -> None:
    frame = build_sensor_experiment_dataset(
        input_dirs,
        output_path=output_path,
        strict_coverage=strict_coverage,
    )
    typer.echo(f"prepared {len(frame)} experiment sensor rows from {len(input_dirs)} runs")


@app.command("run-sensor-ping-experiment")
def run_sensor_ping_experiment_command(
    run_dir: Path,
    target: str,
    interface: str,
    ping_interval_seconds: float = 0.2,
    sudo: bool = False,
) -> None:
    raw_metrics_path = run_sensor_ping_experiment(
        run_dir=run_dir,
        target=target,
        interface=interface,
        ping_interval_seconds=ping_interval_seconds,
        use_sudo=sudo,
    )
    typer.echo(f"captured raw metrics at {raw_metrics_path}")


@app.command("generate-sensor-sample")
def generate_sensor_sample(
    output_path: Path,
    sessions: int = 200,
    rows_per_session: int = 20,
    seed: int = 42,
) -> None:
    frame = build_synthetic_sensor_metrics_frame(
        sessions=sessions,
        rows_per_session=rows_per_session,
        seed=seed,
    )
    save_synthetic_sensor_metrics(frame, output_path)
    typer.echo(f"generated {len(frame)} sensor rows")


@app.command("generate-controlled-sensor-domains")
def generate_controlled_sensor_domains_command(
    output_dir: Path,
    summary_path: Path | None = None,
    domains: Annotated[list[str], typer.Option("--domain")] = [],
    applications: Annotated[list[str], typer.Option("--application")] = [],
    sessions_per_application: int = 24,
    baseline_rows: int = 25,
    impairment_rows: int = 25,
    recovery_rows: int = 15,
    seed: int = 42,
) -> None:
    summary = generate_controlled_sensor_domains(
        output_dir=output_dir,
        domains=domains or None,
        applications=applications or None,
        sessions_per_application=sessions_per_application,
        baseline_rows=baseline_rows,
        impairment_rows=impairment_rows,
        recovery_rows=recovery_rows,
        seed=seed,
        summary_path=summary_path,
    )
    typer.echo(
        "generated controlled domains "
        f"domains={','.join(summary['domains'])} "
        f"applications={','.join(summary['applications'])}"
    )
    typer.echo(
        "available-controlled-domains "
        + ",".join(controlled_domain_names())
    )


@app.command("prepare-browser-iat-from-pcap")
def prepare_browser_iat_from_pcap(
    input_path: Path,
    output_path: Path,
    max_iat_ms: float = 5_000.0,
) -> None:
    iat_frame = build_iat_frame_from_archive(input_path, max_iat_ms=max_iat_ms)
    save_iat_frame(iat_frame, output_path)
    typer.echo(f"prepared {len(iat_frame)} browser iat rows")


@app.command("infer-morpher")
def infer_morpher_command(
    input_path: Path,
    model_path: Path,
    output_path: Path,
    device: str = "auto",
) -> None:
    summary = infer_morpher(
        input_path=input_path,
        model_path=model_path,
        output_path=output_path,
        requested_device=device,
    )
    typer.echo(
        f"inferred morpher rows={summary['rows']} "
        f"device={summary['device']} "
        f"sequence_length={summary['sequence_length']}"
    )


@app.command("train-morpher")
def train_morpher_command(
    input_path: Path,
    output_model: Path,
    output_metadata: Path,
    output_onnx: Path | None = None,
    sequence_length: int = 20,
    hidden_size: int = 32,
    epochs: int = 5,
    batch_size: int = 128,
    learning_rate: float = 1e-3,
    validation_ratio: float = 0.2,
    seed: int = 42,
    device: str = "auto",
) -> None:
    metadata = train_morpher(
        input_path=input_path,
        output_model_path=output_model,
        output_metadata_path=output_metadata,
        output_onnx_path=output_onnx,
        sequence_length=sequence_length,
        hidden_size=hidden_size,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        validation_ratio=validation_ratio,
        seed=seed,
        requested_device=device,
    )
    validation_metrics = metadata.get("validation_metrics")
    typer.echo(f"trained morpher examples={metadata['training_examples']} device={metadata['device']}")
    if validation_metrics:
        typer.echo(
            "validation "
            f"huber={validation_metrics['huber_loss']:.6f} "
            f"mae_ms={validation_metrics['mae_ms']:.3f} "
            f"rmse_ms={validation_metrics['rmse_ms']:.3f}"
        )


@app.command("train-sensor")
def train_sensor_command(
    input_path: Path,
    output_model: Path,
    output_metadata: Path,
    output_onnx: Path | None = None,
    algorithm: str = "lightgbm",
    threshold: float = 0.15,
    window_seconds: int = 5,
    validation_ratio: float = 0.2,
    seed: int = 42,
) -> None:
    metadata = train_sensor(
        input_path=input_path,
        output_model_path=output_model,
        output_metadata_path=output_metadata,
        output_onnx_path=output_onnx,
        algorithm=algorithm,
        threshold=threshold,
        window_seconds=window_seconds,
        validation_ratio=validation_ratio,
        seed=seed,
    )
    typer.echo(
        f"trained sensor algorithm={metadata['model_type']} "
        f"examples={metadata['training_rows']} "
        f"validation={metadata['validation_rows']}"
    )
    validation_metrics = metadata.get("validation_metrics")
    if validation_metrics:
        typer.echo(
            "validation "
            f"roc_auc={validation_metrics['roc_auc'] if validation_metrics['roc_auc'] is not None else 'n/a'} "
            f"avg_precision={validation_metrics['average_precision'] if validation_metrics['average_precision'] is not None else 'n/a'} "
            f"f1={validation_metrics['f1']:.4f} "
            f"precision={validation_metrics['precision']:.4f} "
            f"recall={validation_metrics['recall']:.4f}"
        )
    threshold_sweep = metadata.get("threshold_sweep")
    if threshold_sweep:
        best = threshold_sweep["best_by_f1"]
        typer.echo(
            "best-threshold "
            f"threshold={best['threshold']:.2f} "
            f"f1={best['f1']:.4f} "
            f"precision={best['precision']:.4f} "
            f"recall={best['recall']:.4f} "
            f"positive_rate={best['positive_rate']:.4f}"
        )


@app.command("train-sensor-multidomain")
def train_sensor_multidomain_command(
    input_paths: Annotated[list[Path], typer.Option("--input", "-i")],
    output_model: Path,
    output_metadata: Path,
    output_onnx: Path | None = None,
    algorithm: str = "lightgbm",
    threshold: float = 0.15,
    window_seconds: int = 5,
    validation_ratio: float = 0.2,
    seed: int = 42,
    balance_domains: bool = True,
) -> None:
    metadata = train_sensor_multidomain(
        input_paths=input_paths,
        output_model_path=output_model,
        output_metadata_path=output_metadata,
        output_onnx_path=output_onnx,
        algorithm=algorithm,
        threshold=threshold,
        window_seconds=window_seconds,
        validation_ratio=validation_ratio,
        seed=seed,
        balance_domains=balance_domains,
    )
    typer.echo(
        f"trained multidomain sensor algorithm={metadata['model_type']} "
        f"train={metadata['training_rows']} "
        f"validation={metadata['validation_rows']} "
        f"domains={','.join(metadata['domains'])}"
    )
    pooled = metadata["validation_metrics"]
    typer.echo(
        "validation-pooled "
        f"roc_auc={pooled['roc_auc'] if pooled['roc_auc'] is not None else 'n/a'} "
        f"avg_precision={pooled['average_precision'] if pooled['average_precision'] is not None else 'n/a'} "
        f"f1={pooled['f1']:.4f} "
        f"precision={pooled['precision']:.4f} "
        f"recall={pooled['recall']:.4f}"
    )
    macro = metadata["validation_metrics_macro"]
    typer.echo(
        "validation-macro "
        f"roc_auc={macro['roc_auc'] if macro['roc_auc'] is not None else 'n/a'} "
        f"avg_precision={macro['average_precision'] if macro['average_precision'] is not None else 'n/a'} "
        f"f1={macro['f1'] if macro['f1'] is not None else 'n/a'} "
        f"precision={macro['precision'] if macro['precision'] is not None else 'n/a'} "
        f"recall={macro['recall'] if macro['recall'] is not None else 'n/a'}"
    )
    for domain, domain_entry in metadata["validation_metrics_by_domain"].items():
        metrics = domain_entry["metrics"]
        typer.echo(
            "validation-domain "
            f"domain={domain} "
            f"roc_auc={metrics['roc_auc'] if metrics['roc_auc'] is not None else 'n/a'} "
            f"avg_precision={metrics['average_precision'] if metrics['average_precision'] is not None else 'n/a'} "
            f"f1={metrics['f1']:.4f} "
            f"precision={metrics['precision']:.4f} "
            f"recall={metrics['recall']:.4f}"
        )


@app.command("compare-sensor-models")
def compare_sensor_models_command(
    input_path: Path,
    output_summary: Path,
    algorithms: list[str] | None = None,
    threshold: float = 0.15,
    window_seconds: int = 5,
    validation_ratio: float = 0.2,
    seed: int = 42,
    max_train_rows: int | None = None,
    max_validation_rows: int | None = None,
) -> None:
    summary = compare_sensor_models(
        input_path=input_path,
        output_summary_path=output_summary,
        algorithms=algorithms or [
            "lightgbm",
            "lightgbm_sigmoid",
            "hist_gradient_boosting",
            "extra_trees",
            "random_forest",
            "logistic_regression",
        ],
        threshold=threshold,
        window_seconds=window_seconds,
        validation_ratio=validation_ratio,
        seed=seed,
        max_train_rows=max_train_rows,
        max_validation_rows=max_validation_rows,
    )
    best = summary["ranking"][0]
    best_threshold = best["threshold_sweep"]["best_by_f1"]
    typer.echo(
        f"best={best['algorithm']} "
        f"f1={best_threshold['f1']:.4f} "
        f"precision={best_threshold['precision']:.4f} "
        f"recall={best_threshold['recall']:.4f} "
        f"threshold={best_threshold['threshold']:.2f}"
    )


@app.command("evaluate-sensor-transfer")
def evaluate_sensor_transfer_command(
    train_input_path: Path,
    eval_input_path: Path,
    output_summary: Path,
    algorithm: str = "lightgbm",
    threshold: float = 0.15,
    window_seconds: int = 5,
    seed: int = 42,
    max_train_rows: int | None = None,
    max_eval_rows: int | None = None,
) -> None:
    summary = evaluate_sensor_transfer(
        train_input_path=train_input_path,
        eval_input_path=eval_input_path,
        output_summary_path=output_summary,
        algorithm=algorithm,
        threshold=threshold,
        window_seconds=window_seconds,
        seed=seed,
        max_train_rows=max_train_rows,
        max_eval_rows=max_eval_rows,
    )
    metrics = summary["eval_metrics"]
    best = summary["threshold_sweep"]["best_by_f1"]
    typer.echo(
        f"transfer algorithm={summary['algorithm']} "
        f"train_rows={summary['train_rows']} "
        f"eval_rows={summary['eval_rows']}"
    )
    typer.echo(
        "eval "
        f"roc_auc={metrics['roc_auc'] if metrics['roc_auc'] is not None else 'n/a'} "
        f"avg_precision={metrics['average_precision'] if metrics['average_precision'] is not None else 'n/a'} "
        f"f1={metrics['f1']:.4f} "
        f"precision={metrics['precision']:.4f} "
        f"recall={metrics['recall']:.4f}"
    )
    typer.echo(
        "best-threshold "
        f"threshold={best['threshold']:.2f} "
        f"f1={best['f1']:.4f} "
        f"precision={best['precision']:.4f} "
        f"recall={best['recall']:.4f} "
        f"positive_rate={best['positive_rate']:.4f}"
    )


@app.command("infer-sensor")
def infer_sensor_command(
    input_path: Path,
    model_path: Path,
    output_path: Path,
    metadata_path: Path | None = None,
    threshold: float | None = None,
    window_seconds: int = 5,
) -> None:
    summary = infer_sensor(
        input_path=input_path,
        model_path=model_path,
        output_path=output_path,
        metadata_path=metadata_path,
        threshold=threshold,
        window_seconds=window_seconds,
    )
    typer.echo(
        f"inferred sensor rows={summary['rows']} "
        f"threshold={summary['threshold']:.2f} "
        f"format={summary['model_format']}"
    )


@app.command("evaluate-sensor-loo")
def evaluate_sensor_loo_command(
    input_paths: Annotated[list[Path], typer.Option("--input", "-i")],
    output_summary: Path,
    algorithm: str = "lightgbm",
    threshold: float = 0.15,
    window_seconds: int = 5,
    seed: int = 42,
    balance_domains: bool = True,
) -> None:
    summary = evaluate_sensor_loo(
        input_paths=input_paths,
        output_summary_path=output_summary,
        algorithm=algorithm,
        threshold=threshold,
        window_seconds=window_seconds,
        seed=seed,
        balance_domains=balance_domains,
    )
    macro = summary["macro_metrics"]
    macro_best = summary["macro_best_by_f1"]
    typer.echo(
        f"loo algorithm={summary['algorithm']} "
        f"domains={','.join(summary['domains'])}"
    )
    typer.echo(
        "loo-macro "
        f"roc_auc={macro['roc_auc'] if macro['roc_auc'] is not None else 'n/a'} "
        f"avg_precision={macro['average_precision'] if macro['average_precision'] is not None else 'n/a'} "
        f"f1={macro['f1'] if macro['f1'] is not None else 'n/a'}"
    )
    typer.echo(
        "loo-macro-best "
        f"threshold={macro_best['threshold'] if macro_best['threshold'] is not None else 'n/a'} "
        f"f1={macro_best['f1'] if macro_best['f1'] is not None else 'n/a'} "
        f"precision={macro_best['precision'] if macro_best['precision'] is not None else 'n/a'} "
        f"recall={macro_best['recall'] if macro_best['recall'] is not None else 'n/a'}"
    )
    for domain_name, entry in summary["holdouts"].items():
        metrics = entry["metrics"]
        best = entry["threshold_sweep"]["best_by_f1"]
        typer.echo(
            "loo-domain "
            f"domain={domain_name} "
            f"roc_auc={metrics['roc_auc'] if metrics['roc_auc'] is not None else 'n/a'} "
            f"avg_precision={metrics['average_precision'] if metrics['average_precision'] is not None else 'n/a'} "
            f"f1={metrics['f1']:.4f} "
            f"best_f1={best['f1']:.4f} "
            f"best_threshold={best['threshold']:.2f}"
        )


if __name__ == "__main__":
    app()
