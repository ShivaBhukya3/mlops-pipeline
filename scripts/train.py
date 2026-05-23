#!/usr/bin/env python3
"""CLI entry point for the training pipeline."""

import logging
import sys
from pathlib import Path

import click

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.ingestion import ingest_data, load_raw_data
from src.data.preprocessing import split_data, preprocess, save_splits, load_splits, prepare_xy
from src.data.validation import DataValidator
from src.models.trainer import ModelTrainer
from src.models.registry import ModelRegistry

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


@click.group()
def cli():
    """MLOps Fraud Detection Training Pipeline."""


@cli.command()
@click.option("--force", is_flag=True, help="Re-generate data even if it exists")
@click.option("--config", default="config/config.yaml")
def ingest(force: bool, config: str):
    """Ingest raw data."""
    path = ingest_data(config_path=config, force=force)
    click.echo(f"Data available at: {path}")


@cli.command()
@click.option("--config", default="config/config.yaml")
def preprocess_cmd(config: str):
    """Preprocess & split data."""
    raw = load_raw_data("data/raw/transactions.parquet")
    validator = DataValidator()
    passed = validator.validate(raw)
    if not passed:
        click.echo(click.style("WARNING: Data validation issues found!", fg="yellow"))
        click.echo(validator.summary().to_string())

    train, val, test = split_data(raw)
    X_train, y_train = prepare_xy(train)
    X_val, y_val = prepare_xy(val)

    from sklearn.preprocessing import StandardScaler
    _, scaler = __import__("src.data.preprocessing", fromlist=["preprocess"]).preprocess(X_train, fit_scaler=True)

    save_splits(train, val, test, scaler=scaler)
    click.echo(f"Preprocessing complete: train={len(train)}, val={len(val)}, test={len(test)}")


@cli.command()
@click.option("--run-name", default=None)
@click.option("--config", default="config/config.yaml")
@click.option("--promote", is_flag=True, help="Auto-promote to Staging if AUC > 0.95")
def train(run_name, config, promote):
    """Train model and log to MLflow."""
    import yaml
    with open(config) as f:
        cfg = yaml.safe_load(f)

    train_df, val_df, test_df, scaler = load_splits()
    X_train, y_train = prepare_xy(train_df)
    X_val, y_val = prepare_xy(val_df)
    X_test, y_test = prepare_xy(test_df)

    trainer = ModelTrainer(config_path=config)
    model, metrics, run_id = trainer.train(
        X_train, y_train, X_val, y_val, X_test, y_test, run_name=run_name
    )

    click.echo(click.style("\nTraining Results:", fg="green", bold=True))
    for k, v in metrics.items():
        click.echo(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    if promote and metrics.get("roc_auc", 0) > 0.95:
        registry = ModelRegistry(
            tracking_uri=cfg["mlflow"].get("tracking_uri", "mlruns"),
            model_name=cfg["mlflow"]["model_name"],
        )
        version = registry.get_latest_version(stage="None")
        if version:
            registry.promote_to_staging(version)
            click.echo(click.style(f"Model v{version} promoted to Staging!", fg="cyan"))


@cli.command()
@click.option("--version", required=True)
@click.option("--config", default="config/config.yaml")
def promote(version: str, config: str):
    """Promote a model version to Production."""
    import yaml
    with open(config) as f:
        cfg = yaml.safe_load(f)
    registry = ModelRegistry(
        tracking_uri=cfg["mlflow"].get("tracking_uri", "mlruns"),
        model_name=cfg["mlflow"]["model_name"],
    )
    registry.promote_to_production(version)
    click.echo(click.style(f"Model v{version} is now in Production!", fg="green", bold=True))


if __name__ == "__main__":
    cli()
