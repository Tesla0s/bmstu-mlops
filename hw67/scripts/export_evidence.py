"""Снимок фактических runs, версии реестра и отчётов для сдачи."""

import json
from pathlib import Path
import shutil

import mlflow
from mlflow import MlflowClient

from src.common import connect, load_params, save_json


def main():
    p = load_params()
    connect(p)
    client = MlflowClient()
    summary = json.loads(Path("metrics/sweep.json").read_text())
    snapshots = []
    for item in summary["rows"]:
        run = client.get_run(item["run_id"])
        snapshots.append(
            dict(
                run_id=run.info.run_id,
                name=item["name"],
                status=run.info.status,
                experiment_id=run.info.experiment_id,
                url=f"{mlflow.get_tracking_uri()}/#/experiments/{run.info.experiment_id}/runs/{run.info.run_id}",
                params=run.data.params,
                tags=run.data.tags,
                metrics=run.data.metrics,
                history={
                    key: [
                        dict(step=m.step, value=m.value, timestamp=m.timestamp)
                        for m in client.get_metric_history(run.info.run_id, key)
                    ]
                    for key in ("train_loss", "val_loss")
                },
            )
        )
    save_json("reports/mlflow-runs.json", dict(parent_id=summary["parent_id"], runs=snapshots))
    registry = client.get_registered_model(p["registry"]["name"])
    version = client.get_model_version_by_alias(p["registry"]["name"], "candidate")
    save_json(
        "reports/registry.json",
        dict(
            name=registry.name, aliases=registry.aliases, version=str(version.version), run_id=version.run_id
        ),
    )
    dest = Path("reports/evaluation")
    dest.mkdir(exist_ok=True)
    for source in sorted(Path("metrics").glob("*.json")):
        shutil.copyfile(source, dest / source.name)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 4.5), layout="constrained")
    for run in snapshots:
        h = run["history"]["val_loss"]
        ax.plot([v["step"] for v in h], [v["value"] for v in h], marker="o", label=run["name"])
    ax.set(
        xlabel="Optimizer step",
        ylabel="Validation loss (same 8 examples)",
        title="Qwen3-0.6B: six LoRA configurations",
        xticks=[0, 2, 4],
    )
    ax.grid(alpha=0.25)
    ax.legend(ncol=2)
    fig.savefig("docs/sweep-curves.png", dpi=160)
    plt.close(fig)
    print("Сохранены шесть runs, кривые, состояние реестра и все отчёты.")


if __name__ == "__main__":
    main()
