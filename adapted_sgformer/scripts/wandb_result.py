import argparse
import wandb
import numpy as np


def get_results(project, run_name, entity=None):
    api = wandb.Api()

    project_path = f"{entity}/{project}" if entity else project
    runs = api.runs(project_path)

    # Exact match on run name
    selected_runs = [
        run for run in runs
        if run.name == run_name
    ]

    if len(selected_runs) == 0:
        print(f'No runs found with name "{run_name}"')
        return

    print(f"\n{run_name} — {len(selected_runs)} runs")

    metrics = [
        "test/metric/mAP",
        "test/metric/mAP_50",
        "test/metric/mAP_75",
        "test/metric/mAP_L",
        "test/metric/mAP_M",
        "test/metric/mAP_S",
    ]

    results = {metric: [] for metric in metrics}

    for run in selected_runs:
        for metric in metrics:
            value = run.summary.get(metric)

            if value is not None:
                results[metric].append(value)

    print("\nMean ± std:")

    for metric in metrics:
        values = results[metric]

        if len(values) == 0:
            print(f"{metric:<25} No data")
            continue

        values = np.asarray(values)

        mean = np.mean(values) * 100
        std = np.std(values) * 100

        print(
            f"{metric:<25} "
            f"{mean:.2f} ± {std:.2f} %"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--name",
        type=str,
        required=True,
        help="Exact W&B run name"
    )

    parser.add_argument(
        "--project",
        type=str,
        default="low_latency-ncaltech101-detection",
        help="W&B project name"
    )

    parser.add_argument(
        "--entity",
        type=str,
        default="kevin-hoarau-cole-normale-sup-rieure-paris-saclay",
        help="W&B entity/team name"
    )

    args = parser.parse_args()

    get_results(
        project=args.project,
        run_name=args.name,
        entity=args.entity,
    )