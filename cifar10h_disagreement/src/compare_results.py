import os
import pandas as pd
import matplotlib.pyplot as plt


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(PROJECT_ROOT, "outputs", "results")
PLOTS_DIR = os.path.join(PROJECT_ROOT, "outputs", "plots")

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)


def main():
    files = [
        "kl_test_results.csv",
        "jsd_test_results.csv",
        "custom_test_results.csv"
    ]

    dfs = []

    for file in files:
        path = os.path.join(RESULTS_DIR, file)

        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing file: {path}")

        dfs.append(pd.read_csv(path))

    results = pd.concat(dfs, ignore_index=True)

    save_path = os.path.join(RESULTS_DIR, "all_loss_comparison.csv")
    results.to_csv(save_path, index=False)

    print("\nCombined Results")
    print(results.to_string(index=False))

    print("\nSaved combined comparison:", save_path)

    # Main metric comparison chart
    plot_df = results[
        [
            "model",
            "test_KL_mean",
            "test_JSD_mean",
            "test_cosine_mean",
            "pearson_entropy_corr",
            "spearman_entropy_corr",
            "precision_at_100",
            "precision_at_200",
            "precision_at_500"
        ]
    ].copy()

    metrics = [
        "test_KL_mean",
        "test_JSD_mean",
        "test_cosine_mean",
        "pearson_entropy_corr",
        "spearman_entropy_corr",
        "precision_at_100",
        "precision_at_200",
        "precision_at_500"
    ]

    for metric in metrics:
        plt.figure(figsize=(8, 5))
        plt.bar(plot_df["model"], plot_df[metric])
        plt.xticks(rotation=30, ha="right")
        plt.ylabel(metric)
        plt.title(f"Comparison across losses: {metric}")
        plt.tight_layout()

        fig_path = os.path.join(PLOTS_DIR, f"comparison_{metric}.png")
        plt.savefig(fig_path, dpi=300)
        plt.close()

        print("Saved:", fig_path)

    # A compact grouped chart for the most important metrics
    important_metrics = [
        "test_KL_mean",
        "test_JSD_mean",
        "test_cosine_mean",
        "spearman_entropy_corr",
        "precision_at_500"
    ]

    normalized = plot_df[["model"] + important_metrics].copy()

    # Normalize each metric only for visualization.
    # Lower is better for KL/JSD, so invert them after normalization.
    for metric in important_metrics:
        values = normalized[metric]
        min_v = values.min()
        max_v = values.max()

        if max_v - min_v == 0:
            normalized[metric] = 1.0
        else:
            normalized[metric] = (values - min_v) / (max_v - min_v)

        if metric in ["test_KL_mean", "test_JSD_mean"]:
            normalized[metric] = 1.0 - normalized[metric]

    x = range(len(normalized["model"]))
    width = 0.15

    plt.figure(figsize=(12, 6))

    for i, metric in enumerate(important_metrics):
        positions = [p + i * width for p in x]
        plt.bar(positions, normalized[metric], width=width, label=metric)

    center_positions = [p + width * 2 for p in x]
    plt.xticks(center_positions, normalized["model"], rotation=20, ha="right")
    plt.ylabel("Normalized score, higher is better")
    plt.title("Normalized Comparison of Loss Functions")
    plt.legend()
    plt.tight_layout()

    grouped_path = os.path.join(PLOTS_DIR, "loss_function_grouped_comparison.png")
    plt.savefig(grouped_path, dpi=300)
    plt.close()

    print("Saved:", grouped_path)


if __name__ == "__main__":
    main()