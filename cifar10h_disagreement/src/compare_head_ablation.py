import os
import pandas as pd
import matplotlib.pyplot as plt


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(PROJECT_ROOT, "outputs", "results")
PLOTS_DIR = os.path.join(PROJECT_ROOT, "outputs", "plots")

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)


def main():
    linear_path = os.path.join(RESULTS_DIR, "kl_test_results.csv")
    mlp_path = os.path.join(RESULTS_DIR, "kl_mlp_test_results.csv")

    linear_df = pd.read_csv(linear_path)
    mlp_df = pd.read_csv(mlp_path)

    linear_df["head_type"] = "Linear"
    mlp_df["head_type"] = "MLP"

    results = pd.concat([linear_df, mlp_df], ignore_index=True)

    save_path = os.path.join(RESULTS_DIR, "head_ablation_comparison.csv")
    results.to_csv(save_path, index=False)

    print("\nHead Ablation Comparison")
    print(results.to_string(index=False))
    print("\nSaved:", save_path)

    metrics = [
        "test_KL_mean",
        "test_JSD_mean",
        "test_cosine_mean",
        "spearman_entropy_corr",
        "precision_at_500"
    ]

    for metric in metrics:
        plt.figure(figsize=(7, 5))
        plt.bar(results["head_type"], results[metric])
        plt.xlabel("Prediction Head")
        plt.ylabel(metric)
        plt.title(f"Head Ablation: {metric}")
        plt.tight_layout()

        fig_path = os.path.join(PLOTS_DIR, f"head_ablation_{metric}.png")
        plt.savefig(fig_path, dpi=300)
        plt.close()

        print("Saved:", fig_path)


if __name__ == "__main__":
    main()