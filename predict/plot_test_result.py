import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from glob import glob
from pathlib import Path

# Set style
plt.style.use("ggplot")
sns.set_theme(style="whitegrid")

# Create plots directory
PLOTS_DIR = os.path.join(os.path.dirname(__file__), "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)


def find_test_result_folders():
    """Find all test result folders in the predict directory"""
    predict_dir = os.path.dirname(__file__)
    result_folders = [
        d
        for d in os.listdir(predict_dir)
        if os.path.isdir(os.path.join(predict_dir, d))
        and (d.endswith("_test_result") or d.startswith("results_"))
    ]

    return result_folders


def read_test_results(folder_name):
    """Read the CSV files from a test result folder"""
    folder_path = os.path.join(os.path.dirname(__file__), folder_name)

    # Check for class_metrics.csv
    class_metrics_path = os.path.join(folder_path, "class_metrics.csv")
    class_metrics = None
    if os.path.exists(class_metrics_path):
        class_metrics = pd.read_csv(class_metrics_path)

    # Check for overall_metrics.csv
    overall_metrics_path = os.path.join(folder_path, "overall_metrics.csv")
    overall_metrics = None
    if os.path.exists(overall_metrics_path):
        overall_metrics = pd.read_csv(overall_metrics_path)

    return class_metrics, overall_metrics


def clean_model_name(folder_name):
    """Extract a clean model name from the folder name"""
    name = folder_name.replace("_test_result", "").replace("results_", "")
    # Improve model name formatting for better display
    name = name.replace("densenet", "DenseNet")
    name = name.replace("efficientnet", "EfficientNet")
    # Capitalize first letter if not already done
    if not name[0].isupper():
        name = name.capitalize()
    return name


def plot_overall_metrics(results_dict):
    """Plot overall metrics comparison between models"""
    if not results_dict:
        print("No overall metrics found.")
        return

    # Prepare data for plotting
    models = []
    accuracy_values = []
    f1_values = []

    for model_name, (_, overall_df) in results_dict.items():
        if overall_df is not None:
            models.append(model_name)

            # Extract accuracy and F1 score
            try:
                accuracy = overall_df.loc[
                    overall_df["Metric"] == "Accuracy", "Value"
                ].values[0]
                f1_score = overall_df.loc[
                    overall_df["Metric"] == "F1_Score_Macro", "Value"
                ].values[0]

                accuracy_values.append(accuracy)
                f1_values.append(f1_score)
            except (KeyError, IndexError):
                # If metrics are not found, use NaN
                accuracy_values.append(np.nan)
                f1_values.append(np.nan)

    if not models:
        print("No models with overall metrics found.")
        return

    # Create bar chart
    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(models))
    width = 0.35

    ax.bar(x - width / 2, accuracy_values, width, label="Accuracy")
    ax.bar(x + width / 2, f1_values, width, label="F1 Score (Macro)")

    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_ylabel("Score")
    ax.set_title("Overall Model Performance Comparison")
    ax.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "overall_metrics_comparison.png"), dpi=300)
    plt.close()

    # Create radar chart if more than one model
    if len(models) > 1:
        # Adjust figure size based on number of models
        fig_size = 8 + (len(models) * 0.5)
        fig, ax = plt.subplots(
            figsize=(fig_size, fig_size), subplot_kw=dict(polar=True)
        )

        # Number of metrics (2 in this case: accuracy and F1)
        num_metrics = 2
        angles = np.linspace(0, 2 * np.pi, num_metrics, endpoint=False).tolist()
        angles += angles[:1]  # Close the loop

        # For each model
        for i, model in enumerate(models):
            values = [accuracy_values[i], f1_values[i]]
            values += values[:1]  # Close the loop

            ax.plot(angles, values, linewidth=2, linestyle="solid", label=model)
            ax.fill(angles, values, alpha=0.1)

        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)
        ax.set_thetagrids(np.degrees(angles[:-1]), ["Accuracy", "F1 Score"])

        ax.set_ylim(0, 1)
        ax.set_title("Model Performance Metrics", size=15)

        # Adjust legend position based on number of models
        if len(models) > 2:
            ax.legend(loc="lower right", bbox_to_anchor=(0.1, 0.1))
        else:
            ax.legend(loc="upper right", bbox_to_anchor=(0.1, 0.1))

        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, "radar_chart_comparison.png"), dpi=300)
        plt.close()

        # Add heatmap comparison if more than one model
        if len(models) > 1:
            # Create a heatmap to compare models
            metrics_data = {
                "Model": models,
                "Accuracy": accuracy_values,
                "F1 Score": f1_values,
            }
            metrics_df = pd.DataFrame(metrics_data).set_index("Model")

            plt.figure(figsize=(10, 6))
            sns.heatmap(
                metrics_df, annot=True, cmap="YlGnBu", vmin=0, vmax=1, fmt=".3f"
            )
            plt.title("Model Performance Heatmap")
            plt.tight_layout()
            plt.savefig(
                os.path.join(PLOTS_DIR, "model_heatmap_comparison.png"), dpi=300
            )
            plt.close()


def plot_class_metrics_distribution(results_dict):
    """Plot distribution of class metrics for each model"""
    if not results_dict:
        print("No class metrics found.")
        return

    for model_name, (class_df, _) in results_dict.items():
        if class_df is None:
            continue

        # Create distribution plots for each metric
        metrics = ["Accuracy", "Precision", "Recall", "F1_Score"]

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.flatten()

        for i, metric in enumerate(metrics):
            sns.histplot(class_df[metric], kde=True, ax=axes[i])
            axes[i].set_title(f"{metric} Distribution")
            axes[i].set_xlabel(metric)
            axes[i].set_ylabel("Count")

            # Add mean line
            mean_val = class_df[metric].mean()
            axes[i].axvline(
                mean_val, color="r", linestyle="--", label=f"Mean: {mean_val:.3f}"
            )
            axes[i].legend()

        plt.suptitle(f"{model_name} - Class Metrics Distribution", fontsize=16)
        plt.tight_layout()
        plt.savefig(
            os.path.join(PLOTS_DIR, f"{model_name.lower()}_metrics_distribution.png"),
            dpi=300,
        )
        plt.close()

        # Create box plots
        fig, ax = plt.subplots(figsize=(12, 8))

        df_melted = pd.melt(
            class_df,
            id_vars=["Class", "Count"],
            value_vars=metrics,
            var_name="Metric",
            value_name="Value",
        )

        sns.boxplot(x="Metric", y="Value", data=df_melted, ax=ax)
        ax.set_title(f"{model_name} - Metrics Box Plot")

        plt.tight_layout()
        plt.savefig(
            os.path.join(PLOTS_DIR, f"{model_name.lower()}_metrics_boxplot.png"),
            dpi=300,
        )
        plt.close()


def plot_top_bottom_classes(results_dict):
    """Plot top and bottom performing classes for each model"""
    if not results_dict:
        print("No class metrics found.")
        return

    for model_name, (class_df, _) in results_dict.items():
        if class_df is None:
            continue

        # Get top 10 and bottom 10 classes by F1 score
        top_classes = class_df.nlargest(10, "F1_Score")
        bottom_classes = class_df.nsmallest(10, "F1_Score")

        # Plot top classes
        fig, ax = plt.subplots(figsize=(12, 8))

        sns.barplot(x="F1_Score", y="Class", data=top_classes, ax=ax)
        ax.set_title(f"{model_name} - Top 10 Classes by F1 Score")
        ax.set_xlabel("F1 Score")
        ax.set_ylabel("Class")

        plt.tight_layout()
        plt.savefig(
            os.path.join(PLOTS_DIR, f"{model_name.lower()}_top_classes.png"), dpi=300
        )
        plt.close()

        # Plot bottom classes
        fig, ax = plt.subplots(figsize=(12, 8))

        sns.barplot(x="F1_Score", y="Class", data=bottom_classes, ax=ax)
        ax.set_title(f"{model_name} - Bottom 10 Classes by F1 Score")
        ax.set_xlabel("F1 Score")
        ax.set_ylabel("Class")

        plt.tight_layout()
        plt.savefig(
            os.path.join(PLOTS_DIR, f"{model_name.lower()}_bottom_classes.png"), dpi=300
        )
        plt.close()


def plot_metric_comparison(results_dict):
    """Plot comparison of class metrics between models"""
    if len(results_dict) <= 1:
        print("Not enough models for comparison.")
        return

    # Check if all models have class metrics
    class_dfs = [df for df, _ in results_dict.values() if df is not None]
    if not class_dfs:
        print("No class metrics found for comparison.")
        return

    # Prepare data for scatter plot
    metrics = ["Accuracy", "Precision", "Recall", "F1_Score"]
    model_names = list(results_dict.keys())

    # Ensure we have at least 2 models with class data
    if len(class_dfs) < 2:
        print("Not enough models with class data for comparison.")
        return

    # Get common classes across all models
    common_classes = set(class_dfs[0]["Class"])
    for df in class_dfs[1:]:
        common_classes &= set(df["Class"])

    if not common_classes:
        print("No common classes found across models.")
        return

    # Filter dataframes to include only common classes
    filtered_dfs = []
    for df in class_dfs:
        filtered_dfs.append(df[df["Class"].isin(common_classes)])

    # Create a comparison plot for each metric
    for metric in metrics:
        fig, ax = plt.subplots(figsize=(10, 10))

        # Get data for first two models
        df1 = filtered_dfs[0]
        df2 = filtered_dfs[1]

        # Sort by class name for consistent comparison
        df1 = df1.sort_values("Class")
        df2 = df2.sort_values("Class")

        # Create scatter plot
        plt.scatter(df1[metric], df2[metric], alpha=0.7)

        # Add diagonal line (y=x) for reference
        min_val = min(df1[metric].min(), df2[metric].min())
        max_val = max(df1[metric].max(), df2[metric].max())
        plt.plot([min_val, max_val], [min_val, max_val], "r--")

        # Add labels
        plt.xlabel(f"{model_names[0]} {metric}")
        plt.ylabel(f"{model_names[1]} {metric}")
        plt.title(
            f"Comparison of {metric} between {model_names[0]} and {model_names[1]}"
        )

        # Annotate some points (top 5 differences)
        df_combined = pd.DataFrame(
            {
                "Class": df1["Class"],
                f"{model_names[0]}": df1[metric],
                f"{model_names[1]}": df2[metric],
            }
        )
        df_combined["Diff"] = abs(
            df_combined[f"{model_names[0]}"] - df_combined[f"{model_names[1]}"]
        )

        for _, row in df_combined.nlargest(5, "Diff").iterrows():
            plt.annotate(
                row["Class"],
                (row[f"{model_names[0]}"], row[f"{model_names[1]}"]),
                xytext=(5, 5),
                textcoords="offset points",
            )

        plt.tight_layout()
        plt.savefig(
            os.path.join(PLOTS_DIR, f"comparison_{metric.lower()}.png"), dpi=300
        )
        plt.close()

        # If we have more than 2 models, create additional comparison plots
        if len(class_dfs) > 2:
            # Create pairwise comparisons for all model combinations
            model_pairs = [
                (i, j)
                for i in range(len(model_names))
                for j in range(i + 1, len(model_names))
            ]

            for idx1, idx2 in model_pairs:
                # Skip the first pair as it's already plotted above
                if idx1 == 0 and idx2 == 1:
                    continue

                df1 = filtered_dfs[idx1]
                df2 = filtered_dfs[idx2]

                # Sort by class name for consistent comparison
                df1 = df1.sort_values("Class")
                df2 = df2.sort_values("Class")

                # Create scatter plot
                fig, ax = plt.subplots(figsize=(10, 10))
                plt.scatter(df1[metric], df2[metric], alpha=0.7)

                # Add diagonal line (y=x) for reference
                min_val = min(df1[metric].min(), df2[metric].min())
                max_val = max(df1[metric].max(), df2[metric].max())
                plt.plot([min_val, max_val], [min_val, max_val], "r--")

                # Add labels
                plt.xlabel(f"{model_names[idx1]} {metric}")
                plt.ylabel(f"{model_names[idx2]} {metric}")
                plt.title(
                    f"Comparison of {metric} between {model_names[idx1]} and {model_names[idx2]}"
                )

                # Annotate some points (top 5 differences)
                df_combined = pd.DataFrame(
                    {
                        "Class": df1["Class"],
                        f"{model_names[idx1]}": df1[metric],
                        f"{model_names[idx2]}": df2[metric],
                    }
                )
                df_combined["Diff"] = abs(
                    df_combined[f"{model_names[idx1]}"]
                    - df_combined[f"{model_names[idx2]}"]
                )

                for _, row in df_combined.nlargest(5, "Diff").iterrows():
                    plt.annotate(
                        row["Class"],
                        (row[f"{model_names[idx1]}"], row[f"{model_names[idx2]}"]),
                        xytext=(5, 5),
                        textcoords="offset points",
                    )

                plt.tight_layout()
                plt.savefig(
                    os.path.join(
                        PLOTS_DIR,
                        f"comparison_{metric.lower()}_{model_names[idx1].lower()}_{model_names[idx2].lower()}.png",
                    ),
                    dpi=300,
                )
                plt.close()


def main():
    """Main function to execute all plotting"""
    # Find all test result folders
    result_folders = find_test_result_folders()

    if not result_folders:
        print("No test result folders found.")
        return

    print(f"Found {len(result_folders)} test result folders: {result_folders}")

    # Read results from each folder
    results_dict = {}
    for folder in result_folders:
        class_metrics, overall_metrics = read_test_results(folder)

        # Only include folders with at least one type of metrics
        if class_metrics is not None or overall_metrics is not None:
            model_name = clean_model_name(folder)
            results_dict[model_name] = (class_metrics, overall_metrics)

    if not results_dict:
        print("No valid test results found in any folder.")
        return

    # Create all plots
    print("Creating plots...")
    plot_overall_metrics(results_dict)
    plot_class_metrics_distribution(results_dict)
    plot_top_bottom_classes(results_dict)
    plot_metric_comparison(results_dict)

    print(f"All plots saved to {PLOTS_DIR}")


if __name__ == "__main__":
    main()
