import os
import json
import matplotlib.pyplot as plt
import argparse

def plot_metrics_from_folders(base_dir, output_dir=None):
    # Define metrics to plot
    training_metrics = [
        "training.loss.autoencoder.mean",
        "training.loss.mean",
        "training.loss.topo_error.mean",
        "training.metrics.loss.mean",
        "training.reconstruction_error.mean"
    ]
    validation_metrics = [
        "validation.loss",
        "validation.loss.autoencoder",
        "validation.loss.topo_error",
        "validation.metrics.loss",
        "validation.reconstruction_error"
    ]
    all_metrics = training_metrics + validation_metrics

    # Find all subfolders with metrics.json
    runs = []
    for root, dirs, files in os.walk(base_dir):
        if "metrics.json" in files:
            runs.append(os.path.join(root, "metrics.json"))

    # Load all runs
    run_data = []
    for path in runs:
        with open(path, "r") as f:
            data = json.load(f)
        run_data.append((os.path.basename(os.path.dirname(path)), data))

    # For each metric, plot all runs
    for metric in all_metrics:
        plt.figure(figsize=(8, 5))
        for run_name, data in run_data:
            # Check if metric exists in data
            if metric in data:
                metric_data = data[metric]
                if isinstance(metric_data, dict) and "steps" in metric_data and "values" in metric_data:
                    steps = metric_data["steps"]
                    values = metric_data["values"]
                    if steps and values:
                        plt.plot(steps, values, label=run_name)
        plt.xlabel("Step")
        plt.ylabel(metric)
        plt.yscale('log')  # Set y-axis to logarithmic scale
        plt.grid(True, linestyle='--', color='lightgray', alpha=0.7)  # Add dashed light gray grid
        plt.title(f"{metric} over steps")
        plt.legend()
        plt.tight_layout()
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            out_path = os.path.join(output_dir, f"{metric.replace('.', '_')}.png")
        else:
            out_path = f"{metric.replace('.', '_')}.png"
        plt.savefig(out_path)
        plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot metrics from multiple runs")
    parser.add_argument("input_dir", help="Directory containing subdirectories with metrics.json files")
    parser.add_argument("-o", "--output", help="Output directory for plots (default: plots in input_dir)", default=None)
    
    args = parser.parse_args()
    
    # If no output directory specified, create plots subdirectory in input directory
    if args.output is None:
        output_dir = os.path.join(args.input_dir, "plots")
    else:
        output_dir = args.output
    
    plot_metrics_from_folders(args.input_dir, output_dir=output_dir)
