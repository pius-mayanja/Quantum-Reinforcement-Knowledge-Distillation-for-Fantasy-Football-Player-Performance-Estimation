import json
import os
import pandas as pd
import matplotlib.pyplot as plt

# Directory containing the JSON metrics files
metrics_directory = "./"

# Find all JSON files in the directory
json_files = [f for f in os.listdir(metrics_directory) if os.path.isfile(os.path.join(metrics_directory, f)) and f.endswith(".json")]

# List to store the metrics from each file
all_metrics = []

# Read metrics from each JSON file
for json_file in json_files:
    file_path = os.path.join(metrics_directory, json_file)
    try:
        with open(file_path, 'r') as f:
            metrics = json.load(f)
            metrics['File'] = os.path.splitext(json_file)[0]  # Remove .json extension
            all_metrics.append(metrics)
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON in {json_file}: {e}")

# Prepare and format data for saving
if all_metrics:
    # Convert to Pandas DataFrame
    df = pd.DataFrame(all_metrics)

    # Move 'File' column to the front if it exists
    if 'File' in df.columns:
        cols = ['File'] + [col for col in df.columns if col != 'File']
        df = df[cols]

    # Format numeric values to 2 decimal places
    for col in df.select_dtypes(include=['float64', 'int64']).columns:
        df[col] = df[col].map(lambda x: f"{x:.2f}" if pd.notnull(x) else "-")

    # Save as an image
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.axis('tight')
    ax.axis('off')
    table = ax.table(cellText=df.values, colLabels=df.columns, cellLoc='center', loc='center')

    # Adjust table appearance
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.auto_set_column_width(col=list(range(len(df.columns))))  # Adjust column width

    # Define desktop path
    desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")
    image_path = os.path.join(desktop_path, "metrics_table.png")

    plt.savefig(image_path, bbox_inches='tight', dpi=300)

    print(f"\n✅ Table saved successfully as an image:\n🖼️ {image_path}")

else:
    print("No metrics found.")




