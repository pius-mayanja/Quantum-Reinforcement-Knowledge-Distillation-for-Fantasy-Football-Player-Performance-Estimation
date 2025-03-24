import json
import os
from tabulate import tabulate

# Directory containing the JSON metrics files
metrics_directory = "./"

# Find all JSON files in the directory
json_files = [f for f in os.listdir(metrics_directory) if os.path.isfile(os.path.join(metrics_directory, f)) and f.endswith(".json")]

# List to store the metrics from each file
all_metrics = []

# Read metrics from each JSON file
for json_file in json_files:
    file_path = os.path.join(metrics_directory, json_file)
    with open(file_path, 'r') as f:
        try:
            metrics = json.load(f)
            metrics['File'] = json_file  # Add filename to metrics
            all_metrics.append(metrics)
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON in {json_file}: {e}")

# Prepare and format data for tabulate
if all_metrics:
    # Get all possible keys across files to ensure consistency
    headers = sorted(set().union(*[metrics.keys() for metrics in all_metrics]))
    
    # Ensure order: 'File' comes first, then other metrics
    headers = ['File'] + [h for h in headers if h != 'File']

    table_data = []
    for metrics in all_metrics:
        row = []
        for header in headers:
            value = metrics.get(header, "-")
            if isinstance(value, (int, float)):
                value = f"{value:,.2f}"  # Format numbers
            row.append(value)
        table_data.append(row)

    # Sort data by filename
    table_data.sort(key=lambda x: x[0])

    # Print the table
    print(tabulate(table_data, headers=headers, tablefmt="psql"))
    print(f"\nTotal metrics collected: {len(all_metrics)}")
else:
    print("No metrics found.")


