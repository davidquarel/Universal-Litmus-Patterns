# %%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys
# Load the data
file_path = "results/model_poisonacc_testloss.csv"  # Replace with your file path

try:
    save_path = sys.argv[2]
except:
    save_path = None
    
    
df = pd.read_csv(file_path)

# Rename columns for clarity
df.columns = ['model_name', 'accuracy']

# Function to categorize models
def categorize_model(model_name):
    if 'clean_models/trainval' in model_name:
        return 'clean_models/trainval'
    elif 'clean_models/test' in model_name:
        return 'clean_models/test'
    elif 'poisoned_models/trainval' in model_name:
        return 'poisoned_models/trainval'
    elif 'poisoned_models/test' in model_name:
        return 'poisoned_models/test'
    else:
        return 'unknown'

# Categorize the models
df['category'] = df['model_name'].apply(categorize_model)

# Calculate summary statistics for each category
category_stats = df.groupby('category')['accuracy'].agg(['mean', 'median', 'std', 'min', 'max'])

# Function to annotate plots with statistics
def annotate_stats(ax, stats):
    textstr = '\n'.join((
        f"Mean: {stats['mean']:.4f}",
        f"Median: {stats['median']:.4f}",
        f"Std Dev: {stats['std']:.4f}",
        f"Min: {stats['min']:.4f}",
        f"Max: {stats['max']:.4f}"))

    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=10, verticalalignment='top', bbox=props)

# Set plot style
sns.set(style="whitegrid")

# Create figure for plots
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.ravel()

# Define categories for plotting
categories = ['poisoned_models/test', 'poisoned_models/trainval']

# Plot histograms for each category
for i, category in enumerate(categories):
    ax = axes[i]
    data = df[df['category'] == category]['accuracy']
    sns.histplot(data, bins=30, kde=True, ax=ax)
    ax.set_title(category)
    ax.set_xlim(0.5, 1)
    ax.set_xlabel('Accuracy')
    ax.set_ylabel('Frequency')
    
    # Annotate with statistics
    annotate_stats(ax, category_stats.loc[category])

# Adjust layout
plt.tight_layout()

# Save plot as SVG
if save_path is not None:
    format = 'svg' if save_path.endswith('.svg') else 'png'
    plt.savefig(save_path, format=format)
# %%