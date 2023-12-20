# %%
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import sys

def create_corrected_histograms_with_kde(data, title, file_name):
    fig, axs = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(title, fontsize=16)

    sns.histplot(data, bins=30, kde=True, ax=axs[0])
    axs[0].set_title('Full Range Histogram with KDE')
    axs[0].set_xlabel('Accuracy')
    axs[0].set_ylabel('Frequency')

    sns.histplot(data, bins=20, kde=True, ax=axs[1], binrange=(0.9, 1.0))
    axs[1].set_title('Focused Range Histogram (0.9 to 1) with KDE')
    axs[1].set_xlabel('Accuracy')
    axs[1].set_ylabel('Frequency')
    axs[1].set_xlim(0.9, 1.0)

    for ax in axs:
        stats = data.describe()
        textstr = '\n'.join((
            f'Count: {stats["count"]:.0f}',
            f'Mean: {stats["mean"]:.3f}',
            f'Std: {stats["std"]:.3f}',
            f'Min: {stats["min"]:.3f}',
            f'25%: {stats["25%"]:.3f}',
            f'50% (Median): {stats["50%"]:.3f}',
            f'75%: {stats["75%"]:.3f}',
            f'Max: {stats["max"]:.3f}'
        ))
        ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=12,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.savefig(file_name + '.svg', format='svg')

def main(csv_file):
    df = pd.read_csv(csv_file, header=None, names=['model_name', 'accuracy'])

    trainval_data = df[df['model_name'].str.contains('poisoned_models/trainval')]
    test_data = df[df['model_name'].str.contains('poisoned_models/test')]

    create_corrected_histograms_with_kde(trainval_data['accuracy'], 'Corrected Histograms with KDE for poisoned_models/trainval Models', 'poison_trainval_histo2')
    create_corrected_histograms_with_kde(test_data['accuracy'], 'Corrected Histograms with KDE for poisoned_models/test Models', 'poison_test_histo2')

csv_file = "results/model_poisonacc_testloss2.csv"
main(csv_file)
# %%

# if __name__ == "__main__":
#     if len(sys.argv) != 2:
#         print("Usage: python script.py <path_to_csv_file>")
#         sys.exit(1)

#     csv_file = sys.argv[1]
#     main(csv_file)
