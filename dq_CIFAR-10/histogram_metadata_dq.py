# %%
import glob
import os
import csv
import sys
from tqdm import tqdm
import dq
from dataclasses import dataclass, asdict
import numpy as np
import matplotlib.pyplot as plt
# %%

def summary_stats(data, threshold):
    min_val = np.min(data)
    max_val = np.max(data)
    mean_val = np.mean(data)
    std_val = np.std(data)
    num_over_thresh = np.sum(data > threshold)
    num_models = len(data)
    frac_above_thresh = f"{num_over_thresh}/{num_models}"  # Raw, unsimplified fraction
    dec_above_thresh = num_over_thresh / num_models  # Decimal fraction
    return f"Min: {min_val:.2f}\n\
            Max: {max_val:.2f}\n\
            Mean: {mean_val:.2f}\n\
            Std: {std_val:.2f}\n\
            Frac > {threshold}: {frac_above_thresh}={dec_above_thresh:.4f}"

def read_and_plot(args):
    model_name = args.model_path.split('/')[1]
    data = dq.csv_reader(args.metadata_path, 'model_name')
    model_paths = sorted(glob.glob(f'{args.model_path}/*.pt'))
    assert len(model_paths) == len(data), f"Found {len(model_paths)} models but {len(data)} metadata entries"

    #os.makedirs(args.out_dir, exist_ok=True)
    cleans = []
    poisons = []
    runner = tqdm(model_paths)
    copied = 0
    for model_path in runner:
        model_basename = os.path.basename(model_path).split('.')[0]
        clean_acc = float(data[model_basename].clean_acc)
        poisoned_acc = float(data[model_basename].poisoned_acc)
        
        cleans.append(float(clean_acc))
        poisons.append(float(poisoned_acc))

    cleans = np.array(cleans)
    poisons = np.array(poisons)

    fig, axs = plt.subplots(2, 2, figsize=(10, 10))

    # Plot 1: Clean Histogram
    axs[0, 0].hist(cleans, bins=100)
    axs[0, 0].set_title(f"Clean {model_name}")
    axs[0, 0].annotate(summary_stats(cleans, args.clean_thresh), xy=(0.05, 0.75), xycoords='axes fraction')

    # Plot 2: Poison Histogram
    axs[0, 1].hist(poisons, bins=100)
    axs[0, 1].set_title(f"Poison {model_name}")
    axs[0, 1].annotate(summary_stats(poisons, args.poison_thresh), xy=(0.05, 0.75), xycoords='axes fraction')

    # Plot 3: Clean CDF
    axs[1, 0].hist(cleans, bins=100, cumulative=True, density=True)
    axs[1, 0].set_title(f"Clean {model_name} CDF")
    axs[1, 0].annotate(summary_stats(cleans, args.clean_thresh), xy=(0.05, 0.75), xycoords='axes fraction')

    # Plot 4: Poison CDF
    axs[1, 1].hist(poisons, bins=100, cumulative=True, density=True)
    axs[1, 1].set_title(f"Poison {model_name} CDF")
    axs[1, 1].annotate(summary_stats(poisons, args.poison_thresh), xy=(0.05, 0.75), xycoords='axes fraction')

    metadata_parent = os.path.dirname(args.metadata_path)
    print(metadata_parent)
    clean_good = cleans > args.clean_thresh
    poison_good = poisons > args.poison_thresh
    good_models = np.logical_and(clean_good, poison_good)
    print(f"Found {np.sum(good_models)}/{len(good_models)} for {model_name}")

    plt.tight_layout()
    plt.savefig(os.path.join(metadata_parent, f"{model_name}_histogram.png"))
    plt.show()

    return data

    # plt.hist(cleans, bins=20)
    # plt.title(f"Clean {model_name}")
    # plt.show()
    # plt.hist(poisons, bins=100, range=(0.8, 1))
    # plt.title(f"Poison {model_name}")
    # plt.show()

    # # %%
    # plt.hist(cleans, bins=20, cumulative=True, density=True)
    # plt.title(f"Clean {model_name} cdf")
    # plt.show()
    # plt.hist(poisons, bins=100, range=(0.9, 1), cumulative=True, density=True)
    # plt.title(f"poison {model_name} cdf")
    # plt.show()

# %%

paths = ['models/VGG_replicate/poison/trainval',
         'models/VGG_replicate/poison/test',
         'models/VGG_blending_alpha_25/models_pt',
         'models/VGG_badnet/models_pt',
         'models/VGG_badnets_random/models_pt',
         'models/VGG_blended_run/models_pt',
         'models/VGG_blended_run/models_pt_good',
         'models/VGG_blending_alpha_25/models_pt_good']

metadatas = ['models/VGG_replicate/metadata_poison_trainval.csv',
                'models/VGG_replicate/metadata_poison_test.csv',
                'models/VGG_blending_alpha_25/metadata/metadata_blending25.csv',
                'models/VGG_badnet/metadata/metadata_badnets.csv',
                'models/VGG_badnets_random/metadata/metadata_badnets_random.csv',
                'models/VGG_blended_run/metadata/metadata_blended_run.csv',
                'models/VGG_blended_run/metadata/metadata_blended_run_good.csv',
                'models/VGG_blending_alpha_25/metadata/metadata_blending25_good.csv']

@dataclass
class Args:
    model_path: str = "models/VGG_replicate/poison/trainval"
    metadata_path: str = "models/VGG_replicate/metadata_poison_trainval.csv"
    # out_dir: str = "models/VGG_blending_alpha_25/models_pt_good"
    clean_thresh: float = 0.77
    poison_thresh: float = 0.99

# %%

for path, metadata in zip(paths, metadatas):
    print(f"Plotting {path}...")
    args = Args(path, metadata)
    read_and_plot(args)


    
    
# %%