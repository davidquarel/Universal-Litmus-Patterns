# %%
import glob
import os
import csv
import sys
from tqdm import tqdm
import dq
from dataclasses import dataclass, asdict
# %%

@dataclass
class Args:
    model_path: str = None # "models/VGG_blending_alpha_25/models_pt"
    metadata_path: str = "models/VGG_blending_alpha_25/metadata/metadata_blending25.csv"
    out_dir: str = "models/VGG_blending_alpha_25/models_pt_good"
    out_metadata: str = "models/VGG_blending_alpha_25/metadata/metadata_blending25_good.csv"
    clean_thresh: float = 0.77
    poison_thresh: float = 0.99

# %%

# read csv of metadata
args = Args()
args = dq.parse_args_with_default(Args, args)


# Example usage:
# data = read_csv_to_dict('path_to_your_csv_file.csv', 'model_name')
# print(data['VGG_CIFAR-10_0901_0000'].train_loss)  # Accessing the train loss of the specified model
data = dq.csv_reader(args.metadata_path, 'model_name')
model_paths = sorted(glob.glob(f'{args.model_path}/*.pt'))
# %%

os.makedirs(args.out_dir, exist_ok=True)

with open(args.out_metadata, 'w') as f:
    header = ",".join(list(list(data.items())[0][1]._fields))
    f.write(f"{header}\n")
    cleans = []
    poisons = []
    runner = tqdm(model_paths)
    copied = 0
    for model_path in runner:
        model_basename = os.path.basename(model_path).split('.')[0]
        clean_acc = float(data[model_basename].clean_acc)
        poisoned_acc = float(data[model_basename].poisoned_acc)
        
        if clean_acc > args.clean_thresh and poisoned_acc > args.poison_thresh:
            copied += 1
            # copy model into outdir using os library
            os.system(f"cp {model_path} {args.out_dir}/")
            runner.set_description(f"model {model_basename} copied {copied} clean_acc: {clean_acc:.4f}, poisoned_acc: {poisoned_acc:.4f}")
            # write metadata to csv
            row = ",".join(list(data[model_basename]))
            f.write(f"{row}\n")
    

# %%
# cleans = []
# poisons = []

# for model_path in model_paths:
#     model_basename = os.path.basename(model_path).split('.')[0]
#     clean_acc = data[model_basename].clean_acc
#     poisoned_acc = data[model_basename].poisoned_acc
#     cleans.append(float(clean_acc))
#     poisons.append(float(poisoned_acc))
    
# cleans = np.array(cleans)
# poisons = np.array(poisons)    

# %%

# # histogram
# import matplotlib.pyplot as plt
# import numpy as np
# plt.hist(cleans, bins=20)
# plt.title("Clean Accuracies")
# plt.show()
# plt.hist(poisons, bins=100, range=(0.8, 1))
# plt.title("Poisoned Accuracies")
# plt.show()

# # %%

# # plot cumulative distribution
# import matplotlib.pyplot as plt
# import numpy as np
# plt.hist(cleans, bins=20, cumulative=True, density=True)
# plt.title("Clean Accuracies")
# plt.show()
# plt.hist(poisons, bins=100, range=(0.9, 1), cumulative=True, density=True)
# plt.title("Poisoned Accuracies")
# plt.show()




    
    
# %%