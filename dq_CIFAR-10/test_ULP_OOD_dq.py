# %%
import wandb
from dataclasses import dataclass, asdict
from tqdm import tqdm
import glob
from collections import namedtuple
import pandas as pd
import re
import numpy as np
import torch
import dq
from einops.layers.torch import Rearrange
import torch.nn as nn
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import einops
# %%
"""
Check how well the best trained ULP/meta-models perform on OOD base models.
meta-classifier trained on base models poisoned in same way as from ULP paper
testing on OOD base models poisoned using a blending attack
"""

# %%
def read_ulp_metadata(filepath = 'models/david_ULPs/metadata.csv'):
    # Load the CSV file
    df = pd.read_csv(filepath)

    # Replace invalid characters in column names for the namedtuple
    column_names = [col for col in df.columns if col != 'Name']
    valid_column_names = [re.sub(r'\W|^(?=\d)', '_', col) for col in column_names]
    RunData = namedtuple('RunData', valid_column_names)

    # Function to extract run ID from the "Name" column
    def extract_run_id(name):
        match = re.search(r'\d+$', name)
        return int(match.group()) if match else None

    # Create the dictionary using namedtuple
    run_data_dict = {}
    for _, row in df.iterrows():
        run_id = extract_run_id(row['Name'])
        if run_id is not None:
            run_data = RunData(**{re.sub(r'\W|^(?=\d)', '_', col): row[col] for col in column_names})
            run_data_dict[run_id] = run_data
    return run_data_dict
    
def extract_good_runs(run_data_dict, metric_name="test_acc", metric_threshold=0.99):
    good_runs = []
    for run_id, run_data in run_data_dict.items():
        if run_data.test_acc >= metric_threshold:
            good_runs.append(run_id)
    return good_runs


metadata = read_ulp_metadata()
good_runs = extract_good_runs(metadata)

# %%

@dataclass
class Train_Config:
    wandb_project: str = "ULP-CIFAR10"
    clean_test_dir : str = "models/VGG_replicate/clean/trainval/*.pt"
    poison_test_dir : str = "models/VGG_replicate/poison/test/*.pt"
    badnets_test_dir : str = "models/VGG_badnet/models_pt/*.pt"
    blending_test_dir : str = "models/VGG_blending_alpha_25/models_pt/*.pt"
    
    ulp_dir : str = "models/david_ULPs/ULPs"
    meta_dir : str = "models/david_ULPs/meta"
    num_test : int = 2000
    num_ulps : int = 10

cfg = Train_Config()

# %%

# measure accuracy on hold out test set

def load_raw_set(file_path):
    raw_models = sorted(glob.glob(file_path))
    assert len(raw_models) > 0, "No models found"
    print(f"Found {len(raw_models)} models")
    
    
    
    return raw_models

clean_models_test_all = load_raw_set(cfg.clean_test_dir)
poisoned_models_test_all = load_raw_set(cfg.poison_test_dir)
badnets_models_test_all = load_raw_set(cfg.badnets_test_dir)
blending_models_test_all = load_raw_set(cfg.blending_test_dir)
# %%

def gen_ensemble(model_list, label = 0):
    truncated = np.array(model_list[-cfg.num_test:])
    labels = torch.zeros(cfg.num_test, dtype=torch.long, device=device) + label
    weights = dq.batch_load_models(truncated)
    ensemble = dq.Ensemble(*weights).to(device)
    ensemble.eval()
    return ensemble, labels

clean_ensemble, labels_clean = gen_ensemble(clean_models_test_all, label=0)
poisoned_ensemble, labels_poisoned = gen_ensemble(poisoned_models_test_all, label=1)
badnets_ensemble, labels_badnets = gen_ensemble(badnets_models_test_all, label=1)
blending_ensemble, labels_blending = gen_ensemble(blending_models_test_all, label=1)

ensembles = [clean_ensemble, poisoned_ensemble, badnets_ensemble, blending_ensemble]
labels = [labels_clean, labels_poisoned, labels_badnets, labels_blending]
names = ["clean", "poisoned", "badnets", "blending"]

def evaluate_correctness(ULPs, meta_classifier, ensembles, labels, names):
    
    scores = {}
    
    with torch.no_grad():
        for ensemble, label, name in zip(ensembles, labels, names):
            model_logits = ensemble(ULPs, average=False, split=False)
            meta_logits = meta_classifier(model_logits)
            y_guess = torch.argmax(meta_logits, dim=1)
            correct = torch.sum(y_guess == label).item() / len(y_guess)
            scores[name] = correct
    return scores

# %%
# 25e0ttv0

def evaluate_correctness_no_meta(ULPs, ensembles, labels, names):
        
        scores = {}
        
        with torch.no_grad():
            for ensemble, label, name in zip(ensembles, labels, names):
                meta_logits = ensemble(ULPs, average=False, split=False) #(100, ULP, 10)
                meta_logits = einops.reduce(meta_logits, 'b u c -> b c', 'mean')
                
                y_guess = torch.argmax(meta_logits, dim=-1)
                correct = torch.sum(y_guess == label).item() / len(y_guess)
                scores[name] = correct
                
        return scores

cfg = Train_Config()

def score_ULP(run_id, ulp_dir = cfg.ulp_dir, meta_dir = cfg.meta_dir, num_ulps = cfg.num_ulps, meta = True):
    base_path_ulp = f"{cfg.ulp_dir}/ULPs_{run_id}.pth"
    base_path_meta = f"{cfg.meta_dir}/meta_classifier_{run_id}.pth"
    
    ULPs = torch.load(base_path_ulp , map_location=device)
    
    meta_classifier = nn.Sequential(
        Rearrange('b u c -> b (u c)'),
        nn.Linear(num_ulps * dq.cnn_cfg.nofclasses, 2)
    ).to(device)
    meta_classifier.load_state_dict(torch.load(base_path_meta, map_location=device))
    
    if meta:
        scores = evaluate_correctness(ULPs, meta_classifier, ensembles, labels, names)
    else:
        scores = evaluate_correctness_no_meta(ULPs, poisoned_ensemble, ensembles, labels, names)
    return scores

# %%

ULPs_5 = torch.load("ULP_interrogate/ULPs_5.pt", map_location=device).data
#score_ULP(ULPs_5, meta=False)
# %%
ulp_paths = sorted(glob.glob(f"{cfg.ulp_dir}/ULPs_*.pth"))
meta_paths = sorted(glob.glob(f"{cfg.meta_dir}/meta_classifier_*.pth"))

all_scores = []

for ulp, meta in tqdm(zip(ulp_paths, meta_paths)):
    ulp_id = ulp.split("_")[-1].split(".")[0]
    meta_id = meta.split("_")[-1].split(".")[0]
    assert ulp_id == meta_id, "ULP and meta classifier IDs do not match"
    all_scores.append(score_ULP(ulp_id))
    
# %%
    
scores_df = pd.DataFrame(all_scores)    
# save scores_df as csv
scores_df.to_csv("models/david_ULPs/ULP_OOD_scores.csv")


# %%
