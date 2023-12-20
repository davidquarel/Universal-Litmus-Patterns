import dq
from dataclasses import dataclass, asdict
from train_ULP_dq import generate_test_ensembles, init_ULP_and_meta_classifier, init_models_train, main
import torch
import wandb


@dataclass
class Train_Config:
    wandb_project: str = "ULP-CIFAR10-ood3"
    wandb: bool = True
    wandb_name : str = None
    wandb_desc : str = None
    #====================================
    epochs: int = 20
    clean_train_dir : str = "models/VGG_replicate/clean/trainval"
    clean_test_dir : str = None 
    poison_train_dir : str = "models/VGG_good_blended_alpha_25/models_pt"
    poison_test_dir : str = "models/VGG_good_blended_alpha_25/models_pt"
    _ood_test_dirs : str = None
    
    # clean_dir : str = "new_models/clean/models_pt/*.pt"
    # poison_train_dir : str = "new_models/poison_train/models_pt/*.pt"
    # poison_test_dir : str = "new_models/poison_test/models_pt/*.pt"
    num_train : int = 3000
    num_test : int = 500
    #====================================
    acc_thresh : float = -1 #dummy value model will always exceed
    epoch_thresh : int = 0
    num_ulps: int = 10
    meta_lr : float = 1e-3
    ulp_lr : float = 1e2 #WTF LR=100?
    ulp_scale : float = 1
    tv_reg : float = 1e-6
    meta_bs : int = 100
    sigmoid_no_clip : bool = False #if true, sigmoid the ULP and do not clip to [0,1]
    grad_clip_threshold: float = None  # Set a default value
    hyper_param_search: bool = False
    cache_dataset : bool = False #preload entire dataset into GPU memory
    #====================================
    _model_ext : str = ".pt"
    _debug : bool = False
    _poison_name : str = None


# %%

def experiments(cfg):

    desc = "Experimenting with seeing how well ULPs trained on one poison generalize to other poisons"

    clean_train_dir = "models/VGG_replicate/clean/trainval"

    poisons = {'ulp_train':  "models/VGG_replicate/poison/trainval",
                'ulp_test' : "models/VGG_replicate/poison/test",
                'blending' : "models/VGG_good_blended_alpha_25/models_pt",
                'badnets' : "models/VGG_badnet/models_pt",
                'badnets_random' : "models/VGG_badnets_random/models_pt"}
    
    test_ensembles_exists = False
    
    poison_name, poison_dir = 'blending', poisons['blending']
    
    for run in range(1000):
        
        print(f"Training on {poison_name}")
        
        cfg = Train_Config(clean_train_dir=clean_train_dir, 
                            poison_train_dir=poison_dir, 
                            _ood_test_dirs=poisons,
                            wandb=cfg.wandb,
                            _poison_name=poison_name,
                            wandb_project = f"ULP-CIFAR10-ood3")
                            #wandb_name = f"{poison_name}-{run}")
        
        models_train_paths, labels_train, clean_models_test, _ = init_models_train(cfg)
        
        if not test_ensembles_exists:
            test_ensembles = generate_test_ensembles(clean_models_test, poisons_models=poisons, cfg=cfg)
            test_ensembles_exists = True
        
        
        print(f"Config: {asdict(cfg)}")
        torch.cuda.empty_cache()

        if cfg.wandb:
            wandb.init(project=cfg.wandb_project, config = asdict(cfg), 
                    notes = desc, name=cfg.wandb_name)
            
        ULPs, meta_classifier = init_ULP_and_meta_classifier(cfg)
        
        models_train = dq.batch_load_models(models_train_paths)
        main(ULPs, meta_classifier, models_train, labels_train, test_ensembles, cfg=cfg)
        dq.clear_model_memory(models_train)
    
# %%
cfg  =Train_Config(wandb=False)
experiments(cfg)
# %%