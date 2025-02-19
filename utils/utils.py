import torch
import math
import time, copy, os, json
import xgboost as  xgb
import numpy as np
from torch.nn import DataParallel
from torch.nn.parallel import DistributedDataParallel as DDP

def torch_load_cpu(load_path):
    return torch.load(load_path, map_location=lambda storage, loc: storage)  # Load on CPU

def get_inner_model(model):
    return model.module if isinstance(model, DataParallel) or isinstance(model, DDP) else model

def set_seed(seed=None):
    if seed is None:
        seed=int(time.time())
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)

def move_to(var, device):
    if isinstance(var, dict):
        return {k: move_to(v, device) for k, v in var.items()}
    return var.to(device)

def move_to_cuda(var, device):
    if isinstance(var, dict):
        return {k: move_to(v, device) for k, v in var.items()}
    return var.cuda(device)

def clip_grad_norms(param_groups, max_norm=math.inf):
    """
    Clips the norms for all param groups to max_norm and returns gradient norms before clipping
    :param optimizer:
    :param max_norm:
    :param gradient_norms_log:
    :return: grad_norms, clipped_grad_norms: list with (clipped) gradient norms per group
    """
    grad_norms = [
        torch.nn.utils.clip_grad_norm(
            group['params'],
            max_norm if max_norm > 0 else math.inf,  # Inf so no clipping but still call to calc
            norm_type=2
        )
        for idx, group in enumerate(param_groups)
    ]
    grad_norms_clipped = [min(g_norm, max_norm) for g_norm in grad_norms] if max_norm > 0 else grad_norms
    return grad_norms, grad_norms_clipped


def get_surrogate_gbest(env,dataset,bs,seed,skip_step):
    print('getting surrogate gbest...')
    gbests=[]
    set_seed(seed)
    for pro in dataset:
        action=[{'problem':copy.deepcopy(pro),'sgbest':0} for i in range(bs)]
        env.step(action)
        env.reset()
        is_done=False
        while not is_done:
            action=[{'skip_step':skip_step} for i in range(bs)]
            pop,_,is_done,_=env.step(action)
            is_done=is_done.all()
        # ! change 
        gbest_list=[p.gbest_cost for p in pop]
        gbests.append(np.min(gbest_list))
    print('done...')
    return np.array(gbests)

def get_data(mode,surrogates_dir,root_dir):
    train_set,vali_set,test_set=None,None,None
    if mode == "v3-test":
        train_set,vali_set,test_set,bo_initializations=load_data(root_dir, only_test=True)
    elif mode == "v3-train-augmented":
        train_set,vali_set,test_set,bo_initializations=load_data(root_dir, only_test=False, augmented_train=True)
    elif mode in ["v1", "v2", "v3"]:
        train_set,vali_set,test_set,bo_initializations=load_data(root_dir, version = mode, only_test=False)
    else:
        raise ValueError("Provide a valid mode")

    surrogates_file = surrogates_dir+"summary-stats.json"
    if os.path.isfile(surrogates_file):
        with open(surrogates_file) as f:
            surrogates_stats = json.load(f)

    return train_set,vali_set,test_set,bo_initializations,surrogates_stats

def load_data( rootdir="", version = "v3", only_test = True, augmented_train = False):
    
        """
        Loads data with some specifications.
        Inputs:
            * root_dir: path to directory with the benchmark data.
            * version: name indicating what HPOB version to use. Options: v1, v2, v3).
            * Only test: Whether to load only testing data (valid only for version v3).  Options: True/False
            * augmented_train: Whether to load the augmented train data (valid only for version v3). Options: True/False

        """

        print("Loading data...")
        meta_train_augmented_path = os.path.join(rootdir, "meta-train-dataset-augmented.json")
        meta_train_path = os.path.join(rootdir, "meta-train-dataset.json")
        meta_test_path = os.path.join(rootdir,"meta-test-dataset.json")
        meta_validation_path = os.path.join(rootdir, "meta-validation-dataset.json")
        bo_initializations_path = os.path.join(rootdir, "bo-initializations.json")

        with open(meta_test_path, "rb") as f:
            meta_test_data = json.load(f)

        with open(bo_initializations_path, "rb") as f:
            bo_initializations = json.load(f)

        meta_train_data = None
        meta_validation_data = None
        
        if not only_test:
            if augmented_train or version=="v1":
                with open(meta_train_augmented_path, "rb") as f:
                    meta_train_data = json.load(f)
            else:
                with open(meta_train_path, "rb") as f:
                    meta_train_data = json.load(f)
            with open(meta_validation_path, "rb") as f:
                meta_validation_data = json.load(f)

        if version != "v3":
            temp_data = {}
            for search_space in meta_train_data.keys():
                temp_data[search_space] = {}

                for dataset in meta_train_data[search_space].keys():
                    temp_data[search_space][dataset] =  meta_train_data[search_space][dataset]

                if search_space in meta_test_data.keys():
                    for dataset in meta_test_data[search_space].keys():
                        temp_data[search_space][dataset] = meta_test_data[search_space][dataset]

                    for dataset in meta_validation_data[search_space].keys():
                        temp_data[search_space][dataset] = meta_validation_data[search_space][dataset]

            meta_train_data = None
            meta_validation_data = None
            meta_test_data = temp_data

        search_space_dims = {}

        for search_space in meta_test_data.keys():
            dataset = list(meta_test_data[search_space].keys())[0]
            X = meta_test_data[search_space][dataset]["X"][0]
            search_space_dims[search_space] = len(X)

        return meta_train_data,meta_validation_data,meta_test_data,bo_initializations


def get_bst(surrogates_dir,search_space_id,dataset_id,surrogates_stats):
    surrogate_name='surrogate-'+search_space_id+'-'+dataset_id
    bst_surrogate = xgb.Booster()
    bst_surrogate.load_model(surrogates_dir+surrogate_name+'.json')

    y_min = surrogates_stats[surrogate_name]["y_min"]
    y_max = surrogates_stats[surrogate_name]["y_max"]
    assert y_min is not None, 'y_min is None!!'

    return bst_surrogate,y_min,y_max
