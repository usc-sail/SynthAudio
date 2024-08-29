import json
import yaml
import torch
import random
import numpy as np
import pandas as pd
import torch.nn as nn
import argparse, logging
import torch.multiprocessing
import copy, time, pickle, shutil, sys, os, pdb

from tqdm import tqdm
from pathlib import Path
from copy import deepcopy
from collections import defaultdict, deque
from torch.optim.lr_scheduler import ReduceLROnPlateau


sys.path.append(os.path.join(str(Path(os.path.realpath(__file__)).parents[1])))
sys.path.append(os.path.join(str(Path(os.path.realpath(__file__)).parents[1]), 'models'))
sys.path.append(os.path.join(str(Path(os.path.realpath(__file__)).parents[1]), 'utils'))
sys.path.append(os.path.join(str(Path(os.path.realpath(__file__)).parents[1]), 'dataloader'))

from utils import parse_finetune_args, set_seed, log_epoch_result, log_best_result

# from utils
from ast_models import ASTModel
from evaluation import EvalMetric
from dataloader import load_finetune_audios, load_gen_audios, set_finetune_dataloader, return_weights


# define logging console
import logging
logging.basicConfig(
    format='%(asctime)s %(levelname)-3s ==> %(message)s', 
    level=logging.INFO, 
    datefmt='%Y-%m-%d %H:%M:%S'
)

os.environ["MKL_NUM_THREADS"] = "1" 
os.environ["NUMEXPR_NUM_THREADS"] = "1" 
os.environ["OMP_NUM_THREADS"] = "1" 

def train_epoch(
    dataloader, 
    model, 
    device, 
    optimizer,
    weights
):
    model.train()
    criterion = nn.CrossEntropyLoss(weights).to(device)
    eval_metric = EvalMetric()
    
    logging.info(f'-------------------------------------------------------------------')
    for batch_idx, batch_data in enumerate(dataloader):
        # read data
        model.zero_grad()
        optimizer.zero_grad()
        x, y = batch_data
        x, y = x.to(device), y.to(device)
        
        # forward pass
        outputs = model(x, task="ft_avgtok")
        
        # backward
        loss = criterion(outputs, y)
        loss.backward()
        
        # clip gradients
        optimizer.step()
        
        eval_metric.append_classification_results(y, outputs, loss)
        
        if (batch_idx % 10 == 0 and batch_idx != 0) or batch_idx == len(dataloader) - 1:
            result_dict = eval_metric.classification_summary()
            logging.info(f'Fold {fold_idx} - Current Train Loss at epoch {epoch}, step {batch_idx+1}/{len(dataloader)}: {result_dict["loss"]:.3f}')
            logging.info(f'Fold {fold_idx} - Current Train ACC at epoch {epoch}, step {batch_idx+1}/{len(dataloader)}: {result_dict["acc"]:.2f}%')
            logging.info(f'Fold {fold_idx} - Current Train LR at epoch {epoch}, step {batch_idx+1}/{len(dataloader)}: {scheduler.optimizer.param_groups[0]["lr"]}')
            logging.info(f'-------------------------------------------------------------------')
    
    logging.info(f'-------------------------------------------------------------------')
    result_dict = eval_metric.classification_summary()
    return result_dict
    
def validate_epoch(
    dataloader, 
    model, 
    device,
    weights,
    split:  str="Validation"
):
    model.eval()
    criterion = nn.CrossEntropyLoss(weights).to(device)
    eval_metric = EvalMetric()
    with torch.no_grad():
        for batch_idx, batch_data in enumerate(dataloader):
            # read data
            x, y = batch_data
            x, y = x.to(device), y.to(device)
            
            # forward pass
            outputs = model(x, task="ft_avgtok")
                     
            # backward
            loss = criterion(outputs, y)
            eval_metric.append_classification_results(y, outputs, loss)
        
            if (batch_idx % 50 == 0 and batch_idx != 0) or batch_idx == len(dataloader) - 1:
                result_dict = eval_metric.classification_summary()
                logging.info(f'Fold {fold_idx} - Current {split} Loss at epoch {epoch}, step {batch_idx+1}/{len(dataloader)} {result_dict["loss"]:.3f}')
                logging.info(f'Fold {fold_idx} - Current {split} ACC at epoch {epoch}, step {batch_idx+1}/{len(dataloader)} {result_dict["acc"]:.2f}%')
                logging.info(f'-------------------------------------------------------------------')
    logging.info(f'-------------------------------------------------------------------')
    result_dict = eval_metric.classification_summary()
    scheduler.step(result_dict["loss"])
    return result_dict


if __name__ == '__main__':

    # Argument parser
    args = parse_finetune_args()
    with open("../../config/config.yml", "r") as stream: config = yaml.safe_load(stream)
    args.split_dir  = str(Path(config["project_dir"]).joinpath("train_split"))
    args.data_dir   = str(Path(config["project_dir"]).joinpath("generation", args.dataset, "audio"))
    args.log_dir    = str(Path(config["project_dir"]).joinpath("finetune"))

    # Find device
    device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
    if torch.cuda.is_available(): print('GPU available, use GPU')
    
    best_dict = dict()
    result_hist_dict = dict()
    
    if args.dataset == "esc50": total_folds = 4
    # Read train/dev file list
    train_file_list, _, test_file_list = load_finetune_audios(
        args.split_dir, dataset=args.dataset, fold_idx=1
    )
    
    label_dict = dict()
    for data_info in train_file_list:
        label_dict[data_info[1]] = data_info[-1]

    # We perform cross-folds experiments
    # for fold_idx in range(1, total_folds):
    train_file_list = load_gen_audios(
        Path(args.data_dir), 
        label_dict, 
        gen_method=args.gen_method, 
        num_gen=args.gen_per_class,
        gen_model=args.gen_model
    )

    # Read weights of training data
    weights = return_weights(
        args.split_dir, dataset=args.dataset, fold_idx=1
    )
    
    # Set train/dev/test dataloader
    train_dataloader = set_finetune_dataloader(
        args, train_file_list, is_train=True, freqm=24, timem=96
    )
    
    # Define log dir
    log_dir = Path(args.log_dir).joinpath(
        f"gen-{args.dataset}", 
        args.pretrain_model,
        args.gen_model,
        args.gen_method,
        f"{args.gen_per_class}",
        f"{args.setting}"
    )
    Path.mkdir(log_dir, parents=True, exist_ok=True)
    
    # Define the model wrapper
    if args.pretrain_model == "ssast":
        model = ASTModel(
            label_dim=50,
            fshape=16, tshape=16, fstride=10, tstride=10,
            input_fdim=128, input_tdim=512, model_size='base',
            pretrain_stage=False, load_pretrained_mdl_path='../models/SSAST-Base-Patch-400.pth', 
            finetune_method="ft_avgtok"
        ).to(device)
    # Read trainable params
    model_parameters = list(filter(lambda p: p.requires_grad, model.parameters()))
    params = sum([np.prod(p.size()) for p in model_parameters])
    logging.info(f'Trainable params size: {params/(1e6):.2f} M')
    
    # Define optimizer
    optimizer = torch.optim.Adam(
        list(filter(lambda p: p.requires_grad, model.parameters())),
        lr=args.learning_rate, 
        weight_decay=1e-4,
        betas=(0.9, 0.98)
    )
    # Define scheduler, patient = 5, minimum learning rate 5e-5
    scheduler = ReduceLROnPlateau(
        optimizer, mode='min', patience=5, factor=0.5, verbose=True, min_lr=1e-5
    )

    # Training steps
    for fold_idx in range(1, 6):
        best_dict[fold_idx] = dict()
        best_dict[fold_idx]["acc"] = 0
        best_dict[fold_idx]["epoch"] = 0
        result_hist_dict[fold_idx] = dict()

    for epoch in range(args.num_epochs):
        train_result = train_epoch(
            train_dataloader, model, device, optimizer, weights
        )
        
        # Read train/dev file list
        for fold_idx in range(1, 6):
            _, _, test_file_list = load_finetune_audios(
                args.split_dir, dataset=args.dataset, fold_idx=fold_idx
            )

            test_dataloader = set_finetune_dataloader(
                args, test_file_list, is_train=False
            )
            test_result = validate_epoch(
                test_dataloader, model, device, weights, split="Test"
            )
            # if we get a better results
            if best_dict[fold_idx]["acc"] < test_result["acc"]:
                best_dict[fold_idx]["acc"] = test_result["acc"]
                best_dict[fold_idx]["epoch"] = epoch
                torch.save(model.state_dict(), str(log_dir.joinpath(f'fold_{fold_idx}.pt')))
            
            logging.info(f'-------------------------------------------------------------------')
            logging.info(f"Fold {fold_idx} - Best train epoch {best_dict[fold_idx]['epoch']}, best test ACC {best_dict[fold_idx]['acc']:.2f}%")
            logging.info(f'-------------------------------------------------------------------')
            
            # log the current result
            log_epoch_result(result_hist_dict[fold_idx], epoch, train_result, test_result, test_result, log_dir, fold_idx)

            # log the best results
            log_best_result(result_hist_dict[fold_idx], epoch, best_dict[fold_idx]["epoch"], best_dict[fold_idx]["epoch"], best_dict[fold_idx]["epoch"], best_dict[fold_idx]["epoch"], log_dir, fold_idx)

        # save best results
        jsonString = json.dumps(best_dict, indent=4)
        jsonFile = open(str(log_dir.joinpath(f'results.json')), "w")
        jsonFile.write(jsonString)
        jsonFile.close()

    acc_list = [best_dict[fold_idx]["acc"] for fold_idx in best_dict]
    best_dict["average"] = dict()
    best_dict["average"]["acc"] = np.mean(acc_list)
    
    best_dict["std"] = dict()
    best_dict["std"]["acc"] = np.std(acc_list)
    
    # save best results
    jsonString = json.dumps(best_dict, indent=4)
    jsonFile = open(str(log_dir.joinpath(f'results.json')), "w")
    jsonFile.write(jsonString)
    jsonFile.close()