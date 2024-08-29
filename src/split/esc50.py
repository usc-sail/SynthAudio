import json
import yaml
import numpy as np
import pandas as pd
import pickle, pdb, re

from tqdm import tqdm
from pathlib import Path

# define logging console
import logging
logging.basicConfig(
    format='%(asctime)s %(levelname)-3s ==> %(message)s', 
    level=logging.INFO, 
    datefmt='%Y-%m-%d %H:%M:%S'
)

if __name__ == '__main__':

    # Read data path
    dataset = "esc50"
    with open("../../config/config.yml", "r") as stream:
        config = yaml.safe_load(stream)
    data_path   = Path(config["data_dir"][dataset])
    output_path = Path(config["project_dir"])

    meta_df = pd.read_csv(data_path.joinpath("meta", "esc50.csv"))
    for fold_idx in range(1, 6):
        Path.mkdir(output_path.joinpath('train_split'), parents=True, exist_ok=True)
        train_list, test_list = list(), list()

        for idx in range(len(meta_df)):
            test_fold = meta_df.fold.values[idx]
            filename = meta_df.filename.values[idx]
            category = meta_df.category.values[idx]
            label = meta_df.target.values[idx]
            
            file_path = Path(output_path).joinpath(
                "audio", dataset, filename
            )
            
            file_data = [filename, category, str(file_path), int(label)]
            if test_fold == fold_idx: test_list.append(file_data)
            else: train_list.append(file_data)
        
        return_dict = dict()
        return_dict['train'], return_dict['test'] = train_list, test_list
        logging.info(f'-------------------------------------------------------')
        logging.info(f'Split distribution for ESC50 dataset')
        for split in ['train', 'test']: logging.info(f'Split {split}: Number of files {len(return_dict[split])}')
        logging.info(f'-------------------------------------------------------')
        
        # dump the dictionary
        jsonString = json.dumps(return_dict, indent=4)
        jsonFile = open(str(output_path.joinpath('train_split', f'{dataset}_fold{fold_idx}.json')), "w")
        jsonFile.write(jsonString)
        jsonFile.close()

    