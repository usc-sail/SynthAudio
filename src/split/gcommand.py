import os
import pdb
import tqdm
import yaml
import json
import argparse
import numpy as np

from pathlib import Path
from typing import Tuple

# define logging console
import logging
logging.basicConfig(
    format='%(asctime)s %(levelname)-3s ==> %(message)s', 
    level=logging.INFO, 
    datefmt='%Y-%m-%d %H:%M:%S'
)

def get_classes():
    CLASSES = [
        "backward",
        "bed",
        "bird",
        "cat",
        "dog",
        "down",
        "eight",
        "five",
        "follow",
        "forward",
        "four",
        "go",
        "happy",
        "house",
        "learn",
        "left",
        "marvin",
        "nine",
        "no",
        "off",
        "on",
        "one",
        "right",
        "seven",
        "sheila",
        "six",
        "stop",
        "three",
        "tree",
        "two",
        "up",
        "visual",
        "wow",
        "yes",
        "zero"
    ]
    classes = CLASSES
    weight = None
    class_to_id = {label: i for i, label in enumerate(classes)}
    return classes, weight, class_to_id

def data_partition(data_path, output_path, dataset):
    
    # Fetch all labels
    label_list, weight, label_to_id = get_classes()
    class_num = len(label_list)
    
    # Read dev and test split
    dev_file_path = Path(data_path).joinpath("validation_list.txt")
    test_file_path = Path(data_path).joinpath("testing_list.txt")
    
    dev_file_list, test_file_list = list(), list()
    with open(dev_file_path, 'r') as f:
        # Read contents of file line by line
        lines = f.readlines()
        # Print each line
        for line in lines: dev_file_list.append(line.split("\n")[0])
    with open(test_file_path, 'r') as f:
        # Read contents of file line by line
        lines = f.readlines()
        # Print each line
        for line in lines: test_file_list.append(line.split("\n")[0])
            
    train_data_list, dev_data_list, test_data_list = list(), list(), list()
    # Create data_dict => {key: [key, file_path, label]}
    for label in label_list:
        label_file_list = os.listdir(Path(data_path).joinpath(label))
        for label_file in label_file_list:
            key = f"{label}/{label_file}"
            file_path = str(Path(data_path).joinpath(label, label_file))

            file_data = [key, label, str(file_path), label_to_id[label]]
            if key in test_file_list: test_data_list.append(file_data)
            elif key in dev_file_list: dev_data_list.append(file_data)
            else: train_data_list.append(file_data)

    return_dict = dict()
    return_dict['train'], return_dict['dev'], return_dict['test'] = train_data_list, dev_data_list, test_data_list
    logging.info(f'-------------------------------------------------------')
    logging.info(f'Split distribution for SpeechCommands dataset')
    for split in ['train', 'dev', 'test']: logging.info(f'Split {split}: Number of files {len(return_dict[split])}')
    logging.info(f'-------------------------------------------------------')
    
    # dump the dictionary
    jsonString = json.dumps(return_dict, indent=4)
    jsonFile = open(str(output_path.joinpath('train_split', f'{dataset}.json')), "w")
    jsonFile.write(jsonString)
    jsonFile.close()


if __name__ == "__main__":
    # Read data path
    dataset = "speechcommands"
    with open("../../config/config.yml", "r") as stream:
        config = yaml.safe_load(stream)
    data_path   = Path(config["data_dir"][dataset])
    output_path = Path(config["project_dir"])
    data_partition(data_path, output_path, dataset)
    