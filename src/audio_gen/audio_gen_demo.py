import json
import yaml
import torch
import scipy
import numpy as np
import argparse, logging
import torch.multiprocessing
import copy, time, pickle, shutil, sys, os, pdb

from tqdm import tqdm
from pathlib import Path
from diffusers import AudioLDM2Pipeline


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


def dataset_generate(
    labels,
    gen_per_class
):
    repo_id = "cvssp/audioldm2-large"
    pipe = AudioLDM2Pipeline.from_pretrained(
        repo_id,
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe = pipe.to(device)
    
    for label in labels:
        class_name = label.lower()
        
        for j in tqdm(range(gen_per_class)):
            save_file_path = Path(args.gen_dir).joinpath(
                f'{label}_{j}.wav'
            )
            
            is_skipping = False
            if Path(save_file_path).exists(): is_skipping = True
            if is_skipping: continue
            
            if args.generate_method in ["class_prompt"]:
                prompt = f"a sound of {class_name}"
                audio = pipe(
                    prompt,
                    num_inference_steps=50,
                    audio_length_in_s=10.0,
                    num_waveforms_per_prompt=1,
                ).audios
            scipy.io.wavfile.write(save_file_path, rate=16000, data=audio[0])
            
            
if __name__ == '__main__':

    # Argument parser
    parser = argparse.ArgumentParser(description='Audio generation')
    parser.add_argument(
        '--gen_per_class', 
        default=10,
        type=int, 
        help='Number of generation per class'
    )
    
    parser.add_argument(
        '--generate_method', 
        default="class_prompt",
        type=str, 
        help='generate method: class_prompt, multi_domain, ucg'
    )
    
    parser.add_argument(
        '--dataset', 
        default="esc50",
        type=str, 
        help='dataset name'
    )
    
    args = parser.parse_args()
    with open("../../config/config.yml", "r") as stream: config = yaml.safe_load(stream)
    args.root_dir = str(Path(config["project_dir"]).joinpath(args.dataset))
    args.gen_dir  = str(Path(config["project_dir"]).joinpath(args.dataset, "generation", "audio", args.generate_method))
    
    Path.mkdir(
        Path(args.gen_dir), 
        parents=True, 
        exist_ok=True
    )
    
    # Find device
    device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
    if torch.cuda.is_available(): print('GPU available, use GPU')
    
    gen_labels = [
        "Helicopter", "Chainsaw", "Siren", "Car horn", "Engine", 
        "Train", "Church bells", "Airplane", "Fireworks", "Hand saw"
    ]
    gen_labels.sort()
    
    # pdb.set_trace()
    dataset_generate(
        gen_labels,
        args.gen_per_class,
    )