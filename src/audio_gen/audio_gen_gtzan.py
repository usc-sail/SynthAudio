import json
import yaml
import torch
import scipy
import random
import torchaudio
import numpy as np
import argparse, logging
import torch.multiprocessing
import copy, time, pickle, shutil, sys, os, pdb

from tqdm import tqdm
from pathlib import Path
from diffusers import AudioLDM2Pipeline

from audiocraft.models import AudioGen
from audiocraft.models import MusicGen
from audiocraft.data.audio import audio_write

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
    gen_per_class,
    description=None
):  
    if args.model == "audioldm":
        repo_id = "cvssp/audioldm2-large"
        model = AudioLDM2Pipeline.from_pretrained(
            repo_id,
        )
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)
    elif args.model == "audiogen":
        model = AudioGen.get_pretrained('facebook/audiogen-medium')
        model.set_generation_params(duration=10)
        # model = model.to(device)
    elif args.model == "musicgen":
        model = MusicGen.get_pretrained("medium")
        model.set_generation_params(duration=10)

    
    for label in labels:
        class_name = label.lower()
        save_class_name = class_name.replace(" ", "_").replace(",", "")
        if description is not None:
            description_list = description[label]
        for j in tqdm(range(gen_per_class)):
            save_file_path = Path(args.gen_dir).joinpath(
                f'{save_class_name}_{j}.wav'
            )
            
            is_skipping = False
            if Path(save_file_path).exists(): is_skipping = True
            if is_skipping: continue
            
            
            if args.generate_method in ["class_prompt"]:
                prompt = f"{class_name} music"
                if args.model == "audioldm":
                    audio = model(
                        prompt,
                        num_inference_steps=50,
                        audio_length_in_s=10.0,
                        num_waveforms_per_prompt=1,
                    ).audios
                elif args.model == "audiogen":
                    audio = model.generate([prompt])[0]
                    audio = audio.detach().cpu().numpy()
                elif args.model == "musicgen":
                    audio = model.generate([prompt])[0].detach().cpu()
                    transform_model = torchaudio.transforms.Resample(model.sample_rate, 16000)
                    audio = transform_model(audio.detach().cpu())
                    audio = audio.detach().cpu().numpy()

            elif args.generate_method in ["llm"]:
                sampled_des = random.sample(description_list, 3)
                des = ", ".join(sampled_des)
                prompt = f"{class_name} music, {des}"
                if args.model == "audioldm":
                    audio = model(
                        prompt,
                        num_inference_steps=50,
                        audio_length_in_s=10.0,
                        num_waveforms_per_prompt=1,
                    ).audios
                elif args.model == "audiogen":
                    audio = model.generate([prompt])[0]
                    audio = audio.detach().cpu().numpy()
                elif args.model == "musicgen":
                    audio = model.generate([prompt])[0]
                    transform_model = torchaudio.transforms.Resample(model.sample_rate, 16000)
                    audio = transform_model(audio.detach().cpu())
                    audio = audio.detach().cpu().numpy()
            
            scipy.io.wavfile.write(save_file_path, rate=16000, data=audio[0])
            
            
if __name__ == '__main__':

    # Argument parser
    parser = argparse.ArgumentParser(description='Audio generation')
    parser.add_argument(
        '--gen_per_class', 
        default=150,
        type=int, 
        help='Number of generation per class'
    )
    
    parser.add_argument(
        '--generate_method', 
        default="class_prompt",
        type=str, 
        help='generate method: class_prompt, multi_domain, ucg, llm'
    )
    
    parser.add_argument(
        '--model', 
        default="musicgen",
        type=str, 
        help='generate method: audiogen musicgen audioldm'
    )
    
    parser.add_argument(
        '--dataset', 
        default="gtzan",
        type=str, 
        help='dataset name'
    )
    
    args = parser.parse_args()
    with open("../../config/config.yml", "r") as stream: config = yaml.safe_load(stream)
    args.root_dir = str(Path(config["project_dir"]).joinpath(args.dataset))
    args.gen_dir  = str(Path(config["project_dir"]).joinpath("generation", args.dataset, "audio", args.model, args.generate_method))
    args.des_dir  = str(Path(config["project_dir"]).joinpath("description", args.dataset, "data.json"))
    
    des_dict = None
    if args.generate_method == "llm":
        with open(str(Path(args.des_dir)), "r") as f: 
            des_dict = json.load(f)

    Path.mkdir(
        Path(args.gen_dir), 
        parents=True, 
        exist_ok=True
    )
    
    # Find device
    device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
    if torch.cuda.is_available(): print('GPU available, use GPU')
    
    gen_labels = [
        "blues", "classical", "country", "disco", "hiphop",
        "jazz", "metal", "pop", "reggae", "rock",
    ]
    gen_labels.sort()
    
    dataset_generate(
        gen_labels,
        args.gen_per_class,
        description=des_dict
    )