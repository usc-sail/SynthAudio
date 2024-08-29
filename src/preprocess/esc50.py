import os
import json
import yaml
import torch
import torchaudio
import numpy as np
import pandas as pd
import pickle, pdb, re, argparse

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

    # Parser the dataset
    parser = argparse.ArgumentParser(description='input dataset')
    parser.add_argument(
        '--dataset', 
        default='esc50',
        type=str, 
        help='dataset name'
    )
    args = parser.parse_args()
    
    # Read configs
    with open("../../config/config.yml", "r") as stream:
         config = yaml.safe_load(stream)
    data_path   = Path(config["data_dir"]["esc50"])
    split_path = Path(config["project_dir"]).joinpath("train_split")
    output_path = Path(config["project_dir"]).joinpath("audio")

    # Iterate over different splits
    Path.mkdir(output_path.joinpath(args.dataset), parents=True, exist_ok=True)
    audio_files = os.listdir(data_path.joinpath("audio"))
    for audio_file in tqdm(audio_files):
        # Read data: speaker_id, path
        if not audio_file.endswith(".wav"): continue
        file_path = data_path.joinpath("audio", audio_file)

        # Read wavforms
        waveform, sample_rate = torchaudio.load(str(file_path))

        # If the waveform has multiple channels, compute the mean across channels to create a single-channel waveform.
        if waveform.shape[0] != 1:
            waveform = torch.mean(waveform, dim=0).unsqueeze(0)

        # If the sample rate is not 16000 Hz, resample the waveform to 16000 Hz.
        if sample_rate != 16000:
            transform_model = torchaudio.transforms.Resample(sample_rate, 16000)
            waveform = transform_model(waveform)

        # Set the output path for the processed audio file based on the dataset and other information.
        output_audio_path = output_path.joinpath(args.dataset, audio_file)
        # pdb.set_trace()
        
        # Save the audio file with desired sampling frequency
        torchaudio.save(str(output_audio_path), waveform, 16000)
        