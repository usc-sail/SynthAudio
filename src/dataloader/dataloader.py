import pdb
import json
import copy
import torch
import torchaudio
import numpy as np

from pathlib import Path
from collections import Counter
from torch.utils.data import DataLoader, Dataset

from audiomentations import AddBackgroundNoise, PolarityInversion


ucf101_dict = {
    "BasketballDunk": "Dunking Basketball",
    "Bowling": "Bowling",
    "BoxingPunchingBag": "Boxing with Punching Bag",
    "SoccerPenalty": "Performing Soccer Penalty",
    "TableTennisShot": "Table Tennis",
    "BandMarching": "Band Marching",
    "PlayingCello": "Playing Cello",
    "PlayingDhol": "Playing Dhol",
    "PlayingFlute": "Playing Flute",
    "PlayingSitar": "Playing Sitar",
    "BlowDryHair": "Blowing Dry Hair",
    "BlowingCandles": "Blowing Candles",
    "BrushingTeeth": "Brushing Teeth",
    "CuttingInKitchen": "Cutting In Kitchen",
    "Haircut": "Haircut",
    "Hammering": "Hammering",
    "Knitting": "Knitting",
    "ShavingBeard": "Shaving Beard",
    "Typing": "Typing",
    "WritingOnBoard": "Writing On Board",
}


activitynet_dict = {

    # Outdoor
    "Horseback+riding": "Horseback+riding",
    "Swimming": "Swimming",
    "Tennis+serve+with+ball+bouncing": "doing Tennis+serve+with+ball+bouncing",
    "Riding+bumper+cars": "Riding+bumper+cars",
    "Skateboarding": "Skateboarding",

    # Indoor
    "Playing+badminton": "Playing+badminton",
    "Playing+drums": "Playing+drums",
    "Playing+guitarra": "Playing+guitarra",
    "Playing+pool": "Playing+pool",
    "Volleyball": "Playing Volleyball",

    # Chores and Activity
    "Chopping+wood": "Chopping wood",
    "Hand+washing+clothes": "Hand+washing+clothes",
    "Sharpening+knives": "Sharpening+knives",
    "Vacuuming+floor": "Vacuuming+floor",
    "Washing+dishes": "Washing+dishes",

    # Personal Care and Grooming
    "Blow-drying+hair": "Blow-drying hair",
    "Brushing+teeth": "Brushing teeth",
    "Getting+a+haircut": "Getting+a+haircut",
    "Shaving": "Shaving",
    "Gargling+mouthwash": "Gargling+mouthwash",
}

esc_top_list = [
    "clapping", "brushing_teeth", "pig", "crackling_fire", "pouring_water", 
    "siren", "clock_tick", "fireworks", "can_opening", "sneezing",
    "hand_saw", "breathing", "clock_alarm", "rain", "rooster",
    "thunderstorm", "sea_waves", "engine", "crow", "footsteps"
]


def preemphasis(signal,coeff=0.97):
    """perform preemphasis on the input signal.
    :param signal: The signal to filter.
    :param coeff: The preemphasis coefficient. 0 is none, default 0.97.
    :returns: the filtered signal.
    """
    return np.append(signal[0],signal[1:]-coeff*signal[:-1])

def padding_cropping(
    input_wav, size
):
    if len(input_wav) > size:
        input_wav = input_wav[:size]
    elif len(input_wav) < size:
        input_wav = torch.nn.ConstantPad1d(padding=(0, size - len(input_wav)), value=0)(input_wav)
    return input_wav

def collate_fn(batch):
    # max of 6s of data
    max_audio_len = min(max([b[0].shape[0] for b in batch]), 16000*10)
    data, len_data, taregt = list(), list(), list()
    for idx in range(len(batch)):
        # append data
        data.append(padding_cropping(batch[idx][0], max_audio_len))
        
        # append len
        if len((batch[idx][0])) >= max_audio_len: len_data.append(torch.tensor(max_audio_len))
        else: len_data.append(torch.tensor(len((batch[idx][0]))))
        
        # append target
        taregt.append(torch.tensor(batch[idx][1]))
    
    data = torch.stack(data, dim=0)
    len_data = torch.stack(len_data, dim=0)
    taregt = torch.stack(taregt, dim=0)
    return data, taregt, len_data

class EmotionDatasetGenerator(Dataset):
    def __init__(
        self,
        data_list:              list,
        data_len:               int,
        is_train:               bool=False,
        audio_duration:         int=10,
        noise_audios:           list=None,
        model_type:             str="rnn",
        test_db:                int=5
    ):
        """
        Set dataloader for emotion recognition finetuning.
        :param data_list:       Audio list files
        :param noise_list:      Audio list files
        :param data_len:        Length of input audio file size
        :param is_train:        Flag for dataloader, True for training; False for dev
        :param audio_duration:  Max length for the audio length
        :param model_type:      Type of the model
        """
        self.data_list              = data_list
        self.noise_audios           = noise_audios
        self.data_len               = data_len
        self.is_train               = is_train
        self.audio_duration         = audio_duration
        self.model_type             = model_type
        
        if self.noise_audios is not None:
            if is_train:
                self.transform = AddBackgroundNoise(
                    sounds_path=noise_audios,
                    min_snr_in_db=3.0,
                    max_snr_in_db=30.0,
                    noise_transform=PolarityInversion(),
                    p=1.0
                )
            else:
                self.transform = AddBackgroundNoise(
                    sounds_path=noise_audios,
                    min_snr_in_db=test_db,
                    max_snr_in_db=test_db,
                    noise_transform=PolarityInversion(),
                    p=1.0
                )


    def __len__(self):
        return self.data_len

    def __getitem__(
        self, item
    ):
        # Read original speech in dev
        data, _ = torchaudio.load(self.data_list[item][3])
        data = data[0]
        if data.isnan()[0].item(): data = torch.zeros(data.shape)
        if len(data) > self.audio_duration*16000: data = data[:self.audio_duration*16000]
        if self.noise_audios is not None:
            data = data.detach().cpu().numpy()
            data = self.transform(samples=data, sample_rate=16000)
            data = torch.tensor(data)
        
        return data, self.data_list[item][-1]

    def _padding_cropping(
        self, input_wav, size
    ):
        if len(input_wav) > size:
            input_wav = input_wav[:size]
        elif len(input_wav) < size:
            input_wav = torch.nn.ConstantPad1d(padding=(0, size - len(input_wav)), value=0)(input_wav)
        return input_wav

class AudioDatasetGenerator(Dataset):
    def __init__(
        self,
        data_list:                          list,
        data_len:                           int,
        is_train:                           bool=False,
        target_len:                         int=512,
        freqm:                              int=None,
        timem:                              int=None,
        noise_audios:                       list=None,
        test_db:                            int=5
    ):
        """
        Set dataloader for emotion recognition finetuning.
        :param data_list:               Audio list files
        :param data_len:                Length of input audio file size
        :param is_train:                Flag for dataloader, True for training; False for dev
        :param audio_duration:          Max length for the audio length
        """
        self.data_list              = data_list
        self.data_len               = data_len
        self.is_train               = is_train
        self.target_length          = target_len
        self.freqm                  = freqm
        self.timem                  = timem
        self.noise_audios           = noise_audios

        if self.noise_audios is not None:
            if is_train:
                self.transform = AddBackgroundNoise(
                    sounds_path=noise_audios,
                    min_snr_in_db=3.0,
                    max_snr_in_db=30.0,
                    noise_transform=PolarityInversion(),
                    p=1.0
                )
            else:
                self.transform = AddBackgroundNoise(
                    sounds_path=noise_audios,
                    min_snr_in_db=test_db,
                    max_snr_in_db=test_db,
                    noise_transform=PolarityInversion(),
                    p=1.0
                )

    def __len__(self):
        return self.data_len
    
    def _wav2fbank(self, filename):
        waveform, sr = torchaudio.load(filename)
        if self.noise_audios is not None: 
            waveform = torch.tensor(self.transform(waveform.detach().numpy()[0], 16000)).unsqueeze(0)
        waveform = waveform - waveform.mean()

        # pdb.set_trace()
        fbank = torchaudio.compliance.kaldi.fbank(
            waveform, htk_compat=True, sample_frequency=sr, use_energy=False,
            window_type='hanning', num_mel_bins=128, dither=0.0, frame_shift=10
        )

        # self.target_length = self.targ
        n_frames = fbank.shape[0]
        p = self.target_length - n_frames

        # cut and pad
        if p > 0:
            m = torch.nn.ZeroPad2d((0, 0, 0, p))
            fbank = m(fbank)
        elif p < 0: fbank = fbank[0:self.target_length, :]
        if self.target_length == 1000:
            fbank = (fbank - -2.4507537) / (2.7830634 * 2)
        elif self.target_length == 128:
            fbank = (fbank - -6.845978) / (5.5654526 * 2)
        else:
            fbank = (fbank - -6.6268077) / (5.358466 * 2)
        return fbank

    def __getitem__(self, item):
        # Read original speech
        # print(self.data_list[item][-2])
        audio_data = self._wav2fbank(self.data_list[item][-2])
        if self.is_train and self.freqm is not None:
        # SpecAug, not do for eval set
            freqm = torchaudio.transforms.FrequencyMasking(self.freqm)
            timem = torchaudio.transforms.TimeMasking(self.timem)
            audio_data = torch.transpose(audio_data, 0, 1)
            # this is just to satisfy new torchaudio version.
            audio_data = audio_data.unsqueeze(0)
            if self.freqm != 0:
                audio_data = freqm(audio_data)
            if self.timem != 0:
                audio_data = timem(audio_data)
            # squeeze back
            audio_data = audio_data.squeeze(0)
            audio_data = torch.transpose(audio_data, 0, 1)
        
        return audio_data, self.data_list[item][-1]

def include_for_finetune(
    data: list, dataset: str
):
    """
    Return flag for inlusion of finetune.
    :param data:        Input data entries [key, filepath, labels]
    :param dataset:     Input dataset name
    :return: flag:      True to include for finetuning, otherwise exclude for finetuning
    """
    if dataset in ["iemocap", "iemocap_impro"]:
        # IEMOCAP data include 4 emotions, exc->hap
        if data[-1] in ["neu", "sad", "ang" , "exc", "hap"]: return True
    return False

def map_label(
    data: list, dataset: str
):  
    """
    Return labels for the input data.
    :param data:        Input data entries [key, filepath, labels]
    :param dataset:     Input dataset name
    :return label:      Label index: int
    """
    label_dict = {
        "iemocap": {"neu": 0, "sad": 1, "ang": 2, "exc": 3, "hap": 3},
    }
    if dataset in ["iemocap", "msp-improv", "msp-podcast", "meld", "crema_d", "iemocap_impro", "crema_d_complete"]:
        return label_dict[dataset][data[-1]]

def load_finetune_audios(
    input_path:     str,
    dataset:        str,
    fold_idx:       int,
    audio_path:     str=None,
    data_percent:   float=1.0
):
    """
    Load finetune audio data.
    :param input_path:  Input data path
    :param dataset:     Dataset name
    :param fold_idx:    Fold idx
    :return train_file_list, dev_file_list: train, dev, and test file list
    """
    train_file_list, dev_file_list, test_file_list = list(), list(), list()
    split_set = ['train', 'dev', 'test']
    if dataset in ["esc50"]:
        with open(str(Path(input_path).joinpath(f'{dataset}_fold{fold_idx}.json')), "r") as f: split_dict = json.load(f)
        split_set = ['train', 'test']
    elif dataset in ["ucf101"]:
        with open(str(Path(input_path).joinpath(f'{dataset}_fold{fold_idx}.json')), "r") as f: split_dict = json.load(f)
        split_set = ['train', 'dev', 'test']   
    elif dataset in ["gtzan", "activitynet", "speechcommands"]:
        with open(str(Path(input_path).joinpath(f'{dataset}.json')), "r") as f: split_dict = json.load(f)
        split_set = ['train', 'dev', 'test']  
    elif dataset in ["iemocap"]:
        with open(str(Path(input_path).joinpath(f'{dataset}_fold{fold_idx}.json')), "r") as f: split_dict = json.load(f)
        split_set = ['train', 'dev', 'test']

    if dataset == "ucf101":
        label_list = list(ucf101_dict.keys())
        label_list.sort()
    elif dataset == "activitynet":
        label_list = list(activitynet_dict.keys())
        label_list.sort()

    for split in split_set:
        for data in split_dict[split]:
            skip_file = False
            if dataset in ["ucf101"]:
                data[-2] = str(Path(audio_path).joinpath(data[-2].split('/')[-1]))
                if data[0].split("/")[0] in label_list: data[-1] = label_list.index(data[0].split("/")[0])
                else: skip_file = True
            elif dataset in ["activitynet"]:
                data[-2] = str(Path(audio_path).joinpath(data[-3].split('/')[-1]))
                if data[0].split("/")[0] in label_list: 
                    waveform, _ = torchaudio.load(data[-2])
                    if len(waveform[0]) == 0: skip_file = True
                    data[-1] = label_list.index(data[0].split("/")[0])
                else: skip_file = True
            elif dataset in ["iemocap"]:
                if include_for_finetune(data, dataset):
                    data[-1] = map_label(data, dataset)
                    speaker_id, file_path  = data[1], data[3]
                    if dataset in ['iemocap', 'msp-improv', 'meld', 'crema_d', 'msp-podcast']:
                        output_path = Path(audio_path).joinpath(file_path.split('/')[-1])
                    data[3] = str(output_path)
                else: skip_file = True
            if skip_file: continue
            if split == 'train': train_file_list.append(data)
            elif split == 'dev': dev_file_list.append(data)
            elif split == 'test': test_file_list.append(data)
    if data_percent < 1:
        if dataset == "esc50": np.random.seed(42)
        else: np.random.seed(fold_idx)
        select_idx = np.random.choice(len(train_file_list), int(len(train_file_list)*data_percent), replace=False)
        train_file_list = [train_file_list[idx] for idx in select_idx]
    
    return train_file_list, dev_file_list, test_file_list

def load_noise_audios(
    input_path:     str,
    dataset:        str,
):
    if dataset in ["esc50"]:
        with open(str(Path(input_path).joinpath(f'{dataset}_fold1.json')), "r") as f: split_dict = json.load(f)
        split_set = ['train']
    
    noise_audios = list()
    for split in split_set:
        for data in split_dict[split]:
            noise_audios.append(data[-2])
    return noise_audios

def load_gen_noise_audios(
    input_path:     str,
    label_dict:     dict,
    num_gen:        int=30,
    gen_method:     str=None,
    gen_model:      str="audioldm"
):
    noise_audios = list()
    for label in label_dict:
        for i in range(num_gen):
            if not Path.exists(Path(input_path).joinpath(gen_model, gen_method, f'{label}_{i}.wav')): continue
            data = str(Path(input_path).joinpath(gen_model, gen_method, f'{label}_{i}.wav'))
            noise_audios.append(data)
    return noise_audios

def load_gen_audios(
    input_path:     str,
    label_dict:     dict,
    num_gen:        int=30,
    gen_method:     str=None,
    gen_model:      str="audioldm",
    filter_list:    list=None
):
    """
    Load finetune audio data.
    :param input_path:  Input data path
    :param label_dict:  Label dictionary
    :return train_file_list, dev_file_list: train, dev, and test file list
    """
    train_file_list = list()
    if filter_list is None: filter_list = list(label_dict.keys())
    for label in label_dict:
        for i in range(num_gen):
            if not Path.exists(Path(input_path).joinpath(gen_model, gen_method, f'{label}_{i}.wav')): continue
            if label not in filter_list: continue
            data = [f'{label}_{i}', label, Path(input_path).joinpath(gen_model, gen_method, f'{label}_{i}.wav'), label_dict[label]]
            train_file_list.append(data)
    return train_file_list

def set_finetune_dataloader(
    args:                   dict,
    input_file_list:        list,
    is_train:               bool,
    target_len:             int=512,
    freqm:                  int=None,
    timem:                  int=None,
    noise_audios:           list=None,
    test_db:                int=5
):
    """
    Return dataloader for finetune experiments.
    :param data:                    Input data entries [key, filepath, labels]
    :param is_train:                Flag for training or not
    :param is_distributed:          Flag for distributed training or not
    :return dataloader:             Dataloader
    """
    if args.dataset == "iemocap":
        data_generator = EmotionDatasetGenerator(
            data_list       = input_file_list, 
            data_len        = len(input_file_list),
            is_train        = is_train,
            audio_duration  = 10,
            noise_audios    = noise_audios,
            test_db         = test_db
        )
    else:
        data_generator = AudioDatasetGenerator(
            data_list       = input_file_list, 
            data_len        = len(input_file_list),
            is_train        = is_train,
            target_len      = target_len,
            freqm           = freqm,
            timem           = timem,
            noise_audios    = noise_audios,
            test_db         = test_db
        )
    if args.dataset == "gtzan": batch_size = 20
    elif args.dataset == "speechcommands": batch_size = 256
    elif args.dataset == "iemocap": batch_size = 32
    else: batch_size = 32

    if is_train:
        if args.dataset == "iemocap":
            dataloader = DataLoader(
                data_generator, 
                batch_size=32, 
                num_workers=6, 
                shuffle=is_train,
                collate_fn=collate_fn,
                drop_last=is_train
            )
        else:
            dataloader = DataLoader(
                data_generator, 
                batch_size=batch_size, 
                num_workers=5, 
                shuffle=is_train,
                drop_last=is_train,
            )
    else:
        if args.dataset == "iemocap":
            dataloader = DataLoader(
                data_generator, 
                batch_size=1, 
                num_workers=5, 
                shuffle=is_train,
                drop_last=is_train
            )
        else:
            dataloader = DataLoader(
                data_generator, 
                batch_size=batch_size, 
                num_workers=5, 
                shuffle=is_train,
                drop_last=is_train
            )
    return dataloader

def return_weights(
    input_path:     str,
    dataset:        str,
    fold_idx:       int,
    input_data:     list=None
):
    """
    Return training weights.
    :param input_path:  Input data path
    :param dataset:     Dataset name
    :param fold_idx:    Fold idx
    :return weights:    Class weights
    """
    if dataset in ["esc50"]:
        with open(str(Path(input_path).joinpath(f'esc50_fold{fold_idx}.json')), "r") as f: split_dict = json.load(f)
    elif dataset in ["ucf101"]:
        with open(str(Path(input_path).joinpath(f'ucf101_fold{fold_idx}.json')), "r") as f: split_dict = json.load(f)
    elif dataset in ["gtzan"]:
        with open(str(Path(input_path).joinpath(f'gtzan.json')), "r") as f: split_dict = json.load(f)
    elif dataset in ["speechcommands"]:
        with open(str(Path(input_path).joinpath(f'speechcommands.json')), "r") as f: split_dict = json.load(f)
    
    weights_stats = dict()
    if dataset in ["activitynet"]:
        for i in range(len(activitynet_dict)):
            weights_stats[i] = 1
            
    if input_data is None: data_dict = split_dict['train']
    else: data_dict = input_data
    for data in data_dict:
        if data[-1] not in weights_stats: weights_stats[data[-1]] = 0
        weights_stats[data[-1]] += 1
    # pdb.set_trace()
    weights = torch.tensor([weights_stats[c] for c in range(len(weights_stats))]).float()
    weights = weights.sum() / weights
    weights = weights / weights.sum()
    return weights