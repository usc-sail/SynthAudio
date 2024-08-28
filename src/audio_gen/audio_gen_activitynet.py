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
from audiocraft.data.audio import audio_write



activitynet_dict = {
    "Applying+sunscreen":  "Applying sunscreen",
    "Archery":  "Archerying",
    "Assembling+bicycle": "Assembling bicycle",
    "Arm+wrestling": "Arm wrestling",
    "BMX": "Riding BMX",
    "Baking+cookies": "Baking cookies",
    "Ballet": "Dancing Ballet",
    "Bathing+dog": "Bathing dog",
    "Baton+twirling": "Baton twirling",
    "Beach+soccer": "Playing Beach soccer",
    "Beer+pong": "Playing Beer pong",
    "Belly+dance": "Belly dance",
    "Blow-drying+hair": "Blow-drying hair",
    "Blowing+leaves": "Blowing leaves",
    "Braiding+hair": "Braiding hair",
    "Breakdancing": "Breakdancing",
    "Brushing+hair": "Brushing hair",
    "Brushing+teeth": "Brushing teeth",
    "Building+sandcastles": "Building sandcastles",
    "Bullfighting": "Bull fighting",
    "Bungee+jumping": "Bungee jumping",
    "Calf+roping": "Calf roping",
    "Camel+ride": "Riding Camel",
    "Canoeing": "Canoeing",
    "Capoeira": "Playing Capoeira",
    "Carving+jack-o-lanterns": "Carving jack-o-lanterns",
    "Changing+car+wheel": "Changing car wheel",
    "Cheerleading": "Cheer leading",
    "Chopping+wood": "Chopping wood",
    "Clean+and+jerk": "Clean and jerk",
    "Cleaning+shoes": "Cleaning shoes",
    "Cleaning+sink": "Cleaning sink",
    "Cleaning+windows": "Cleaning windows",
    "Clipping+cat+claws": "Clipping cat claws",
    "Cricket": "Playing Cricket",
    "Croquet": "Playing Croquet",
    "Cumbia": "Playing Cumbia",
    "Curling": "Curling",
    "Cutting+the+grass": "Cutting the grass",
    "Decorating+the+Christmas+tree": "Decorating the Christmas tree",
    "Disc+dog": "Playing Disc dog",
    "Discus+throw": "throwing Discus",
    "Dodgeball": "Playing Dodgeball",
    "Doing+a+powerbomb": "Doing a powerbomb",
    "Doing+crunches": "Doing crunches",
    "Doing+fencing": "Doing fencing",
    "Doing+karate": "Doing karate",
    "Doing+kickboxing": "Doing+kickboxing",
    "Doing+motocross": "Doing+motocross",
    "Doing+nails": "Doing+nails",
    "Doing+step+aerobics": "Doing+step+aerobics",
    "Drinking+beer": "Drinking+beer",
    "Drinking+coffee": "Drinking+coffee",
    "Drum+corps": "Doing Drum+corps",
    "Elliptical+trainer": "Performing Elliptical+trainer",
    "Fixing+bicycle": "Fixing+bicycle",
    "Fixing+the+roof": "Fixing+the+roof",
    "Fun+sliding+down": "Fun+sliding+down",
    "Futsal": "Playing Futsal",
    "Gargling+mouthwash": "Gargling+mouthwash",
    "Getting+a+haircut": "Getting+a+haircut",
    "Getting+a+piercing": "Getting+a+piercing",
    "Getting+a+tattoo": "Getting+a+tattoo",
    "Grooming+dog": "Grooming+dog",
    "Grooming+horse": "Grooming+horse",
    "Hammer+throw": "throwing Hammer",
    "Hand+car+wash": "Hand washing car",
    "Hand+washing+clothes": "Hand+washing+clothes",
    "Hanging+wallpaper": "Hanging+wallpaper",
    "Having+an+ice+cream": "Having+an+ice+cream",
    "Drinking+coffee": "Drinking+coffee",
    "High+jump": "Doing High+jump",
    "Hitting+a+pinata": "Hitting+a+pinata",
    "Hopscotch": "Playing Hopscotch",
    "Horseback+riding": "Horseback+riding",
    "Hula+hoop": "Doing Hula+hoop",
    "Hurling": "Hurling",
    "Ice+fishing": "Ice+fishing",
    "Installing+carpet": "Installing+carpet",
    "Ironing+clothes": "Ironing+clothes",
    "Javelin+throw": "throwing Javelin",
    "Kayaking": "Kayaking",
    "Kite+flying": "flying Kite",
    "Kneeling": "Kneeling",
    "Knitting": "Knitting",
    "Laying+tile": "Laying+on tile",
    "Layup+drill+in+basketball": "Layup+drill+in+basketball",
    "Long+jump": "Doing Long+jump",
    "Longboarding": "Long boarding",
    "Making+a+cake": "Making+a+cake",
    "Making+a+lemonade": "Making+a+lemonade",
    "Making+a+sandwich": "Making+a+sandwich",
    "Making+an+omelette": "Making+an+omelette",
    "Mixing+drinks": "Mixing+drinks",
    "Mooping+floor": "Mooping+floor",
    "Mowing+the+lawn": "Mowing+the+lawn",
    "Paintball": "Playing Paintball",
    "Painting": "Painting",
    "Painting+fence": "Painting+fence",
    "Painting+furniture": "Painting+furniture",
    "Peeling+potatoes": "Peeling+potatoes",
    "Ping-pong": "Playing Ping-pong",
    "Plastering": "Plastering",
    "Plataform+diving": "Plataform+diving",
    "Playing+accordion": "Playing+accordion",
    "Playing+badminton": "Playing+badminton",
    "Playing+bagpipes": "Playing+bagpipes",
    "Playing+beach+volleyball": "Playing+beach+volleyball",
    "Playing+blackjack": "Playing+blackjack",
    "Playing+congas": "Playing+congas",
    "Playing+drums": "Playing+drums",
    "Playing+field+hockey": "Playing+field+hockey",
    "Playing+flauta": "Playing+flauta",
    "Playing+guitarra": "Playing+guitarra",
    "Playing+harmonica": "Playing+harmonica",
    "Playing+ice+hockey": "Playing+ice+hockey",
    "Playing+kickball": "Playing+kickball",
    "Playing+lacrosse": "Playing+lacrosse",
    "Playing+piano": "Playing+piano",
    "Playing+polo": "Playing+polo",
    "Playing+pool": "Playing+pool",
    "Playing+racquetball": "Playing+racquetball",
    "Playing+rubik+cube": "Playing+rubik+cube",
    "Playing+saxophone": "Playing+saxophone",
    "Playing+squash": "Playing+squash",
    "Playing+ten+pins": "Playing+ten+pins",
    "Playing+violin": "Playing+violin",
    "Playing+water+polo": "Playing+water+polo",
    "Pole+vault": "Doing Pole+vault",
    "Polishing+forniture": "Polishing+forniture",
    "Polishing+shoes": "Polishing+shoes",
    "Powerbocking": "Powerbocking",
    "Preparing+pasta": "Preparing+pasta",
    "Preparing+salad": "Preparing+salad",
    "Putting+in+contact+lenses": "Putting+in+contact+lenses",
    "Putting+on+makeup": "Putting+on+makeup",
    "Putting+on+shoes": "Putting+on+shoes",
    "Rafting": "Rafting",
    "Raking+leaves": "Raking+leaves",
    "Removing+curlers": "Removing+curlers",
    "Removing+ice+from+car": "Removing+ice+from+car",
    "Riding+bumper+cars": "Riding+bumper+cars",
    "River+tubing": "doing River+tubing",
    "Rock+climbing": "Rock+climbing",
    "Rock-paper-scissors": "playing Rock-paper-scissors",
    "Rollerblading": "doing Rollerblading",
    "Roof+shingle+removal": "removing Roof+shingle",
    "Rope+skipping": "doing Rope+skipping",
    "Running+a+marathon": "Running+a+marathon",
    "Sailing": "Sailing",
    "Scuba+diving": "Scuba+diving",
    "Sharpening+knives": "Sharpening+knives",
    "Shaving": "Shaving",
    "Shaving+legs": "Shaving+legs",
    "Shot+put": "doing Shot+put",
    "Shoveling+snow": "Shoveling+snow",
    "Shuffleboard": "doing Shuffleboard",
    "Skateboarding": "Skateboarding",
    "Skiing": "Skiing",
    "Slacklining": "Slacklining",
    "Smoking+a+cigarette": "Smoking+a+cigarette",
    "Smoking+hookah": "Smoking+hookah",
    "Snatch": "doing Snatch",
    "Snow+tubing": "Snow+tubing",
    "Snowboarding": "Snowboarding",
    "Spinning": "Spinning",
    "Spread+mulch": "doing Spread+mulch",
    "Springboard+diving": "Springboard+diving",
    "Starting+a+campfire": "Starting+a+campfire",
    "Sumo": "plaing Sumo",
    "Surfing": "Surfing",
    "Swimming": "Swimming",
    "Swinging+at+the+playground": "Swinging+at+the+playground",
    "Table+soccer": "Playing Table+soccer",
    "Tai+chi": "Playing Tai+chi",
    "Tango": "Doing Tango",
    "Tennis+serve+with+ball+bouncing": "doing Tennis+serve+with+ball+bouncing",
    "Throwing+darts": "Throwing+darts",
    "Trimming+branches+or+hedges": "Trimming+branches+or+hedges",
    "Triple+jump": "Doing Triple+jump",
    "Tug+of+war": "Playing Tug+of+war",
    "Tumbling": "Tumbling",
    "Using+parallel+bars": "Using+parallel+bars",
    "Using+the+balance+beam": "Using+the+balance+beam",
    "Using+the+monkey+bar": "Using+the+monkey+bar",
    "Using+the+pommel+horse": "Using+the+pommel+horse",
    "Using+the+rowing+machine": "Using+the+rowing+machine",
    "Using+uneven+bars": "Using+uneven+bars",
    "Vacuuming+floor": "Vacuuming+floor",
    "Volleyball": "Playing Volleyball",
    "Wakeboarding": "Playing Wakeboarding",
    "Walking+the+dog": "Walking+the+dog",
    "Washing+dishes": "Washing+dishes",
    "Washing+face": "Washing+face",
    "Washing+hands": "Washing+hands",
    "Waterskiing": "Waterskiing",
    "Waxing+skis": "Waxing+skis",
    "Welding": "Welding",
    "Windsurfing": "Wind surfing",
    "Wrapping+presents": "Wrapping+presents",
    "Zumba": "Playing Zumba",
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
    if args.dataset == "gtzan":
        duration = 10
    else:
        duration = 5
        
    if args.model == "audioldm":
        repo_id = "cvssp/audioldm2-large"
        model = AudioLDM2Pipeline.from_pretrained(
            repo_id,
        )
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)
    elif args.model == "audiogen":
        model = AudioGen.get_pretrained('facebook/audiogen-medium')
        model.set_generation_params(duration=duration)
        # model = model.to(device)
    
    for label in labels:
        class_name = activitynet_dict[label].lower().replace("+", " ")
        if description is not None:
            description_list = description[label]
        for j in tqdm(range(gen_per_class)):
            save_file_path = Path(args.gen_dir).joinpath(
                f'{label}_{j}.wav'
            )
            
            is_skipping = False
            if Path(save_file_path).exists(): is_skipping = True
            if is_skipping: continue
            
            
            if args.generate_method in ["class_prompt"]:
                prompt = f"{class_name}"
                if args.model == "audioldm":
                    audio = model(
                        prompt,
                        num_inference_steps=50,
                        audio_length_in_s=duration,
                        num_waveforms_per_prompt=1,
                    ).audios
                elif args.model == "audiogen":
                    audio = model.generate([prompt])[0]
                    audio = audio.detach().cpu().numpy()
            elif args.generate_method in ["llm"]:
                sampled_des = random.sample(description_list, 3)
                des = ", ".join(sampled_des)
                prompt = f"{class_name}, {des}"
                if args.model == "audioldm":
                    audio = model(
                        prompt,
                        num_inference_steps=50,
                        audio_length_in_s=duration,
                        num_waveforms_per_prompt=1,
                    ).audios
                elif args.model == "audiogen":
                    audio = model.generate([prompt])[0]
                    audio = audio.detach().cpu().numpy()
            scipy.io.wavfile.write(save_file_path, rate=16000, data=audio[0])
            
            
if __name__ == '__main__':

    # Argument parser
    parser = argparse.ArgumentParser(description='Audio generation')
    parser.add_argument(
        '--gen_per_class', 
        default=300,
        type=int, 
        help='Number of generation per class'
    )
    
    parser.add_argument(
        '--generate_method', 
        default="llm",
        type=str, 
        help='generate method: class_prompt, multi_domain, ucg, llm'
    )
    
    parser.add_argument(
        '--model', 
        default="audiogen",
        type=str, 
        help='generate method: audiogen audioldm'
    )
    
    parser.add_argument(
        '--dataset', 
        default="activitynet",
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
    
    gen_labels = list(activitynet_dict.keys())
    gen_labels.sort()
    
    # pdb.set_trace()
    dataset_generate(
        gen_labels,
        args.gen_per_class,
        description=des_dict
    )