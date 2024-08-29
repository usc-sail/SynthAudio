import pdb
import yaml
import json
import time
import pathlib
import argparse
import textwrap
import PIL.Image
from pathlib import Path

import google.generativeai as genai

# define logging console
import logging
logging.basicConfig(
    format='%(asctime)s %(levelname)-3s ==> %(message)s', 
    level=logging.INFO, 
    datefmt='%Y-%m-%d %H:%M:%S'
)

GOOGLE_API_KEY = "CHANGE_ME"

gen_labels = [
    "helicopter", "Chainsaw", "Siren", "Car horn", "Engine", "Train", "Church bells", "Airplane", "Fireworks", "Hand saw",
    "dog", "Rooster", "Pig", "Cow", "Frog", "Cat", "Hen", "Insects", "Sheep", "Crow",
    "Rain", "Sea waves", "Crackling fire", "Crickets", "Chirping birds", "Water drops", "Wind", "Pouring water", "Toilet flush", "Thunderstorm",
    "Crying baby", "Sneezing", "Clapping", "Breathing", "Coughing", "Footsteps", "Laughing", "Brushing teeth", "Snoring", "Drinking sipping",
    "Door wood knock", "Mouse click", "Keyboard typing", "Door, wood creaks", "Can opening", "Washing machine", "Vacuum cleaner", "Clock alarm", "Clock tick", "Glass breaking"
]

gen_labels = [
    "blues", "classical", "country", "disco", "hiphop",
    "jazz", "metal", "pop", "reggae", "rock",
]

gen_labels = {
    "ApplyEyeMakeup": "Applying Eye Makeup",
    "ApplyLipstick": "Applying Lipstick",
    "Archery":  "Archerying",
    "BandMarching": "Band Marching",
    "BabyCrawling": "Baby Crawling",
    "BalanceBeam": "Playing Balance Beam",
    "BasketballDunk": "Dunking Basketball",
    "BlowDryHair": "Blowing Dry Hair",
    "BlowingCandles": "Blowing Candles",
    "BodyWeightSquats": "Body Weight Squats",
    "Bowling": "Bowling",
    "BoxingPunchingBag": "Boxing with Punching Bag",
    "BoxingSpeedBag": "Boxing with Speed Bag",
    "BrushingTeeth": "Brushing Teeth",
    "CliffDiving": "Cliff Diving",
    "CricketBowling": "Cricket Bowling",
    "CricketShot": "Cricket Shot",
    "CuttingInKitchen": "Cutting In Kitchen",
    "FieldHockeyPenalty": "Field Hockey Penalty",
    "FloorGymnastics": "Floor Gymnastics",
    "FrisbeeCatch": "Catching Frisbee",
    "FrontCrawl": "Performing Front Crawl",
    "Haircut": "Haircut",
    "HammerThrow": "Throwing Hammer",
    "Hammering": "Hammering",
    "HandstandPushups": "Handstand Pushups",
    "HandstandWalking": "Handstand Walking",
    "HeadMassage": "Massaging Head",
    "IceDancing": "Ice Dancing",
    "Knitting": "Knitting",
    "LongJump": "Long Jump",
    "MoppingFloor": "Mopping Floor",
    "ParallelBars": "Parallel Bars",
    "PlayingCello": "Playing Cello",
    "PlayingDaf": "Playing Daf",
    "PlayingDhol": "Playing Dhol",
    "PlayingFlute": "Playing Flute",
    "PlayingSitar": "Playing Sitar",
    "Rafting": "Rafting",
    "ShavingBeard": "Shaving Beard",
    "Shotput": "Shot put",
    "SkyDiving": "Sky Diving",
    "SoccerPenalty": "Performing Soccer Penalty",
    "StillRings": "Playing StillRings",
    "SumoWrestling": "Sumo Wrestling",
    "Surfing": "Surfing",
    "TableTennisShot": "Table Tennis",
    "Typing": "Typing",
    "UnevenBars": "Uneven Bars",
    "WallPushups": "Wall Pushups",
    "WritingOnBoard": "Writing On Board",
}


gen_labels = {
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

if __name__ == '__main__':
    
    # Argument parser
    parser = argparse.ArgumentParser(description='Audio generation')
    parser.add_argument(
        '--dataset', 
        default="activitynet",
        type=str, 
        help='dataset name'
    )
    
    args = parser.parse_args()
    with open("../../config/config.yml", "r") as stream: config = yaml.safe_load(stream)
    args.root_dir = str(Path(config["project_dir"]).joinpath(args.dataset))
    args.gen_dir  = str(Path(config["project_dir"]).joinpath("description", args.dataset))
    Path.mkdir(
        Path(args.gen_dir), 
        parents=True, 
        exist_ok=True
    )

    genai.configure(api_key=GOOGLE_API_KEY)

    for m in genai.list_models():
        if 'generateContent' in m.supported_generation_methods:
            print(m.name)

    # Define the models
    # model = genai.GenerativeModel('gemini-pro')
    model = genai.GenerativeModel('models/gemini-1.0-pro')
    
    save_dict = dict()
    for label in gen_labels:

        while True:
            if args.dataset == "ucf101":
                label_des = gen_labels[label].lower()
            elif args.dataset == "activitynet":
                label_des = gen_labels[label].lower().replace("+", " ")
            try:
                if args.dataset == "gtzan":
                    response = model.generate_content(f"List 20 characteristics in single word or phrase of the {label_des} music", stream=True)
                else:
                    response = model.generate_content(f"List 10 characteristics in single word of the {label_des} sound", stream=True)
                response.resolve()
                response_text = response.text.split("\n")
                response_text = [text.split(".")[-1].strip().lower().replace("-", "").replace("*", "") for text in response_text]
            except Exception as e:
                print(f'{type(e).__name__}: {e}')
            time.sleep(5)

            if len(response_text) != 10: continue
            
            save_dict[label] = response_text
            logging.info(response_text)

            # dump the dictionary
            jsonString = json.dumps(save_dict, indent=4)
            jsonFile = open(str(Path(args.gen_dir).joinpath(f'data.json')), "w")
            jsonFile.write(jsonString)
            jsonFile.close()

            break
    
