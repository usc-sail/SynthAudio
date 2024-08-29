# Synth Audio
### This repo includes part of the code for IS24 paper: Can Synthetic Audio From Generative Foundation Models Assist Audio Recognition and Speech Modeling? [[Paper Link](https://arxiv.org/abs/2406.08800)]


#### The core idea behind this work is simple, and we are aiming to answer whether we can use synthetic audio as training and augmentation data?


#### The major part of the work is audio generation, so we recommend to use the code under src/audio_gen

### Audio Generation
We provide code to generate the audio used in our paper: ESC50, GTZAN, and UCF101. 
```
src/audio_gen/audio_gen_esc50.py
src/audio_gen/audio_gen_gtzan.py
src/audio_gen/audio_gen_ucf101.py
```

You can specify the argument for the generation
```
gen_per_class: number of generation per class
generate_method: class_prompt (class-guided), llm (llm-assisted)
model: audiogen or audioldm (musicgen for musics as well)
```

For using LLM, as this research was performed when Gemini was first launched, we notice there are some new things to configure in recent Gemini release. 


### Synthetic Audio Release

For now, we only release the audios from ESC50, due to the large size of synthetic audios.

[[Dropbox Download Link](https://www.dropbox.com/scl/fi/5wakpkvppjadrj90t8ck2/esc50.zip?rlkey=m5qgtecauhpdp18u778ydgwyb&st=pll7tdgd&dl=0)]


### Audio Training
We provided sample experiment code for audio training. You will need to run split, preprocess, and finally finetune scripts under experiments. You would need to download the SSAST-Base-Patch-400.pth from the SSAST repo.



