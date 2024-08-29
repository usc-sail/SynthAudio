# Synth Audio
### This repo includes part of the code for IS24 paper: Can Synthetic Audio From Generative Foundation Models Assist Audio Recognition and Speech Modeling? [[Paper Link](https://arxiv.org/abs/2406.08800)]


#### The core idea behind this work is simple, and we are aiming to answer whether we can use synthetic audio as training and augmentation data?


#### The major part of the work is audio generation, so we recommend to use the code under src/audio_gen

### Audio Generation
We provide code to generate the audio used in our paper: ESC50, GTZAN, and UCF101. 

You can specify the argument for the generation
```
gen_per_class: number of generation per class
generate_method: class_prompt (class-guided), llm (llm-assisted)
model: audiogen or audioldm (musicgen for musics as well)

```

