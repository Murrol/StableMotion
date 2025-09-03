<!-- **StableMotion: Training Motion Cleanup Models with Unpaired Corrupted Data** -->

# <p align="center"> StableMotion: Training Motion Cleanup Models with Unpaired Corrupted Data </p>

> :mechanic: **You don’t need a clean dataset to train a motion cleanup model.**  
> StableMotion learns to fix corrupted motions directly from raw mocap data — no handcrafted data pairs, no synthetic artifact augmentation.


![Teaser](assets/teaser.jpg)

<p align="center"> :x: &nbsp; Raw corrupted data &emsp;&emsp; :white_check_mark: &nbsp; Clean results!</p>

## Table of Contents
- [ StableMotion: Training Motion Cleanup Models with Unpaired Corrupted Data ](#-stablemotion-training-motion-cleanup-models-with-unpaired-corrupted-data-)
  - [Table of Contents](#table-of-contents)
  - [Installation](#installation)
    - [Environment Setup](#environment-setup)
    - [Dependencies](#dependencies)
  - [Quick Start](#quick-start)
    - [0. Get BrokenAMASS](#0-get-brokenamass)
    - [1. Training](#1-training)
    - [2. Inference](#2-inference)
  - [Evaluation](#evaluation)
  - [Visualization](#visualization)
  - [Acknowlegements](#acknowlegements)
  - [License](#license)
  - [Citation](#citation)

:hammer_and_wrench::gear::two_women_holding_hands::rocket::broom::broom::soap::mechanic::x::white_check_mark::dart::bar_chart:
## Installation :gear:

### Environment Setup
Create and activate a new conda environment:

```bash
conda create --name stablemotion python=3.11.8
conda activate stablemotion
```

### Dependencies
Install the required packages:

```bash
pip install -r requirements.txt 
```

### SMPL Dependency

<details><summary>The SMPL model is required for preprocessing, evaluation, and visualization. </summary>

Please follow the [README from TEMOS](https://github.com/Mathux/TEMOS?tab=readme-ov-file#4-optional-smpl-body-model) to obtain the `deps` folder with SMPL+H downloaded, and place the `deps` folder under ``./data_loaders/amasstools``.

</details>

### TMR Dependency

<details><summary>Text-to-Motion Retrieval (TMR) is used for evaluation. </summary>

Please follow the [README from TMR](https://github.com/Mathux/TMR?tab=readme-ov-file#pretrained-models-dvd) to download pretrained TMR models. After downloading, place the models in the following structure: 
```
StableMotion/
└── tmr_models/
    └── tmr_humanml3d_guoh3dfeats
    └── tmr_kitml_guoh3dfeats
```
</details>

### Pretrained Checkpoint: StableMotion-BrokenAMASS

To play around, download a **StableMotion** checkpoint trained on BrokenAMASS from [OneDrive](https://1sfu-my.sharepoint.com/:u:/g/personal/yma101_sfu_ca/EXhMWi9T749No18jTtPGr-EBvz9aEGueCWvLbzueUbpLcw?e=C9ZnKj) and place it under the `./save` directory. 


## Quick Start

### 0. Get Benchmark Dataset: BrokenAMASS

Please follow the [README for DATA](./data_loaders/amasstools/README.md) to download and preprocess the original AMASS dataset.

Then, run the following scripts to build **BrokenAMASS**:

```bash
python -m data_loaders.corrupting_globsmpl_dataset --mode train
python -m data_loaders.corrupting_globsmpl_dataset --mode test
```

After preprocessing and corruption, your dataset folder should look like this:
```
dataset/
├── AMASS
├── AMASS_20.0_fps_nh
├── AMASS_20.0_fps_nh_smpljoints_neutral_nobetas
└── AMASS_20.0_fps_nh_globsmpl_base_cano
├── AMASS_20.0_fps_nh_globsmpl_corrupted_cano
└── meta_AMASS_20.0_fps_nh_globsmpl_corrupted_cano/
    └── mean.pt
    └── std.pt
```

<details><summary>misc. </summary>

The released version of BrokenAMASS may differ slightly from the version used in the experiments reported in the paper, due to different random seeds. Contact yma101@sfu.ca for further questions.

</details>

### 0.5 Customized Dataset :dart:

If you want to clean up your **own motion data**, we strongly recommend preparing the training data with quality labels and training your **own StableMotion model** on that dataset — this is exactly what **StableMotion** was designed for!


### 1. Training :rocket:
Train the StableMotion model on BrokenAMASS:

```bash
python -m train.train_stablemotion_smpl_glob \
  --save_dir save/stablemotion \
  --data_dir dataset/AMASS_20.0_fps_nh_globsmpl_corrupted_cano \
  --normalizer_dir dataset/meta_AMASS_20.0_fps_nh_globsmpl_corrupted_cano \
  --l1_loss \
  --model_ema \
  --gradient_clip \
  --batch_size 128 \
  --num_steps 1_000_000 \
  --train_platform_type TensorboardPlatform
```

### 2. Inference :soap:
Clean up corrupted motion sequences using the trained model:

```bash
# Basic inference
python -m sample.fix_globsmpl \
  --model_path save/stablemotion/ema001000000.pt \
  --use_ema \
  --batch_size 32 \
  --testdata_dir dataset/AMASS_20.0_fps_nh_globsmpl_corrupted_cano \
  --output_dir ./output/stablemotion_vanilla

# Enhanced inference with ensemble and adaptive cleanup
python -m sample.fix_globsmpl \
  --model_path save/stablemotion/ema001000000.pt \
  --use_ema \
  --batch_size 32 \
  --testdata_dir dataset/AMASS_20.0_fps_nh_globsmpl_corrupted_cano \
  --ensemble \
  --enable_sits \
  --classifier_scale 100 \
  --output_dir ./output/stablemotion_hack
```

## Evaluation :bar_chart:

Evaluate the quality of cleaned motion sequences:

```bash
python -m eval.eval_scripts --data_path ./output/stablemotion_vanilla/results.npy
```

Content preservation metrics:

<details><summary>Collect clean ground truth</summary>

To evaluate content preservation, first record the clean ground-truth data from `dataset/AMASS_20.0_fps_nh_globsmpl_base_cano`:

```bash
python -m sample.fix_globsmpl \
  --model_path save/stablemotion/ema001000000.pt \
  --use_ema \
  --batch_size 32 \
  --testdata_dir dataset/AMASS_20.0_fps_nh_globsmpl_base_cano \
  --output_dir ./output/benchmark_clean
  --collect_dataset
```

</details>

Then run evaluation with ground truth:

```bash
python -m eval.eval_scripts \
  --data_path ./output/stablemotion_vanilla/results.npy \
  --gt_data_path ./output/benchmark_clean/results_collected.npy

```

## Visualization :two_women_holding_hands:

Generate visual renderings of the cleaned motion data:

```bash
python -m visualize.render_scripts \
  --data_path ./output/stablemotion_vanilla/results.npy \
  --rendersmpl
```

<!-- ## Project Structure

```
StableMotion/
├── train/
│   └── train_stablemotion_smpl_glob.py
├── sample/
│   └── fix_globsmpl.py
├── eval/
│   └── eval_scripts.py
├── visualize/
│   └── render_scripts.py
├── dataset/
│   ├── AMASS_20.0_fps_nh_globsmpl_corrupted_cano/
│   └── meta_AMASS_20.0_fps_nh_globsmpl_corrupted_cano/
├── save/
└── output/
``` -->

## Acknowlegements

We sincerely thank the open-sourcing of these works where our code is based on: 

[MDM](https://github.com/GuyTevet/motion-diffusion-model/tree/main), [stmc](https://github.com/nv-tlabs/stmc.git), [diffusers](https://github.com/huggingface/diffusers), [TMR](https://github.com/Mathux/TMR), [humor](https://github.com/davrempe/humor), [PixArt-α](https://github.com/PixArt-alpha/PixArt-alpha), [stable-audio-tools](https://github.com/Stability-AI/stable-audio-tools),

## License
This code is distributed under an [MIT LICENSE](LICENSE).

Note that our code depends on other libraries, including TMR, SMPL, SMPL-X, and uses datasets which each have their own respective licenses that must also be followed.

## Citation

If you find our work helpful, please cite:

```bibtex
@article{mu2025stablemotion,
  title={StableMotion: Training Motion Cleanup Models with Unpaired Corrupted Data},
  author={Mu, Yuxuan and Ling, Hung Yu and Shi, Yi and Ojeda, Ismael Baira and Xi, Pengcheng and Shu, Chang and Zinno, Fabio and Peng, Xue Bin},
  journal={arXiv preprint arXiv:2505.03154},
  year={2025}
}
```