# Prepare AMASS

*Adapted from stmc (https://github.com/nv-tlabs/stmc/tree/main)*

## Download

The motions all come from the AMASS dataset. Please download all "SMPL-H G" motions from the [AMASS website](https://amass.is.tue.mpg.de/download.php) and place them in the folder ``dataset/AMASS``.

<details><summary>It should look like this:</summary>

```bash
dataset
└── AMASS
    ├── ACCAD
    ├── BioMotionLab_NTroje
    ├── BMLhandball
    ├── BMLmovi
    ├── CMU
    ├── DanceDB
    ├── DFaust_67
    ├── EKUT
    ├── Eyes_Japan_Dataset
    ├── HumanEva
    ├── KIT
    ├── MPI_HDM05
    ├── MPI_Limits
    ├── MPI_mosh
    ├── SFU
    ├── SSM_synced
    ├── TCD_handMocap
    ├── TotalCapture
    └── Transitions_mocap
```

Each file contains a "poses" field with 156 (52x3) parameters (1x3 for global orientation, 21x3 for the whole body, 15x3 for the right hand and 15x3 for the left hand).

</details>

## SMPL dependency
Please follow the [README from TEMOS](https://github.com/Mathux/TEMOS?tab=readme-ov-file#4-optional-smpl-body-model) to obtain the ``deps`` folder with SMPL+H downloaded, and place ``deps`` in the ``data_loaders/amasstools`` directory.


## Preprocessing

Then, launch these commands:

```bash
python -m data_loaders.amasstools.fix_fps 
python -m data_loaders.amasstools.smpl_mirroring
python -m data_loaders.amasstools.extract_joints
python -m data_loaders.amasstools.get_globsmplrifke_base
```

<details><summary>Click here for more information on these commands</summary>

### Fix FPS

The script will interpolate the SMPL pose parameters and translation to obtain a constant FPS (=20.0). It will also remove the hand pose parameters, as they are not captured for most AMASS sequences. The SMPL pose parameters now have 66 (22x3) parameters (1x3 for global orientation and 21x3 for full body). It will create and save all the files in the folder ``dataset/AMASS_20.0_fps_nh``.


### SMPL mirroring

This command will mirror SMPL pose parameters and translations, to enable data augmentation with SMPL (as done by the authors of HumanML3D with joint positions).
The mirrored motions will be saved in ``dataset/AMASS_20.0_fps_nh/M`` and will have a structure similar than the enclosing folder.


### Extract joints

The script extracts the joint positions from the SMPL pose parameters with the SMPL layer (24x3=72 parameters). It will save the joints in .npy format in this folder: ``dataset/AMASS_20.0_fps_nh_smpljoints_neutral_nobetas``.


### Prepare Canonicalized Global SMPL RIFKE Base Features

This step parses **joints + SMPL pose parameters** and applies **canonicalization** (rotation) to produce Global SMPL RIFKE base features.  
See `data_loaders/amasstools/get_globsmplrifke_base.py` for implementation details.

</details>

The dataset folder should look like this:
```bash
dataset
├── AMASS
├── AMASS_20.0_fps_nh
├── AMASS_20.0_fps_nh_smpljoints_neutral_nobetas
└── AMASS_20.0_fps_nh_globsmpl_base_cano
```

</details>
