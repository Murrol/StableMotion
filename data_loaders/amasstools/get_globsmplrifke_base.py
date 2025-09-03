import os
import numpy as np
import torch

from .loop_amass import loop_amams
from .globsmplrifke_base_feats import (
    smpldata_to_globsmplrifkefeats,
    globsmplrifkefeats_to_smpldata,
    group,
    ungroup,
    smpldata_to_alignglobsmplrifkefeats,
    alignglobsmplrifkefeats_to_smpldata,
    canonicalize_rotation,
)
import einops

from data_loaders.amasstools.geometry import (
    axis_angle_to_matrix,
    matrix_to_rotation_6d,
    rotation_6d_to_matrix,
    matrix_to_euler_angles,
    axis_angle_rotation,
    matrix_to_axis_angle,
    axis_angle_to_quaternion,
    quaternion_to_axis_angle,
)


def get_features(folder, base_name, joints_name, new_name, force_redo=False):
    """
    Canonicalize AMASS SMPL sequences and save results.

    Args:
        folder (str): Root dataset directory (e.g., "dataset").
        base_name (str): Folder containing resampled SMPL data (e.g., "AMASS_20.0_fps_nh").
        joints_name (str): Folder with extracted joints (e.g., "AMASS_20.0_fps_nh_smpljoints_neutral_nobetas").
        new_name (str): Output folder name for canonicalized results (e.g., "AMASS_20.0_fps_nh_cano").
        force_redo (bool): If True, overwrite existing outputs.

    Inputs (per sequence):
        - SMPL npz with keys "poses" (T, 66), "trans" (T, 3).
        - Joints npy with shape (T, 24, 3).

    Output:
        - Saves canonicalized SMPL dict as npz to `folder/new_name/...`.
          The saved dict mirrors the input keys (e.g., poses/trans/joints).
    """
    base_folder = os.path.join(folder, base_name)
    joints_folder = os.path.join(folder, joints_name)
    new_folder = os.path.join(folder, new_name)

    print("Prepare SMPL joint features and canonicalize rotations.")
    print(f"The processed motions will be stored in: {new_folder}")

    if not os.path.exists(base_folder):
        print(f"{base_folder} folder does not exist")
        print("Run fix_fps.py")
        exit()

    if not os.path.exists(joints_folder):
        print(f"{joints_folder} folder does not exist")
        print("Run extract_joints.py")
        exit()

    iterator = loop_amams(
        base_folder, new_folder, ext=".npz", newext=".npy", force_redo=force_redo
    )

    for motion_path, new_motion_path in iterator:
        smpl_data = np.load(motion_path)

        # Load paired joints file
        joint_path = motion_path.replace(base_name, joints_name).replace(".npz", ".npy")
        joints = np.load(joint_path)

        # Expect 22×3 SMPL (no hands)
        assert smpl_data["poses"].shape[-1] == 66
        if len(smpl_data["poses"]) < 5:
            continue

        smpl_data = {
            "poses": torch.from_numpy(smpl_data["poses"]).to(torch.double),
            "trans": torch.from_numpy(smpl_data["trans"]).to(torch.double),
            "joints": torch.from_numpy(joints).to(torch.double),
        }

        # Canonicalize rotation/translation and save
        can_smpl_data = canonicalize_rotation(smpl_data)
        np.savez(motion_path.replace(base_name, new_name), **can_smpl_data)


def main():
    """
    Default entry: canonicalize AMASS sequences using prepared inputs.
    Adjust names below only if your folder names differ.
    """
    base_name = "AMASS_20.0_fps_nh"
    joints_name = "AMASS_20.0_fps_nh_smpljoints_neutral_nobetas"
    new_name = "AMASS_20.0_fps_nh_globsmpl_base_cano"
    folder = "dataset"

    force_redo = False
    get_features(folder, base_name, joints_name, new_name, force_redo=force_redo)


if __name__ == "__main__":
    main()