import os
import argparse
import numpy as np

import torch
from .smplh_layer import load_smplh_gender
from .loop_amass import loop_amams


def extract_joints(
    base_folder,
    new_base_folder,
    smplh_folder,
    jointstype,
    batch_size,
    gender,
    use_betas,
    device,
    force_redo,
):
    print(
        "Extract joint position ({}) from SMPL pose parameter, {} betas and {}".format(
            jointstype,
            "with" if use_betas else "without",
            "with {gender} body shape"
            if gender != "gendered"
            else "with original gender",
        )
    )
    print("The processed motions will be stored in this folder:")
    print(new_base_folder)

    smplh = load_smplh_gender(gender, smplh_folder, jointstype, batch_size, device)

    iterator = loop_amams(
        base_folder,
        new_base_folder,
        ext=".npz",
        newext=".npy",
        force_redo=force_redo,
    )

    for motion_path, new_motion_path in iterator:
        data = np.load(motion_path)

        # process sequences
        poses = torch.from_numpy(data["poses"]).to(torch.float).to(device)
        trans = torch.from_numpy(data["trans"]).to(torch.float).to(device)

        if use_betas and "betas" in data and data["betas"] is not None:
            betas = torch.from_numpy(data["betas"]).to(torch.float).to(device)
        else:
            betas = None

        if gender == "gendered":
            gender_motion = str(data["gender"])
            smplh_layer = smplh[gender_motion]
        else:
            smplh_layer = smplh

        joints = smplh_layer(poses, trans, betas).cpu().numpy()
        np.save(new_motion_path, joints)

def extract_joints_smpldata(
    smpldata, fps, value_from="joints", smpl_layer=None, first_angle=np.pi, root_offset=np.zeros([1,1,3]), device='cpu', **kwargs
):
    if value_from == "smpl":
        assert smpl_layer is not None

    smpldata["mocap_framerate"] = fps
    poses = smpldata["poses"].to(device)
    trans = smpldata["trans"].to(device)
    joints = smpldata["joints"]

    if value_from == "smpl":
        with torch.no_grad():
            vertices, joints = smpl_layer(poses.float(), trans.float())
        output = {
            "vertices": vertices.cpu().numpy()-root_offset, #root offset inconsistency from smplrifke feature extraction
            "joints": joints.cpu().numpy()-root_offset,
            "smpldata": smpldata,
        }
    elif value_from == "joints":
        output = {"joints": joints.numpy()}
    else:
        raise NotImplementedError
    return output

def main():
    base_folder = "dataset/AMASS_20.0_fps_nh"
    smplh_folder = "data_loaders/amasstools/deps/smplh"
    jointstype = "smpljoints"
    batch_size = 4096
    gender = "neutral"
    use_betas = False
    force_redo = False

    name = os.path.split(base_folder)[1]
    new_base_folder = f"dataset/{name}_{jointstype}_{gender}_{'betas' if use_betas else 'nobetas'}"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    extract_joints(
        base_folder,
        new_base_folder,
        smplh_folder,
        jointstype,
        batch_size,
        gender,
        use_betas,
        device,
        force_redo,
    )


if __name__ == "__main__":
    main()
