import torch
import numpy as np
import os
import random
import codecs as cs
from tqdm import tqdm
from torch.utils.data import Dataset

from data_loaders.smpl_collate import collate_motion
from data_loaders.amasstools.globsmplrifke_feats import (
    smpldata_to_alignglobsmplrifkefeats,
)
from smplx.lbs import batch_rigid_transform
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
from utils.normalizer import Normalizer


class AMASSMotionLoader:
    """
    Loads a SMPL motion clip, crops a segment, computes joints (if missing),
    converts to aligned Global SMPL RIFKE features, appends label channel,
    and (optionally) normalizes.

    Args:
        base_dir (str): Root folder of AMASS .npz files.
        fps (int): Frames per second of stored motions (used for crop lengths).
        disable (bool): If True, skip normalization/label-channel adjustments.
        ext (str): File extension for motion files (default: ".npz").
        umin_s (float): Min crop length (seconds).
        umax_s (float): Max crop length (seconds).
        mode (str): Split name, e.g., "train" or "test".
        **kwargs:
            normalizer_dir (str): Directory for Normalizer stats (required if not disable).
    """

    def __init__(
        self, base_dir, fps=20, disable: bool = False, ext=".npz", umin_s=5.0, umax_s=5.0, mode="train", **kwargs
    ):
        self.fps = fps
        self.base_dir = base_dir
        self.motions = {}
        self.disable = disable
        self.ext = ext
        self.umin = int(self.fps * umin_s)
        assert self.umin > 0
        self.umax = int(self.fps * umax_s)
        self.mode = mode

        j_regressor_stat = np.load("data_loaders/amasstools/smpl_neutral_nobetas_24J.npz")
        self.J_regressor = torch.from_numpy(j_regressor_stat["J"]).to(torch.double)
        self.parents = torch.from_numpy(j_regressor_stat["parents"])
        if not disable:
            normalizer_dir = kwargs.get("normalizer_dir", None)
            assert normalizer_dir is not None, "Please provide normalizer_dir when not disable"
            self.motion_normalizer = Normalizer(base_dir=normalizer_dir)
            self.motion_normalizer.add_label_channel()

    def __call__(self, path):
        """
        Load and process a single motion clip.

        Args:
            path (str): Relative path (without extension). May include ",start,duration" for slicing.

        Returns:
            dict: {"x": Tensor[T, F], "length": int}
        """
        path_meta = path.strip().split(",")
        data_path = path_meta[0]
        motion_path = os.path.join(self.base_dir, data_path + self.ext)
        smpl_data = np.load(motion_path)

        poses = smpl_data["poses"].copy()
        trans = smpl_data["trans"].copy()
        labels = smpl_data["labels"].copy() if "labels" in smpl_data else np.zeros_like(trans[:, 0])
        joints = smpl_data["joints"].copy() if "joints" in smpl_data else None

        poses = torch.from_numpy(poses)
        trans = torch.from_numpy(trans)  # (T, 3)
        labels = torch.from_numpy(labels)  # (T,)
        joints = torch.from_numpy(joints) if joints is not None else None

        # Determine crop [start:start+duration]
        mlen = len(trans)
        if len(path_meta) == 3:
            start = eval(path_meta[1])
            duration = eval(path_meta[2])
        else:
            duration = random.randint(min(self.umin, mlen), min(self.umax, mlen))
            start = random.randint(0, max(mlen - duration, 0))

        poses = poses[start : start + duration]
        trans = trans[start : start + duration]
        joints = joints[start : start + duration] if joints is not None else None
        labels = labels[start : start + duration]

        # Compute joints if missing, using SMPL kinematics
        if joints is None:
            _poses = einops.rearrange(poses, "k (l t) -> k l t", t=3)
            rot_mat = axis_angle_to_matrix(_poses)
            T = rot_mat.shape[0]
            zero_hands_rot = torch.eye(3)[None, None].expand(T, 2, -1, -1)
            rot_mat = torch.concat((rot_mat, zero_hands_rot), dim=1).to(torch.double)
            joints, _ = batch_rigid_transform(
                rot_mat,
                self.J_regressor[None].expand(T, -1, -1),
                self.parents,
            )
            joints = joints.squeeze() + trans.unsqueeze(1)
            joints = joints.float()

        smpl_data = {"poses": poses, "joints": joints, "trans": trans}

        # Convert to aligned Global SMPL RIFKE features.
        motion = smpldata_to_alignglobsmplrifkefeats(smpl_data).to(torch.float)
        motion = torch.cat([motion, labels[:, None]], axis=-1)

        # Optional normalization
        if not self.disable:
            motion = self.motion_normalizer(motion)

        return {"x": motion, "length": len(motion)}


def read_split(path, split):
    """
    Read IDs from data_loaders/splits/{split}.txt.
    """
    split_file = os.path.join(path, "splits", split + ".txt")
    id_list = []
    with cs.open(split_file, "r") as f:
        for line in f.readlines():
            id_list.append(line.strip())
    return id_list


class MotionDataset(Dataset):
    """
    Thin dataset wrapper around a motion loader callable.
    """

    def __init__(self, motion_loader, split: str = "train", preload: bool = False):
        self.collate_fn = collate_motion
        self.split = split
        self.keyids = read_split("data_loaders", split)
        self.motion_loader = motion_loader
        self.is_training = "train" in split

        if preload:
            for _ in tqdm(self, desc="Preloading the dataset"):
                continue

    def __len__(self):
        return len(self.keyids)

    def __getitem__(self, index):
        keyid = self.keyids[index]
        return self.load_keyid(keyid)

    def load_keyid(self, keyid):
        """
        Load a single example by key id.
        """
        file_path = keyid.strip(".npy")
        motion_x_dict = self.motion_loader(path=file_path)
        x = motion_x_dict["x"]
        length = motion_x_dict["length"]
        return {"x": x, "keyid": keyid, "length": length}


if __name__ == "__main__":
    from utils.normalizer import Normalizer

    # Example: compute normalization stats from corrupted-canonicalized data.
    motion_loader = AMASSMotionLoader(
        base_dir="dataset/AMASS_20.0_fps_nh_globsmpl_corrupted_cano",
        disable=True,
    )
    motion_dataset = MotionDataset(motion_loader=motion_loader, split="train")

    motion_normalizer = Normalizer(
        base_dir="dataset/meta_AMASS_20.0_fps_nh_globsmpl_corrupted_cano",
        disable=True,
    )

    # Accumulate features and compute mean/std (excluding label channel).
    # Tip: you may manually adjust the scale of root features standard deviation
    data_bank = []
    for _ in range(3):
        data_bank += [x["x"] for x in tqdm(motion_dataset)]
    motionfeats = torch.cat(data_bank)
    mean_motionfeats = motionfeats.mean(0)[:-1]
    std_motionfeats = motionfeats.std(0)[:-1]

    motion_normalizer.save(mean_motionfeats, std_motionfeats)