import torch
import numpy as np
import os
import codecs as cs
from tqdm import tqdm
from torch.utils.data import Dataset

from data_loaders.smpl_collate import collate_motion
from data_loaders.dataset_utils import foot_slidedetect_zup, motion_artifacts_smpl
from data_loaders.amasstools.globsmplrifke_base_feats import (
    smpldata_to_globsmplrifkefeats,
    ungroup,
    canonicalize_rotation,
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


class UtilAMASSMotionLoader:
    """
    Utility loader that:
      1) Loads SMPL sequences (.npz).
      2) Injects synthetic artifacts (optional).
      3) Recomputes joints.
      4) Canonicalizes rotation/translation.
      5) Saves corrupted + canonicalized sequences to `save_dir`.
      6) Returns Global SMPL RIFKE features (+ optional label channel).

    Args:
        base_dir (str): Root folder with input sequences (npz).
        fps (int): Motion framerate (unused here; kept for parity).
        disable (bool): If True, skip normalization/label add-ons (not used here).
        ext (str): Extension of input files ('.npz' expected).
        mode (str): 'train' or 'test' (controls artifact ranges).
        artifacts (bool): Enable/disable artifact injection.
        save_dir (str): Output root to store processed npz files.
        **kwargs: Ignored extra args for interface compatibility.
    """

    def __init__(
        self, base_dir, fps=20, disable: bool = False, ext=".npz", mode="train", artifacts=True, save_dir=None, **kwargs
    ):
        self.fps = fps
        self.base_dir = base_dir
        self.motions = {}
        self.disable = disable
        self.ext = ext
        self.artifacts = artifacts
        self.mode = mode
        self.save_dir = save_dir
        assert self.save_dir is not None, "Please provide save_dir to store corrupted data"

        j_regressor_stat = np.load("data_loaders/amasstools/smpl_neutral_nobetas_24J.npz")
        self.J_regressor = torch.from_numpy(j_regressor_stat["J"]).to(torch.double)
        self.parents = torch.from_numpy(j_regressor_stat["parents"])

    def __call__(self, path):
        """
        Process a single sequence by relative path (without extension).
        Returns:
            dict: {"x": Tensor[T, F], "length": T}
        """
        if path not in self.motions:
            motion_path = os.path.join(self.base_dir, path + self.ext)
            try:
                motion = np.load(motion_path)
                self.motions[path] = motion
            except Exception:
                print("Cannot loaded")

        motion = self.motions[path]

        if self.ext == ".npz":
            smpl_data = motion
            poses = smpl_data["poses"].copy()   # (T, 66) axis-angle flattened
            trans = smpl_data["trans"].copy()   # (T, 3)

            # Convert axis-angle -> quaternion for artifact injection, then back.
            _poses = einops.rearrange(poses, "k (l t) -> k l t", t=3)  # (T, 22, 3)
            poses_quat = axis_angle_to_quaternion(torch.from_numpy(_poses)).numpy()

            poses_quat, trans, det_mask = motion_artifacts_smpl(poses_quat, trans, self.mode, self.artifacts)
            _poses = quaternion_to_axis_angle(torch.from_numpy(poses_quat))  # (T, 22, 3)
            poses = einops.rearrange(_poses, "k l t -> k (l t)", t=3)
            trans = torch.from_numpy(trans)  # (T, 3)
            det_mask = torch.from_numpy(det_mask)

            # Recompute joints (append two dummy hand joints to match 24).
            rot_mat = axis_angle_to_matrix(_poses)  # (T, 22, 3, 3)
            T = rot_mat.shape[0]
            zero_hands_rot = torch.eye(3)[None, None].expand(T, 2, -1, -1)
            rot_mat = torch.concat((rot_mat, zero_hands_rot), dim=1).to(torch.double)

            joints, _ = batch_rigid_transform(
                rot_mat,
                self.J_regressor[None].expand(T, -1, -1),
                self.parents,
            )
            joints = joints.squeeze() + trans.unsqueeze(1)

            smpl_data = {
                "poses": poses.to(torch.double),
                "trans": trans.to(torch.double),
                "joints": joints.to(torch.double),
            }

            # Canonicalize; optionally detect foot sliding in canonical space.
            cano_smpl_data = canonicalize_rotation(smpl_data)
            if ENABLE_SLIDEDET:
                slide_label = foot_slidedetect_zup(cano_smpl_data["joints"].clone())
            else:
                slide_label = torch.zeros_like(det_mask)

            det_mask = ((det_mask + slide_label.squeeze()) > 0).to(torch.float)
            cano_smpl_data["labels"] = det_mask
            cano_smpl_data = {k: v.numpy() for k, v in cano_smpl_data.items()}

            # Save canonicalized (possibly corrupted) npz.
            motion_path = os.path.join(self.base_dir, path + self.ext)
            save_path = os.path.join(self.save_dir, path + self.ext)
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            np.savez(save_path, **cano_smpl_data)

            # Return features (+ 1 label channel).
            motion = smpldata_to_globsmplrifkefeats(smpl_data).to(torch.float)
            motion = torch.cat([motion, det_mask[:, None]], axis=-1)
        else:
            raise NotImplementedError

        return {"x": motion, "length": len(motion)}


def read_split(path, split):
    """
    Read a split file and return the list of ids (one per line).
    """
    split_file = os.path.join(path, "splits", split + ".txt")
    id_list = []
    with cs.open(split_file, "r") as f:
        for line in f.readlines():
            id_list.append(line.strip())
    return id_list


class MotionDataset(Dataset):
    """
    Thin dataset wrapper around a motion loader.
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
        file_path = keyid.strip(".npy")
        motion_x_dict = self.motion_loader(path=file_path)
        x = motion_x_dict["x"]
        length = motion_x_dict["length"]
        return {"x": x, "keyid": keyid, "length": length}


if __name__ == "__main__":
    from utils.fixseed import fixseed
    from torch.utils.data import DataLoader
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--mode", type=str)
    args = parser.parse_args()

    fixseed(42)
    mode = args.mode
    ENABLE_SLIDEDET = True

    motion_loader = UtilAMASSMotionLoader(
        base_dir="dataset/AMASS_20.0_fps_nh_globsmpl_base_cano",
        ext=".npz",
        mode=mode,
        save_dir="dataset/AMASS_20.0_fps_nh_globsmpl_corrupted_cano",
    )
    dataset = MotionDataset(motion_loader=motion_loader, split=mode, preload=False)
    loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=8, drop_last=False, collate_fn=dataset.collate_fn)

    for _ in tqdm(loader):
        pass