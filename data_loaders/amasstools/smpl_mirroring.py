import os
import numpy as np

import torch
from .loop_amass import loop_amams

# FROM
# https://github.com/mkocabas/PARE/blob/master/pare/core/constants.py
# Permutation of SMPL pose parameters when flipping the shape
# fmt: off
SMPL_JOINTS_FLIP_PERM = [
    0, 2, 1, 3, 5, 4, 6, 8, 7, 9,
    11, 10, 12, 14, 13, 15, 17, 16, 19, 18,
    21, 20,
]  # Hands are removed: # , 23, 22]
# fmt: on

SMPL_POSE_FLIP_PERM = []
for i in SMPL_JOINTS_FLIP_PERM:
    SMPL_POSE_FLIP_PERM.append(3 * i)
    SMPL_POSE_FLIP_PERM.append(3 * i + 1)
    SMPL_POSE_FLIP_PERM.append(3 * i + 2)

# FROM
# https://github.com/mkocabas/PARE/blob/master/pare/utils/image_utils.py#L220


def flip_pose(_pose):
    """Flip pose.
    The flipping is based on SMPL parameters.
    """
    pose = _pose.clone()
    flipped_parts = SMPL_POSE_FLIP_PERM
    pose = pose[..., flipped_parts]
    # we also negate the second and the third dimension of the axis-angle
    pose[..., 1::3] = -pose[..., 1::3]
    pose[..., 2::3] = -pose[..., 2::3]
    return pose


def mirror_smpl(
    base_folder,
    new_base_folder,
    force_redo,
):
    print("Mirror SMPL pose parameters")
    print("The processed motions will be stored in this folder:")
    print(new_base_folder)

    iterator = loop_amams(
        base_folder,
        new_base_folder,
        ext=".npz",
        newext=".npz",
        force_redo=force_redo,
    )

    for motion_path, new_motion_path in iterator:
        if "humanact12" in motion_path:
            continue

        if new_base_folder in motion_path:
            continue

        # not mirroing again
        if "M/" in motion_path:
            continue

        data = {x: y for x, y in np.load(motion_path).items()}

        # process sequences
        poses = torch.from_numpy(data["poses"])
        trans = torch.from_numpy(data["trans"])

        # flip poses
        assert poses.shape[-1] == 66
        mirror_poses = flip_pose(poses)

        # flip trans
        x, y, z = torch.unbind(trans, dim=-1)
        mirrored_trans = torch.stack((-x, y, z), axis=-1)

        data["poses"] = mirror_poses.numpy()
        data["trans"] = mirrored_trans.numpy()

        np.savez(new_motion_path, **data)

    return new_base_folder


def main():
    base_folder = "dataset/AMASS_20.0_fps_nh"
    new_base_folder = os.path.join(base_folder, "M/")
    # put the mirror motions in the M/ subfolder

    force_redo = False

    mirror_smpl(
        base_folder,
        new_base_folder,
        force_redo,
    )


if __name__ == "__main__":
    main()
