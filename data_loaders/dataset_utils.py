from data_loaders.humanml_utils import HML_LOWER_BODY_JOINTS, HML_LEFT_LEG_JOINTS, HML_RIGHT_LEG_JOINTS
from scipy.ndimage import gaussian_filter1d
import scipy.interpolate as interpolate
import random
import torch
import numpy as np
import einops


import random
import numpy as np
import einops
from scipy.ndimage import gaussian_filter1d
import scipy.interpolate as interpolate


def motion_artifacts_smpl(poses, trans, mode='train', enable=True):
    """Add motion artifacts (jittering / foot sliding / over smooth / drifting)."""
    N, J, D = poses.shape
    mlen = len(poses)
    candidates = ['jittering', 'foot sliding', 'over smooth', 'drifting']
    aug_types = random.sample(candidates, random.randint(1, len(candidates)))
    det_mask = np.zeros_like(trans[:, 0])

    if not enable:
        poses_copy, trans_copy, det_mask_copy = poses.copy(), trans.copy(), det_mask.copy()
    if mlen < 15:
        return poses, trans, det_mask

    for aug_type in aug_types:
        if mode == 'train':
            aug_length = min(mlen - 2, int(random.randint(5, min(50, mlen - 2)) * 5))
        else:
            aug_length = random.randint(min(20, int(0.8 * mlen)), min(40, int(0.8 * mlen)))

        aug_interval = random.randint(1, mlen - aug_length)
        det_mask[aug_interval - 1: aug_interval + aug_length + 1] = 1

        if aug_type == 'jittering':
            base_scale = 0.5
            gaussian_noise_std = 0.1
            rd = np.random.random() * 0.5 + 0.5
            noise_term = np.clip(np.random.randn(N, J, D) * gaussian_noise_std * rd, -base_scale, base_scale)

            joints_select_id = np.random.choice(np.array([0, 1, 2, 3]), p=np.array([0.4, 0.3, 0.15, 0.15]))
            if joints_select_id == 0:
                joints_selected = random.sample(list(range(J)), random.randint(1, J))
            elif joints_select_id == 1:
                from data_loaders.humanml_utils import HML_LOWER_BODY_JOINTS
                joints_selected = HML_LOWER_BODY_JOINTS[random.choice((0, 1, 3, 5, 7)):]
            elif joints_select_id == 2:
                from data_loaders.humanml_utils import HML_LEFT_LEG_JOINTS
                joints_selected = HML_LEFT_LEG_JOINTS[random.choice(range(len(HML_LEFT_LEG_JOINTS))):]
            else:
                from data_loaders.humanml_utils import HML_RIGHT_LEG_JOINTS
                joints_selected = HML_RIGHT_LEG_JOINTS[random.choice(range(len(HML_RIGHT_LEG_JOINTS))):]

            poses[aug_interval: aug_interval + aug_length][:, joints_selected] += \
                noise_term[aug_interval: aug_interval + aug_length][:, joints_selected]

            if np.random.random() < 0.25:
                radius = round(6 * (np.random.random() * 2 + 2))
                poses[aug_interval: aug_interval + aug_length][:, joints_selected] = gaussian_filter1d(
                    poses[:, joints_selected],
                    sigma=4,
                    axis=0,
                    radius=radius,
                    mode='nearest',
                )[aug_interval: aug_interval + aug_length]

        elif aug_type == 'over smooth':
            radius = round(6 * (np.random.random() * 2 + 2))
            poses[aug_interval: aug_interval + aug_length] = gaussian_filter1d(
                poses,
                sigma=4,
                axis=0,
                radius=radius,
                mode='nearest',
            )[aug_interval: aug_interval + aug_length]

        elif aug_type == 'foot sliding':
            root_disp = trans[:, :2].copy()
            root_origin = trans.copy()
            root_origin[:, :2] -= root_disp
            root_vel = root_disp[1:] - root_disp[:-1]
            root_vel = np.concatenate((np.zeros_like(root_vel[[0]]), root_vel), axis=0)

            scale = 0.1
            trans_out = np.zeros_like(trans)
            temp_len = len(trans_out)
            diag_vec = np.ones((temp_len,))
            diag_vec[aug_interval: aug_interval + aug_length] += scale * np.random.random((aug_length,))
            disp_matrix = np.triu(np.broadcast_to(diag_vec[:, None], (temp_len, temp_len)))
            trans_out[:, :2] += einops.einsum(root_vel, disp_matrix, "n c, n t -> t c")
            trans_out[:, 2] += trans[:, 2]
            trans = trans_out

        elif aug_type == 'drifting':
            root_drift_dir = np.random.randn(1, 2) + np.random.randn(aug_length, 2) * 0.1
            root_drift_vel = np.random.random((aug_length, 1)) * 0.025
            root_drift_dir /= np.linalg.norm(root_drift_dir, keepdims=True)
            root_drift_vel = root_drift_vel * root_drift_dir
            root_drift_dist = np.cumsum(root_drift_vel, axis=0)
            trans[aug_interval: aug_interval + aug_length][:, :2] += root_drift_dist
            trans[aug_interval + aug_length:][:, :2] += root_drift_dist[-1:]
        else:
            raise NotImplementedError

    if not enable:
        return poses_copy, trans_copy, det_mask_copy
    return poses, trans, det_mask

def foot_slidedetect_zup(_positions, velfactor=0.002):
    """
    Simplest boosted detector using horizon velocity + height thresholds (meters).
    """
    thresh_height = 0.10
    thresh_vel = 0.10
    fps = 20
    FEET2METER = 0.3048
    foot_joint_index_list = [7, 10, 8, 11]
    joints = _positions.clone()
    joints_foot = joints[:, foot_joint_index_list, :] * FEET2METER
    offseth = joints[:, :, 2].min() * FEET2METER
    joints_feet_horizon_vel = torch.linalg.norm(
        joints_foot[1:, :, [0, 1]] - joints_foot[:-1, :, [0, 1]], dim=-1
    ) * fps
    joints_feet_height = joints_foot[0:-1, :, 2] - offseth
    skating_left = (joints_feet_horizon_vel[:, 0] > thresh_vel) * (joints_feet_horizon_vel[:, 1] > thresh_vel) * \
                   (joints_feet_height[:, 0] < (thresh_height + 0.05)) * (joints_feet_height[:, 1] < thresh_height)
    skating_right = (joints_feet_horizon_vel[:, 2] > thresh_vel) * (joints_feet_horizon_vel[:, 3] > thresh_vel) * \
                    (joints_feet_height[:, 2] < (thresh_height + 0.05)) * (joints_feet_height[:, 3] < thresh_height)
    slide_fc = torch.logical_and(skating_left, skating_right).unsqueeze(-1)
    slide_fc = torch.cat([slide_fc, slide_fc[-1:]], dim=0)
    return slide_fc