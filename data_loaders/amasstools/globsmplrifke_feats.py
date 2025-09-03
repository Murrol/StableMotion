import torch
import einops
from torch import Tensor
from data_loaders.amasstools.geometry import (
    axis_angle_to_matrix,
    matrix_to_rotation_6d,
    rotation_6d_to_matrix,
    matrix_to_euler_angles,
    axis_angle_rotation,
    matrix_to_axis_angle,
)
import numpy as np


def my_diff(vector):
    vel_vector = torch.diff(vector, dim=0)
    # Repeat last acceleration for the final velocity step
    last_acceleration = vel_vector[-1] - vel_vector[-2]
    future_vel_vector = vel_vector[-1] + last_acceleration
    vel_vector = torch.cat((vel_vector, future_vel_vector[None]), dim=0)
    return vel_vector


def smpldata_to_alignglobsmplrifkefeats(smpldata, rotaug=False) -> Tensor:
    poses = smpldata["poses"].clone()
    trans = smpldata["trans"].clone()
    joints = smpldata["joints"].clone()

    # Sequence level checks
    assert poses.shape[-1] == 66  # 22 * 3 -> SMPL without hands
    assert len(poses.shape) == 2
    assert len(trans.shape) == 2
    assert len(joints.shape) == 3

    # --- JOINTS PROCESS ---

    # Remove ground
    ground = joints[:, :, 2].min()
    joints[:, :, 2] -= ground
    root_grav_axis = joints[:, 0, 2].clone()

    # Consistency checks
    _val = abs((trans[:, 2] - trans[0, 2]) - (root_grav_axis - root_grav_axis[0])).mean()
    assert _val < 1e-6

    trajectory = joints[:, 0, :2].clone()
    _val = torch.abs((trajectory - trajectory[0]) - (trans[:, :2] - trans[0, :2])).mean()
    assert _val < 1e-6

    # Pelvis coordinate system
    joints[:, :, [0, 1]] -= trajectory[..., None, :]
    joints[:, :, 2] -= joints[:, [0], 2]
    assert (joints[:, 0] == 0).all()  # pelvis all zeros

    # Remove pelvis
    joints = joints[:, 1:]

    # Shift trajectory to start at origin
    trajectory = trajectory - trajectory[0]

    # --- ROTATIONS ---
    poses = einops.rearrange(poses, "k (l t) -> k l t", t=3)
    poses_mat = axis_angle_to_matrix(poses)
    global_orient = poses_mat[:, 0]

    # Decompose ZYX rotations
    global_euler = matrix_to_euler_angles(global_orient, "ZYX")
    rotZ_angle, rotY_angle, rotX_angle = torch.unbind(global_euler, -1)
    og_rotZ = axis_angle_rotation("Z", rotZ_angle)

    # Random augmentation around Z axis
    rand_rot = np.random.random() * 2 * np.pi if rotaug else 0
    rotZ_angle[0] += rand_rot
    init_rotZ = axis_angle_rotation("Z", rotZ_angle[[0]]).repeat_interleave(rotZ_angle.shape[0], dim=0)
    rotZ_angle = rotZ_angle - rotZ_angle[0] - rand_rot

    rotZ = axis_angle_rotation("Z", rotZ_angle)
    rotY = axis_angle_rotation("Y", rotY_angle)
    rotX = axis_angle_rotation("X", rotX_angle)
    rotZ_2d = rotZ[:, :2, 0]

    # Global orient without Z rotation
    global_orient_local = rotY @ rotX

    # Rotate trajectory to canonical facing (1,0)
    trajectory = torch.einsum("tkj,tk->tj", init_rotZ[:, :2, :2], trajectory)

    # Rotate joints
    joints_local = torch.einsum("tkj,tlk->tlj", og_rotZ[:, :2, :2], joints[:, :, [0, 1]])
    joints_local = torch.stack(
        (joints_local[..., 0], joints_local[..., 1], joints[..., 2]), axis=-1
    )

    # Local pose representation
    poses_mat_local = torch.cat((global_orient_local[:, None], poses_mat[:, 1:]), dim=1)
    poses_local = matrix_to_rotation_6d(poses_mat_local)

    foot_idxes = [7, 10, 6, 9]  # left/right foot joints
    foot_global = joints_local[:, foot_idxes].clone()
    foot_global[:, :, [0, 1]] += trajectory.unsqueeze(1)
    foot_global[:, :, 2] += root_grav_axis.unsqueeze(1)

    vel_traj = my_diff(trajectory)
    vel_foot_global = my_diff(foot_global)

    # Stack all features
    features = group(
        root_grav_axis, trajectory, rotZ_2d,
        poses_local, joints_local, foot_global,
        vel_foot_global, vel_traj
    )
    return features


def globsmplrifkefeats_to_smpldata(features: Tensor):
    (
        root_grav_axis, 
        trajectory, 
        rotZ_2D,
        poses_local, 
        joints_local,
        foot_global, 
        vel_foot_global, 
        vel_traj
    ) = ungroup(features)

    poses_mat_local = rotation_6d_to_matrix(poses_local)
    global_orient_local = poses_mat_local[:, 0]

    zero = torch.zeros_like(rotZ_2D[..., 0])
    one = torch.ones_like(rotZ_2D[..., 0])
    rotZ_2D = rotZ_2D / torch.norm(rotZ_2D, dim=-1, keepdim=True)
    cos = rotZ_2D[..., 0]
    sin = rotZ_2D[..., 1]

    # Rotation matrix from 2D Z-rotation
    R_flat = (cos, -sin, zero, sin, cos, zero, zero, zero, one)
    rotZ = torch.stack(R_flat, -1).reshape(rotZ_2D.shape[:-1] + (3, 3))

    # Rotate back joints
    joints = torch.einsum("bjk,blk->blj", rotZ[:, :2, :2], joints_local[..., [0, 1]])
    joints = torch.stack(
        (joints[..., 0], joints[..., 1], joints_local[..., 2]), axis=-1
    )

    # Add pelvis (all zero initially)
    joints = torch.cat((0 * joints[:, [0]], joints), axis=1)

    # Add Z component and trajectory
    joints[:, :, 2] = joints[:, :, 2] + root_grav_axis[:, None]
    joints[:, :, [0, 1]] = joints[:, :, [0, 1]] + trajectory[:, None]

    # Translation from trajectory and root height
    trans = torch.cat([trajectory, root_grav_axis[..., None]], dim=1)

    # Rebuild global orientation
    global_euler_local = matrix_to_euler_angles(global_orient_local, "ZYX")
    _, rotY_angle, rotX_angle = torch.unbind(global_euler_local, -1)
    rotY = axis_angle_rotation("Y", rotY_angle)
    rotX = axis_angle_rotation("X", rotX_angle)
    global_orient = rotZ @ rotY @ rotX

    poses_mat = torch.cat(
        (global_orient[..., None, :, :], poses_mat_local[..., 1:, :, :]), dim=-3
    )
    poses = matrix_to_axis_angle(poses_mat)
    poses = einops.rearrange(poses, "k l t -> k (l t)")

    smpldata = {"poses": poses, "trans": trans, "joints": joints}
    return smpldata


def group(root_grav_axis, trajectory, rotZ_2D, poses_local, joints_local, foot_global, vel_foot_global, vel_traj):
    # Flatten tensors
    poses_local_flatten = einops.rearrange(poses_local, "k l t -> k (l t)")
    joints_local_flatten = einops.rearrange(joints_local, "k l t -> k (l t)")
    foot_global_flatten = einops.rearrange(foot_global, "k l t -> k (l t)")
    vel_foot_global_flatten = einops.rearrange(vel_foot_global, "k l t -> k (l t)")

    # Pack into a single feature vector
    features, _ = einops.pack(
        [
            root_grav_axis,
            trajectory,
            rotZ_2D,
            poses_local_flatten,
            joints_local_flatten,
            foot_global_flatten,
            vel_foot_global_flatten,
            vel_traj,
        ],
        "k *",
    )
    assert features.shape[-1] == 232
    return features


def ungroup(features: Tensor) -> tuple[Tensor]:
    assert features.shape[-1] == 232
    (
        root_grav_axis,
        trajectory,
        rotZ_2D,
        poses_local_flatten,
        joints_local_flatten,
        foot_global_flatten,
        vel_foot_global_flatten,
        vel_traj,
    ) = einops.unpack(features, [[], [2], [2], [132], [69], [12], [12], [2]], "k *")

    poses_6d = einops.rearrange(poses_local_flatten, "k (l t) -> k l t", t=6)
    joints = einops.rearrange(joints_local_flatten, "k (l t) -> k l t", t=3)
    foot_global = einops.rearrange(foot_global_flatten, "k (l t) -> k l t", t=3)
    vel_foot_global = einops.rearrange(vel_foot_global_flatten, "k (l t) -> k l t", t=3)

    return root_grav_axis, trajectory, rotZ_2D, poses_6d, joints, foot_global, vel_foot_global, vel_traj