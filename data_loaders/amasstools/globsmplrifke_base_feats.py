import torch
import einops
from torch import Tensor
import warnings
from data_loaders.amasstools.geometry import (
    axis_angle_to_matrix,
    matrix_to_rotation_6d,
    rotation_6d_to_matrix,
    matrix_to_euler_angles,
    axis_angle_rotation,
    matrix_to_axis_angle,
)
import numpy as np


def smpldata_to_globsmplrifkefeats(smpldata) -> Tensor:
    poses = smpldata["poses"].clone()
    trans = smpldata["trans"].clone()
    joints = smpldata["joints"].clone()

    # Sequence-level checks
    assert poses.shape[-1] == 66  # 22 * 3 (SMPL without hands)
    assert len(poses.shape) == 2
    assert len(trans.shape) == 2
    assert len(joints.shape) == 3

    # --- Joints preprocessing ---
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
    assert (joints[:, 0] == 0).all()
    joints = joints[:, 1:]

    # --- Rotations ---
    poses = einops.rearrange(poses, "k (l t) -> k l t", t=3)
    poses_mat = axis_angle_to_matrix(poses)
    global_orient = poses_mat[:, 0]

    global_euler = matrix_to_euler_angles(global_orient, "ZYX")
    rotZ_angle, rotY_angle, rotX_angle = torch.unbind(global_euler, -1)
    assert ~torch.any(torch.isnan(rotZ_angle))

    rotZ = axis_angle_rotation("Z", rotZ_angle)
    rotY = axis_angle_rotation("Y", rotY_angle)
    rotX = axis_angle_rotation("X", rotX_angle)
    rotZ_2d = rotZ[:, :2, 0]

    # Reconstruction check
    global_orient_recons = rotZ @ rotY @ rotX
    if torch.abs(global_orient - global_orient_recons).mean() > 1e-6:
        warnings.warn("sanity check: global_orient recon error > 1e-6", Warning)

    # Local global orientation (remove final Z)
    global_orient_local = rotY @ rotX

    # Rotate joints into local frame
    joints_local = torch.einsum("tkj,tlk->tlj", rotZ[:, :2, :2], joints[:, :, [0, 1]])
    joints_local = torch.stack((joints_local[..., 0], joints_local[..., 1], joints[..., 2]), axis=-1)

    # Local pose representation
    poses_mat_local = torch.cat((global_orient_local[:, None], poses_mat[:, 1:]), dim=1)
    poses_local = matrix_to_rotation_6d(poses_mat_local)
    assert ~torch.any(torch.isnan(rotZ_2d))

    # Pack features
    features = group(root_grav_axis, trajectory, rotZ_2d, poses_local, joints_local)
    return features


def smpldata_to_alignglobsmplrifkefeats(smpldata, rotaug=False) -> Tensor:
    poses = smpldata["poses"].clone()
    trans = smpldata["trans"].clone()
    joints = smpldata["joints"].clone()

    # Sequence-level checks
    assert poses.shape[-1] == 66
    assert len(poses.shape) == 2
    assert len(trans.shape) == 2
    assert len(joints.shape) == 3

    # --- Joints preprocessing ---
    ground = joints[:, :, 2].min()
    joints[:, :, 2] -= ground
    root_grav_axis = joints[:, 0, 2].clone()

    _val = abs((trans[:, 2] - trans[0, 2]) - (root_grav_axis - root_grav_axis[0])).mean()
    assert _val < 1e-6

    trajectory = joints[:, 0, :2].clone()
    _val = torch.abs((trajectory - trajectory[0]) - (trans[:, :2] - trans[0, :2])).mean()
    assert _val < 1e-6

    joints[:, :, [0, 1]] -= trajectory[..., None, :]
    joints[:, :, 2] -= joints[:, [0], 2]
    assert (joints[:, 0] == 0).all()
    joints = joints[:, 1:]

    # Move trajectory to origin
    trajectory = trajectory - trajectory[0]

    # --- Rotations ---
    poses = einops.rearrange(poses, "k (l t) -> k l t", t=3)
    poses_mat = axis_angle_to_matrix(poses)
    global_orient = poses_mat[:, 0]

    global_euler = matrix_to_euler_angles(global_orient, "ZYX")
    rotZ_angle, rotY_angle, rotX_angle = torch.unbind(global_euler, -1)
    og_rotZ = axis_angle_rotation("Z", rotZ_angle)

    rand_rot = np.random.random() * 2 * np.pi if rotaug else 0
    rotZ_angle[0] += rand_rot
    init_rotZ = axis_angle_rotation("Z", rotZ_angle[[0]]).repeat_interleave(rotZ_angle.shape[0], dim=0)
    rotZ_angle = rotZ_angle - rotZ_angle[0] - rand_rot

    rotZ = axis_angle_rotation("Z", rotZ_angle)
    rotY = axis_angle_rotation("Y", rotY_angle)
    rotX = axis_angle_rotation("X", rotX_angle)
    rotZ_2d = rotZ[:, :2, 0]

    global_orient_local = rotY @ rotX

    # Rotate trajectory to canonical facing
    trajectory = torch.einsum("tkj,tk->tj", init_rotZ[:, :2, :2], trajectory)

    # Rotate joints
    joints_local = torch.einsum("tkj,tlk->tlj", og_rotZ[:, :2, :2], joints[:, :, [0, 1]])
    joints_local = torch.stack((joints_local[..., 0], joints_local[..., 1], joints[..., 2]), axis=-1)

    # Local pose representation
    poses_mat_local = torch.cat((global_orient_local[:, None], poses_mat[:, 1:]), dim=1)
    poses_local = matrix_to_rotation_6d(poses_mat_local)

    features = group(root_grav_axis, trajectory, rotZ_2d, poses_local, joints_local)
    return features


def globsmplrifkefeats_to_smpldata(features: Tensor):
    (root_grav_axis, trajectory, rotZ_2D, poses_local, joints_local) = ungroup(features)

    poses_mat_local = rotation_6d_to_matrix(poses_local)
    global_orient_local = poses_mat_local[:, 0]

    # Rebuild Z rotation from 2D vector
    zero = torch.zeros_like(rotZ_2D[..., 0])
    one = torch.ones_like(rotZ_2D[..., 0])
    rotZ_2D = rotZ_2D / torch.norm(rotZ_2D, dim=-1, keepdim=True)
    cos = rotZ_2D[..., 0]
    sin = rotZ_2D[..., 1]
    R_flat = (cos, -sin, zero, sin, cos, zero, zero, zero, one)
    rotZ = torch.stack(R_flat, -1).reshape(rotZ_2D.shape[:-1] + (3, 3))

    # Rotate back joints and restore pelvis/translation
    joints = torch.einsum("bjk,blk->blj", rotZ[:, :2, :2], joints_local[..., [0, 1]])
    joints = torch.stack((joints[..., 0], joints[..., 1], joints_local[..., 2]), axis=-1)
    joints = torch.cat((0 * joints[:, [0]], joints), axis=1)
    joints[:, :, 2] = joints[:, :, 2] + root_grav_axis[:, None]
    joints[:, :, [0, 1]] = joints[:, :, [0, 1]] + trajectory[:, None]
    trans = torch.cat([trajectory, root_grav_axis[..., None]], dim=1)

    # Rebuild global orientation
    global_euler_local = matrix_to_euler_angles(global_orient_local, "ZYX")
    _, rotY_angle, rotX_angle = torch.unbind(global_euler_local, -1)
    rotY = axis_angle_rotation("Y", rotY_angle)
    rotX = axis_angle_rotation("X", rotX_angle)
    global_orient = rotZ @ rotY @ rotX

    poses_mat = torch.cat((global_orient[..., None, :, :], poses_mat_local[..., 1:, :, :]), dim=-3)
    poses = matrix_to_axis_angle(poses_mat)
    poses = einops.rearrange(poses, "k l t -> k (l t)")

    smpldata = {"poses": poses, "trans": trans, "joints": joints}
    return smpldata


def alignglobsmplrifkefeats_to_smpldata(features: Tensor):
    (root_grav_axis, trajectory, rotZ_2D, poses_local, joints_local) = ungroup(features)

    poses_mat_local = rotation_6d_to_matrix(poses_local)
    global_orient_local = poses_mat_local[:, 0]

    # Normalize and build Z-rotation
    zero = torch.zeros_like(rotZ_2D[..., 0])
    one = torch.ones_like(rotZ_2D[..., 0])
    rotZ_2D /= torch.norm(rotZ_2D, dim=-1, keepdim=True)
    cos = rotZ_2D[..., 0]
    sin = rotZ_2D[..., 1]
    R_flat = (cos, -sin, zero, sin, cos, zero, zero, zero, one)
    rotZ = torch.stack(R_flat, -1).reshape(rotZ_2D.shape[:-1] + (3, 3))

    # Align by first-frame rotation
    init_rotZ = rotZ[[0]].repeat_interleave(rotZ.shape[0], dim=0)
    rotZ = torch.einsum("jl,bjk->blk", rotZ[0], rotZ)
    trajectory = torch.einsum("bkj,bk->bj", init_rotZ[:, :2, :2], trajectory)

    # Rotate back joints and restore pelvis/translation
    joints = torch.einsum("bjk,blk->blj", rotZ[:, :2, :2], joints_local[..., [0, 1]])
    joints = torch.stack((joints[..., 0], joints[..., 1], joints_local[..., 2]), axis=-1)
    trajectory = trajectory - trajectory[0]
    joints = torch.cat((0 * joints[:, [0]], joints), axis=1)
    joints[:, :, 2] += root_grav_axis[:, None]
    joints[:, :, [0, 1]] += trajectory[:, None]
    trans = torch.cat([trajectory, root_grav_axis[..., None]], dim=1)

    # Rebuild global orientation
    global_euler_local = matrix_to_euler_angles(global_orient_local, "ZYX")
    _, rotY_angle, rotX_angle = torch.unbind(global_euler_local, -1)
    rotY = axis_angle_rotation("Y", rotY_angle)
    rotX = axis_angle_rotation("X", rotX_angle)
    global_orient = rotZ @ rotY @ rotX

    poses_mat = torch.cat((global_orient[..., None, :, :], poses_mat_local[..., 1:, :, :]), dim=-3)
    poses = matrix_to_axis_angle(poses_mat)
    poses = einops.rearrange(poses, "k l t -> k (l t)")

    smpldata = {"poses": poses, "trans": trans, "joints": joints}
    return smpldata


def group(root_grav_axis, trajectory, rotZ_2D, poses_local, joints_local):
    # Flatten and pack features: total size should be 206
    poses_local_flatten = einops.rearrange(poses_local, "k l t -> k (l t)")
    joints_local_flatten = einops.rearrange(joints_local, "k l t -> k (l t)")
    features, _ = einops.pack(
        [root_grav_axis, trajectory, rotZ_2D, poses_local_flatten, joints_local_flatten],
        "k *",
    )
    assert features.shape[-1] == 206
    return features


def ungroup(features: Tensor) -> tuple[Tensor]:
    assert features.shape[-1] == 206
    (
        root_grav_axis,
        trajectory,
        rotZ_2D,
        poses_local_flatten,
        joints_local_flatten,
    ) = einops.unpack(features, [[], [2], [2], [132], [69]], "k *")

    poses_6d = einops.rearrange(poses_local_flatten, "k (l t) -> k l t", t=6)
    joints = einops.rearrange(joints_local_flatten, "k (l t) -> k l t", t=3)

    return root_grav_axis, trajectory, rotZ_2D, poses_6d, joints


def canonicalize_rotation(smpldata):
    # Canonicalize global rotation (and translation) via feature round-trip.
    features = smpldata_to_globsmplrifkefeats(smpldata)
    return globsmplrifkefeats_to_smpldata(features)