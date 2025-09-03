from typing import Optional

import numpy as np
import torch
from torch import nn

from einops import rearrange, repeat
from torch import Tensor

from functools import reduce
import operator

from .smplx_hack import SMPLHLayer
from .geometry import to_matrix

# EXTRACTOR from SMPLH layer
# replace the "left_hand", "right_hand" by "left_index1", "right_index1" of SMPLH
# fmt: off
JOINTS_EXTRACTOR = {
    "smpljoints": np.array([0, 1, 2, 3, 4, 5,
                            6, 7, 8, 9, 10, 11,
                            12, 13, 14, 15, 16,
                            17, 18, 19, 20, 21,
                            22, 37])
}
# fmt: on


def call_by_chunks(
    function,
    nelements: int,
    batch_size: int,
    parameters_dict_to_chunk: dict,
    other_parameters: dict = {},
):
    for chunk in range(int((nelements - 1) / batch_size) + 1):
        params = other_parameters.copy()
        cslice = slice(chunk * batch_size, (chunk + 1) * batch_size)

        for key, val in parameters_dict_to_chunk.items():
            params[key] = val[cslice] if val is not None else val
        yield function(**params)


def extract_data(smpl_data, jointstype):
    assert jointstype in ["smpljoints", "vertices", "both"]

    if jointstype == "vertices":
        return smpl_data.vertices

    joints = smpl_data.joints
    if jointstype == "both":
        extractor = JOINTS_EXTRACTOR["smpljoints"]
    else:
        extractor = JOINTS_EXTRACTOR[jointstype]
    joints = joints[..., extractor, :]

    if jointstype == "both":
        return smpl_data.vertices, joints
    return joints


class SMPLH(nn.Module):
    def __init__(
        self,
        path: str,
        jointstype: str = "smpljoints",
        input_pose_rep: str = "matrix",
        batch_size: int = 512,
        num_betas: int = 16,
        gender: str = "neutral",
        **kwargs
    ) -> None:
        super().__init__()
        self.batch_size = batch_size
        self.input_pose_rep = input_pose_rep
        self.jointstype = jointstype

        # Remove annoying print
        # with contextlib.redirect_stdout(None):
        self.smplh = SMPLHLayer(
            path, ext="npz", gender=gender, num_betas=num_betas
        ).eval()

        self.faces = self.smplh.faces

        self.eval()

        for p in self.parameters():
            p.requires_grad = False

    def train(self, mode: bool = True):
        # override it to be always false
        self.training = False
        for module in self.children():
            module.train(False)
        return self

    def forward(
        self,
        poses,
        trans,
        betas: Optional = None,
        jointstype: Optional[str] = None,
        input_pose_rep: Optional[str] = None,
        batch_size: Optional[int] = None,
    ) -> Tensor:
        # Take values from init if not specified there
        jointstype = self.jointstype if jointstype is None else jointstype
        batch_size = self.batch_size if batch_size is None else batch_size
        input_pose_rep = (
            self.input_pose_rep if input_pose_rep is None else input_pose_rep
        )

        needs_to_squeeze = False
        if len(trans.shape) == 2:
            needs_to_squeeze = True
            poses = poses[None]
            trans = trans[None]

        if len(poses.shape) == len(trans.shape):
            poses = rearrange(poses, "b l (p t) -> b l p t", t=3)

        matrix_poses = to_matrix(input_pose_rep, poses)

        save_shape_bs_len = matrix_poses.shape[:-3]
        nposes = reduce(operator.mul, save_shape_bs_len, 1)

        if matrix_poses.shape[-3] == 52:
            nohands = False
        elif matrix_poses.shape[-3] == 22:
            nohands = True
        else:
            print(matrix_poses.shape)
            raise NotImplementedError("Could not parse the poses.")

        # Reshaping
        matrix_poses = matrix_poses.reshape((nposes, *matrix_poses.shape[-3:]))
        global_orient = matrix_poses[:, 0]

        if trans is None:
            trans = torch.zeros(
                (*save_shape_bs_len, 3), dtype=poses.dtype, device=poses.device
            )

        trans_all = trans.reshape((nposes, *trans.shape[-1:]))

        body_pose = matrix_poses[:, 1:22]
        if nohands:
            # still axis angle
            left_hand_pose = self.smplh.left_hand_mean.reshape(15, 3)
            left_hand_pose = to_matrix("axisangle", left_hand_pose)
            left_hand_pose = left_hand_pose[None].repeat((nposes, 1, 1, 1))

            right_hand_pose = self.smplh.right_hand_mean.reshape(15, 3)
            right_hand_pose = to_matrix("axisangle", right_hand_pose)
            right_hand_pose = right_hand_pose[None].repeat((nposes, 1, 1, 1))
        else:
            hand_pose = matrix_poses[:, 22:]
            left_hand_pose = hand_pose[:, :15]
            right_hand_pose = hand_pose[:, 15:]

        n = len(body_pose)

        if betas is not None:
            if len(betas.shape) == 1:
                # repeat betas
                betas = repeat(betas, "x -> b x", b=len(global_orient))
            else:
                # need to implement
                __import__("ipdb").set_trace()

        parameters = {
            "global_orient": global_orient,
            "body_pose": body_pose,
            "left_hand_pose": left_hand_pose,
            "right_hand_pose": right_hand_pose,
            "transl": trans_all,
            "betas": betas,
        }

        # run smplh model, split by chunks to fit in memory
        outputs = []
        for smpl_output in call_by_chunks(self.smplh, n, batch_size, parameters):
            outputs.append(extract_data(smpl_output, jointstype))

        if jointstype != "both":
            outputs = torch.cat(outputs)
            outputs = outputs.reshape((*save_shape_bs_len, *outputs.shape[1:]))

            if needs_to_squeeze:
                outputs = outputs.squeeze(0)
            return outputs
        else:
            out = []
            for idx in range(2):
                output = torch.cat([x[idx] for x in outputs])
                output = output.reshape((*save_shape_bs_len, *output.shape[1:]))

                if needs_to_squeeze:
                    output = output.squeeze(0)
                out.append(output)
            return (*out,)


def load_smplh_gender(
    gender, smplh_folder, jointstype, batch_size, device, input_pose_rep="axisangle"
):
    if gender != "gendered":
        # only load one
        smplh = SMPLH(
            smplh_folder,
            input_pose_rep=input_pose_rep,
            jointstype=jointstype,
            gender=gender,
            batch_size=batch_size,
        ).to(device)
        return smplh

    # else load all of them
    smplh = {
        g: SMPLH(
            smplh_folder,
            input_pose_rep=input_pose_rep,
            jointstype=jointstype,
            gender=g,
            batch_size=batch_size,
        ).to(device)
        for g in ["male", "female", "neutral"]
    }
    return smplh
