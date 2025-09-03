# HACK to make it work with 16 blend shape for AMASS

#  -*- coding: utf-8 -*-

# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# You can only use this computer program if you have closed
# a license agreement with MPG or you get the right to use the computer
# program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and
# liable to prosecution.
#
# Copyright©2019 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems. All rights reserved.
#
# Contact: ps-license@tuebingen.mpg.de

from typing import Optional, Dict, Union
import os
import os.path as osp

import pickle

import numpy as np

import torch
import torch.nn as nn

from smplx.lbs import (
    lbs,
    blend_shapes,
)

from smplx.vertex_ids import vertex_ids as VERTEX_IDS
from smplx.utils import (
    Struct,
    to_np,
    to_tensor,
    Tensor,
    Array,
    SMPLOutput,
    SMPLHOutput,
)
from smplx.vertex_joint_selector import VertexJointSelector


class SMPL(nn.Module):
    NUM_JOINTS = 23
    NUM_BODY_JOINTS = 23
    SHAPE_SPACE_DIM = 300

    def __init__(
        self,
        model_path: str,
        kid_template_path: str = "",
        data_struct: Optional[Struct] = None,
        create_betas: bool = True,
        betas: Optional[Tensor] = None,
        num_betas: int = 10,
        create_global_orient: bool = True,
        global_orient: Optional[Tensor] = None,
        create_body_pose: bool = True,
        body_pose: Optional[Tensor] = None,
        create_transl: bool = True,
        transl: Optional[Tensor] = None,
        dtype=torch.float32,
        batch_size: int = 1,
        joint_mapper=None,
        gender: str = "neutral",
        age: str = "adult",
        vertex_ids: Dict[str, int] = None,
        v_template: Optional[Union[Tensor, Array]] = None,
        **kwargs,
    ) -> None:
        """SMPL model constructor

        Parameters
        ----------
        model_path: str
            The path to the folder or to the file where the model
            parameters are stored
        data_struct: Strct
            A struct object. If given, then the parameters of the model are
            read from the object. Otherwise, the model tries to read the
            parameters from the given `model_path`. (default = None)
        create_global_orient: bool, optional
            Flag for creating a member variable for the global orientation
            of the body. (default = True)
        global_orient: torch.tensor, optional, Bx3
            The default value for the global orientation variable.
            (default = None)
        create_body_pose: bool, optional
            Flag for creating a member variable for the pose of the body.
            (default = True)
        body_pose: torch.tensor, optional, Bx(Body Joints * 3)
            The default value for the body pose variable.
            (default = None)
        num_betas: int, optional
            Number of shape components to use
            (default = 10).
        create_betas: bool, optional
            Flag for creating a member variable for the shape space
            (default = True).
        betas: torch.tensor, optional, Bx10
            The default value for the shape member variable.
            (default = None)
        create_transl: bool, optional
            Flag for creating a member variable for the translation
            of the body. (default = True)
        transl: torch.tensor, optional, Bx3
            The default value for the transl variable.
            (default = None)
        dtype: torch.dtype, optional
            The data type for the created variables
        batch_size: int, optional
            The batch size used for creating the member variables
        joint_mapper: object, optional
            An object that re-maps the joints. Useful if one wants to
            re-order the SMPL joints to some other convention (e.g. MSCOCO)
            (default = None)
        gender: str, optional
            Which gender to load
        vertex_ids: dict, optional
            A dictionary containing the indices of the extra vertices that
            will be selected
        """

        self.gender = gender
        self.age = age

        if data_struct is None:
            if osp.isdir(model_path):
                model_fn = "SMPL_{}.{ext}".format(gender.upper(), ext="pkl")
                smpl_path = os.path.join(model_path, model_fn)
            else:
                smpl_path = model_path
            assert osp.exists(smpl_path), "Path {} does not exist!".format(smpl_path)

            with open(smpl_path, "rb") as smpl_file:
                data_struct = Struct(**pickle.load(smpl_file, encoding="latin1"))

        super(SMPL, self).__init__()
        self.batch_size = batch_size
        shapedirs = data_struct.shapedirs
        # Remove issues here
        num_betas = min(num_betas, self.SHAPE_SPACE_DIM)

        if self.age == "kid":
            v_template_smil = np.load(kid_template_path)
            v_template_smil -= np.mean(v_template_smil, axis=0)
            v_template_diff = np.expand_dims(
                v_template_smil - data_struct.v_template, axis=2
            )
            shapedirs = np.concatenate(
                (shapedirs[:, :, :num_betas], v_template_diff), axis=2
            )
            num_betas = num_betas + 1

        self._num_betas = num_betas
        shapedirs = shapedirs[:, :, :num_betas]
        # The shape components
        self.register_buffer("shapedirs", to_tensor(to_np(shapedirs), dtype=dtype))

        if vertex_ids is None:
            # SMPL and SMPL-H share the same topology, so any extra joints can
            # be drawn from the same place
            vertex_ids = VERTEX_IDS["smplh"]

        self.dtype = dtype

        self.joint_mapper = joint_mapper

        self.vertex_joint_selector = VertexJointSelector(
            vertex_ids=vertex_ids, **kwargs
        )

        self.faces = data_struct.f
        self.register_buffer(
            "faces_tensor",
            to_tensor(to_np(self.faces, dtype=np.int64), dtype=torch.long),
        )

        if create_betas:
            if betas is None:
                default_betas = torch.zeros([batch_size, self.num_betas], dtype=dtype)
            else:
                if torch.is_tensor(betas):
                    default_betas = betas.clone().detach()
                else:
                    default_betas = torch.tensor(betas, dtype=dtype)

            self.register_parameter(
                "betas", nn.Parameter(default_betas, requires_grad=True)
            )

        # The tensor that contains the global rotation of the model
        # It is separated from the pose of the joints in case we wish to
        # optimize only over one of them
        if create_global_orient:
            if global_orient is None:
                default_global_orient = torch.zeros([batch_size, 3], dtype=dtype)
            else:
                if torch.is_tensor(global_orient):
                    default_global_orient = global_orient.clone().detach()
                else:
                    default_global_orient = torch.tensor(global_orient, dtype=dtype)

            global_orient = nn.Parameter(default_global_orient, requires_grad=True)
            self.register_parameter("global_orient", global_orient)

        if create_body_pose:
            if body_pose is None:
                default_body_pose = torch.zeros(
                    [batch_size, self.NUM_BODY_JOINTS * 3], dtype=dtype
                )
            else:
                if torch.is_tensor(body_pose):
                    default_body_pose = body_pose.clone().detach()
                else:
                    default_body_pose = torch.tensor(body_pose, dtype=dtype)
            self.register_parameter(
                "body_pose", nn.Parameter(default_body_pose, requires_grad=True)
            )

        if create_transl:
            if transl is None:
                default_transl = torch.zeros(
                    [batch_size, 3], dtype=dtype, requires_grad=True
                )
            else:
                default_transl = torch.tensor(transl, dtype=dtype)
            self.register_parameter(
                "transl", nn.Parameter(default_transl, requires_grad=True)
            )

        if v_template is None:
            v_template = data_struct.v_template
        if not torch.is_tensor(v_template):
            v_template = to_tensor(to_np(v_template), dtype=dtype)
        # The vertices of the template model
        self.register_buffer("v_template", v_template)

        j_regressor = to_tensor(to_np(data_struct.J_regressor), dtype=dtype)
        self.register_buffer("J_regressor", j_regressor)

        # Pose blend shape basis: 6890 x 3 x 207, reshaped to 6890*3 x 207
        num_pose_basis = data_struct.posedirs.shape[-1]
        # 207 x 20670
        posedirs = np.reshape(data_struct.posedirs, [-1, num_pose_basis]).T
        self.register_buffer("posedirs", to_tensor(to_np(posedirs), dtype=dtype))

        # indices of parents for each joints
        parents = to_tensor(to_np(data_struct.kintree_table[0])).long()
        parents[0] = -1
        self.register_buffer("parents", parents)

        lbs_weights = to_tensor(to_np(data_struct.weights), dtype=dtype)
        self.register_buffer("lbs_weights", lbs_weights)

    @property
    def num_betas(self):
        return self._num_betas

    @property
    def num_expression_coeffs(self):
        return 0

    def create_mean_pose(self, data_struct) -> Tensor:
        pass

    def name(self) -> str:
        return "SMPL"

    @torch.no_grad()
    def reset_params(self, **params_dict) -> None:
        for param_name, param in self.named_parameters():
            if param_name in params_dict:
                param[:] = torch.tensor(params_dict[param_name])
            else:
                param.fill_(0)

    def get_num_verts(self) -> int:
        return self.v_template.shape[0]

    def get_num_faces(self) -> int:
        return self.faces.shape[0]

    def extra_repr(self) -> str:
        msg = [
            f"Gender: {self.gender.upper()}",
            f"Number of joints: {self.J_regressor.shape[0]}",
            f"Betas: {self.num_betas}",
        ]
        return "\n".join(msg)

    def forward_shape(
        self,
        betas: Optional[Tensor] = None,
    ) -> SMPLOutput:
        betas = betas if betas is not None else self.betas
        v_shaped = self.v_template + blend_shapes(betas, self.shapedirs)
        return SMPLOutput(vertices=v_shaped, betas=betas, v_shaped=v_shaped)

    def forward(
        self,
        betas: Optional[Tensor] = None,
        body_pose: Optional[Tensor] = None,
        global_orient: Optional[Tensor] = None,
        transl: Optional[Tensor] = None,
        return_verts=True,
        return_full_pose: bool = False,
        pose2rot: bool = True,
        **kwargs,
    ) -> SMPLOutput:
        """Forward pass for the SMPL model

        Parameters
        ----------
        global_orient: torch.tensor, optional, shape Bx3
            If given, ignore the member variable and use it as the global
            rotation of the body. Useful if someone wishes to predicts this
            with an external model. (default=None)
        betas: torch.tensor, optional, shape BxN_b
            If given, ignore the member variable `betas` and use it
            instead. For example, it can used if shape parameters
            `betas` are predicted from some external model.
            (default=None)
        body_pose: torch.tensor, optional, shape Bx(J*3)
            If given, ignore the member variable `body_pose` and use it
            instead. For example, it can used if someone predicts the
            pose of the body joints are predicted from some external model.
            It should be a tensor that contains joint rotations in
            axis-angle format. (default=None)
        transl: torch.tensor, optional, shape Bx3
            If given, ignore the member variable `transl` and use it
            instead. For example, it can used if the translation
            `transl` is predicted from some external model.
            (default=None)
        return_verts: bool, optional
            Return the vertices. (default=True)
        return_full_pose: bool, optional
            Returns the full axis-angle pose vector (default=False)

        Returns
        -------
        """
        # If no shape and pose parameters are passed along, then use the
        # ones from the module
        global_orient = (
            global_orient if global_orient is not None else self.global_orient
        )
        body_pose = body_pose if body_pose is not None else self.body_pose
        betas = betas if betas is not None else self.betas

        apply_trans = transl is not None or hasattr(self, "transl")
        if transl is None and hasattr(self, "transl"):
            transl = self.transl

        full_pose = torch.cat([global_orient, body_pose], dim=1)

        batch_size = max(betas.shape[0], global_orient.shape[0], body_pose.shape[0])

        if betas.shape[0] != batch_size:
            num_repeats = int(batch_size / betas.shape[0])
            betas = betas.expand(num_repeats, -1)

        vertices, joints = lbs(
            betas,
            full_pose,
            self.v_template,
            self.shapedirs,
            self.posedirs,
            self.J_regressor,
            self.parents,
            self.lbs_weights,
            pose2rot=pose2rot,
        )

        joints = self.vertex_joint_selector(vertices, joints)
        # Map the joints to the current dataset
        if self.joint_mapper is not None:
            joints = self.joint_mapper(joints)

        if apply_trans:
            joints += transl.unsqueeze(dim=1)
            vertices += transl.unsqueeze(dim=1)

        output = SMPLOutput(
            vertices=vertices if return_verts else None,
            global_orient=global_orient,
            body_pose=body_pose,
            joints=joints,
            betas=betas,
            full_pose=full_pose if return_full_pose else None,
        )

        return output


class SMPLLayer(SMPL):
    def __init__(self, *args, **kwargs) -> None:
        # Just create a SMPL module without any member variables
        super(SMPLLayer, self).__init__(
            create_body_pose=False,
            create_betas=False,
            create_global_orient=False,
            create_transl=False,
            *args,
            **kwargs,
        )

    def forward(
        self,
        betas: Optional[Tensor] = None,
        body_pose: Optional[Tensor] = None,
        global_orient: Optional[Tensor] = None,
        transl: Optional[Tensor] = None,
        return_verts=True,
        return_full_pose: bool = False,
        pose2rot: bool = True,
        **kwargs,
    ) -> SMPLOutput:
        """Forward pass for the SMPL model

        Parameters
        ----------
        global_orient: torch.tensor, optional, shape Bx3x3
            Global rotation of the body.  Useful if someone wishes to
            predicts this with an external model. It is expected to be in
            rotation matrix format.  (default=None)
        betas: torch.tensor, optional, shape BxN_b
            Shape parameters. For example, it can used if shape parameters
            `betas` are predicted from some external model.
            (default=None)
        body_pose: torch.tensor, optional, shape BxJx3x3
            Body pose. For example, it can used if someone predicts the
            pose of the body joints are predicted from some external model.
            It should be a tensor that contains joint rotations in
            rotation matrix format. (default=None)
        transl: torch.tensor, optional, shape Bx3
            Translation vector of the body.
            For example, it can used if the translation
            `transl` is predicted from some external model.
            (default=None)
        return_verts: bool, optional
            Return the vertices. (default=True)
        return_full_pose: bool, optional
            Returns the full axis-angle pose vector (default=False)

        Returns
        -------
        """
        model_vars = [betas, global_orient, body_pose, transl]
        batch_size = 1
        for var in model_vars:
            if var is None:
                continue
            batch_size = max(batch_size, len(var))
        device, dtype = self.shapedirs.device, self.shapedirs.dtype
        if global_orient is None:
            global_orient = (
                torch.eye(3, device=device, dtype=dtype)
                .view(1, 1, 3, 3)
                .expand(batch_size, -1, -1, -1)
                .contiguous()
            )
        if body_pose is None:
            body_pose = (
                torch.eye(3, device=device, dtype=dtype)
                .view(1, 1, 3, 3)
                .expand(batch_size, self.NUM_BODY_JOINTS, -1, -1)
                .contiguous()
            )
        if betas is None:
            betas = torch.zeros(
                [batch_size, self.num_betas], dtype=dtype, device=device
            )
        if transl is None:
            transl = torch.zeros([batch_size, 3], dtype=dtype, device=device)
        full_pose = torch.cat(
            [
                global_orient.reshape(-1, 1, 3, 3),
                body_pose.reshape(-1, self.NUM_BODY_JOINTS, 3, 3),
            ],
            dim=1,
        )

        vertices, joints = lbs(
            betas,
            full_pose,
            self.v_template,
            self.shapedirs,
            self.posedirs,
            self.J_regressor,
            self.parents,
            self.lbs_weights,
            pose2rot=False,
        )

        joints = self.vertex_joint_selector(vertices, joints)
        # Map the joints to the current dataset
        if self.joint_mapper is not None:
            joints = self.joint_mapper(joints)

        if transl is not None:
            joints += transl.unsqueeze(dim=1)
            vertices += transl.unsqueeze(dim=1)

        output = SMPLOutput(
            vertices=vertices if return_verts else None,
            global_orient=global_orient,
            body_pose=body_pose,
            joints=joints,
            betas=betas,
            full_pose=full_pose if return_full_pose else None,
        )

        return output


class SMPLH(SMPL):
    # The hand joints are replaced by MANO
    NUM_BODY_JOINTS = SMPL.NUM_JOINTS - 2
    NUM_HAND_JOINTS = 15
    NUM_JOINTS = NUM_BODY_JOINTS + 2 * NUM_HAND_JOINTS

    def __init__(
        self,
        model_path,
        kid_template_path: str = "",
        data_struct: Optional[Struct] = None,
        create_left_hand_pose: bool = True,
        left_hand_pose: Optional[Tensor] = None,
        create_right_hand_pose: bool = True,
        right_hand_pose: Optional[Tensor] = None,
        use_pca: bool = True,
        num_pca_comps: int = 6,
        flat_hand_mean: bool = False,
        batch_size: int = 1,
        gender: str = "neutral",
        age: str = "adult",
        dtype=torch.float32,
        vertex_ids=None,
        use_compressed: bool = True,
        ext: str = "pkl",
        **kwargs,
    ) -> None:
        """SMPLH model constructor

        Parameters
        ----------
        model_path: str
            The path to the folder or to the file where the model
            parameters are stored
        data_struct: Strct
            A struct object. If given, then the parameters of the model are
            read from the object. Otherwise, the model tries to read the
            parameters from the given `model_path`. (default = None)
        create_left_hand_pose: bool, optional
            Flag for creating a member variable for the pose of the left
            hand. (default = True)
        left_hand_pose: torch.tensor, optional, BxP
            The default value for the left hand pose member variable.
            (default = None)
        create_right_hand_pose: bool, optional
            Flag for creating a member variable for the pose of the right
            hand. (default = True)
        right_hand_pose: torch.tensor, optional, BxP
            The default value for the right hand pose member variable.
            (default = None)
        num_pca_comps: int, optional
            The number of PCA components to use for each hand.
            (default = 6)
        flat_hand_mean: bool, optional
            If False, then the pose of the hand is initialized to False.
        batch_size: int, optional
            The batch size used for creating the member variables
        gender: str, optional
            Which gender to load
        dtype: torch.dtype, optional
            The data type for the created variables
        vertex_ids: dict, optional
            A dictionary containing the indices of the extra vertices that
            will be selected
        """

        self.num_pca_comps = num_pca_comps
        # If no data structure is passed, then load the data from the given
        # model folder
        if data_struct is None:
            # Load the model
            if osp.isdir(model_path):
                model_fn = "SMPLH_{}.{ext}".format(gender.upper(), ext=ext)
                smplh_path = os.path.join(model_path, model_fn)
            else:
                smplh_path = model_path
            assert osp.exists(smplh_path), "Path {} does not exist!".format(smplh_path)

            if ext == "pkl":
                with open(smplh_path, "rb") as smplh_file:
                    model_data = pickle.load(smplh_file, encoding="latin1")
            elif ext == "npz":
                model_data = np.load(smplh_path, allow_pickle=True)
            else:
                raise ValueError("Unknown extension: {}".format(ext))
            data_struct = Struct(**model_data)

        if vertex_ids is None:
            vertex_ids = VERTEX_IDS["smplh"]

        super(SMPLH, self).__init__(
            model_path=model_path,
            kid_template_path=kid_template_path,
            data_struct=data_struct,
            batch_size=batch_size,
            vertex_ids=vertex_ids,
            gender=gender,
            age=age,
            use_compressed=use_compressed,
            dtype=dtype,
            ext=ext,
            **kwargs,
        )

        self.use_pca = use_pca
        self.num_pca_comps = num_pca_comps
        self.flat_hand_mean = flat_hand_mean

        left_hand_components = data_struct.hands_componentsl[:num_pca_comps]
        right_hand_components = data_struct.hands_componentsr[:num_pca_comps]

        self.np_left_hand_components = left_hand_components
        self.np_right_hand_components = right_hand_components
        if self.use_pca:
            self.register_buffer(
                "left_hand_components", torch.tensor(left_hand_components, dtype=dtype)
            )
            self.register_buffer(
                "right_hand_components",
                torch.tensor(right_hand_components, dtype=dtype),
            )

        if self.flat_hand_mean:
            left_hand_mean = np.zeros_like(data_struct.hands_meanl)
        else:
            left_hand_mean = data_struct.hands_meanl

        if self.flat_hand_mean:
            right_hand_mean = np.zeros_like(data_struct.hands_meanr)
        else:
            right_hand_mean = data_struct.hands_meanr

        self.register_buffer(
            "left_hand_mean", to_tensor(left_hand_mean, dtype=self.dtype)
        )
        self.register_buffer(
            "right_hand_mean", to_tensor(right_hand_mean, dtype=self.dtype)
        )

        # Create the buffers for the pose of the left hand
        hand_pose_dim = num_pca_comps if use_pca else 3 * self.NUM_HAND_JOINTS
        if create_left_hand_pose:
            if left_hand_pose is None:
                default_lhand_pose = torch.zeros(
                    [batch_size, hand_pose_dim], dtype=dtype
                )
            else:
                default_lhand_pose = torch.tensor(left_hand_pose, dtype=dtype)

            left_hand_pose_param = nn.Parameter(default_lhand_pose, requires_grad=True)
            self.register_parameter("left_hand_pose", left_hand_pose_param)

        if create_right_hand_pose:
            if right_hand_pose is None:
                default_rhand_pose = torch.zeros(
                    [batch_size, hand_pose_dim], dtype=dtype
                )
            else:
                default_rhand_pose = torch.tensor(right_hand_pose, dtype=dtype)

            right_hand_pose_param = nn.Parameter(default_rhand_pose, requires_grad=True)
            self.register_parameter("right_hand_pose", right_hand_pose_param)

        # Create the buffer for the mean pose.
        pose_mean_tensor = self.create_mean_pose(
            data_struct, flat_hand_mean=flat_hand_mean
        )
        if not torch.is_tensor(pose_mean_tensor):
            pose_mean_tensor = torch.tensor(pose_mean_tensor, dtype=dtype)
        self.register_buffer("pose_mean", pose_mean_tensor)

    def create_mean_pose(self, data_struct, flat_hand_mean=False):
        # Create the array for the mean pose. If flat_hand is false, then use
        # the mean that is given by the data, rather than the flat open hand
        global_orient_mean = torch.zeros([3], dtype=self.dtype)
        body_pose_mean = torch.zeros([self.NUM_BODY_JOINTS * 3], dtype=self.dtype)

        pose_mean = torch.cat(
            [
                global_orient_mean,
                body_pose_mean,
                self.left_hand_mean,
                self.right_hand_mean,
            ],
            dim=0,
        )
        return pose_mean

    def name(self) -> str:
        return "SMPL+H"

    def extra_repr(self):
        msg = super(SMPLH, self).extra_repr()
        msg = [msg]
        if self.use_pca:
            msg.append(f"Number of PCA components: {self.num_pca_comps}")
        msg.append(f"Flat hand mean: {self.flat_hand_mean}")
        return "\n".join(msg)

    def forward(
        self,
        betas: Optional[Tensor] = None,
        global_orient: Optional[Tensor] = None,
        body_pose: Optional[Tensor] = None,
        left_hand_pose: Optional[Tensor] = None,
        right_hand_pose: Optional[Tensor] = None,
        transl: Optional[Tensor] = None,
        return_verts: bool = True,
        return_full_pose: bool = False,
        pose2rot: bool = True,
        **kwargs,
    ) -> SMPLHOutput:
        """ """

        # If no shape and pose parameters are passed along, then use the
        # ones from the module
        global_orient = (
            global_orient if global_orient is not None else self.global_orient
        )
        body_pose = body_pose if body_pose is not None else self.body_pose
        betas = betas if betas is not None else self.betas
        left_hand_pose = (
            left_hand_pose if left_hand_pose is not None else self.left_hand_pose
        )
        right_hand_pose = (
            right_hand_pose if right_hand_pose is not None else self.right_hand_pose
        )

        apply_trans = transl is not None or hasattr(self, "transl")
        if transl is None:
            if hasattr(self, "transl"):
                transl = self.transl

        if self.use_pca:
            left_hand_pose = torch.einsum(
                "bi,ij->bj", [left_hand_pose, self.left_hand_components]
            )
            right_hand_pose = torch.einsum(
                "bi,ij->bj", [right_hand_pose, self.right_hand_components]
            )

        full_pose = torch.cat(
            [global_orient, body_pose, left_hand_pose, right_hand_pose], dim=1
        )
        full_pose += self.pose_mean

        vertices, joints = lbs(
            betas,
            full_pose,
            self.v_template,
            self.shapedirs,
            self.posedirs,
            self.J_regressor,
            self.parents,
            self.lbs_weights,
            pose2rot=pose2rot,
        )

        # Add any extra joints that might be needed
        joints = self.vertex_joint_selector(vertices, joints)
        if self.joint_mapper is not None:
            joints = self.joint_mapper(joints)

        if apply_trans:
            joints += transl.unsqueeze(dim=1)
            vertices += transl.unsqueeze(dim=1)

        output = SMPLHOutput(
            vertices=vertices if return_verts else None,
            joints=joints,
            betas=betas,
            global_orient=global_orient,
            body_pose=body_pose,
            left_hand_pose=left_hand_pose,
            right_hand_pose=right_hand_pose,
            full_pose=full_pose if return_full_pose else None,
        )

        return output


class SMPLHLayer(SMPLH):
    def __init__(self, *args, **kwargs) -> None:
        """SMPL+H as a layer model constructor"""
        super(SMPLHLayer, self).__init__(
            create_global_orient=False,
            create_body_pose=False,
            create_left_hand_pose=False,
            create_right_hand_pose=False,
            create_betas=False,
            create_transl=False,
            *args,
            **kwargs,
        )

    def forward(
        self,
        betas: Optional[Tensor] = None,
        global_orient: Optional[Tensor] = None,
        body_pose: Optional[Tensor] = None,
        left_hand_pose: Optional[Tensor] = None,
        right_hand_pose: Optional[Tensor] = None,
        transl: Optional[Tensor] = None,
        return_verts: bool = True,
        return_full_pose: bool = False,
        pose2rot: bool = True,
        **kwargs,
    ) -> SMPLHOutput:
        """Forward pass for the SMPL+H model

        Parameters
        ----------
        global_orient: torch.tensor, optional, shape Bx3x3
            Global rotation of the body. Useful if someone wishes to
            predicts this with an external model. It is expected to be in
            rotation matrix format. (default=None)
        betas: torch.tensor, optional, shape BxN_b
            Shape parameters. For example, it can used if shape parameters
            `betas` are predicted from some external model.
            (default=None)
        body_pose: torch.tensor, optional, shape BxJx3x3
            If given, ignore the member variable `body_pose` and use it
            instead. For example, it can used if someone predicts the
            pose of the body joints are predicted from some external model.
            It should be a tensor that contains joint rotations in
            rotation matrix format. (default=None)
        left_hand_pose: torch.tensor, optional, shape Bx15x3x3
            If given, contains the pose of the left hand.
            It should be a tensor that contains joint rotations in
            rotation matrix format. (default=None)
        right_hand_pose: torch.tensor, optional, shape Bx15x3x3
            If given, contains the pose of the right hand.
            It should be a tensor that contains joint rotations in
            rotation matrix format. (default=None)
        transl: torch.tensor, optional, shape Bx3
            Translation vector of the body.
            For example, it can used if the translation
            `transl` is predicted from some external model.
            (default=None)
        return_verts: bool, optional
            Return the vertices. (default=True)
        return_full_pose: bool, optional
            Returns the full axis-angle pose vector (default=False)

        Returns
        -------
        """
        model_vars = [
            betas,
            global_orient,
            body_pose,
            transl,
            left_hand_pose,
            right_hand_pose,
        ]
        batch_size = 1
        for var in model_vars:
            if var is None:
                continue
            batch_size = max(batch_size, len(var))
        device, dtype = self.shapedirs.device, self.shapedirs.dtype
        if global_orient is None:
            global_orient = (
                torch.eye(3, device=device, dtype=dtype)
                .view(1, 1, 3, 3)
                .expand(batch_size, -1, -1, -1)
                .contiguous()
            )
        if body_pose is None:
            body_pose = (
                torch.eye(3, device=device, dtype=dtype)
                .view(1, 1, 3, 3)
                .expand(batch_size, 21, -1, -1)
                .contiguous()
            )
        if left_hand_pose is None:
            left_hand_pose = (
                torch.eye(3, device=device, dtype=dtype)
                .view(1, 1, 3, 3)
                .expand(batch_size, 15, -1, -1)
                .contiguous()
            )
        if right_hand_pose is None:
            right_hand_pose = (
                torch.eye(3, device=device, dtype=dtype)
                .view(1, 1, 3, 3)
                .expand(batch_size, 15, -1, -1)
                .contiguous()
            )
        if betas is None:
            betas = torch.zeros(
                [batch_size, self.num_betas], dtype=dtype, device=device
            )
        if transl is None:
            transl = torch.zeros([batch_size, 3], dtype=dtype, device=device)

        # Concatenate all pose vectors
        full_pose = torch.cat(
            [
                global_orient.reshape(-1, 1, 3, 3),
                body_pose.reshape(-1, self.NUM_BODY_JOINTS, 3, 3),
                left_hand_pose.reshape(-1, self.NUM_HAND_JOINTS, 3, 3),
                right_hand_pose.reshape(-1, self.NUM_HAND_JOINTS, 3, 3),
            ],
            dim=1,
        )

        vertices, joints = lbs(
            betas,
            full_pose,
            self.v_template,
            self.shapedirs,
            self.posedirs,
            self.J_regressor,
            self.parents,
            self.lbs_weights,
            pose2rot=False,
        )

        # Add any extra joints that might be needed
        joints = self.vertex_joint_selector(vertices, joints)
        if self.joint_mapper is not None:
            joints = self.joint_mapper(joints)

        if transl is not None:
            joints += transl.unsqueeze(dim=1)
            vertices += transl.unsqueeze(dim=1)

        output = SMPLHOutput(
            vertices=vertices if return_verts else None,
            joints=joints,
            betas=betas,
            global_orient=global_orient,
            body_pose=body_pose,
            left_hand_pose=left_hand_pose,
            right_hand_pose=right_hand_pose,
            full_pose=full_pose if return_full_pose else None,
        )

        return output
