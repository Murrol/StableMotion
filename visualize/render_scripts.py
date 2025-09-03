import numpy as np
import os
from utils.renderer.humor import HumorRenderer
from utils.renderer.matplotlib import MatplotlibRender
import torch

from data_loaders.amasstools.smplh_layer import SMPLH
from data_loaders.amasstools.extract_joints import extract_joints_smpldata
import multiprocessing
from argparse import ArgumentParser

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--data_path", type=str)
    parser.add_argument("--gt_data_path", type=str)
    parser.add_argument("--motiontypes", nargs='+', default=['motion_fix'])
    parser.add_argument("--rendersmpl", action='store_true')
    parser.add_argument("--ext", default='', type=str)
    args = parser.parse_args()
    plot = True
    FPS = 20
    jointstype = 'smpljoints'
    smplh_folder = "data_loaders/amasstools/deps/smplh"
    value_from = 'smpl'
    data_path = args.data_path
    motiontypes = args.motiontypes

    exp_folder_name = os.path.dirname(data_path)
    if plot:
        for motiontype in motiontypes:
            video_dir = os.path.join(
                exp_folder_name, f'video_{motiontype}_from_{value_from}'
            )
            if args.ext != '':
                video_dir += f'_{args.ext}'
            print(f"save in {video_dir}")
            os.makedirs(video_dir, exist_ok=True)

            smpl_renderer = HumorRenderer(
                fps = FPS,
                imw = 224,
                imh = 224,
                cam_offset = [0.0, -2.2, 0.9],
                cam_rot = [  
                    [1.0000000,  0.0000000,  0.0000000],
                    [0.0000000,  0.0000000, -1.0000000],
                    [0.0000000,  1.0000000,  0.0000000],
                    ],
            )
            joints_renderer = MatplotlibRender(
                jointstype = jointstype,
                fps = FPS,
                colors = ["black", "magenta", "red", "green", "blue"],
                figsize = 4,
                canonicalize = True,
            )
            smplh = SMPLH(
                path = smplh_folder,
                jointstype = 'both',
                input_pose_rep = "axisangle",
                gender = "neutral",
            )
            datas = np.load(data_path, allow_pickle=True).item()
            smpldatas = datas[motiontype]
            if motiontype == 'motion':
                _gen_labels = datas.get('gt_labels', None)
            else:
                _gen_labels = datas.get('label', None)
            if _gen_labels is not None:
                gen_labels = []
                for glst in _gen_labels:
                    gen_labels += glst.tolist()
            lengths = datas['lengths']
            n_motions = len(smpldatas)
            vext = ".mp4"
            joints_video_paths = [
                os.path.join(video_dir, f"{idx}_joints{vext}")
                for idx in range(n_motions)
            ]
            smpl_video_paths = [
                os.path.join(video_dir, f"{idx}_smpl{vext}")
                for idx in range(n_motions)
            ]
            for idx, smpldata in enumerate(smpldatas):
                output = extract_joints_smpldata(
                    smpldata = smpldata,
                    fps = FPS,
                    value_from = value_from, #could be joint
                    smpl_layer = smplh,
                    )
                if args.rendersmpl:
                    if value_from == 'smpl':
                        vertices = output["vertices"][:lengths[idx]]
                        # print(np.min(vertices, axis=(0,1)))
                        vertices[..., 2] -= np.min(vertices[..., 2])
                        smpl_renderer(
                                    vertices, 
                                    title="", 
                                    output=smpl_video_paths[idx], 
                                    gen_label=gen_labels[idx][:lengths[idx]] if _gen_labels is not None else None
                                )
                else:
                    os.makedirs(os.path.dirname(joints_video_paths[idx]), exist_ok=True)
                    joints = output['joints'][:lengths[idx]]
                    joints_renderer(
                                    joints, title="", output=joints_video_paths[idx], canonicalize=False
                                )

    