"""
Evaluate motion quality/accuracy metrics for SMPL/SMPL-H sequences.

Metrics:
- Foot sliding distance
- Skating ratio
- Ground penetration ratio & distance
- Jittering
- (Optional, if gt provided) MPJPE, global MPJPE, RTE (aligned/unaligned), accel error, TMR m2m

Usage:
    python -m eval.eval_scripts --data_path <pred.npy> [--gt_data_path <gt.npy>] \
        --motiontypes motion_fix other_type --force_redo
"""

import numpy as np
import os
from eval.eval_motion import (
    compute_foot_sliding_wrapper,
    compute_jitter_wrapper,
    compute_label_metrics,
    compute_skating_ratio_wrapper,
    compute_gpenetration_wrapper,
    compute_rte_wrapper,
    compute_jpe_wrapper,
    compute_error_accel_wrapper,
    tmr_m2m_score_wrapper,
)
from data_loaders.amasstools.extract_joints import extract_joints_smpldata
from data_loaders.amasstools.smplh_layer import SMPLH
from tqdm import tqdm
from argparse import ArgumentParser


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--data_path", type=str, help="Path to predictions .npy (dict-like).")
    parser.add_argument("--gt_data_path", type=str, default=None, help="Path to GT .npy (dict-like).")
    parser.add_argument("--motiontypes", nargs='+', default=['motion_fix'], help="Keys inside predictions to evaluate.")
    parser.add_argument("--force_redo", action="store_true", help="Recompute joints even if cached.")
    args = parser.parse_args()

    data_path = args.data_path
    gt_data_path = args.gt_data_path

    # --------------------
    # Config
    # --------------------
    FPS = 20
    root_offset = np.array([-0.00179506, -0.22333382, 0.02821918])
    smplh_folder = "data_loaders/amasstools/deps/smplh"
    value_from = 'smpl'
    motiontypes = args.motiontypes
    device = 'cuda'

    # --------------------
    # Load prediction data (+ optional GT)
    # --------------------
    datas = np.load(data_path, allow_pickle=True).item()
    mpjpe_enable = False if gt_data_path is None else True
    if mpjpe_enable:
        gt_datas = np.load(gt_data_path, allow_pickle=True).item()

    lengths = datas['lengths']

    # SMPL-H layer
    smplh = SMPLH(
        path=smplh_folder,
        jointstype='both',
        input_pose_rep="axisangle",
        gender="neutral",
    )
    smplh.to(device)

    # --------------------
    # Optional quality indicator prediction metrics (if present in predictions)
    # --------------------
    pred_labels, gt_lables = [], []
    if 'gt_labels' in datas.keys():
        assert len(datas['gt_labels']) == len(datas['label'])
        for i in range(len(datas['gt_labels'])):
            pred_labels += datas['label'][i].tolist()
            gt_lables += datas['gt_labels'][i].tolist()
        acc, precision, recall, f1 = compute_label_metrics(
            preds=np.array(pred_labels),
            gts=np.array(gt_lables),
            lengths=lengths,
        )
        print(f"Label prediction: Accuracy: {acc:.5f}; Precision: {precision:.5f}, Recall: {recall:.5f}, f1-score: {f1:.5f}")

    # --------------------
    # Motion artifacts metrics
    # --------------------
    for motiontype in motiontypes:
        joints_from_smpl_path = data_path.replace('.npy', f'_{motiontype}joints.npz')

        # Load cached joints if available (and allowed)
        if os.path.exists(joints_from_smpl_path) and value_from == 'smpl' and not args.force_redo:
            joints = np.load(joints_from_smpl_path, allow_pickle=True)['joints']
        else:
            if motiontype not in datas.keys():
                print(f"Motion type {motiontype} does not exist in results file")
                break

            smpldatas = datas[motiontype]
            assert len(lengths) == len(smpldatas)

            joints = []
            for idx, smpldata in tqdm(enumerate(smpldatas)):
                output = extract_joints_smpldata(
                    smpldata=smpldata,
                    fps=FPS,
                    value_from=value_from,  # could be "joint"
                    smpl_layer=smplh,
                    root_offset=root_offset,
                    device=device,
                )
                joints.append(output['joints'][:, :22])  # (L, 24, 3) -> first 22 joints
            np.savez(joints_from_smpl_path, joints=joints)

        # Compute motion quality metrics
        motion_slide_dis = compute_foot_sliding_wrapper(joints, lengths, upaxis=2, ankle_h=0.1)  # mm
        motion_jitter = compute_jitter_wrapper(joints, lengths)  # (x10 m/s^3)
        motion_skating_ratio = compute_skating_ratio_wrapper(joints, lengths)
        motion_gp_ratio, motoin_gp_dist = compute_gpenetration_wrapper(joints, lengths)

        print(
            f"Motion from {motiontype}"
            f"\n Foot Sliding Dist (mm): {motion_slide_dis:.5f}"
            f"\n Skating Ratio (%): {motion_skating_ratio*100:.5f}"
            f"\n Ground Penetration Ratio (%): {motion_gp_ratio*100:.5f}"
            f"\n Ground Penetration Dist (mm): {motoin_gp_dist:.5f}"
            f"\n Jittering measurement (x10 m/s^3): {motion_jitter.mean():.5f}"
        )

        # --------------------
        # Content presevation metrics by comparing with GT (if provided)
        # --------------------
        if mpjpe_enable:
            gt_joints_from_smpl_path = gt_data_path.replace('.npy', f'_motionjoints.npz')
            if os.path.exists(gt_joints_from_smpl_path) and value_from == 'smpl':
                gt_joints = np.load(gt_joints_from_smpl_path, allow_pickle=True)['joints']
                assert len(gt_joints) == len(joints)
            else:
                smpldatas = gt_datas['motion']
                assert len(lengths) == len(smpldatas)
                gt_joints = []
                for idx, smpldata in tqdm(enumerate(smpldatas)):
                    output = extract_joints_smpldata(
                        smpldata=smpldata,
                        fps=FPS,
                        value_from=value_from,  # could be "joint"
                        smpl_layer=smplh,
                        root_offset=root_offset,
                        device=device,
                    )
                    gt_joints.append(output['joints'][:, :22])  # (L, 24, 3) -> first 22 joints
                np.savez(gt_joints_from_smpl_path, joints=gt_joints)

            mpjpe = compute_jpe_wrapper(gt_joints, joints, lengths)
            gb_mpjpe = compute_jpe_wrapper(gt_joints, joints, lengths, local=False)
            rte = compute_rte_wrapper(gt_joints, joints, lengths)
            raw_rte = compute_rte_wrapper(gt_joints, joints, lengths, align=False)
            error_accel = compute_error_accel_wrapper(gt_joints, joints, lengths)
            m2m_score, m2m_r1, m2m_r3 = tmr_m2m_score_wrapper(joints, gt_joints, lengths, y_is_z_axis=False, device='cpu')

            print(
                f"Motion from {motiontype}"
                f"\n MPJPE (cm): {mpjpe:.5f}"
                f"\n GBMPJPE (cm): {gb_mpjpe:.5f}"
                f"\n RTE (m): {rte:.5f}"
                f"\n Unaligned RTE (m): {raw_rte:.5f}"
                f"\n Accel Error (m/s^2): {error_accel:.5f}"
                f"\n m2m_score: {m2m_score:.5f}"
                f"\n m2m_r1: {m2m_r1*100:.5f}"
                f"\n m2m_r3: {m2m_r3*100:.5f}"
            )