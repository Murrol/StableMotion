
import os
from copy import deepcopy
from functools import partial
import numpy as np
import torch
from utils import dist_util
from copy import deepcopy
from data_loaders.amasstools.globsmplrifke_feats import globsmplrifkefeats_to_smpldata
from eval.eval_motion import compute_foot_sliding_wrapper_torch
from smplx.lbs import batch_rigid_transform
import einops
from tqdm import tqdm
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

def build_output_dir(args) -> str:
    """Derive output directory name from args and model checkpoint name."""
    out_path = args.output_dir
    if out_path != "":
        return out_path

    name = os.path.basename(os.path.dirname(args.model_path))
    niter = os.path.basename(args.model_path).replace("model", "").replace(".pt", "")

    out_path = os.path.join(
        os.path.dirname(args.model_path), f"Fix_{name}_{niter}_seed{args.seed}"
    )
    if args.dataset_path != "":
        out_path += f'_{os.path.split(args.dataset_path)[-1]}'
    if args.skip_timesteps != 0:
        out_path += f"_skip{args.skip_timesteps}"
    if args.enable_cfg:
        out_path += f"_cfg{args.guidance_param}"
    if args.classifier_scale:
        out_path += f"_cs{args.classifier_scale}"
    if args.ProbDetNum:
        out_path += f"_pdn{args.ProbDetNum}"
    if args.enable_sits:
        out_path += "_softi"
    if args.ensemble:
        out_path += "_esnb"
    if args.ext != "":
        out_path += f"_ext{args.ext}"
    return out_path


def prepare_cond_fn(args, motion_normalizer, device):
    """
    Prepare optional foot-locking guidance function (for classifier guidance).
    Returns cond_fn or None.
    """
    if not args.classifier_scale:
        return None

    print("Preparing cond function ...")
    j_regressor_stat = np.load("data_loaders/amasstools/smpl_neutral_nobetas_24J.npz")
    J_regressor = torch.from_numpy(j_regressor_stat["J"]).to(device)
    parents = torch.from_numpy(j_regressor_stat["parents"])
    root_offset = torch.tensor([-0.00179506, -0.22333382, 0.02821918]).to(device)
    std = motion_normalizer.std.clone().to(device)
    mean = motion_normalizer.mean.clone().to(device)

    return partial(
        footlocking_fn,
        model=None,  # set later when model is ready
        mean=mean,
        std=std,
        classifier_scale=args.classifier_scale,
        J_regressor=J_regressor,
        parents=parents,
        root_offset=root_offset,
    )

def choose_sampler(diffusion, ts_respace: bool):
    """Pick DDIM or ancestral sampler."""
    return diffusion.ddim_sample_loop if ts_respace else diffusion.p_sample_loop

def batch_expander(model_kwargs, repeat_times):
    out_model_kwargs = deepcopy(model_kwargs)
    if 'y' in out_model_kwargs:
        out_model_kwargs['y'] = batch_expander(model_kwargs['y'], repeat_times)
    for k, v in model_kwargs.items():
        if k == 'y':
            continue
        else:
            if isinstance(v, list):
                out_model_kwargs[k] = v * repeat_times
            elif isinstance(v, (torch.Tensor, np.ndarray)):
                out_model_kwargs[k] = einops.repeat(v, 'b ... -> (repeat b) ...', repeat=repeat_times)
    return out_model_kwargs

def footlocking_fn(x, t, model=None, mean=None, std=None, classifier_scale=0., J_regressor=None, parents=None, root_offset=None, **kwargs):
    lengths = kwargs["length"]
    eps = 1e-12
    with torch.autograd.set_detect_anomaly(False):
        with torch.enable_grad():
            inpaint_cond = kwargs['inpaint_cond']
            x_gt = kwargs['y']['inpainted_motion']
            x_in = x.detach().requires_grad_(True)
            loss = 0.
            x_in = torch.where(inpaint_cond, x_in, x_gt)
            x_0 = model(x_in, t, **kwargs)
            x_0 = torch.where(inpaint_cond, x_0, x_gt)
            denorm_x0 = x_0.transpose(1, 2) * (std + eps) + mean
            joints = []
            loss = 0
            B = denorm_x0.shape[0]
            denorm_x0_flatten = einops.rearrange(denorm_x0, "b n d -> (b n) d")
            smpldata = globsmplrifkefeats_to_smpldata(denorm_x0_flatten[..., :-1])
            poses = smpldata["poses"]
            trans = smpldata["trans"]
            poses = einops.rearrange(poses, "k (l t) -> k l t", t=3)
            rot_mat = axis_angle_to_matrix(poses)
            T = rot_mat.shape[0]
            zero_hands_rot = torch.eye(3)[None, None].expand(T, 2, -1, -1).to(dist_util.dev())
            rot_mat = torch.concat((rot_mat, zero_hands_rot), dim=1)
            joints, _ = batch_rigid_transform(
                rot_mat,
                J_regressor[None].expand(T, -1, -1),
                parents,
            )
            joints = joints.squeeze() + trans.unsqueeze(1) - root_offset
            joints = einops.rearrange(joints, "(b n) j d -> b n j d", b=B)
            slide_dist = compute_foot_sliding_wrapper_torch(joints, lengths, upaxis=2, ankle_h=0.1)
            loss = sum(slide_dist)
            grad = torch.autograd.grad(-loss, x_in)[0] * classifier_scale
        
        grad = torch.nan_to_num(grad)
        grad = torch.clip(grad, min=-10, max=10) # [b d n] 
        grad[:, 0] = 0.

    return grad

def run_cleanup_selection(
    model,
    model_kwargs_detmode,
    model_kwargs,
    motion_normalizer,
    sample_fn,
    cond_fn,
    args,
    bs,
    nfeats,
    nframes,
):
    """
    Wraps the detection->fixing pipeline with repeated sampling and candidate selection.

    Returns:
        sample (Tensor): The selected inpainted motion [bs, nfeats, nframes].
    """
    sample_candidates = []
    forward_rp_times = 5  # Hardcode
    eval_times = 25  # Hardcode

    rp_model_kwargs_detmode = batch_expander(model_kwargs_detmode, forward_rp_times)

    with torch.no_grad():
        _re_sample = 0
        _re_t = torch.ones((bs * forward_rp_times,), device=dist_util.dev()) * 49
        for _ in tqdm(range(eval_times)):  # quick det sampler
            x = torch.randn_like(rp_model_kwargs_detmode['y']['inpainted_motion'])
            inpaint_cond = rp_model_kwargs_detmode['inpaint_cond']
            x_gt = rp_model_kwargs_detmode['y']['inpainted_motion']
            x = torch.where(inpaint_cond, x, x_gt)
            _re_sample += model(x, _re_t, **rp_model_kwargs_detmode)
        _re_sample = _re_sample / eval_times

    _sample = motion_normalizer.inverse(_re_sample.transpose(1, 2).cpu())
    _label = _sample[..., -1] > args.ProbDetTh

    # -------------------------------
    # Preparing for Fixing Mode
    # -------------------------------
    temp_labels = _label.clone()
    _label[..., 1:] += temp_labels[..., :-1]
    _label[..., :-1] += temp_labels[..., 1:]
    for mids, mlen in enumerate(rp_model_kwargs_detmode['length'].cpu().numpy()):
        _label[mids, ..., mlen - 1] = 0

    det_good_frames_per_sample = {
        sample_i: np.nonzero(~_label.numpy()[sample_i].squeeze())[0].tolist()
        for sample_i in range(len(_label))
    }

    # Break Frame Fix
    inpainting_mask_fixmode = torch.zeros_like(_re_sample).bool().to()  # True to keep, False to re-paint
    for sample_i in range(len(_re_sample)):
        inpainting_mask_fixmode[sample_i, ..., det_good_frames_per_sample[sample_i]] = True
    inpainting_mask_fixmode[:, -1] = True

    inpaint_motion_fixmode = rp_model_kwargs_detmode['y']['inpainted_motion'].clone()
    inpaint_motion_fixmode[:, -1] = -1.0
    inpaint_cond_fixmode = (~inpainting_mask_fixmode) & rp_model_kwargs_detmode['attention_mask'].unsqueeze(-2)

    rp_model_kwargs = batch_expander(model_kwargs, forward_rp_times)
    rp_model_kwargs['y']['inpainting_mask'] = inpainting_mask_fixmode.clone()
    rp_model_kwargs['y']['inpainted_motion'] = inpaint_motion_fixmode.clone()
    rp_model_kwargs['inpaint_cond'] = inpaint_cond_fixmode.clone()

    if args.enable_sits:
        soft_inpaint_ts = einops.repeat(_re_sample[:, [-1]], 'b c l -> b (repeat c) l', repeat=nfeats)
        soft_inpaint_ts = torch.clip((soft_inpaint_ts + 1 / 2), min=0.0, max=1.0)
        soft_inpaint_ts = torch.ceil((torch.sin(soft_inpaint_ts * torch.pi * 0.5)) * args.diffusion_steps).long()
    else:
        soft_inpaint_ts = None

    sample = sample_fn(
        model,
        (bs * forward_rp_times, nfeats, nframes),
        clip_denoised=False,
        model_kwargs=rp_model_kwargs,
        skip_timesteps=args.skip_timesteps,  # 0 is default
        init_image=rp_model_kwargs['y']['inpainted_motion'],
        progress=True,
        dump_steps=None,
        noise=None,
        const_noise=False,
        soft_inpaint_ts=soft_inpaint_ts,
        cond_fn=cond_fn if args.classifier_scale else None,
    )

    _inpaint_motion_detmode = sample.clone()
    _inpaint_motion_detmode[:, -1] = 1.0  # For safety, corrupt the label from input.
    rp_model_kwargs_detmode['y']['inpainted_motion'] = _inpaint_motion_detmode.clone()

    score = 0
    with torch.no_grad():
        _re_t = torch.ones((bs * forward_rp_times,), device=dist_util.dev()) * 49
        for _ in tqdm(range(eval_times)):  # quick det sampler
            x = torch.randn_like(rp_model_kwargs_detmode['y']['inpainted_motion'])
            inpaint_cond = rp_model_kwargs_detmode['inpaint_cond']
            x_gt = rp_model_kwargs_detmode['y']['inpainted_motion']
            x = torch.where(inpaint_cond, x, x_gt)
            score += model(x, _re_t, **rp_model_kwargs_detmode)[:, -1]
    score /= eval_times
    score = torch.sum((score > 0.0) * rp_model_kwargs_detmode['attention_mask'], dim=-1)
    score = einops.rearrange(score, "(repeat b) -> repeat b", repeat=forward_rp_times)  # [forward_rp_times, bs]

    sample_candidates = einops.rearrange(sample, "(repeat b) c l -> repeat b c l", repeat=forward_rp_times)
    selected_id = torch.argmin(score, dim=0)  # [bs]
    selected_id = selected_id[..., None, None].expand(sample_candidates.shape[1:]).unsqueeze(0)  # [1, bs, nfeats, nframes]
    sample = torch.gather(sample_candidates, dim=0, index=selected_id).squeeze(0)

    return sample