"""
Detect-and-fix pipeline for AMASS motions using a diffusion model.

- Loads test data (aligned Global SMPL RIFKE feats + labels channel).
- Runs a detection pass to find bad frames.
- Builds an inpainting mask and fixes bad frames.
- Optionally runs an ensemble cleanup path.
- Saves .npy results and triggers evaluation/visualization script.
"""

import os
import numpy as np


import torch
import einops
torch.backends.cuda.enable_flash_sdp(True)
torch.backends.cudnn.benchmark = True

from utils.fixseed import fixseed
from utils.parser_util import det_args
from utils.model_util import create_model_and_diffusion
from utils import dist_util
from data_loaders.get_data import get_dataset_loader
from data_loaders.amasstools.globsmplrifke_feats import globsmplrifkefeats_to_smpldata

from ema_pytorch import EMA
from sample.utils import run_cleanup_selection, prepare_cond_fn, choose_sampler, build_output_dir



@torch.no_grad()
def detect_labels(
    *,
    model,
    diffusion,
    args,
    input_motions,           # [B, C, N]
    length,                  # Tensor[int] of shape [B]
    attention_mask,          # [B, N] bool
    motion_normalizer,
):
    """
    Detection pass:
      - inpaint only label channel
      - optional MC averaging (ProbDetNum)
      - return binary label mask, decoded detected motions (list of dicts), re_sample (features)
    """
    device = input_motions.device
    bs, nfeats, nframes = input_motions.shape

    # Prepare det-mode kwargs
    inpaint_motion_detmode = input_motions.clone()
    inpaint_motion_detmode[:, -1] = 1.0  # corrupt label channel

    inpainting_mask_detmode = torch.ones_like(input_motions).bool()
    inpainting_mask_detmode[:, -1] = False  # predict label channel

    inpaint_cond_detmode = (~inpainting_mask_detmode) & attention_mask.unsqueeze(-2)

    model_kwargs_detmode = {
        "y": {
            "inpainting_mask": inpainting_mask_detmode,
            "inpainted_motion": inpaint_motion_detmode,
        },
        "inpaint_cond": inpaint_cond_detmode,
        "length": length,
        "attention_mask": attention_mask,
    }

    sample_fn = choose_sampler(diffusion, args.ts_respace)

    # Single detection pass
    re_sample = sample_fn(
        model,
        (bs, nfeats, nframes),
        clip_denoised=False,
        model_kwargs=model_kwargs_detmode,
        skip_timesteps=0,
        init_image=None,
        progress=True,
        dump_steps=None,
        noise=None,
        const_noise=False,
    )

    # Monte-Carlo averaging if requested
    if args.ProbDetNum:
        for _ in range(args.ProbDetNum):
            re_sample += sample_fn(
                model,
                (bs, nfeats, nframes),
                clip_denoised=False,
                model_kwargs=model_kwargs_detmode,
                skip_timesteps=0,
                init_image=None,
                progress=True,
                dump_steps=None,
                noise=None,
                const_noise=False,
            )
        re_sample = re_sample / (args.ProbDetNum + 1)

    # Read label channel
    sample_det = motion_normalizer.inverse(re_sample.transpose(1, 2).cpu())
    label = sample_det[..., -1] > args.ProbDetTh
    sample_body = sample_det[..., :-1]

    recon_input_motion = [globsmplrifkefeats_to_smpldata(_s) for _s in sample_body]

    # Pack outputs
    out = {
        "label": label,                              # [B, N] bool (on CPU)
        "recon_input_motion": recon_input_motion,        # list[dict]
        "re_sample_det_feats": re_sample,            # [B, C, N] feats (device)
    }
    return out


@torch.no_grad()
def fix_motion(
    *,
    model,
    diffusion,
    args,
    input_motions,           # [B, C, N]
    length,                  # Tensor[int] of shape [B]
    attention_mask,          # [B, N] bool
    motion_normalizer,
    label,                   # [B, N] bool (CPU or GPU ok)
    re_sample_det_feats,     # [B, C, N] (from detect pass)
    cond_fn=None,            # optional
):
    """
    Fix pass:
      - dilate detections
      - build inpainting mask
      - optional soft-inpaint schedule
      - run sampler or ensemble cleanup
      - return sample_fix (feats) + decoded fixed motions
    """
    device = input_motions.device
    bs, nfeats, nframes = input_motions.shape

    # Ensure label on same device for ops below
    label = label.to(device)

    # Slightly dilate detections by 1 frame on both sides; last frame forced to good
    temp_labels = label.clone()
    label[..., 1:] += temp_labels[..., :-1]
    label[..., :-1] += temp_labels[..., 1:]
    for mids, mlen in enumerate(length.cpu().numpy()):
        label[mids, ..., mlen - 1] = 0

    # Build mask: True=keep, False=repaint (label channel always kept)
    det_good_frames_per_sample = {
        s_i: np.nonzero(~label.cpu().numpy()[s_i].squeeze())[0].tolist()
        for s_i in range(len(label))
    }

    inpainting_mask_fixmode = torch.zeros_like(input_motions).bool()
    for s_i in range(len(input_motions)):
        inpainting_mask_fixmode[s_i, ..., det_good_frames_per_sample[s_i]] = True
    inpainting_mask_fixmode[:, -1] = True

    inpaint_motion_fixmode = input_motions.clone()
    inpaint_motion_fixmode[:, -1] = -1.0
    inpaint_cond_fixmode = (~inpainting_mask_fixmode) & attention_mask.unsqueeze(-2)

    model_kwargs_fix = {
        "y": {
            "inpainting_mask": inpainting_mask_fixmode.clone(),
            "inpainted_motion": inpaint_motion_fixmode.clone(),
        },
        "inpaint_cond": inpaint_cond_fixmode.clone(),
        "length": length,
        "attention_mask": attention_mask,
    }

    # Optional Soft-inpaint schedule
    if args.enable_sits and args.ProbDetNum:
        soft_inpaint_ts = einops.repeat(re_sample_det_feats[:, [-1]], "b c l -> b (repeat c) l", repeat=nfeats)
        soft_inpaint_ts = torch.clip((soft_inpaint_ts + 1 / 2), min=0.0, max=1.0)
        soft_inpaint_ts = torch.ceil((torch.sin(soft_inpaint_ts * torch.pi * 0.5)) * args.diffusion_steps).long()
    else:
        soft_inpaint_ts = None

    # Sampler
    sample_fn = choose_sampler(diffusion, args.ts_respace)

    # Ensemble cleanup
    if args.ensemble:
        sample_fix = run_cleanup_selection(
            model=model,
            model_kwargs_detmode={
                "y": {
                    "inpainting_mask": torch.ones_like(input_motions).bool().index_fill_(1, torch.tensor([nfeats-1], device=device), False),
                    "inpainted_motion": input_motions.clone().index_fill_(1, torch.tensor([nfeats-1], device=device), 1.0),
                },
                "inpaint_cond": ((~torch.ones_like(input_motions).bool().index_fill_(1, torch.tensor([nfeats-1], device=device), False))
                                 & attention_mask.unsqueeze(-2)),
                "length": length,
                "attention_mask": attention_mask,
            },
            model_kwargs=model_kwargs_fix,
            motion_normalizer=motion_normalizer,
            args=args,
            bs=bs,
            nfeats=nfeats,
            nframes=nframes,
            sample_fn=sample_fn,
            cond_fn=cond_fn if args.classifier_scale else None,
        )
    else:
        sample_fix = sample_fn(
            model,
            (bs, nfeats, nframes),
            clip_denoised=False,
            model_kwargs=model_kwargs_fix,
            skip_timesteps=args.skip_timesteps,
            init_image=model_kwargs_fix["y"]["inpainted_motion"],
            progress=True,
            dump_steps=None,
            noise=None,
            const_noise=False,
            soft_inpaint_ts=soft_inpaint_ts,
            cond_fn=cond_fn if args.classifier_scale else None,
        )

    # Decode
    sample_fix_det = motion_normalizer.inverse(sample_fix.transpose(1, 2).cpu())
    sample_fix_body = sample_fix_det[..., :-1]
    fixed_motion = [globsmplrifkefeats_to_smpldata(_s) for _s in sample_fix_body]

    return {
        "sample_fix_feats": sample_fix,   # [B, C, N]
        "fixed_motion": fixed_motion,       # list[dict]
    }


# ---------------------------
# Main
# ---------------------------
def main():
    args = det_args()
    fixseed(args.seed)

    # Device / dist
    dist_util.setup_dist(args.device)
    device = dist_util.dev()

    # Flags
    collect_dataset = args.collect_dataset

    # Output path
    out_path = build_output_dir(args)

    # Data
    print(f"Loading dataset from {args.testdata_dir}...")
    data = get_dataset_loader(
        name="globsmpl",
        batch_size=args.batch_size,
        split="test_benchmark",
        data_dir=args.testdata_dir,
        normalizer_dir=args.normalizer_dir,
        shuffle=False,
    )
    motion_normalizer = data.dataset.motion_loader.motion_normalizer

    # Model + diffusion
    print("Creating model and diffusion...")
    print("USING Sampler: ", args.ts_respace)
    _model, diffusion = create_model_and_diffusion(args)
    model = EMA(_model, include_online_model=False) if args.use_ema else _model

    print(f"Loading checkpoints from [{args.model_path}]...")
    state_dict = torch.load(args.model_path, map_location="cpu")
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()

    # Optional guidance
    cond_fn = prepare_cond_fn(args, motion_normalizer, device)
    if cond_fn is not None:
        cond_fn.keywords["model"] = model

    # Buffers
    all_motions = []            # decoded SMPL dicts (detected)
    all_lengths = []            # list of np arrays
    all_input_motions_vec = []  # raw feature sequences (pre-fix)
    gt_labels_buf = []          # GT boolean labels per frame

    all_motions_fix = []        # decoded SMPL dicts (fixed)
    all_fix_motions_vec = []    # raw feature sequences (post-fix)
    labels = []                 # predicted boolean labels per frame

    # Loop
    for i, input_batch in enumerate(data):
        input_batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in input_batch.items()}
        input_motions = input_batch["x"]                 # [B, C, N]

        attention_mask = input_batch["mask"].squeeze().bool().clone()
        length = input_batch["length"]

        # For collecting GT motions only
        temp_sample = motion_normalizer.inverse(input_motions.transpose(1, 2).cpu())
        if collect_dataset:
            for _sample in temp_sample:
                all_motions.append(globsmplrifkefeats_to_smpldata(_sample[..., :-1]))
            all_lengths.append(length.cpu().numpy())
            continue

        # Cache GT labels from label channel
        gt_labels = (temp_sample[..., -1] > 0.5).numpy()
        gt_labels_buf.append(gt_labels)

        # --- Detect ---
        det_out = detect_labels(
            model=model,
            diffusion=diffusion,
            args=args,
            input_motions=input_motions,
            length=length,
            attention_mask=attention_mask,
            motion_normalizer=motion_normalizer,
        )
        label = det_out["label"]
        recon_input_motion = det_out["recon_input_motion"]
        re_sample_det_feats = det_out["re_sample_det_feats"]

        all_motions += recon_input_motion
        all_lengths.append(length.cpu().numpy())
        labels.append(label.numpy().copy())
        all_input_motions_vec.append(input_motions.transpose(1, 2).cpu().numpy())
        print(f"Detected {len(all_motions)} samples")

        # Optionally override with GT labels
        if args.gtdet:
            print("use gt label")
            label = torch.from_numpy(gt_labels.copy())

        # --- Fix ---
        fix_out = fix_motion(
            model=model,
            diffusion=diffusion,
            args=args,
            input_motions=input_motions,
            length=length,
            attention_mask=attention_mask,
            motion_normalizer=motion_normalizer,
            label=label,
            re_sample_det_feats=re_sample_det_feats,
            cond_fn=cond_fn,
        )
        sample_fix_feats = fix_out["sample_fix_feats"]
        fixed_motion = fix_out["fixed_motion"]

        all_fix_motions_vec.append(sample_fix_feats.transpose(1, 2).cpu().numpy())
        all_motions_fix += fixed_motion
        print(f"Fixed {len(all_motions_fix)} samples")

        if args.num_samples and args.batch_size * (i + 1) >= args.num_samples:
            break

    # Save results
    all_lengths = np.concatenate(all_lengths, axis=0)
    os.makedirs(out_path, exist_ok=True)

    if collect_dataset:
        npy_path = os.path.join(out_path, "results_collected.npy")
        print(f"saving collected motion file to [{npy_path}]")
        np.save(npy_path, {"motion": all_motions, "lengths": all_lengths})
        with open(npy_path.replace(".npy", "_len.txt"), "w") as fw:
            fw.write("\n".join([str(l) for l in all_lengths]))
        return

    npy_path = os.path.join(out_path, "results.npy")
    print(f"saving results file to [{npy_path}]")
    np.save(
        npy_path,
        {
            "motion": all_motions,
            "motion_fix": all_motions_fix,
            "label": labels,
            "gt_labels": gt_labels_buf,
            "lengths": all_lengths,
            "all_fix_motions_vec": all_fix_motions_vec,
            "all_input_motions_vec": all_input_motions_vec,
        },
    )
    with open(npy_path.replace(".npy", "_len.txt"), "w") as fw:
        fw.write("\n".join([str(l) for l in all_lengths]))

    # Launch eval
    os.system(f"python -m eval.eval_scripts  --data_path {npy_path} --force_redo")


if __name__ == "__main__":
    main()