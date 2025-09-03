from argparse import ArgumentParser
import argparse
import os
import json


# ----------------------------
# High-level helpers
# ----------------------------
def parse_and_load_from_model(parser: ArgumentParser):
    """
    Parse args for detection/eval, then overwrite dataset/model/diffusion
    options from the checkpoint's saved args.json.
    """
    # Make sure these groups exist so we can collect their keys
    add_data_options(parser)
    add_model_options(parser)
    add_diffusion_options(parser)

    args = parser.parse_args()

    # Collect the option names per group so we know what to overwrite
    args_to_overwrite = []
    for group_name in ["dataset", "model", "diffusion"]:
        args_to_overwrite += get_args_per_group_name(parser, args, group_name)

    # Load args.json saved next to the provided model checkpoint
    model_path = get_model_path_from_args()
    args_path = os.path.join(os.path.dirname(model_path), "args.json")
    assert os.path.exists(args_path), "Arguments json file was not found!"
    with open(args_path, "r") as fr:
        model_args = json.load(fr)

    # Overwrite only the collected keys with checkpoint values (if present)
    for key in args_to_overwrite:
        if key in model_args:
            setattr(args, key, model_args[key])
        else:
            print(
                f"Warning: was not able to load [{key}], using default value [{args.__dict__[key]}] instead."
            )

    return args


def get_args_per_group_name(parser: ArgumentParser, args, group_name: str):
    """
    Return the list of argument names that belong to a given argparse group.
    """
    for group in parser._action_groups:
        if group.title == group_name:
            group_dict = {a.dest: getattr(args, a.dest, None) for a in group._group_actions}
            return list(argparse.Namespace(**group_dict).__dict__.keys())
    return ValueError("group_name was not found.")


def get_model_path_from_args() -> str:
    """
    Pull only `model_path` from CLI without interfering with the main parser.
    Expects a CLI arg named 'model_path' to be present.
    """
    try:
        dummy_parser = ArgumentParser()
        dummy_parser.add_argument("model_path")
        dummy_args, _ = dummy_parser.parse_known_args()
        return dummy_args.model_path
    except Exception:
        raise ValueError("model_path argument must be specified.")


# ----------------------------
# Option groups
# ----------------------------
def add_base_options(parser: ArgumentParser):
    """General runtime knobs."""
    group = parser.add_argument_group("base")
    group.add_argument("--cuda", default=True, type=bool, help="Use CUDA if available; otherwise CPU.")
    group.add_argument("--device", default=0, type=int, help="Device id (e.g., CUDA index).")
    group.add_argument("--seed", default=10, type=int, help="Random seed.")
    group.add_argument("--batch_size", default=64, type=int, help="Batch size.")
    group.add_argument("--ext", default="", type=str, help="Optional name extension for output paths.")


def add_diffusion_options(parser: ArgumentParser):
    """Scheduler and objective for diffusion."""
    group = parser.add_argument_group("diffusion")
    group.add_argument(
        "--noise_schedule",
        default="cosine",
        choices=["linear", "cosine"],
        type=str,
        help="Noise schedule type.",
    )
    group.add_argument(
        "--diffusion_steps",
        default=50,
        type=int,
        help="Number of diffusion steps (T).",
    )
    group.add_argument("--sigma_small", default=True, type=bool, help="Use smaller sigma values.")
    group.add_argument("--predict_xstart", default=1, type=int, help="Use predict_xstart. Set 0 to disable.")


def add_model_options(parser: ArgumentParser):
    """Model architecture size and init choices."""
    group = parser.add_argument_group("model")
    group.add_argument("--layers", default=8, type=int, help="Number of transformer layers.")
    group.add_argument("--heads", default=8, type=int, help="Number of attention heads.")
    group.add_argument("--zero_init", action="store_true", help="Zero-init some layers (e.g., output).")


def add_data_options(parser: ArgumentParser):
    """Dataset paths and splits."""
    group = parser.add_argument_group("dataset")
    group.add_argument(
        "--data_dir",
        default="dataset/AMASS_20.0_fps_nh_globsmpl_corrupted_cano",
        type=str,
        help="Path to preprocessed dataset.",
    )
    group.add_argument("--data_split", default="train", type=str, help="Dataset split for training.")
    group.add_argument(
        "--normalizer_dir",
        default="dataset/meta_AMASS_20.0_fps_nh_globsmpl_corrupted_cano",
        type=str,
        help="Directory containing mean.pt and std.pt.",
    )


def add_training_options(parser: ArgumentParser):
    """Training loop, logging, EMA, and loss-related settings."""
    group = parser.add_argument_group("training")
    group.add_argument("--save_dir", required=True, type=str, help="Directory to save checkpoints and results.")
    group.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow writing into an existing save_dir.",
    )
    group.add_argument(
        "--train_platform_type",
        default="NoPlatform",
        choices=["NoPlatform", "ClearmlPlatform", "TensorboardPlatform"],
        type=str,
        help="Logging backend.",
    )
    group.add_argument("--lr", default=1e-4, type=float, help="Learning rate.")
    group.add_argument("--weight_decay", default=0.0, type=float, help="AdamW weight decay.")
    group.add_argument("--lr_anneal_steps", default=0, type=int, help="LR anneal steps.")
    group.add_argument("--log_interval", default=1_000, type=int, help="Log every N steps.")
    group.add_argument("--save_interval", default=50_000, type=int, help="Save/eval every N steps.")
    group.add_argument("--num_steps", default=1_000_000, type=int, help="Total training steps.")
    group.add_argument(
        "--eval_during_training",
        action="store_true",
        help="Run evaluation while training.",
    )
    group.add_argument(
        "--resume_checkpoint",
        default="",
        type=str,
        help="Path to model###.pt to resume from.",
    )
    group.add_argument(
        "--weighted_loss",
        action="store_true",
        help="Enable per-feature weighting (requires feature_w.pt).",
    )
    group.add_argument("--feature_w_file", default="feature_w.pt", type=str, help="Feature weight file name.")
    group.add_argument("--snr_gamma", default=0, type=float, help="SNR loss weighting gamma (0 disables).")
    group.add_argument("--l1_loss", action="store_true", help="Use L1 loss (default: L2).")
    group.add_argument("--gradient_clip", action="store_true", help="Clip grad-norm at 1.0.")
    group.add_argument(
        "--fraction",
        default=4,
        type=int,
        help="1/fraction of batch used for detection; rest for inpainting.",
    )

    # EMA
    group.add_argument(
        "--model_ema",
        action="store_true",
        help="Track EMA of model parameters.",
    )
    group.add_argument(
        "--model_ema_steps",
        type=int,
        default=10,
        help="How often to update EMA.",
    )
    group.add_argument(
        "--model_ema_decay",
        type=float,
        default=0.995,
        help="EMA decay.",
    )
    group.add_argument(
        "--model_ema_update_after",
        type=float,
        default=5000,
        help="Start EMA updates after N steps.",
    )


def add_sampling_options(parser: ArgumentParser):
    """Inference-time controls and experiment bookkeeping."""
    group = parser.add_argument_group("sampling")
    group.add_argument(
        "--testdata_dir",
        default="dataset/AMASS_20.0_fps_nh_globsmpl_corrupted_cano",
        type=str,
        help="Path to evaluation dataset.",
    )
    group.add_argument(
        "--model_path",
        required=True,
        type=str,
        help="Path to model####.pt to sample from.",
    )
    group.add_argument(
        "--output_dir",
        default="",
        type=str,
        help="Directory for results (auto-created). If empty, created next to checkpoint.",
    )
    group.add_argument("--num_samples", default=0, type=int, help="Max number of samples to process.")
    group.add_argument(
        "--skip_timesteps",
        default=0,
        type=int,
        help="Start from this diffusion step (0 = full denoise).",
    )
    group.add_argument(
        "--ts_respace",
        type=str,
        help="DDIM steps, format 'ddimN'. If unset, ancestral sampling is used.",
    )
    group.add_argument("--use_ema", action="store_true", help="Use EMA weights for inference.")
    group.add_argument("--enable_cfg", action="store_true", help="Enable classifier-free guidance.")
    group.add_argument("--guidance_param", default=2.5, type=float, help="CFG scale.")
    group.add_argument("--classifier_scale", default=0, type=float, help="Classifier guidance scale.")
    group.add_argument("--ProbDetNum", default=0, type=int, help="MC samples for detection pass.")
    group.add_argument("--ProbDetTh", default=0.5, type=float, help="Threshold for detection label.")
    group.add_argument(
        "--enable_sits",
        action="store_true",
        help="Enable soft inpainting timesteps (requires larger ProbDetNum or ensemble).",
    )
    group.add_argument("--ensemble", action="store_true", help="Use det-fix ensemble.")
    group.add_argument("--gtdet", action="store_true", help="Use ground-truth detection labels.")
    group.add_argument(
        "--collect_dataset",
        action="store_true",
        help="Collect GT motions from dataset (no fixing).",
    )


# ----------------------------
# Public entry points
# ----------------------------
def train_args():
    """
    CLI for training scripts.
    """
    parser = ArgumentParser()
    add_base_options(parser)
    add_data_options(parser)
    add_model_options(parser)
    add_diffusion_options(parser)
    add_training_options(parser)
    return parser.parse_args()


def det_args():
    """
    CLI for detection/fixing scripts.
    NOTE: dataset/model/diffusion options are overwritten from checkpoint's args.json.
    """
    parser = ArgumentParser()
    add_base_options(parser)
    add_sampling_options(parser)
    return parse_and_load_from_model(parser)