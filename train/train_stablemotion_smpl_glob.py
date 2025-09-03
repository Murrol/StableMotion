import os
import json
from utils.fixseed import fixseed
from utils.parser_util import train_args
from utils import dist_util
from train.training_loop_smpl import TrainLoop
from data_loaders.get_data import get_dataset_loader
from utils.model_util import create_model_and_diffusion
from train.train_platforms import ClearmlPlatform, TensorboardPlatform, NoPlatform  # required for the eval operation

import torch
torch.backends.cuda.enable_flash_sdp(True)   # enable FlashAttention-style SDP when available
torch.backends.cudnn.benchmark = True        # speed up convs for fixed input sizes

def main():
    args = train_args()
    fixseed(args.seed)

    # Resolve logging/metrics platform from string, e.g. "TensorboardPlatform"
    train_platform_type = eval(args.train_platform_type)
    
    # Prepare output dir (safe by default unless --overwrite)
    if args.save_dir is None:
        raise FileNotFoundError('save_dir was not specified.')
    elif os.path.exists(args.save_dir) and not args.overwrite:
        raise FileExistsError('save_dir [{}] already exists.'.format(args.save_dir))
    elif not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
        print(f"Saving to {args.save_dir}")

    # Initialize metrics platform and persist full args for reproducibility
    train_platform = train_platform_type(args.save_dir)
    train_platform.report_args(args, name='Args')
    args_path = os.path.join(args.save_dir, 'args.json')
    with open(args_path, 'w') as fw:
        json.dump(vars(args), fw, indent=4, sort_keys=True)

    # Initialize distributed / device context
    dist_util.setup_dist(args.device)

    print("creating data loader...")
    data = get_dataset_loader(
        name='globsmpl',
        batch_size=args.batch_size,
        split=args.data_split,
        data_dir=args.data_dir,
        normalizer_dir=args.normalizer_dir,
    )

    print("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(args)
    model.to(dist_util.dev())
    print('Total params: %.2fM' % (model.num_parameters() / 1_000_000.0))

    print("Training...")
    TrainLoop(args, train_platform, model, diffusion, data).run_loop()
    train_platform.close()

if __name__ == "__main__":
    main()